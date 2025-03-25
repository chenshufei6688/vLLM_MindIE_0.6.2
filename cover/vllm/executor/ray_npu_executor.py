# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of codes in this file was copied from project [vLLM Team][vllm]

import asyncio
import os
import pickle
from collections import defaultdict
from itertools import islice, repeat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import vllm.envs as envs
from vllm.executor.distributed_npu_executor import DistributedNPUExecutor, DistributedNPUExecutorAsync
from vllm.executor.ray_utils import RayWorkerWrapper, ray
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
    get_vllm_instance_id,
    make_async,
    _run_task_with_lock,
)

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

USE_RAY_COMPILED_DAG = envs.VLLM_USE_RAY_COMPILED_DAG


class RayNPUExecutor(DistributedNPUExecutor):
    uses_ray: bool = True
    def execute_model(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        all_outputs = self._run_workers(
            "execute_model",
            driver_kwargs={"execute_model_req": execute_model_req},
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG,
        )

        # Only the driver worker returns the sampling results.
        return all_outputs[0]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()

    def _init_executor(self) -> None:
        self.forward_dag: Optional["ray.dag.CompiledDAG"] = None
        # If the env var is set, it uses the Ray's compiled DAG API
        # which optimizes the control plane overhead.
        # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
        # Currently, this requires USE_RAY_SPMD_WORKER=True.
        self.use_ray_compiled_dag = envs.VLLM_USE_RAY_COMPILED_DAG
        # If the env var is set, then we do not distinguish between the
        # "driver worker" vs other workers. Also, the rank 0 worker will
        # be executed in a remote Ray worker. Currently this requires
        # USE_RAY_COMPILED_DAG=True.
        self.use_ray_spmd_worker = envs.VLLM_USE_RAY_SPMD_WORKER
        if self.use_ray_compiled_dag:
            assert self.use_ray_spmd_worker, "VLLM_USE_RAY_COMPILED_DAG=1 requires " "VLLM_USE_RAY_SPMD_WORKER=1"
        if self.use_ray_spmd_worker:
            # TODO: Support SPMD worker for non-DAG Ray executor.
            assert self.use_ray_compiled_dag, "VLLM_USE_RAY_SPMD_WORKER=1 requires " "VLLM_USE_RAY_COMPILED_DAG=1"

        assert not self.speculative_config, "Speculative decoding not yet supported for RayNPU backend."

        assert self.parallel_config.use_ray
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel NPU workers.
        self._init_workers_ray(placement_group)

        self.forward_dag = None
        if USE_RAY_COMPILED_DAG:
            self.forward_dag = self._compiled_ray_dag()

    def _get_worker_wrapper_args(self) -> Dict[str, Any]:
        (worker_module_name, worker_class_name, worker_class_fn) = self._get_worker_module_and_class()

        return dict(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            worker_class_fn=worker_class_fn,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        if self.parallel_config.tensor_parallel_size == 1:
            # For single NPU case, we use a ray worker with constrained memory.
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full NPU.
            num_gpus = 1

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(ray_remote_kwargs)

        # Create the workers.
        driver_ip = get_ip()
        worker_wrapper_kwargs = self._get_worker_wrapper_args()
        for bundle_id, _ in enumerate(placement_group.bundle_specs):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(**worker_wrapper_kwargs)

            if self.use_ray_spmd_worker:
                self.workers.append(worker)
            else:
                worker_ip = ray.get(worker.get_node_ip.remote())
                if worker_ip == driver_ip and self.driver_dummy_worker is None:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = RayWorkerWrapper(**worker_wrapper_kwargs)
                else:
                    # Else, added to the list of workers.
                    self.workers.append(worker)

        logger.debug("workers: %s", self.workers)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)
        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any NPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "NPU node."
            )

        # Get the set of NPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids", use_dummy_driver=True)

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [
            (
                {
                    "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, node_gpus[node_id])),
                    "VLLM_INSTANCE_ID": VLLM_INSTANCE_ID,
                    "VLLM_TRACE_FUNCTION": str(envs.VLLM_TRACE_FUNCTION),
                },
            )
            for (node_id, _) in worker_node_and_gpu_ids
        ]
        self._run_workers("update_environment_variables", all_args=all_args_to_update_environment_variables)

        distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
            )
            for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model", max_concurrent_workers=self.parallel_config.max_parallel_loading_workers)

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                for tp_rank in range(self.parallel_config.tensor_parallel_size):
                    # PP=2, TP=4
                    # pp_tp_workers will be [[0, 1, 2, 3], [4, 5, 6, 7]]
                    rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

    def _driver_execute_model(self, execute_model_req: Optional[ExecuteModelRequest]) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        # assert not self.use_ray_spmd_worker, (
        #     "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1")
        return self.driver_worker.execute_method("execute_model", execute_model_req)

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        - args/kwargs: All workers share the same args/kwargs
        - args/kwargs and driver_args/driver_kwargs: Driver worker has
          different args
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        if max_concurrent_workers:
            raise NotImplementedError("max_concurrent_workers is not supported yet.")

        if driver_args is None:
            driver_args = args if all_args is None else all_args[0]
        if driver_kwargs is None:
            driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

        count = len(self.workers)
        all_worker_args = repeat(args, count) if all_args is None else islice(all_args, 1, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None else islice(all_kwargs, 1, None)

        if use_ray_compiled_dag:
            # Right now, compiled DAG can only accept a single
            # input. TODO(sang): Fix it.
            assert self.forward_dag is not None
            output_channels = self.forward_dag.execute(1)
        else:
            # Start the ray workers first.
            ray_worker_outputs = [
                worker.execute_method.remote(method, *worker_args, **worker_kwargs)
                for (worker, worker_args, worker_kwargs) in zip(self.workers, all_worker_args, all_worker_kwargs)
            ]

        # Start the driver worker after all the ray workers.
        if not use_dummy_driver:
            driver_worker_output = self.driver_worker.execute_method(method, *driver_args, **driver_kwargs)
        else:
            assert self.driver_dummy_worker is not None
            driver_worker_output = ray.get(
                self.driver_dummy_worker.execute_method.remote(method, *driver_args, **driver_kwargs)
            )
        # Get the results of the ray workers.
        if self.workers:
            if use_ray_compiled_dag:
                try:
                    ray_worker_outputs = [pickle.loads(chan.begin_read()) for chan in output_channels]
                finally:
                    # Has to call end_read in order to reuse the DAG.
                    for chan in output_channels:
                        chan.end_read()
            else:
                ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        ray.get(parallel_worker_tasks)

    def _compiled_ray_dag(self):
        assert self.parallel_config.worker_use_ray
        self._check_ray_adag_installation()
        from ray.dag import InputNode, MultiOutputNode

        # Right now, compiled DAG requires at least 1 arg. We send
        # a dummy value for now. It will be fixed soon.
        with InputNode() as input_data:
            forward_dag = MultiOutputNode(
                [
                    worker.execute_model_compiled_dag_remote.bind(input_data)  # type: ignore[attr-defined]
                    for worker in self.workers
                ]
            )
        return forward_dag.experimental_compile()

    def _check_if_any_actor_is_dead(self):
        if not self.workers:
            return

        dead_actors = []
        for actor in self.workers:
            actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
            if actor_state["State"] == "DEAD":
                dead_actors.append(actor)
        if dead_actors:
            raise RuntimeError("At least one Worker is dead. " f"Dead Workers: {dead_actors}. ")


class RayNPUExecutorAsync(RayNPUExecutor, DistributedNPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_executor = make_async(self.driver_worker.execute_method)
        if not self.use_ray_compiled_dag:
            self.driver_exec_method = make_async(self.driver_worker.execute_method)

    def __del__(self):
        self.shutdown()

    async def _driver_execute_model_async(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        assert not self.use_ray_spmd_worker, "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1"
        if not self.tp_driver_workers:
            return await self.driver_exec_method("execute_model", execute_model_req)
        if self.pp_locks is None:
            # This locks each pipeline parallel stage so multiple virtual
            # engines can't execute on the same stage at the same time
            # We create the locks here to avoid creating them in the constructor
            # which uses a different asyncio loop.
            self.pp_locks = [asyncio.Lock() for _ in range(self.parallel_config.pipeline_parallel_size)]

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(self.driver_exec_method, self.pp_locks[0], "execute_model", execute_model_req)
            )
        ]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers, start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(
                        driver_worker.execute_method.remote, self.pp_locks[pp_rank], "execute_model", execute_model_req
                    )
                )
            )

        results = await asyncio.gather(*tasks)

        # Only the last PP stage has the final results.
        return results[-1]

    async def _start_worker_execution_loop(self):
        assert not self.use_ray_spmd_worker, "worker loop is disabled for VLLM_USE_RAY_SPMD_WORKER=1"
        coros = [worker.execute_method.remote("start_worker_execution_loop") for worker in self.non_driver_workers]
        return await asyncio.gather(*coros)