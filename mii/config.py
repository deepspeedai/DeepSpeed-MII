# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import os
import string
from typing import List, Optional, Dict, Any
from pydantic import validator, root_validator, BaseModel
import mii
from mii.constants import DeploymentType, TaskType, MII_MODEL_PATH_DEFAULT

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.inference.config import DtypeEnum
from deepspeed.launcher.runner import DLTS_HOSTFILE, fetch_hostfile


class ReplicaConfig(DeepSpeedConfigModel):
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []


class DeploymentConfig(DeepSpeedConfigModel):
    # Deployment configs
    deployment_name: str
    load_with_sys_mem: bool = False
    meta_tensor: bool = False
    hf_auth_token: Optional[str] = None
    deploy_rank: Optional[List[int]] = None
    torch_dist_port: int = 29500
    replica_num: int = 1
    replica_configs: List[ReplicaConfig] = []
    profile_model_time: bool = False
    skip_model_check: bool = False
    trust_remote_code: bool = False

    # Model configs
    model: str
    task: TaskType
    dtype: DtypeEnum = DtypeEnum.fp32
    model_path: str = ""
    checkpoint_dict: Optional[Dict[str, Any]] = None
    max_tokens: int = 1024
    GPU_index_map: dict = None

    # Performance configs
    enable_deepspeed: bool = True
    enable_zero: bool = False
    ds_config: Dict[str, Any] = {}
    tensor_parallel: int = 1
    enable_cuda_graph: bool = False
    replace_with_kernel_inject: bool = True

    class Config:
        json_encoders = {torch.dtype: lambda x: str(x)}

    @validator("checkpoint_dict")
    def checkpoint_dict_valid(cls, field_value, values):
        if field_value is None:
            return field_value
        if field_value.get("base_dir", ""):
            raise ValueError(
                "please unset 'base_dir' it will be set w.r.t. the deployment 'model_path'"
            )
        for k in ["checkpoints", "parallelization", "version", "type"]:
            if not field_value.get(k, ""):
                raise ValueError(f"Missing key={k} in checkpoint_dict")
        return field_value

    @validator("deploy_rank", pre=True)
    def deploy_rank_to_list(cls, field_value, values):
        if field_value and not isinstance(field_value, list):
            field_value = [field_value]
        return field_value

    @root_validator
    def zero_or_meta(cls, values):
        if values.get("enable_zero"):
            assert not values.get(
                "meta_tensor"
            ), "ZeRO-Inference does not support meta tensors."
        return values

    @root_validator
    def bloom_model_valid(cls, values):
        if "bigscience/bloom" in values.get("model"):
            # TODO: SHould be albe to use DtypeEnum here
            assert values.get("dtype") in [
                torch.int8,
                torch.float16,
            ], "Bloom models only support fp16/int8."
            assert (not values.get(
                "enable_cuda_graph"
            )), "Bloom models do not support CUDA Graph."
        return values

    @root_validator
    def deploy_rank_valid(cls, values):
        tensor_parallel = values.get("tensor_parallel")
        deploy_rank = values.get("deploy_rank")

        # if deploy rank is not given, default to align with TP value
        if deploy_rank is None:
            deploy_rank = list(range(tensor_parallel))

        # number of ranks provided must be equal to TP size, DP is handled outside MII currently
        assert tensor_parallel == len(
            deploy_rank
        ), f"{len(deploy_rank)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {tensor_parallel}"

        values["deploy_rank"] = deploy_rank
        return values

    @root_validator
    def set_model_path(cls, values):
        model_path = values.get("model_path")
        if not model_path:
            if values.get("deployment_type") == DeploymentType.AML:
                model_path = "model"
            else:
                model_path = MII_MODEL_PATH_DEFAULT
        aml_model_dir = os.environ.get("AZUREML_MODEL_DIR", None)
        if aml_model_dir:
            assert os.path.isabs(
                aml_model_dir
            ), "AZUREML_MODEL_DIR={aml_model_dir} must be an absolute path."
            assert not os.path.isabs(
                model_path
            ), f"model_path={model_path} must be relative to append w/ AML path."
            model_path = os.path.join(aml_model_dir, model_path)

        values["model_path"] = model_path
        return values

    @root_validator
    def validate_model_and_task(cls, values):
        task = values.get("task")
        model = values.get("model")
        if not values.get("skip_model_check"):
            mii.utils.check_if_task_and_model_is_valid(task, model)
            if values.get("enable_deepspeed"):
                mii.utils.check_if_task_and_model_is_supported(task, model)
        # Skip any future checks
        values["skip_model_check"] = True
        return values

    @root_validator
    def meta_tensor_or_sys_mem(cls, values):
        if values.get("meta_tensor") and values.get("load_with_sys_mem"):
            raise ValueError(
                "`meta_tensor` and `load_with_sys_mem` cannot be active at the same time."
            )
        return values

    @root_validator
    def zero_dtype_valid(cls, values):
        if values.get("enable_zero"):
            if values.get("ds_config").get("fp16", {}).get("enabled", False):
                # TODO: We should be able to use DtypeEnum instead of torch.float
                assert (
                    values.get("dtype") == torch.float16
                ), "ZeRO FP16 enabled, `dtype` must be set to `torch.float16`"
            else:
                assert (
                    values.get("dtype") == torch.float32
                ), "ZeRO FP16 disabled, `dtype` must be set to `torch.float32`"
        return values

    @root_validator
    def deepspeed_or_zero(cls, values):
        assert not (
            values.get("enable_deepspeed") and values.get("enable_zero")
        ), "DeepSpeed and ZeRO cannot both be enabled, select only one"
        return values


"""
    @root_validator
    def index_map_valid(cls, values):
        if values.get("GPU_index_map"):
            for host in values.get("GPU_index_map"):
                assert host in resource_pool, f"Host: {host} was not found"
                assert resource_pool[host] >= tensor_parallel, f"Host {host} has {resource_pool[host]} slot(s), but {tensor_parallel} slot(s) are required"
        return values
"""


class MIIConfig(DeepSpeedConfigModel):
    deployment_configs: dict[str, DeploymentConfig] = None
    deployment_type: DeploymentType = DeploymentType.LOCAL
    deployment_tag: str = None
    hf_auth_token: Optional[str] = None
    port_number: int = 50050
    enable_restful_api: bool = False
    restful_api_port: int = 51080
    hostfile: str = DLTS_HOSTFILE
    version: int = 1
    port_map: dict = {}

    @root_validator(skip_on_failure=True)
    def propagate_hf_auth(cls, values):
        # This validator is for when we support multiple models in a deployment
        hf_auth_token = values.get("hf_auth_token")
        deployment_config_list = values.get("deployment_configs")
        print(deployment_config_list)
        for deployment_config in deployment_config_list.values():
            if not deployment_config.hf_auth_token:
                deployment_config.hf_auth_token = hf_auth_token
        return values

    @root_validator(skip_on_failure=True)
    def AML_name_valid(cls, values):
        if values.get("deployment_type") == DeploymentType.AML:
            allowed_chars = set(string.ascii_lowercase + string.ascii_uppercaes +
                                string.digits + "-")
            assert (
                set(values.get("deployment_configs").deployment_name) <= allowed_chars
            ), "AML deployment names can only contain a-z, A-Z, 0-9, and '-'."
        return values

    @root_validator()
    def generate_replica_configs(cls, values):
        port_map = values.get("port_map")
        hostfile = values.get("hostfile")
        port_number = values.get("port_number")
        port_offset = 1
        for deployment_config in values.get("deployment_configs").values():

            replica_configs = deployment_config.replica_configs
            replica_num = deployment_config.replica_num
            if replica_configs:
                assert len(replica_configs) == replica_num
                return values

            torch_dist_port = deployment_config.torch_dist_port
            tensor_parallel = deployment_config.tensor_parallel
            replica_num = deployment_config.replica_num
            GPU_index_map = deployment_config.GPU_index_map
            replica_pool, GPU_index_map = _allocate_processes(hostfile, tensor_parallel, replica_num, GPU_index_map)
            deployment_config.GPU_index_map = GPU_index_map
            replica_configs = []
            print(replica_pool)
            for i, (hostname, gpu_indices) in enumerate(replica_pool):
                # Reserver port for a LB proxy when replication is enabled
                if hostname not in port_map:
                    port_map[hostname] = set()
                base_port = port_number + i * tensor_parallel + port_offset
                if base_port in port_map[hostname]:
                    base_port = max(port_map[hostname]) + 1
                tensor_parallel_ports = list(
                    range(base_port,
                          base_port + tensor_parallel))
                for i in range(base_port, base_port + tensor_parallel):
                    port_map[hostname].add(i)
                replica_torch_dist_port = torch_dist_port + (100 * i)
                replica_configs.append(
                    ReplicaConfig(
                        hostname=hostname,
                        tensor_parallel_ports=tensor_parallel_ports,
                        torch_dist_port=replica_torch_dist_port,
                        gpu_indices=gpu_indices,
                    ))

            deployment_config.replica_configs = replica_configs
        return values


def _allocate_processes(hostfile_path, tensor_parallel, replica_num, GPU_index_map=None):
    resource_pool = fetch_hostfile(hostfile_path)
    assert (
        resource_pool is not None and len(resource_pool) > 0
    ), f"No hosts found in {hostfile_path}"

    replica_pool = []

    if GPU_index_map is not None:
        for host in GPU_index_map:
            assert host in resource_pool, f"Host: {host} was not found"
            assert resource_pool[host] >= tensor_parallel, f"Host {host} has {resource_pool[host]} slot(s), but {tensor_parallel} slot(s) are required"
        for host in GPU_index_map:
            replica_pool.append((host, GPU_index_map[host]))
        return replica_pool, GPU_index_map

    allocated_num = 0
    for host, slots in resource_pool.items():
        available_on_host = slots
        while available_on_host >= tensor_parallel:
            if allocated_num >= replica_num:
                break
            if slots < tensor_parallel:
                raise ValueError(
                    f"Host {host} has {slots} slot(s), but {tensor_parallel} slot(s) are required"
                )

            allocated_num_on_host = slots - available_on_host
            replica_pool.append((
                host,
                [
                    i for i in range(
                        allocated_num_on_host,
                        allocated_num_on_host + tensor_parallel,
                    )
                ],
            ))
            allocated_num += 1

            available_on_host -= tensor_parallel

    if allocated_num < replica_num:
        raise ValueError(
            f"Not sufficient GPUs for {replica_num} replica(s), only {allocated_num} replica(s) can be deployed"
        )

    return replica_pool, GPU_index_map
