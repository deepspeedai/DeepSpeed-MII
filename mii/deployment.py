# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import string
import os
import mii

from deepspeed.launcher.runner import fetch_hostfile

from .constants import DeploymentType, MII_MODEL_PATH_DEFAULT, MODEL_PROVIDER_MAP
from .utils import logger, get_task_name, get_provider_name
from .models.score import create_score_file
from .models import load_models
from .config import ReplicaConfig, LoadBalancerConfig, DeploymentConfig


def deploy(task=None,
           model=None,
           deployment_name=None,
           enable_deepspeed=True,
           enable_zero=False,
           ds_config=None,
           mii_config={},
           deployment_tag=None,
           deployments=[],
           deployment_type=DeploymentType.LOCAL,
           model_path=None,
           version=1):
    """Deploy a task using specified model. For usage examples see:

        mii/examples/local/text-generation-example.py


    Arguments:
        task: Name of the machine learning task to be deployed.Currently MII supports the following list of tasks
            ``['text-generation', 'text-classification', 'question-answering', 'fill-mask', 'token-classification', 'conversational', 'text-to-image']``

        model: Name of a supported model for the task. Models in MII are sourced from multiple open-source projects
            such as Huggingface Transformer, FairSeq, EluetherAI etc. For the list of supported models for each task, please
            see here [TODO].

        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

        deployment_type: One of the ``enum mii.DeploymentTypes: [LOCAL]``.
            *``LOCAL`` uses a grpc server to create a local deployment, and query the model must be done by creating a query handle using
              `mii.mii_query_handle` and posting queries using ``mii_request_handle.query`` API,

        model_path: Optional: In LOCAL deployments this is the local path where model checkpoints are available. In AML deployments this
            is an optional relative path with AZURE_MODEL_DIR for the deployment.

        enable_deepspeed: Optional: Defaults to True. Use this flag to enable or disable DeepSpeed-Inference optimizations

        enable_zero: Optional: Defaults to False. Use this flag to enable or disable DeepSpeed-ZeRO inference

        ds_config: Optional: Defaults to None. Use this to specify the DeepSpeed configuration when enabling DeepSpeed-ZeRO inference

        force_register_model: Optional: Defaults to False. For AML deployments, set it to True if you want to re-register your model
            with the same ``aml_model_tags`` using checkpoints from ``model_path``.

        mii_config: Optional: Dictionary specifying optimization and deployment configurations that should override defaults in ``mii.config.MIIConfig``.
            mii_config is future looking to support extensions in optimization strategies supported by DeepSpeed Inference as we extend mii.
            As of now, it can be used to set tensor-slicing degree using 'tensor_parallel' and port number for deployment using 'port_number'.

        version: Optional: Version to be set for AML deployment, useful if you want to deploy the same model with different settings.
    Returns:
        If deployment_type is `LOCAL`, returns just the name of the deployment that can be used to create a query handle using `mii.mii_query_handle(deployment_name)`

    """
    if not mii_config:
        mii_config = mii.config.MIIConfig(**{})

    if model_path is None and deployment_type == DeploymentType.LOCAL:
        model_path = MII_MODEL_PATH_DEFAULT
    elif model_path is None and deployment_type == DeploymentType.AML:
        model_path = "model"

    deployment_tag, deployments = validate_deployment(task=task,
                                                      model=model,
                                                      deployment_name=deployment_name,
                                                      enable_deepspeed=enable_deepspeed,
                                                      enable_zero=enable_zero,
                                                      ds_config=ds_config,
                                                      mii_config=mii_config,
                                                      deployment_tag=deployment_tag,
                                                      deployments=deployments,
                                                      deployment_type=deployment_type,
                                                      model_path=model_path,
                                                      version=version)

    if not deployments:  #Empty deployment
        create_score_file(deployment_tag=deployment_tag,
                          deployment_type=deployment_type,
                          deployments=None,
                          model_path=model_path,
                          port_map=None,
                          lb_config=None)
        print(f"Starting empty deployment, deployment_tag -> {deployment_tag}")
        return None

    # parse and validate mii config
    for deployment in deployments:
        #mii_config = getattr(deployment, mii.constants.MII_CONFIGS_KEY)
        if getattr(deployment, mii.constants.ENABLE_DEEPSPEED_ZERO_KEY):
            if getattr(deployment,
                       mii.constants.DEEPSPEED_CONFIG_KEY).get("fp16",
                                                               {}).get("enabled",
                                                                       False):
                assert (deployment.dtype == torch.half), "MII Config Error: MII dtype and ZeRO dtype must match"
            else:
                assert (deployment.dtype == torch.float), "MII Config Error: MII dtype and ZeRO dtype must match"
        assert not (enable_deepspeed and enable_zero), "MII Config Error: DeepSpeed and ZeRO cannot both be enabled, select only one"

    # aml only allows certain characters for deployment names
    if deployment_type == DeploymentType.AML:
        assert len(deployments == 1), "mii does not currently support empty/multi-model deployment on AML"
        allowed_chars = set(string.ascii_lowercase + string.ascii_uppercase +
                            string.digits + '-')
        assert set(deployment_name) <= allowed_chars, "AML deployment names can only contain a-z, A-Z, 0-9, and '-'"

        if not mii_config.skip_model_check:
            mii.utils.check_if_task_and_model_is_valid(
                getattr(deployment,
                        mii.constants.TASK_NAME_KEY),
                getattr(deployment,
                        mii.constants.MODEL_NAME_KEY))
            if enable_deepspeed:
                mii.utils.check_if_task_and_model_is_supported(
                    deployment.task,
                    deployment.model)

        if enable_deepspeed:
            logger.info(
                f"************* MII is using DeepSpeed Optimizations to accelerate your model: {deployment.model} *************"
            )
        else:
            logger.info(
                f"************* DeepSpeed Optimizations not enabled. Please use enable_deepspeed to get better performance for: {deployment.model} *************"
            )

    deps = {deployment.deployment_name: deployment for deployment in deployments}
    # In local deployments use default path if no model path set

    # add fields for replica deployment
    port_map = {}
    lb_config, port_map = allocate_processes(deps, port_map, mii_config)

    if deployment_type != DeploymentType.NON_PERSISTENT:
        create_score_file(deployment_tag=deployment_tag,
                          deployment_type=deployment_type,
                          deployments=deps,
                          model_path=model_path,
                          port_map=port_map,
                          lb_config=lb_config,
                          mii_configs=mii_config)

    if deployment_type == DeploymentType.AML:
        _deploy_aml(deployment_tag=deployment_tag, model_name=model, version=version)
    elif deployment_type == DeploymentType.LOCAL:
        return _deploy_local(deployment_tag, model_path=model_path)
    elif deployment_type == DeploymentType.NON_PERSISTENT:
        assert int(os.getenv('WORLD_SIZE', '1')) == mii_config.tensor_parallel, "World Size does not equal number of tensors. When using non-persistent deployment type, please launch with `deepspeed --num_gpus <tensor_parallel>`"
        provider = MODEL_PROVIDER_MAP[get_provider_name(model, task)]
        mii.non_persistent_models[deployment_name] = (load_models(
            task,
            model,
            model_path,
            enable_deepspeed,
            enable_zero,
            provider,
            deployment),
                                                      task)
    else:
        raise Exception(f"Unknown deployment type: {deployment_type}")


def allocate_processes(deployments, port_map, mii_config):
    replica_configs = []
    port_offset = 1
    for deployment in deployments.values():
        #mii_config = getattr(deployment, mii.constants.MII_CONFIGS_KEY)
        replica_pool = _allocate_processes(
            deployment.hostfile,
            deployment.tensor_parallel,
            deployment.replica_num,
            getattr(deployment,
                    mii.constants.GPU_INDEX_KEY))

        for i, (hostname, gpu_indices) in enumerate(replica_pool):
            # Reserver port for a LB proxy when replication is enabled
            if hostname not in port_map:
                port_map[hostname] = set()
            base_port = mii_config.port_number + i * deployment.tensor_parallel + port_offset
            if base_port in port_map[hostname]:
                base_port = max(port_map[hostname]) + 1
            tensor_parallel_ports = list(
                range(base_port,
                      base_port + deployment.tensor_parallel))
            for i in range(base_port, base_port + deployment.tensor_parallel):
                port_map[hostname].add(i)
            torch_dist_port = mii_config.torch_dist_port + i
            replica_configs.append(
                ReplicaConfig(
                    task=get_task_name(getattr(deployment,
                                               mii.constants.TASK_NAME_KEY)),
                    deployment_name=(getattr(deployment,
                                             mii.constants.DEPLOYMENT_NAME_KEY)),
                    hostname=hostname,
                    tensor_parallel_ports=tensor_parallel_ports,
                    torch_dist_port=torch_dist_port,
                    gpu_indices=gpu_indices))
    lb_config = LoadBalancerConfig(port=mii_config.port_number,
                                   replica_configs=replica_configs)
    return lb_config, port_map


def validate_deployment(task=None,
                        model=None,
                        deployment_name=None,
                        enable_deepspeed=True,
                        enable_zero=False,
                        ds_config=None,
                        mii_config={},
                        deployment_tag=None,
                        deployments=[],
                        deployment_type=DeploymentType.LOCAL,
                        model_path=None,
                        version=1):

    if deployments and any((model, task, deployment_name)):
        assert False, "Do not input deployments and model/task/deployment_name at the same time"

    elif deployments:
        assert deployment_tag, "deployment_tag must be set to for multiple models"
        return deployment_tag, deployments

    elif not any((model, task, deployment_name)):
        assert deployment_tag, "deployment_tag must be set for an empty deployment"
        create_score_file(deployment_tag=deployment_tag,
                          deployment_type=deployment_type,
                          deployments=None,
                          model_path=model_path,
                          mii_configs={},
                          port_map=None,
                          lb_config=None)
        return deployment_tag, None

    assert all((model, task, deployment_name)), "model, task, and deployment_name must be set for a single model"
    deployments = [
        DeploymentConfig(DEPLOYMENT_NAME_KEY=deployment_name,
                         TASK_NAME_KEY=task,
                         MODEL_NAME_KEY=model,
                         ENABLE_DEEPSPEED_KEY=enable_deepspeed,
                         ENABLE_DEEPSPEED_ZERO_KEY=enable_zero,
                         GPU_INDEX_KEY=None,
                         MII_CONFIGS_KEY=mii.config.MIIConfig(**mii_config),
                         DEEPSPEED_CONFIG_KEY=ds_config,
                         VERSION_KEY=version)
    ]
    if deployment_tag is None:
        deployment_tag = deployment_name
    return deployment_tag, deployments


def _deploy_local(deployment_tag, model_path):
    mii.utils.import_score_file(deployment_tag).init()


def _deploy_aml(deployment_tag, model_name, version):
    acr_name = mii.aml_related.utils.get_acr_name()
    mii.aml_related.utils.generate_aml_scripts(acr_name=acr_name,
                                               deployment_name=deployment_tag,
                                               model_name=model_name,
                                               version=version)
    print(
        f"AML deployment assets at {mii.aml_related.utils.aml_output_path(deployment_tag)}"
    )
    print("Please run 'deploy.sh' to bring your deployment online")


def _allocate_processes(hostfile_path,
                        tensor_parallel,
                        num_replicas,
                        gpu_index_map=None):
    resource_pool = fetch_hostfile(hostfile_path)
    assert resource_pool is not None and len(
        resource_pool) > 0, f'No hosts found in {hostfile_path}'

    replica_pool = []

    if gpu_index_map is not None:
        for host in gpu_index_map:
            assert host in resource_pool, f"Host: {host} was not found"
            assert resource_pool[host] >= tensor_parallel, f"Host {host} has {resource_pool[host]} slot(s), but {tensor_parallel} slot(s) are required"
        for host in gpu_index_map:
            replica_pool.append((host, gpu_index_map[host]))
        return replica_pool

    allocated_num = 0
    for host, slots in resource_pool.items():
        available_on_host = slots
        while available_on_host >= tensor_parallel:
            if allocated_num >= num_replicas:
                break
            if slots < tensor_parallel:
                raise ValueError(
                    f'Host {host} has {slots} slot(s), but {tensor_parallel} slot(s) are required'
                )

            allocated_num_on_host = slots - available_on_host
            replica_pool.append(
                (host,
                 [
                     i for i in range(allocated_num_on_host,
                                      allocated_num_on_host + tensor_parallel)
                 ]))
            allocated_num += 1

            available_on_host -= tensor_parallel

    if allocated_num < num_replicas:
        raise ValueError(
            f'No sufficient GPUs for {num_replicas} replica(s), only {allocated_num} replica(s) can be deployed'
        )

    return replica_pool
