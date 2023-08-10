# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import mii
import pprint
from mii.utils import logger
from mii.constants import DeploymentType


def create_score_file(deployment_tag,
                      deployment_type,
                      deployments,
                      model_path,
                      port_map,
                      lb_config,
                      deployed=False):

    config_dict = {}
    config_dict[mii.constants.MODEL_PATH_KEY] = model_path
    config_dict[mii.constants.DEPLOYMENT_TAG_KEY] = deployment_tag
    config_dict[mii.constants.DEPLOYED_KEY] = deployed
    config_dict[mii.constants.DEPLOYMENTS_KEY] = {}
    if port_map is not None:
        config_dict[mii.constants.PORT_MAP_KEY] = port_map

    if deployments is not None:
        for deployment in deployments.values():
            deployment_config = {
                mii.constants.DEPLOYMENT_NAME_KEY:
                getattr(deployment,
                        mii.constants.DEPLOYMENT_NAME_KEY),
                mii.constants.TASK_NAME_KEY:
                mii.utils.get_task_name(getattr(deployment,
                                                mii.constants.TASK_NAME_KEY)),
                mii.constants.MODEL_NAME_KEY:
                getattr(deployment,
                        mii.constants.MODEL_NAME_KEY),
                mii.constants.ENABLE_DEEPSPEED_KEY:
                getattr(deployment,
                        mii.constants.ENABLE_DEEPSPEED_KEY),
                mii.constants.MII_CONFIGS_KEY:
                getattr(deployment,
                        mii.constants.MII_CONFIGS_KEY).dict(),
                mii.constants.ENABLE_DEEPSPEED_ZERO_KEY:
                getattr(deployment,
                        mii.constants.ENABLE_DEEPSPEED_ZERO_KEY),
                mii.constants.DEEPSPEED_CONFIG_KEY:
                getattr(deployment,
                        mii.constants.DEEPSPEED_CONFIG_KEY),
                mii.constants.GPU_INDEX_KEY:
                getattr(deployment,
                        mii.constants.GPU_INDEX_KEY)
            }
            config_dict[mii.constants.DEPLOYMENTS_KEY][
                deployment.deployment_name] = deployment_config

    if lb_config is not None:
        config_dict[mii.constants.LOAD_BALANCER_CONFIG_KEY] = lb_config

    if len(mii.__path__) > 1:
        logger.warning(
            f"Detected mii path as multiple sources: {mii.__path__}, might cause unknown behavior"
        )

    with open(os.path.join(mii.__path__[0],
                           "models/score/score_template.py"),
              "r") as fd:
        score_src = fd.read()

    # update score file w. global config dict
    source_with_config = f"{score_src}\n"
    source_with_config += f"configs = {pprint.pformat(config_dict, indent=4)}"

    with open(generated_score_path(deployment_tag, deployment_type), "w") as fd:
        fd.write(source_with_config)
        fd.write("\n")


def generated_score_path(deployment_tag, deployment_type):
    if deployment_type == DeploymentType.LOCAL:
        score_path = os.path.join(mii.utils.mii_cache_path(), deployment_tag)
    elif deployment_type == DeploymentType.AML:
        score_path = os.path.join(mii.aml_related.utils.aml_output_path(deployment_tag),
                                  "code")
    if not os.path.isdir(score_path):
        os.makedirs(score_path)
    return os.path.join(score_path, "score.py")
