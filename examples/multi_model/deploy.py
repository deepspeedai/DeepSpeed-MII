# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

gpu_index_map1 = {'master': [0]}
gpu_index_map2 = {'master': [1]}
gpu_index_map3 = {'master': [0, 1]}

deployments = []

mii_configs1 = {"tensor_parallel": 2, "dtype": "fp16"}
mii_configs2 = {"tensor_parallel": 1}

name = "bigscience/bloom-560m"
deployments.append(
    mii.DeploymentConfig(task='text-generation',
                         model=name,
                         deployment_name=name + "_deployment",
                         GPU_index_map=gpu_index_map3,
                         mii_configs=mii.config.MIIConfig(**mii_configs1)))

# gpt2
name = "microsoft/DialogRPT-human-vs-rand"
deployments.append(
    mii.DeploymentConfig(task='text-classification',
                         model=name,
                         deployment_name=name + "_deployment",
                         GPU_index_map=gpu_index_map2))

name = "microsoft/DialoGPT-large"
deployments.append(
    mii.DeploymentConfig(task='conversational',
                         model=name,
                         deployment_name=name + "_deployment",
                         GPU_index_map=gpu_index_map1,
                         mii_configs=mii.config.MIIConfig(**mii_configs2)))

name = "deepset/roberta-large-squad2"
deployments.append(
    mii.DeploymentConfig(task="question-answering",
                         model=name,
                         deployment_name=name + "-qa-deployment",
                         GPU_index_map=gpu_index_map2))

mii.deploy(deployment_tag="multi_models", deployments=deployments)
