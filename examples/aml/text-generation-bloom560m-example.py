import mii

mii_configs = {
    "tensor_parallel": 1,
    "dtype": "fp16",
    "aml_model_path": "models/bloom-560m"
}
mii.deploy(task='text-generation',
           model="bigscience/bloom-560m",
           deployment_name="bloom560m_deployment",
           deployment_type=mii.constants.DeploymentType.AML,
           mii_config=mii_configs)
