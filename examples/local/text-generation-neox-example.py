import mii

mii_configs = {"tensor_parallel": 4, "port_number": 50050, "dtype": "fp16"}

name = "gpt-neox"
mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path="/data/20b",
           mii_configs=mii_configs)
