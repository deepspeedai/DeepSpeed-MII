import mii

mii_config = {"dtype": "fp16"}

name = "distilgpt2"
mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           mii_configs=mii_config,
           local_model_path=".cache/models/" + name)
