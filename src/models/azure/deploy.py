from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run, Model
from azureml.core.webservice import LocalWebservice
from azureml.core import Environment
from azureml.core.model import InferenceConfig

from azureml.core.conda_dependencies import CondaDependencies 

ws = Workspace.from_config()

#load model
model = ws.models['yesno_model']

#Configure port
deployment_config = LocalWebservice.deploy_configuration(port=6789)

#Configure environment
env = Environment(name="PythonEnv")

python_packages = ['transformers', 'joblib','torch']
for package in python_packages:
    env.python.conda_dependencies.add_pip_package(package)

dummy_inference_config = InferenceConfig(
    environment=env,
    source_directory=".",
    entry_script="api.py",
)

#Deploy model
service = Model.deploy(
    ws,
    "myservice",
    [model],
    dummy_inference_config,
    deployment_config,
    overwrite=True,
)

service.wait_for_deployment(show_output=True)
