from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run
from azureml.core.resource_configuration import ResourceConfiguration
ws = Workspace.from_config()

#Input the run ID here:
run = Run.get(ws,"day2_1623924304_a5fde560")

model = run.register_model(model_name='yesno_model',
                           model_path='outputs/yesno_model.pkl',
                           resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=1))
