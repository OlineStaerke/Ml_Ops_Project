from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails

# Create a Python environment for the experiment
torch_env = Environment("transformer-env")

# Ensure the required packages are installed (we need pip, torch and Azure ML defaults)
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults', 'torch', 'transformers'])
torch_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='./',
                                script='src/models/train_model.py',
                                environment=torch_env) 

# Submit the experiment run
experiment_name = 'first-training-transfomer'
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

# Show the running experiment run in the notebook widget
#RunDetails(run).show(verbose = False)

# Block until the experiment run has completed
run.wait_for_completion()