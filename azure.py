from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1')
    config = ScriptRunConfig(source_directory='./src/models',
                             script='train_model.py',
                             compute_target='root')

    # use curated pytorch environment 
    env = ws.environments['AzureML-PyTorch-1.6-CPU']
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)