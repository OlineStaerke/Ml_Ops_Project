from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1')
    config = ScriptRunConfig(source_directory='.',
                             script='train_model.py',
                             compute_target='ClusterGPUHigh',
                             environment='EnvironmentPython')

    # use curated pytorch environment 
    # env = ws.environments['environmentPython']
    # config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)