from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day3')
    config = ScriptRunConfig(source_directory='../../../',
                             script='src/models/train_model.py',
                             compute_target='ClusterGPUHigh'
            )
    # use curated pytorch environment 
    env = ws.environments['PythonEnv']
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)