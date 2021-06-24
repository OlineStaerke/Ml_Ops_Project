### ML Operations Project at DTU Compute.

S164472 Helena Hansen\
S165352 Matthias Adamsen\
S174398 Natasha Klingenbrunn \
S174388 Oline Stærke \
S174450 Simon Larsen

==============================

This model is inspired by Vincent Micheli's Article [Deep Learning has (almost) all the answers: Yes/No Question Answering with Transformers](https://medium.com/illuin/deep-learning-has-almost-all-the-answers-yes-no-question-answering-with-transformers-223bebb70189)

About this project:
- BoolQ is a reading comprehension dataset composed of a question, a passage and a yes/no answer. The goal is to answer the yes/no question based on the context of the passage.
- The RoBERTa model is built on top of the BERT model for NLP tasks. RoBERTa differs from BERT in terms of dynamic masking, larger batch sizes and longer pretraining, amongst other changes. This improved model has demonstrated better performance than BERT on many key NLP tasks, such as the GLUE Benchmark.
- We will us RoBERTa on the BoolQ dataset with a robust full-scale Deep Learning Framework, following good coding practices as per the MLOps coursework. 

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# How to use our model in Azure
Azure Machine Learning can be used for any kind of machine learning. You can do training, validation and deployment of any machine learning model in Azure. 
This has all been added to Azure_branch and only works from here.

-  Training in the Cloud
    - 1: Insert credentials in config.json (Create a workspace before in AzureMl)
    - 2: Create a compute target and an environment (I created a Python env containing the env.txt script)
    - 3: run src/data/upload-data.py // This will upload the data to Azure as json files 
    - 4: Create a compute target (I used coompute clusters under compute in Azure Ml)
    - 5: run src/models/azure/train_azure.py - Remember to update the name of your compute Target and Environment // This will start an expriment run

- Deploying to the cloud
    - 6: After the training run is complete, find the runId and insert that into src/models/azure/register_model.py
    - 7: Now run deploy.py, and insert your own environment name. 
    - 8: You now have a running model deployed at port 6789
    - 9: Run this from your Terminal to test: curl -X POST -d '{"this":"is a test"}' -H "Content-Type: application/json" http://localhost:6789/score

## OPTUNA 
Optuna is used to do opimization of hyperparameters. 
A method is added to train_model.py, where variable OPTUNA needs to be set to TRUE, if wanting to do hyperparameter optimization.
If OPTUNA == False, Hydra is used instead to load the hyperparameters.


## Weights & Biases
Weights and biases, or wandb for short, is a logging tool to track the progress of model training. We have used it to track individually the performance of one batch of hyperparameters, as well as a hyperparameter tuning study using Optuna. Wandb has an inbuilt hyperparameter sweep tool, but we focused on Optuna before deciding wandb was the logging tool we wanted to use. Thus, we had to be creative in how to write the optuna study so that the hyperparameter plot was created properly.

## Data Drifting
Data Drifting is one of the main reasons model accuracy decreases over time. Therefore, it is good practice to implement a drift detection method to monitor the data . Here, the TorchDrift implementation is added to the model.py but not used, due to time constraints. Drift Detection is measured in terms of score and p-value when using the TrochDrift.   

## Continuous Integration
The project is covered by a python continuous integration building framework, provided by GitHub. Additionally, Unit Tests making sure that data is properly loaded as expected and training is done properly.

## Meme

![jphwuu0wk8w61](https://user-images.githubusercontent.com/49098682/123227871-4a069500-d4d5-11eb-9a24-ffd38c87a7da.jpg)
