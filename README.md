ML
==============================

ML Operations Project at DTU Compute.

This model is inspired by Vincent Micheli's Article [Deep Learning has (almost) all the answers: Yes/No Question Answering with Transformers](https://medium.com/illuin/deep-learning-has-almost-all-the-answers-yes-no-question-answering-with-transformers-223bebb70189)

Project Organization
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



#How to use our model in Azure
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