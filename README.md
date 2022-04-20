# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the first project of ML DevOps Engineer Nanodegree Udacity. The aim of this project is to demonstrate learners' competencies in **writing clean code**. To be specific, a learner has to demonstrate:

- The ability to write modular and efficient code, with proper documentation and style check (using `pylint` or `autopep8` for example)
- Implement best practices such as errors handling, testing and logging

The project target is to identify credit card customers that are most likely to churn. The given data is first preprocessed and feature engineered. Then, models are trained to find potential customers that are likely to churn. In the process, EDA and results analysis are also carried out on the data as well as the models.

## Files and data description
The directories structure are list as below:
```bash
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── conftest.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   └── results
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── README.md
├── requirements_py3.6.txt
└── requirements_py3.8.txt
```
The original given file (`churn_notebook.ipynb`) is refactored into `.py` files for modularity, reusability, as well as ease of testing and logging.

More description on some files and folders included in the project:
* `churn_library.py`: contains utility functions that help with data analysis and model training
* `churn_script_logging_and_tests.py`: used for testing functions in churn_library module
* `conftest.py`: contains fixture functions for testing. Also contains common variables used by other modules
* `data`: the directory that contains the data file used in this project
* `images`: used for saving images of data analysis and model analysis. Contains 2 folders: `eda` and `results`
* `logs`: contains log generated from running testing script
* `models`: used for storing model that are ready to use in production

## Running Files
First, install the necessary dependencies:
```bash
python -m pip install -r requirements_py3.8.txt
```
Use this command to update library if you encounter version conflicts:
```
pip install -U numpy seaborn
```
For a complete run to do data analysis and train the model, use `python` or `ipython` command: 
```
python churn_library.py
```
To test the functions inside the `churn_library.py`, run either one the following command. The first command will produce a `.log` file inside `logs` directory
```
# using python command
python churn_script_logging_and_tests.py
# using pytest library 
pytest churn_script_logging_and_tests.py
```
Run these commands to check for clean code criteria. They should produce scores of more than 7 for both files:
```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```
