# Predict Customer Churn

## ML DevOps Engineer Nanodegree Udacity

### Project Description

Course #1: Writing Clean Code

This project refactors an existing code base written in a jupyter notebook. The original notebook code identifies credit card customers that are most likely to churn based on data from Kaggle. The completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). 

This project consists of (2) python source scripts that train, test and run scikit-learn
models used to prodict customer churn from sample data. Logging is incorporated to
both scripts.

The python source files include refactored functions from the original code along with logging and unit testing.

The programs are typically run from the command-line interface (CLI). To run the scripts, jump to the [running](#running-the-python-scripts) section.

### File and directory structure

```
├── README.md                              This file
├── churn_library.py                       Library of refactored functions
├── churn_script_logging_and_tests.py      Unit tests for the functions defined above
├── data
│   └── bank_data.csv                      The data used to train the churn model
├── images                                 Plots and graphs
├── logs                                   Log files
├── models                                 Model artifacts
```

### Development Environment

- Apple Silicon M1 MacBookAir
- MacOS Monterey 12.3.1
- conda 4.11.0
- Python 3.8.13 

### Installation

This project was completed by first installing [Conda miniforge](https://github.com/conda-forge/miniforge) with a python 3.8 base.

The following manual steps were used to create the Conda environment.

```
conda create --name=ml-devops-eng python=3.8 shap scikit-learn joblib pandas numpy matplotlib seaborn pylint autopep8 jupyterlab
```

### Running the python scripts

The main churn code script can be run using the following command. This may take several minutes to complete. 
To watch the progress run `tail -f logs/churn_library.log` in a separate terminal.
```
python churn_library.py
```

The script should create the following files.

```
logs
└── churn_library.log
images
├── churn_histogram.png
├── customer_age_histogram.png
├── cv_feature_importance.png
├── heatmap.png
├── lr_classification_report.png
├── marital_status_counts.png
├── rf_classification_report.png
└── total_transaction_histogram.png
models
├── logistic_model.pkl
└── rfc_model.pkl
```

After the model is trained (several minutes) and saved a summary report should be displayed.

Example
```
random forest results
test results
              precision    recall  f1-score   support

           0       0.96      0.99      0.98      2543
           1       0.93      0.80      0.86       496

    accuracy                           0.96      3039
   macro avg       0.95      0.90      0.92      3039
weighted avg       0.96      0.96      0.96      3039

train results
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5957
           1       1.00      1.00      1.00      1131

    accuracy                           1.00      7088
   macro avg       1.00      1.00      1.00      7088
weighted avg       1.00      1.00      1.00      7088

logistic regression results
test results
              precision    recall  f1-score   support

           0       0.90      0.96      0.93      2543
           1       0.71      0.45      0.55       496

    accuracy                           0.88      3039
   macro avg       0.81      0.71      0.74      3039
weighted avg       0.87      0.88      0.87      3039

train results
              precision    recall  f1-score   support

           0       0.91      0.96      0.94      5957
           1       0.72      0.50      0.59      1131

    accuracy                           0.89      7088
   macro avg       0.82      0.73      0.76      7088
weighted avg       0.88      0.89      0.88      7088
```

### Testing

The `pytest` program is used for unit testing. The example run command should cause logging 
to write to the `logs/churn_library.log` file.
```
pytest churn_script_logging_and_tests.py --log-file=logs/churn_library.log --log-level=INFO
```

Example output.
```
================================================ test session starts ================================================
platform darwin -- Python 3.8.13, pytest-7.1.2, pluggy-1.0.0
rootdir: /Users/koz/src/github/ml-devops-eng/01-clean-code/project_churn
collected 5 items                                                                                                   
churn_script_logging_and_tests.py .....                                                                       [100%]
================================================= warnings summary ==================================================
...
...
...
=========================================== 5 passed, 4 warnings in 3.00s ===========================================
```

### Clean Code 

#### Checking code for cleanliness.
```
pylint churn_library.py
```

#### After acheiving a high score with `pylint` run `autopep8` to format.
```
autopep8 --in-place --aggressive --aggressive churn_library.py

