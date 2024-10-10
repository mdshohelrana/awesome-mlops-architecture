# MLflow Recipe Pipelines

This repository contains two MLflow recipe pipelines: one for regression tasks and another for classification tasks. Follow the instructions below to set up the environment and run the pipelines.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Pipelines](#running-the-pipelines)
  - [Classification](#classification)
  - [Regression](#regression)
- [Recipe Files](#recipe-files)
- [Conclusion](#conclusion)

## Prerequisites

Before you begin, ensure you have the following installed on your machine:
- Python (version 3.x)
- MLflow

## Setup

1. **Create a Virtual Environment:**
   Create a virtual environment and activate it using the `requirements.txt` file provided.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Pipelines

### Classification

To run the classification pipeline, you can either execute the Python script directly or use the MLflow command.

#### Using Python Command:

```bash
python run_classification.py
```

#### Using MLflow Command:

```bash
mlflow recipes run --profile local_classification
```

### Regression

To run the regression pipeline, similarly, you can execute the Python script or use the MLflow command.

#### Using Python Command:

```bash
python run_regression.py
```

#### Using MLflow Command:

```bash
mlflow recipes run --profile local_regression
```

## Recipe Files

There are two recipe files in this repository:

- `recipe_classification.yaml`
- `recipe_regression.yaml`

Before running the pipelines, make sure to copy the content of the respective recipe file into a single `recipe.yaml` file to execute the desired pipeline.

```bash
# For Classification
cp recipe_classification.yaml recipe.yaml

# For Regression
cp recipe_regression.yaml recipe.yaml
```

## Conclusion

You are now ready to run the MLflow recipe pipelines for both regression and classification tasks. If you encounter any issues, please refer to the MLflow documentation or open an issue in this repository. Happy coding!
