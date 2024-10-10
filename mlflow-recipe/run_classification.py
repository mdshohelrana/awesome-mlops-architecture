import os
from mlflow.recipes import Recipe

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def run_mlflow_recipe():

    """
    This function runs the MLflow recipe which handles data ingestion, cleaning, model training,
    and evaluation using the structured MLflow Recipes workflow.
    """
    # Set the working directory to the root of your recipe repository
    # os.chdir("C:\\Users\\UseR\Desktop\\development\\trading-ml-flow")
    print(os.getcwd())

    # Create an instance of the Recipe for classification, using the 'local' profile
    classification_recipe = Recipe(profile="local_classification")

    # Run each step of the recipe sequentially or all together
    logging.info("Starting the pipeline...")
    classification_recipe.run()
    # Inspect the model training results
    classification_recipe.inspect(step="train")
    # Load the trained model
    regression_model_recipe = classification_recipe.get_artifact("model")

if __name__ == "__main__":
    run_mlflow_recipe()
