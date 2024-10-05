import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import importlib


# Custom Transformer for Dropping Unnecessary Features
class DropColumnsTransformer:
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)


# Custom Transformer for Selecting Specific Features
class SelectColumnsTransformer:
    def __init__(self, columns_to_select):
        self.columns_to_select = columns_to_select

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.columns_to_select]


class TradingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None

    # Step 1: Data Ingestion
    def ingest(self):
        print("Step 1: Ingesting Data - Loading and preprocessing data...")
        data = pd.read_csv(self.cfg.data.file_path)
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data = data.set_index("Datetime")
        data["Price_Change"] = (data["Close"].diff() > 0).astype(int)
        data = data.dropna()
        self.data = data

    # Step 2: Data Splitting
    def split(self):
        print("Step 2: Splitting Data - Creating training and test sets...")
        if self.data is None:
            raise ValueError("Data has not been ingested. Call the 'ingest' method first.")
        X = self.data.drop(columns=["Price_Change"])
        y = self.data["Price_Change"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.cfg.data.train_test_split.test_size,
            random_state=self.cfg.data.train_test_split.random_state
        )

    # Step 3: Dynamic Pipeline Construction
    def build_pipeline(self):
        print("Step 3: Building pipeline dynamically from config...")
        steps = []
        for step in self.cfg.pipeline.steps:
            module_name, class_name = step.class_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            if 'params' in step:
                steps.append((step.name, class_(**step.params)))
            else:
                steps.append((step.name, class_()))

        self.pipeline = Pipeline(steps)

    # Step 4: Train the model using GridSearchCV
    def train(self):
        print("Step 4: Training Model - Using GridSearchCV to find best parameters...")
        if self.X_train is None or self.pipeline is None:
            raise ValueError("Data has not been split or the pipeline is not built.")
        param_grid = OmegaConf.to_container(self.cfg.param_grid, resolve=True)
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=self.cfg.grid_search.cv, scoring=self.cfg.grid_search.scoring, n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_
        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_}")

    # Step 5: Evaluate the trained model
    def evaluate(self):
        print("Step 5: Evaluating Model - Checking accuracy on test set...")
        if self.model is None:
            raise ValueError("Model has not been trained. Call the 'train' method first.")
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Accuracy on the test set: {accuracy}")
        return accuracy

    # Step 6: Register (Save) the trained model
    def register(self):
        print("Step 6: Registering Model - Saving the trained model...")
        dump(self.model, self.cfg.model_output_path)
        print("Model saved successfully.")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Instantiate TradingPipeline with config values
    pipeline = TradingPipeline(cfg)

    # Execute steps dynamically as defined in the YAML file
    for step in cfg.pipeline_flow:
        method = getattr(pipeline, step)
        method()


if __name__ == "__main__":
    main()
