import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin  # <-- Add this import
from joblib import dump

# Custom Transformer for Dropping Unnecessary Features
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop the specified columns, but ignore errors if the columns do not exist
        return X.drop(self.columns_to_drop, axis=1, errors='ignore')

# Custom Transformer for Selecting Specific Features
class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_select):
        self.columns_to_select = columns_to_select

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.columns_to_select]

class SVRTradingPipeline:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
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

        # Load the data from the CSV file
        data = pd.read_csv(self.data_file_path)

        # Convert 'Datetime' to datetime object and set it as the index
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data = data.set_index("Datetime")

        # Drop irrelevant columns like 'Timestamp' and 'Gmtoffset', ignore errors if not present
        data = data.drop(columns=['Timestamp', 'Gmtoffset'], errors='ignore')

        # Drop rows with NaN values
        data = data.dropna()

        # Store the preprocessed data
        self.data = data

    # Step 2: Data Splitting
    def split(self):
        print("Step 2: Splitting Data - Creating training and test sets...")

        # Ensure data has been ingested
        if self.data is None:
            raise ValueError("Data has not been ingested. Call the 'ingest' method first.")

        # Split into features (X) and target (y)
        X = self.data.drop(columns=["Close"])  # Features
        y = self.data["Close"]  # Target: predicting 'Close' prices

        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Data Transformation (Feature Selection and Scaling)
    def transform(self):
        print("Step 3: Transforming Data - Applying feature selection and scaling...")

        # Specify the columns to drop and select for feature engineering
        columns_to_drop = ["Timestamp", "Gmtoffset"]  # Removed 'Close' here since it's already the target
        columns_to_select = self.data.columns.difference(columns_to_drop + ["Close"])

        # Custom transformers for preprocessing
        drop_columns_transformer = DropColumnsTransformer(columns_to_drop=columns_to_drop)
        select_columns_transformer = SelectColumnsTransformer(columns_to_select=columns_to_select)

        # Define the feature processing pipeline
        preprocessing_pipeline = FeatureUnion([
            ('drop_columns', drop_columns_transformer),
            ('select_columns', select_columns_transformer),
        ])

        # Build the complete pipeline with preprocessing, scaling, and SVR model
        self.pipeline = Pipeline(steps=[
            ('preprocessing', preprocessing_pipeline),  # Preprocessing steps
            ('scaler', StandardScaler()),  # Feature scaling
            ('svr', SVR())  # SVR model for regression
        ])

    # Step 4: Train the SVR model using GridSearchCV
    def train(self):
        print("Step 4: Training Model - Using GridSearchCV to find best parameters...")

        # Ensure data has been split and transformed
        if self.X_train is None or self.pipeline is None:
            raise ValueError("Data has not been split or transformed. Ensure 'split' and 'transform' methods have been called.")

        # Define the hyperparameters for GridSearchCV
        param_grid = {
            'svr__C': [0.1, 1, 10],  # Regularization parameter
            'svr__kernel': ['rbf', 'linear'],  # SVR kernels
            'svr__gamma': ['scale', 'auto'],  # Kernel coefficient
            'svr__epsilon': [0.01, 0.1]  # Epsilon-tube
        }

        # Set up GridSearchCV with the pipeline
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=3)
        grid_search.fit(self.X_train, self.y_train)

        # Store the best model found by GridSearchCV
        self.model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score (R2): {grid_search.best_score_}")

    # Step 5: Evaluate the trained model
    def evaluate(self):
        print("Step 5: Evaluating Model - Checking regression metrics on test set...")

        # Ensure the model has been trained
        if self.model is None:
            raise ValueError("Model has not been trained. Call the 'train' method first.")

        # Make predictions and evaluate metrics
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared on the test set: {r2}")
        return mse, r2

    # Step 6: Register (Save) the trained model
    def register(self):
        print("Step 6: Registering Model - Saving the trained model...")
        dump(self.model, "svr_best_model.joblib")
        print("Model saved successfully.")

# Specify the file path for the data
data_file_path = "data/5m_intraday_data.csv"

# Instantiate the SVRTradingPipeline class
svr_pipeline = SVRTradingPipeline(data_file_path)

# Step-by-step execution of the pipeline
svr_pipeline.ingest()      # Step 1: Ingest data
svr_pipeline.split()       # Step 2: Split data
svr_pipeline.transform()   # Step 3: Transform data
svr_pipeline.train()       # Step 4: Train the model
svr_pipeline.evaluate()    # Step 5: Evaluate the model
svr_pipeline.register()    # Step 6: Register the trained model


# This is what we are looking for:
# pipeline = Pipeline(steps=[
#     ('ingest', svr_pipeline.ingest()),       # Step 1: Ingest data
#     ('split', svr_pipeline.split()),         # Step 2: Split data
#     ('transform', svr_pipeline.transform()), # Step 3: Transform data (preprocessing)
#     ('scaler', StandardScaler()),            # Optional: Scaling features
#     ('train', svr_pipeline.train()),         # Step 4: Train the model
#     ('evaluate', svr_pipeline.evaluate()),   # Step 5: Evaluate the model
#     ('register', svr_pipeline.register())    # Step 6: Register the trained model
# ])

# pipeline.predict()
# pipeline.score()
# pipeline.metrics()