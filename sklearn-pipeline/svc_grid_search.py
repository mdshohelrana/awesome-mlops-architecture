import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump

# Ingest the data from CSV file and format Datetime


def ingest(file_path):
    print("Ingesting and preprocessing data...")
    data = pd.read_csv(file_path)

    # Convert 'Datetime' to datetime object and set it as the index
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    data = data.set_index("Datetime")

    # Drop irrelevant columns like 'Timestamp' and 'Gmtoffset'
    data = data.drop(columns=["Timestamp", "Gmtoffset"])

    # Create binary classification labels: 1 if Close increased, 0 otherwise
    data["Price_Change"] = (data["Close"].diff() > 0).astype(int)

    # Drop rows with NaN values (first row because of diff() and any other NaNs)
    data = data.dropna()

    return data


# Split the data into features and target


def split(data):
    print("Splitting data into features and target...")
    # Drop the 'Close' column, only keep features
    X = data.drop(columns=["Price_Change", "Close"])
    y = data["Price_Change"]  # Binary classification target

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Train the SVC model using GridSearchCV with a pipeline


def train(X_train, y_train):
    print("Setting up pipeline and performing GridSearchCV...")

    # Set up the pipeline with the SVC model
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            # SVC with probability=True for classification
            ("svc", SVC(probability=True)),
        ]
    )

    # Set up the hyperparameter grid for GridSearchCV
    param_grid = {"svc__C": [1, 10], "svc__kernel": ["rbf", "linear"]}

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # Cross-validation folds
        scoring="accuracy",  # Scoring method: accuracy
        n_jobs=-1,  # Use all available cores
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and cross-validation score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_


# Evaluate the model on the test set


def evaluate(model, X_test, y_test):
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on test set: {accuracy}")
    print(f"Test set predictions shape: {predictions.shape}")
    return accuracy


# Save the trained model


def register(model):
    print("Saving the model...")
    dump(model, "svc_best_model.joblib")
    print("Model saved successfully.")


# Specify the file path
file_path = "data/5m_intraday_data.csv"

# Execute the steps
data = ingest(file_path)
X_train, X_test, y_train, y_test = split(data)

# Train, evaluate, and register the SVC model using GridSearchCV and pipeline
best_model = train(X_train, y_train)
evaluate(best_model, X_test, y_test)
register(best_model)
