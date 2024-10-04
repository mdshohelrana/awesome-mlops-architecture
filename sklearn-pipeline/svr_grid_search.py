import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Ingest the data from CSV file and format Datetime
def ingest(file_path):
    print("Ingesting and preprocessing data...")
    data = pd.read_csv(file_path)
    
    # Convert 'Datetime' to datetime object and set it as the index
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    
    # Drop irrelevant columns like 'Timestamp' and 'Gmtoffset'
    data = data.drop(columns=['Timestamp', 'Gmtoffset'])
    
    # Drop rows with NaN values
    data = data.dropna()

    return data

# Split the data into features and target
def split(data):
    print("Splitting data into features and target...")
    X = data.drop(columns=['Close'])  # Features
    y = data['Close']  # Regression target: predicting the 'Close' price
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVR model using GridSearchCV with a reduced pipeline and parameter grid
def train(X_train, y_train):
    print("Setting up pipeline and performing GridSearchCV...")

    # Set up the pipeline with scaling and SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Add scaling
        ('svr', SVR())  # SVR for regression
    ])

    # Reduced hyperparameter grid for GridSearchCV
    param_grid = {
        'svr__C': [0.1, 1, 10],  # Reduced range for Regularization parameter
        'svr__kernel': ['rbf', 'linear'],  # Focusing on rbf and linear kernels
        'svr__gamma': ['scale', 'auto'],  # Kernel coefficient
        'svr__epsilon': [0.01, 0.1],  # Epsilon-tube for training loss
    }

    # Set up GridSearchCV with reduced search space and folds
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # Reduced cross-validation folds
        scoring='r2',  # Scoring method: R-squared
        n_jobs=-1,  # Use all available cores for faster computation
        verbose=3  # Show detailed training progress
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and cross-validation score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score (R2):", grid_search.best_score_)
    
    # Return the best model
    return grid_search.best_estimator_

# Evaluate the model on the test set
def evaluate(model, X_test, y_test):
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    
    # Calculate and print metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared on test set: {r2}")
    print(f"Test set predictions shape: {predictions.shape}")
    
    return mse, r2

# Save the trained model
def register(model):
    print("Saving the model...")
    dump(model, "svr_best_model.joblib")
    print("Model saved successfully.")

# Specify the file path
file_path = "data/5m_intraday_data.csv"

# Execute the steps
data = ingest(file_path)
X_train, X_test, y_train, y_test = split(data)

# Train, evaluate, and register the SVR model using GridSearchCV and pipeline
best_model = train(X_train, y_train)
evaluate(best_model, X_test, y_test)
register(best_model)
