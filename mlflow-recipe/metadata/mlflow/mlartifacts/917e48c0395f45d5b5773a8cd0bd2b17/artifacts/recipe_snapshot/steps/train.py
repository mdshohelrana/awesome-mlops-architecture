from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

def estimator_fn_classification(estimator_params):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ('svc', SVC(probability=True))  # SVC with probability=True for classification
    ])
    param_grid = {
        'svc__C': [0.2, 10],
        'svc__kernel': ['rbf', 'linear']
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # Cross-validation folds
        scoring='accuracy',  # Scoring method: accuracy
        n_jobs=-1,  # Use all available cores
    )
    return grid_search


def estimator_fn_regression():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Add scaling
        ('svr', SVR())  # SVR for regression
    ])

    param_grid = {
        'svr__C': [0.1, 1, 10],  # Reduced range for Regularization parameter
        'svr__kernel': ['rbf', 'linear'],  # Focusing on rbf and linear kernels
        'svr__gamma': ['scale', 'auto'],  # Kernel coefficient
        'svr__epsilon': [0.01, 0.1],  # Epsilon-tube for training loss
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # Reduced cross-validation folds
        scoring='r2',  # Scoring method: R-squared
        n_jobs=-1,  # Use all available cores for faster computation
        verbose=3  # Show detailed training progress
    )
    
    return grid_search