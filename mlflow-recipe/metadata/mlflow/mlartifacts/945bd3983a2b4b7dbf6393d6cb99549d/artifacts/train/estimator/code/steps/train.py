from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def estimator_fn(estimator_params):
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
