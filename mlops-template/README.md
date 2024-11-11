├── api                     # API code to interact with the model
│   ├── app.py              # Main API file for routing and interaction
│   ├── models              # Deployment-ready models (from src/models)
│   └── utils.py            # Helper functions for the API
├── config                  # Configuration files for the project
│   ├── environments        # Environment-specific configuration files
│   └── config.yaml         # General project configurations
├── credentials.yaml.gpg    # Encrypted credentials for accessing APIs
├── deployment              # Deployment and CI/CD related files
│   ├── docker              # Docker-related files for containerization
│   │   └── Dockerfile      # Dockerfile for building the project image
│   ├── pipelines           # CI/CD pipeline definitions
│   │   ├── cd_pipeline.yaml # CD pipeline configuration
│   │   └── ci_pipeline.yaml # CI pipeline configuration
│   └── scripts             # Helper scripts for deployment tasks
│       ├── build_image.sh  # Script for building Docker images
│       └── deploy.sh       # Script for deploying the model
├── docs                    # Project documentation
│   ├── api_docs.md         # Documentation for API
│   ├── architecture.md     # High-level project architecture documentation
│   └── usage_guide.md      # Guide on how to use the application
├── infrastructure          # Infrastructure as Code (IaC) scripts
│   └── terraform           # Terraform scripts for cloud resources
│       ├── dev             # Development environment configurations
│       └── prod            # Production environment configurations
├── metadata.yaml           # Metadata configuration for model tracking
├── packages                # Custom utility and common function libraries
│   ├── backtesting         # Backtesting utilities for trading strategies
│   │   ├── signals         # Signal generation for backtesting
│   │   │   └── base_signal.py # Base signal class
│   │   └── strategies      # Strategy definitions for backtesting
│   │       └── base_strategie.py # Base strategy class
│   ├── data_preparation    # Data preparation functions and scripts
│   ├── feature_engineering # Feature engineering functions and scripts
│   └── utils               # General utility functions
├── requirements.txt        # Python dependencies
├── results                 # Generated results and performance metrics
│   ├── metrics.json        # JSON file containing model metrics
│   └── roc_curve.png       # ROC curve plot
├── run.sh                  # Shell script to run the application
├── src                     # Source code for model development
│   ├── __init__.py         # Initialization for the source module
│   ├── config.yml          # Additional configuration file
│   ├── constants           # Constants used across the project
│   ├── data                # Data collection and preprocessing scripts
│   │   ├── external        # External data sources
│   │   ├── interim         # Intermediate processed data
│   │   ├── processed       # Final processed data
│   │   └── raw             # Raw collected data
│   ├── datasets            # Example or sample datasets
│   │   └── foo.csv         # Sample dataset file
│   ├── drift               # Drift detection reports and tools
│   │   ├── production_drift.html # Drift report for production data
│   │   └── test_drift.html       # Drift report for test data
│   ├── main.py             # Main entry point for running the application
│   ├── models              # Trained models stored with semantic versioning
│   ├── notebooks           # Jupyter notebooks for EDA and prototyping
│   ├── results             # Additional results, such as plots and reports
│   │   ├── metrics.json    # Model evaluation metrics
│   │   └── roc_curve.png   # ROC curve image
│   ├── scripts             # Scripts for training and evaluation
│   │   ├── tests           # Test scripts for validation
│   │   └── training        # Training scripts for models
│   ├── steps               # Modular pipeline steps
│   │   ├── 01_data_feeder.py      # Step for data feeding
│   │   ├── 02_data_preprocessing.py # Step for data preprocessing
│   │   ├── 03_feature_engineering.py # Step for feature engineering
│   │   ├── 04_model_training.py   # Step for model training
│   │   ├── 05_model_usage.py      # Step for model usage
│   │   ├── 06_model_tracker.py    # Step for model tracking
│   │   └── 07_post_processing.py  # Step for post-processing results
│   ├── tests               # Unit and integration tests
│   └── utils               # Utility functions specific to `src`
│       ├── cassandra_utils.py # Utilities for Cassandra operations
│       ├── dataset_utils.py   # Utilities for dataset manipulation
│       ├── kafka_clients.py   # Kafka client implementations
│       ├── kafka_utils.py     # Helper functions for Kafka operations
│       ├── misc.py            # Miscellaneous utilities
│       ├── types.py           # Custom data types and type hints
│       └── websocket_utils.py # Utilities for WebSocket operations
└── ui                      # User interface for interacting with the API
    ├── static              # Static assets (CSS, JS, images)
    │   └── styles.css      # Stylesheet for the UI
    ├── templates           # HTML templates
    │   └── index.html      # Main template for the web UI
    └── main.py             # Main script for running the UI