# Classification Model Project

This project consists of three main components running in Docker containers:

1. **Model Trainer** (`model-trainer`) – Trains and saves a classification model.
2. **Model Classifier** (`model-classifier`) – Loads a trained model and serves it via API.
3. **Streamlit Client** (`streamlit-client`) – Provides a simple user interface for interacting with the classifier.

---

## Project Structure

```
.
├── .dockerignore # Specifies files for Docker to ignore during the build process
├── .gitattributes # Defines attributes for files in Git
├── .gitignore # Specifies files and directories for Git to ignore
├── README.md # This file, the main documentation for the project
└── run_all.sh # A script to run all services (likely runs docker-compose)
│
├── # ----------------------------------------------------
├── # Classifier Service - Root Level Files
├── # ----------------------------------------------------
├── app.py # The API endpoint for the classifier service (FastAPI)
├── Dockerfile # Build instructions for the classifier service's Docker image
├── naive_bayes_classifier.py # The core logic for the classification task
└── requirements.txt # Python dependencies for the classifier service
│
├── # ----------------------------------------------------
├── # Trainer Service
├── # ----------------------------------------------------
└── trainer_server/
├── .dockerignore
├── app.py # The API endpoint for the trainer service (FastAPI)
├── Dockerfile # Build instructions for the trainer service's Docker image
├── manager.py # A manager script that orchestrates the entire training pipeline
├── requirements.txt # Python dependencies for the trainer service
│
├── builder/
│ └── naive_bayes_builder.py # Logic for building and training the model
│
├── dat/
│ ├── data_loader.py # Code for loading the raw data from the CSV file
│ └── default_data.csv # The raw dataset used for training
│
├── data_handler/
│ └── data_splitter.py # Code for splitting the data into train/test sets
│
└── evaluator/
└── evaluate_model.py # Code for evaluating the performance of the trained model
```

---

## Running the Project

To build and run all services:

```bash
chmod +x run_all.sh
./run_all.sh
```

This script does the following:

- Removes existing containers (if any)
- Creates a Docker network for communication
- Creates a Docker volume for model persistence
- Builds Docker images for all components
- Runs all containers with proper port mappings

### Ports

| Service          | Port |
|------------------|------|
| Model Trainer    | 8510 |
| Model Classifier | 8010 |
| Streamlit Client | 8011 |



## Requirements

Make sure you have the following installed:

- Docker
- Bash (on Windows, use WSL or Git Bash)

---



