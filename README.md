# Classification Model Project

This project consists of three main components running in Docker containers:

1. **Model Trainer** (`model-trainer`) – Trains and saves a classification model.
2. **Model Classifier** (`model-classifier`) – Loads a trained model and serves it via API.
3. **Streamlit Client** (`streamlit-client`) – Provides a simple user interface for interacting with the classifier.

---

## Project Structure

```
.
├── trainer_server/         # Model training service
├── cls/                    # Model classification service (API)
├── client/                 # Streamlit UI client
├── run_all.sh              # Bash script to build and run all containers
├── .dockerignore
├── .gitignore
└── README.md
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



