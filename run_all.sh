#!/bin/bash

# Remove existing containers if running (avoid conflicts)
docker rm -f model-trainer model-classifier streamlit 2>/dev/null

# Create a new Docker network for the project
docker network create cls-project-network

# Create a Docker volume for persistent model data storage
docker volume create model_data_volume

# Build Docker images from their respective directories
docker build -t model-trainer-image ./trainer_server
docker build -t model-classifier-image ./cls
docker build -t streamlit-client-image ./client

# Run the model-trainer container with network and volume attached, exposing port 8510
docker run -d --name model-trainer --network cls-project-network -v model_data_volume:/app/models -p 8510:8510 model-trainer-image
echo "Waiting 15 seconds for service to start..."

# Sleep for 15 seconds to allow the container to initialize
sleep 15

# Run the model-classifier container on the same network, exposing port 8010
docker run -d --name model-classifier --network cls-project-network -p 8010:8010 model-classifier-image

echo "Waiting 15 seconds for service to start..."

# Sleep another 15 seconds for this container to be ready
sleep 15

# Run the Streamlit client container on the network, exposing port 8011
docker run -d --name streamlit --network cls-project-network -p 8011:8011 streamlit-client-image

echo "All services started successfully."






