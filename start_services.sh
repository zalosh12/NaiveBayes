#!/bin/bash

docker rm -f model-trainer model-classifier streamlit 2>/dev/null

docker network create cls-project-network

docker volume create model_data_volume


docker build -t model-trainer-image ./trainer_server

docker build -t model-classifier-image ./cls

docker build -t streamlit-client-image ./client

docker run -d --name model-trainer --network cls-project-network -v model_data_volume:/app/models -p 8510:8510 model-trainer-image
echo "Waiting 15 seconds for service to start..."

sleep 15

docker run -d --name model-classifier --network cls-project-network -p 8010:8010 model-classifier-image

echo "Waiting 15 seconds for service to start..."

sleep 15

docker run -d --name streamlit --network cls-project-network -p 8011:8011 streamlit-client-image

echo "All services started successfully."




