#!/usr/bin/env bash
echo 'y' | docker image prune
docker build -f docker/test_attack/Dockerfile --tag test_attack .
docker run --rm -it -v "$(pwd)/resources":/app/resources test_attack /bin/bash