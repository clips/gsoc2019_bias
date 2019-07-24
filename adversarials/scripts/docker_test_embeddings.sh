#!/usr/bin/env bash
echo 'y' | docker image prune
docker build -f docker/test_embeddings/Dockerfile --tag test_embeddings .
docker run --rm -it -v /resources:/app/resources test_embeddings /bin/bash