#!/usr/bin/env bash
docker image prune
docker build -f /docker/test_embeddings/Dockerfile -t test_embeddings
docker run --rm -it -v /resources:/app/resources test_embeddings /bin/bash
