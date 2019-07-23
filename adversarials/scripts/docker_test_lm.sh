#!/usr/bin/env bash
docker image prune
docker build -f /docker/test_lm/Dockerfile -t test_language_model
docker run --rm -it -v /resources:/app/resources test_language_model /bin/bash