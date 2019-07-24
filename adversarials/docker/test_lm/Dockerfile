FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

COPY . /app

ENV MAT_DTYPE "float16"

RUN cd /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt