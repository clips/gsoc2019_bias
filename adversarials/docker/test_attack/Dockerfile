FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

COPY . /app

ENV MAT_DTYPE "float16"
ENV RESOURCE_PATH "resources/"
ENV GLOVE_PATH "embed/counter-fitted-vectors.txt,embed/glove.twitter.27B.25d.txt,embed/glove.twitter.27B.50d.txt,embed/glove.twitter.27B.100d.txt,embed/glove.twitter.27B.200d.txt,embed/glove.840B.300d.txt"
ENV WORDS_PATH "gendered-word-pairs.csv"
ENV DATA_PATH "hatespeech-data.csv"
ENV SAMPLES_PATH "sample-sentences.csv"

RUN cd /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt