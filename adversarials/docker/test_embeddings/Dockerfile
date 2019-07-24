FROM tensorflow/tensorflow:latest-py3

WORKDIR /app

COPY . /app

ENV MAT_DTYPE "float16"
ENV GLOVE_PATH "resources/embed/counter-fitted-vectors.txt,embed/glove.twitter.27B.25d.txt,embed/glove.twitter.27B.50d.txt,embed/glove.twitter.27B.100d.txt,embed/glove.twitter.27B.200d.txt,embed/glove.840B.300d.txt"
ENV WORDS_PATH "resources/gender-switched-words.txt"
ENV DATA_PATH "resources/hatespeech-data.csv"

RUN cd /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt