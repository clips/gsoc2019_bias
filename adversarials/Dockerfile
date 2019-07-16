FROM tensorflow/tensorflow:latest-py3

WORKDIR /app

COPY . /app


ENV MAT_DTYPE = float16
ENV GLOVE_PATH = "embed/counter-fitted-vectors.txt,embed/glove.twitter.27b.25d.txt,embed/glove.twitter.27b.50d.txt,embed/glove.twitter.27b.200d.txt, embed/glove.840B.300d.txt"
ENV WORDS_PATH = "gender-switched-words.txt"

RUN cd /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN chmod +x download_counterfitted.sh, chmod +x download_glove.sh, chmod +x download_glove_twitter.sh, ./download_counterfitted.sh, ./download_glove.sh, ./download_glove_twitter.sh