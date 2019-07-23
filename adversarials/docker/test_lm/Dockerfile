FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

COPY /src /app

ENV MAT_DTYPE "float16"
ENV GLOVE_PATH "embed/counter-fitted-vectors.txt,embed/glove.twitter.27B.25d.txt,embed/glove.twitter.27B.50d.txt,embed/glove.twitter.27B.100d.txt,embed/glove.twitter.27B.200d.txt, embed/glove.840B.300d.txt"
ENV WORDS_PATH "gender-switched-words.txt"

RUN cd /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD python3 -m src.utils.language_model.google_language_model