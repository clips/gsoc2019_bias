#!/usr/bin/env bash
test -d resources || mkdir resources
cd resources
test -d embed || mkdir embed
cd embed
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
rm glove.twitter.27B.zip