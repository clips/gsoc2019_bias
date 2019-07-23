#!/usr/bin/env bash
test -d resources || mkdir resources
cd resources
test -d embed || mkdir embed
cd embed
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip