#!/usr/bin/env bash
test -d embed || mkdir embed
cd embed
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip