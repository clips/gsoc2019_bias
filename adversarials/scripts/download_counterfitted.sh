#!/usr/bin/env bash
test -d resources || mkdir resources
cd resources
test -d embed || mkdir embed
cd embed
wget https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip
rm counter-fitted-vectors.txt.zip