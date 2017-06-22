#!/bin/sh

# Get blaze
git submodule init
git submodule update

# Build blaze
cd external/blaze
mkdir build
cd build
cmake .. -DBLAZE_SMP_THREADS=C++11
cd ../../..

# Generate the thrift files
thrift -r --gen cpp qa.thrift
mkdir build
cd build
ln -s ../gen-cpp .
cd ..

# Generate the header file
avrogencpp -i weights.avsc -o nnweights.hxx -n coconut

# Get the word2vec embeddings
echo "Download the word2vec embeddings into the current directory from: https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing"
