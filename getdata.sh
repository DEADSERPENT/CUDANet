#!/bin/sh
# Download MNIST or Fashion-MNIST dataset
# Usage:
#   bash getdata.sh          -> downloads MNIST
#   bash getdata.sh fashion  -> downloads Fashion-MNIST

mkdir -p data

if [ "$1" = "fashion" ]; then
    # Fashion-MNIST via GitHub mirror (same binary format as MNIST)
    BASE_URL="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    echo "Downloading Fashion-MNIST dataset..."
    wget "${BASE_URL}train-images-idx3-ubyte.gz" -O ./data/fashion-train-images-idx3-ubyte.gz
    wget "${BASE_URL}train-labels-idx1-ubyte.gz" -O ./data/fashion-train-labels-idx1-ubyte.gz
    wget "${BASE_URL}t10k-images-idx3-ubyte.gz"  -O ./data/fashion-t10k-images-idx3-ubyte.gz
    wget "${BASE_URL}t10k-labels-idx1-ubyte.gz"  -O ./data/fashion-t10k-labels-idx1-ubyte.gz
    echo "Extracting Fashion-MNIST files..."
    gunzip ./data/fashion-train-images-idx3-ubyte.gz
    gunzip ./data/fashion-train-labels-idx1-ubyte.gz
    gunzip ./data/fashion-t10k-images-idx3-ubyte.gz
    gunzip ./data/fashion-t10k-labels-idx1-ubyte.gz
    echo "Fashion-MNIST downloaded:"
    ls -lh ./data/fashion-*
else
    # Standard MNIST
    BASE_URL="http://ossci-datasets.s3.amazonaws.com/mnist/"
    echo "Downloading MNIST dataset..."
    wget -q --show-progress "${BASE_URL}train-images-idx3-ubyte.gz" -P ./data/
    wget -q --show-progress "${BASE_URL}train-labels-idx1-ubyte.gz" -P ./data/
    wget -q --show-progress "${BASE_URL}t10k-images-idx3-ubyte.gz"  -P ./data/
    wget -q --show-progress "${BASE_URL}t10k-labels-idx1-ubyte.gz"  -P ./data/
    echo "Extracting MNIST files..."
    gunzip ./data/train-images-idx3-ubyte.gz \
           ./data/train-labels-idx1-ubyte.gz \
           ./data/t10k-images-idx3-ubyte.gz  \
           ./data/t10k-labels-idx1-ubyte.gz  2>/dev/null || true
    echo "MNIST downloaded:"
    ls -lh ./data/train-* ./data/t10k-*
fi
