#! /bin/bash

# prepare directory
CS_CKPT_DIR="./cs_ckpt"
QA_CKPT_DIR="./qa_ckpt"
if [ ! -d $CS_CKPT_DIR ]; then
    mkdir $CS_CKPT_DIR
fi
if [ ! -d $QA_CKPT_DIR ]; then
    mkdir $QA_CKPT_DIR
fi

# download checkpoint
TARGET_CS=./cs_ckpt/macbert-large-5epoch
TARGET_QA=./qa_ckpt/macbert-large-5epoch
if [ ! -d $TARGET_CS ]; then
    echo "downloading checkpoint files to ${TARGET_CS}..."
    gdown 1ZHw_sc6VfOp6T7qXUXyL5_2EOI6shahj -O "${TARGET_CS}.zip"
    echo "unzipping checkpoint files..."
    unzip -q "${TARGET_CS}.zip" -d $CS_CKPT_DIR
    rm -f "${TARGET_CS}.zip"
else
    echo "checkpoint ${TARGET_CS} already exists, skipping..."
fi
if [ ! -d $TARGET_QA ]; then
    echo "downloading checkpoint files to ${TARGET_QA}..."
    gdown 1znxpn4nw0c7jI9cmH3QmX7j9kxSQxa6V -O "${TARGET_QA}.zip"
    echo "unzipping checkpoint files..."
    unzip -q "${TARGET_QA}.zip" -d $QA_CKPT_DIR
    rm -f "${TARGET_QA}.zip"
else
    echo "checkpoint ${TARGET_QA} already exists, skipping..."
fi