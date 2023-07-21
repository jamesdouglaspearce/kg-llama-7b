#!/bin/bash

wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-train.tsv -o ../data/quadruples-train.tsv
wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-validation.tsv -o ../data/quadruples-validation.tsv
wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-test.tsv -o ../data/quadruples-test.tsv
wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/kelm_generated_corpus.jsonl -o ../data/elm_generated_corpus.jsonl