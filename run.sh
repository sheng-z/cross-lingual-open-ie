#!/bin/bash


printf "\n\n\n===============================\n"
printf "Prepare data...`date`\n\n"
tar zxvf ./data/corpus.tar.gz -C ./data
python ./mt_ie/data_utils.py 2>/dev/null
printf "\n\n\n===============================\n"
printf "Train...`date`\n"
python -m mt_ie 2>/dev/null
