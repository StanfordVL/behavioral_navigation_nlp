#!/usr/bin/env bash

# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo 'header directory' $HEAD_DIR
CODE_DIR=$HEAD_DIR/codes
DATA_DIR=$HEAD_DIR/data
EXP_DIR=$HEAD_DIR/experiments

mkdir -p $EXP_DIR

# Creates the environment
virtualenv --python=/usr/bin/python2.7 bnnlp

# Activates the environment
#source activate bnnlp
source bnnlp/bin/activate

# pip install into environment
pip install -r requirements.txt
python -m nltk.downloader wordnet

echo 'data directory created at' $DATE_DIR
# Download GloVe vectors to data/
mkdir -p $DATA_DIR
#rm -rf $DATA_DIR
python $CODE_DIR/preprocessing/download_wordvecs.py --download_dir $DATA_DIR
#rm $DATA_DIR/glove.6B.zip

# Download training, val, test data in data/
wget -v -O data_split.zip -L https://stanford.box.com/shared/static/ye2z4zhsdm0mtc9kf44yjar7j9a5nuc3.zip
unzip data_split.zip -d data_split
mv data_split/data/* data/
rm -r data_split
rm data_split.zip
