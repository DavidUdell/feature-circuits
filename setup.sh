#!/bin/bash

# Setup pipeline script
pip install --requirement requirements.txt
git submodule update --init
pip install --requirement dictionary_learning/requirements.txt

pip uninstall --yes torchvision
pip install --upgrade torch

python3 circuit.py > j_log.txt
