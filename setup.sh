#!/bin/bash

# Setup pipeline script
pip install -r requirements.txt
git submodule update --init
pip install -r dictionary_learning/requirements.txt

pip uninstall --yes torchvision
pip install -U torch

python3 circuit.py > j_log.txt
