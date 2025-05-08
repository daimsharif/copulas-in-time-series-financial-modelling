#!/bin/zsh

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


python3 src/main.py

USE_SYNTHETIC=1 N_ASSETS=5 N_OBS=1000 DIST=t DF=6 CORRELATION=high USE_GARCH=1 python main_updated.py
