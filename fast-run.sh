#!/bin/bash
# This is a script for fast running all steps.
rm .\model\model.h5
python .\generate-blur-img.py
docker-compose up
