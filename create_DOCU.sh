#!/bin/bash
jupyter-nbconvert --to markdown ./03_docu/DOCU-HRTF_ML.ipynb
mv ./03_docu/DOCU-HRTF_ML.md ./README.md

rm -r ./DOCU-HRTF_ML_files/
mv ./03_docu/DOCU-HRTF_ML_files/ ./

git add ./DOCU-HRTF_ML_files/ README.md

