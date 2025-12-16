# BERT for Financial Sentiment Analysis

Team 24 Machine Learning Final Project

## Team Members
- Colby Wise (bpx9hy)
- Flavien Moise (nvu5gw)

## Overview 
Our project is determining the correlation between financial news headlines and stock performance. We are using BERT to determine news headline sentiment, and statistical methods to analyze the correlation. 

## How to run our code

Install the requirements (`pip install -r requirements.txt`). It is recommended to use conda to create a Python 3.10 environment for this project. 

1. First, run the data preparer
    - `python3 ./data_preparer.py`
    - set `MAX_ROWS` on line 229 to the amount of data you wish to process, or `None` to process all data
2. Next, run the BERT model
    - `python3 ./bert.py`
3. Last, run the statistical analysis
    - `python3 ./statistical_analysis.py`

Note that Python is commonly installed as `py` on Windows
