# Mobile Network Traffic Forecasting

This repository contains the code, data processing pipeline, and final research report for an empirical study on mobile network traffic forecasting. It evaluates three distinct sequential neural network architectures (LSTM, TCN, and a Time-Series Transformer) using the Milan Telecommunications dataset to predict short-term network loads across different urban environments.

## Repository Structure
* `traffic_forecasting.ipynb`: The primary Google Colab notebook containing the data extraction pipeline, exploratory data analysis (EDA), PyTorch model classes, and evaluation loops.
* `report.pdf`: The final empirical research report detailing the methodology, statistical findings, and architectural comparisons.
* `README.md`: Setup and execution instructions.

## Dataset Handling
The original Milan dataset comprises 62 daily text files totaling roughly 20 GB. To execute this project without exhausting RAM limits, the data is heavily downcasted (`int16` and `float32`) and serialized into a compressed Parquet file.

1. The raw dataset is sourced from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV).
2. The notebook includes the initial preprocessing block to filter for internet traffic, compress the data types, and generate `milan_internet_optimized.parquet`. 

## Setup and Requirements
The pipeline is written in Python 3 and built on PyTorch. If running locally, install the required dependencies:

```bash
pip install torch pandas numpy matplotlib seaborn statsmodels scikit-learn
