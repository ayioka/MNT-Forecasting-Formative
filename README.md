# Mobile Network Traffic Forecasting

This repository contains the code, data-processing pipeline, and final research report for an empirical study on mobile network traffic forecasting. It evaluates three sequential neural network architectures—LSTM, TCN, and a Time-Series Transformer—using the Milan Telecommunications dataset to predict short-term network loads across different urban environments.

## Repository Structure

- `traffic_forecasting.ipynb`: Primary Google Colab notebook containing the data extraction pipeline, exploratory data analysis (EDA), PyTorch model classes, and evaluation loops.
- `report.pdf`: Final empirical research report detailing the methodology, statistical findings, and architectural comparisons.
- `README.md`: Setup and execution instructions.

## Dataset Handling

The original Milan dataset contains 62 daily text files totaling roughly 20 GB. To run the project without exhausting RAM limits, the data is heavily downcasted (`int16` and `float32`) and serialized into a compressed Parquet file.

1. The raw dataset is sourced from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV).
2. The notebook includes the preprocessing block to filter internet traffic, compress the data types, and generate `milan_internet_optimized.parquet`.

## Setup and Requirements

The pipeline is written in Python 3 and built on PyTorch. If running locally, install the required dependencies:

```bash
pip install torch pandas numpy matplotlib seaborn statsmodels scikit-learn
```

Note: The notebook is optimized for Google Colab. If you run it there, the required packages are already pre-installed. You only need to mount your Google Drive to store the dataset and save the output plots.

## Running the Project

1. Open the `.ipynb` notebook in Google Colab.
2. Run the Data Processing section once. This downloads the raw text files, compresses the memory footprint, and saves the `.parquet` file to your Drive.
3. For all future sessions, skip the download step and load the `.parquet` file directly into memory.
4. Run the Exploratory Data Analysis (EDA) cells to generate the spatial distribution graphs, time-series plots, and ACF/PACF stationarity checks.
5. Execute the Model Training block. This initializes the PyTorch DataLoaders (using a 144-step / 24-hour lookback window) and trains the LSTM, TCN, and Transformer models.
6. Run the final evaluation cell to produce performance metrics (MAE, RMSE, MAPE), execution times, and a 3x3 grid of overlaid prediction plots for the three highest-traffic grid squares.

## Key Findings

- LSTM: Provided the most stable and robust baseline across different geographic zones, including commercial and mixed-use areas.
- TCN: Logged the fastest training and inference times because of parallelized 1D convolutions, but slightly under-predicted the sharpest traffic peaks.
- Transformer: Produced erratic results on this univariate dataset. It required 600% more training time and struggled with noise, although it performed well in one specific mixed-use area.
- Calendar anomalies: All purely univariate models failed to anticipate the severe weekend traffic collapse in commercial districts (for example, Square ID 5259) because they lacked exogenous day-of-week variables.

## Project Deliverables

- Full Research Report: [Insert Link to PDF]
- Video Presentation: [Insert Link to Video]
