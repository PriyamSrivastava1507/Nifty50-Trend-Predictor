# Nifty 50 Stock Price Predictor

This project implements a machine learning pipeline to predict the daily price movements of the NIFTY 50 index. Moving beyond basic tutorials, this project focuses on **discernment**—the ability of a model to accurately identify downward trends in a naturally bullish market rather than just "guessing" upward growth.

##  Features
- **Data Cleaning**: Specialized handling of historical Nifty 50 CSVs, including comma removal, date formatting, and resolving missing data (`-` strings).
- **Advanced Feature Engineering**: 
  - **RSI (Relative Strength Index)**: To detect overbought/oversold conditions.
  - **Moving Averages (SMA 10/50)**: To identify long-term and short-term trends.
  - **Momentum Metrics**: Daily and 5-day percentage returns.
- **Model Comparison**: Implements Logistic Regression and Random Forest Classifiers.
- **Ethics of Data**: Incorporates the concept of "Intellectual Courage"—balancing the model between "cowardice" (missing gains) and "recklessness" (ignoring risks).

##  Performance & Evaluation
Unlike basic models that achieve high accuracy by simply predicting "Up" every day, this model uses a **Confusion Matrix** to ensure high **Recall** for downward moves. 

- **Discernment**: The model is tuned to detect market corrections, providing a more realistic tool for risk management.
- **Metrics**: Evaluated using AUC-ROC and Precision-Recall scores to ensure the model isn't just following a "Random Walk."

## Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/your-username/nifty50-predictor.git](https://github.com/your-username/nifty50-predictor.git)

2. Install dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
