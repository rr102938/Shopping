# Online Shopping Purchaser Predictor (CS50 AI)

## What it does
* This is a Python program that uses machine learning (k-Nearest Neighbors classifier) to predict whether online shopping customers will complete a purchase based on their session analytics, page values, exit rates, and browsing behavior.
* It parses and converts categorical data (such as months, visitor types, and boolean flags) into numerical features, trains a model, and evaluates its performance by calculating sensitivity (True Positive Rate) and specificity (True Negative Rate).

## Technologies Used
* Python
* scikit-learn (K-Nearest Neighbors classifier, data splitting)
* CSV data parsing and processing
* Git / GitHub

## How to Run It
* Make sure you have the required dependencies installed by running this in your      terminal:
* "```bash"
* pip install scikit-learn numpy
* Place your shopping dataset CSV file in your project directory.
* Run the main script with the data file path: python shopping.py shopping.csv
