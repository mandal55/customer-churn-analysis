# Customer Churn Analysis

## Intern ID
CITS9341

## Project Overview
A Machine Learning project that predicts whether a customer will churn (leave the service) based on their account details. Built using Python and Scikit-learn (Random Forest Classifier).

## Features
- Loads customer dataset (`data/customers.csv`)
- Encodes categorical data (contract type, churn label)
- Trains a Random Forest classification model
- Evaluates model using accuracy and classification report
- Shows feature importance (which factors matter most for churn)
- Predicts churn for a new sample customer

## Files
- `customer_churn_analysis.py` — main script
- `data/customers.csv` — dataset of customer records
- `requirements.txt` — Python dependencies

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the script:
   ```
   python customer_churn_analysis.py
   ```

## Tech Used
- Python
- Pandas
- Scikit-learn (Random Forest)

## Author
Submitted as part of CodTech IT Solutions Internship Program.
