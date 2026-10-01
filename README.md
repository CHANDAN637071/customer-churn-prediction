# Customer Churn Prediction & Retention Analysis

An end-to-end machine learning project that predicts customer churn, explains individual predictions, and estimates the financial impact of a targeted retention campaign — built on the IBM Telco Customer Churn dataset.

## Problem Statement

Customer churn directly impacts recurring revenue. This project builds a model that:
1. Predicts which customers are likely to churn
2. Explains *why* each customer is at risk (SHAP)
3. Segments customers into risk tiers
4. Estimates the ROI of a targeted retention campaign

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features, 26.5% churn rate (imbalanced).

The dataset is not included in this repo (see `.gitignore`). Download it from Kaggle and place it at `data/raw/telco_churn.csv`.

## Project Structure