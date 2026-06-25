# Walmart Sales Prediction

## Overview

This project explores historical Walmart sales data to build a machine learning model that predicts weekly sales. The main objective was to understand which factors influence sales and compare different regression models to find the best-performing one.

The project covers the complete machine learning workflow, including data cleaning, exploratory data analysis, feature engineering, model training, and evaluation.

---

## Dataset

The dataset contains weekly sales information from multiple Walmart stores along with factors that may affect sales.

**Features**

* Store
* Date
* Weekly Sales
* Holiday Flag
* Temperature
* Fuel Price
* CPI
* Unemployment

**Target Variable**

* Weekly Sales

---

## Project Workflow

### 1. Data Exploration

I started by exploring the dataset to understand its structure, identify missing values, check data types, and look for any inconsistencies.

### 2. Data Cleaning

* Converted the date column to datetime format
* Checked for missing values
* Removed duplicate records (if any)
* Verified the dataset before modeling

### 3. Exploratory Data Analysis

To better understand the data, I created several visualizations, including:

* Distribution of weekly sales
* Store-wise sales comparison
* Holiday vs non-holiday sales
* Monthly sales trends
* Correlation heatmap
* Boxplots for detecting outliers

These visualizations helped identify patterns and relationships between different variables.

### 4. Feature Engineering

Additional features were extracted from the date column, such as:

* Year
* Month
* Week

These features helped improve the model's ability to capture seasonal trends.

### 5. Model Building

The dataset was divided into training and testing sets before training multiple regression models.

The models used were:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

### 6. Model Evaluation

The models were evaluated using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

Among the models tested, Random Forest produced the most accurate predictions for this dataset.

---

## Key Insights

Some observations from the analysis include:

* Holiday weeks generally have different sales patterns than regular weeks.
* Sales vary significantly across different stores.
* Seasonal trends have a noticeable impact on weekly sales.
* Economic indicators such as CPI and unemployment show some relationship with sales performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Plotly
* Scikit-learn
* Jupyter Notebook

---

## Project Structure

```
Walmart-Sales-Prediction/
│
├── data/
│   └── Walmart.csv
│
├── notebook/
│   └── Walmart_Sales_Prediction.ipynb
│
├── images/
│
├── README.md
└── requirements.txt
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/aftab3964/Walmart-Sales-Prediction.git
```

Go to the project folder:

```bash
cd Walmart-Sales-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook and open the notebook to reproduce the analysis.

---

## Future Improvements

Some possible improvements for this project are:

* Hyperparameter tuning
* Trying boosting algorithms such as XGBoost
* Building a Streamlit dashboard
* Deploying the model as a web application
* Using time-series forecasting techniques for comparison

---

## Author

**Aftab Ahmad**

This project was completed as part of my machine learning practice to strengthen my understanding of data preprocessing, exploratory analysis, regression models, and model evaluation.
