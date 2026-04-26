# Airline Passenger Time Series Analysis

This project focuses on analyzing and forecasting airline passenger data using classical time series methods.

## Project Goal

The main objective is to:

- Perform Exploratory Data Analysis (EDA)
- Understand trend and seasonality in the data
- Apply classical models (ARIMA and Linear Regression)
- Evaluate model performance using RMSE, MAPE, and R²

---

## Dataset

- Source: Airline Passengers Dataset
- Monthly passenger counts
- Time-based dataset suitable for forecasting problems

---

## 🔍 Exploratory Data Analysis (EDA)

During the EDA phase:

- Checked missing values and duplicates
- Converted date column to datetime format
- Created new features:
  - Year
  - Month number
  - Month name
  - Time index

### Key Insights:

- Strong upward trend over time
- Clear seasonality (yearly pattern)
- Passenger counts increase significantly in later years
- Positive skewness due to growth in demand

---

## Models Used

### 1. ARIMA (1,1,1)

- Classical time series model
- Captures basic temporal structure
- Struggles with strong seasonality in this dataset

### 2. Linear Regression

- Uses `Time_index` as input
- Captures overall upward trend effectively
- Simpler but performs better in this case

---

## Model Evaluation

Models were evaluated using:

- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

### Result:

- Linear Regression outperformed ARIMA
- ARIMA had difficulty capturing seasonal patterns
- Regression worked well due to strong linear trend

---

## Conclusion

This project shows that:

- Time series data often contains both trend and seasonality
- Simple models can sometimes outperform more complex ones
- Feature engineering (like time index) plays an important role

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn

---

## Notes

- This project was developed as part of an academic assignment
- Focus was on understanding the logic behind time series modeling rather than using advanced models

---
