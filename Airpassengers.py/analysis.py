import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 
# DATA LOADING
# 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)

print(df.head())
print(df.info())  # Month column is initially object type

# Convert Month column to datetime
df["Month"] = pd.to_datetime(df["Month"])

# print(df["Month"].dtype)  # check dtype

# 
# DATA CLEANING
# 
print(df.isnull().sum())       # no missing values found
print(df.duplicated().sum())   # no duplicate rows found

# ----------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------
df["Year"] = df["Month"].dt.year
df["Month_num"] = df["Month"].dt.month
df["Month_name"] = df["Month"].dt.month_name()  # useful for seasonal interpretation

df["Time_index"] = range(len(df))
# A time index variable was created to represent the progression of time.
# This helps the model capture the trend over time.

# print(df.head())  # control

#
# TRAIN-TEST SPLIT
# 
train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

print(len(train))
print(len(test))  #control

print(train.head())
print(train.tail())

print(test.head())


# 
# EDA
# 

print(df["Passengers"].describe())

print("Mean:", df["Passengers"].mean())
print("Median:", df["Passengers"].median())
print("Variance:", df["Passengers"].var())
print("Skewness:", df["Passengers"].skew()) #it is positive (right-skewed)

result = seasonal_decompose(df["Passengers"], model="multiplicative", period=12)

result.plot()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["Month"], df["Passengers"])
plt.title("Passenger Trend Over Time")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()


plt.figure(figsize=(8,5))
plt.hist(df["Passengers"], bins=15)
plt.title("Distribution of Passenger Counts")
plt.xlabel("Passengers")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(6,4))
plt.boxplot(df["Passengers"])
plt.title("Boxplot of Passenger Counts")
plt.show()




corr = df[["Passengers", "Year", "Month_num", "Time_index"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()


# 
# MODELLING
# 

#model1---------------------
ts = df["Passengers"]
train_ts = train["Passengers"]
test_ts = test["Passengers"]
model = ARIMA(train_ts, order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test_ts))

plt.figure(figsize=(10,5))

plt.plot(train_ts, label="Train")
plt.plot(test_ts, label="Test")
plt.plot(forecast, label="Forecast")

plt.legend()
plt.title("ARIMA Forecast")
plt.show()

#model2-------------------------
X_train = train[["Time_index"]]
y_train = train["Passengers"]

X_test = test[["Time_index"]]
y_test = test["Passengers"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(10,5))

plt.plot(train["Time_index"], y_train, label="Train")
plt.plot(test["Time_index"], y_test, label="Test")
plt.plot(test["Time_index"], y_pred, label="Regression")

plt.legend()
plt.title("Regression Forecast")
plt.show()


# 
# METRICS
# 

# ARIMA metrics
rmse_arima = np.sqrt(mean_squared_error(test_ts, forecast))
r2_arima = r2_score(test_ts, forecast)
mape_arima = np.mean(np.abs((test_ts - forecast) / test_ts)) * 100

# Regression metrics
rmse_reg = np.sqrt(mean_squared_error(y_test, y_pred))
r2_reg = r2_score(y_test, y_pred)
mape_reg = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("ARIMA RMSE:", rmse_arima)
print("ARIMA R2:", r2_arima)
print("ARIMA MAPE:", mape_arima)

print("Regression RMSE:", rmse_reg)
print("Regression R2:", r2_reg)
print("Regression MAPE:", mape_reg)