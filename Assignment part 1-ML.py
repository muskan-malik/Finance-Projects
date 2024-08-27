#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:00:44 2024

@author: muskan
"""

# Part 1: Data Analysis and Visualization 

#import all the necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
from sklearn.linear_model import LinearRegression


# load the data
df= pd.read_csv("stock.csv")

stock_data=df 

# Print the data
# stock_data

# Question 1: Calculate the average return of the S&P500

# Ensure that the 'Date' column is of datetime type
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Sort data by date to ensure chronological order
stock_data = stock_data.sort_values(by='Date')

# Calculate daily returns (percentage change)
stock_data['sp500_return'] = stock_data['sp500'].pct_change() * 100

# Calculate the average return
average_sp500_return = stock_data['sp500_return'].mean()

print("The average daily percentage return of the S&P 500 is:",
      average_sp500_return)

# Question 2: Which stock or index has the minimum dispersion 
#(standard deviation) from the mean in dollar value? 

# Selecting only the columns with stock and index prices
price_columns = stock_data.columns[1:-1]  # Excluding 'Date' and 'sp500_return'

# Calculating the standard deviation for each stock and index
std_deviations = stock_data[price_columns].std()

# Finding the stock or index with the minimum standard deviation
min_std_deviation_stock = std_deviations.idxmin()
min_std_deviation_value = std_deviations.min()

print("The stock or index with the minimum dispersion (standard deviation) \
from the mean in dollar value is:", min_std_deviation_stock, "with a standard\
 deviation of", min_std_deviation_value)

#Question 3: Plot the daily price data for all stocks (incl the index)
# without and with normalization

# Plot daily price data for all stocks (including the index)
plt.figure(figsize=(14, 8))
for column in price_columns:
    plt.plot(stock_data['Date'], stock_data[column], label=column)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Daily Price Data for All Stocks (Including S&P 500 Index)')
plt.legend()
plt.show()

# Selecting only the columns with stock and index prices
price_columns = stock_data.columns[1:-1]  # Excluding 'Date' and 'sp500_return'

# Normalize the price data by dividing each price by its initial value
normalized_data = stock_data[price_columns].div(stock_data[price_columns].iloc[0])

# Plot normalized daily price data for all stocks (including the index)
plt.figure(figsize=(14, 8))
for column in normalized_data.columns:
    plt.plot(stock_data['Date'], normalized_data[column], label=column)
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.title('Normalized Daily Price Data for All Stocks (Including S&P 500 Index)')
plt.legend()
plt.show()


#Question 4: Calculate and plot daily returns for all stocks (incl the index)

# Calculate daily returns for all stocks (including the index)
daily_returns = stock_data[price_columns].pct_change()

# Plot daily returns for all stocks (including the index)
plt.figure(figsize=(14, 8))
for column in daily_returns.columns:
    plt.plot(stock_data['Date'], daily_returns[column], label=column)
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Daily Returns for All Stocks (Including S&P 500 Index)')
plt.legend()
plt.show()

#Question 5: Plot the Correlation Table for daily returns for all stocks
#(incl the index) 

# Calculate the correlation matrix for daily returns
correlation_matrix = daily_returns.corr()

# Display the correlation matrix as a table
print("Correlation Matrix for Daily Returns:")
print(correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns,\
           rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.colorbar()
plt.title('Correlation Matrix for Daily Returns of All Stocks (Including S&P \
          500 Index)', pad=20)
plt.show()

# Question 6: What are the top 3 stocks that are positively correlated with
# the S&P500? 

# Find the top 3 stocks that are positively correlated with the S&P 500
sp500_correlations = correlation_matrix['sp500'].sort_values(ascending=False)

# Exclude the correlation of S&P 500 with itself
top_3_positive_correlated_stocks = sp500_correlations[1:4]

top_3_positive_correlated_stocks

# Question 7: Comparing T and TSLA, which stock is riskier and why?

# Calculate daily returns for all stocks (including the index)
daily_returns = stock_data[price_columns].pct_change()

# Calculate the standard deviation of daily returns for T and TSLA
std_T = daily_returns['T'].std()
std_TSLA = daily_returns['TSLA'].std()

print(f"Standard deviation of daily returns for T (AT&T): {std_T}")
print(f"Standard deviation of daily returns for TSLA (Tesla): {std_TSLA}")

# Determine which stock is riskier
if std_T > std_TSLA:
    print("T (AT&T) is riskier than TSLA (Tesla).")
else:
    print("TSLA (Tesla) is riskier than T (AT&T).")
    





    
    
    
