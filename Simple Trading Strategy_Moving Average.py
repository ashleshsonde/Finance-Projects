#!/usr/bin/env python
# coding: utf-8

# In[147]:


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta


# # IMPORT DATA FROM YAHOO FINANCE

# In[148]:


# Define the ticker and date range
ticker = "^FTSE"  # FTSE 100 ticker symbol on Yahoo Finance
start_date = "2024-01-01"  # Start date
end_date = "2024-12-31"    # End date


# In[149]:


# Fetch data
ftse_data = yf.download(ticker, start=start_date, end=end_date)


# In[150]:


# Display the first few rows
print(ftse_data.head())


# In[153]:


print(ftse_data.describe()) # print summary statistics of FTSE 100


# # PLOTTING GRAPH FOR CLOSING PRICE

# In[154]:


# Plot the Closing Prices
plt.figure(figsize=(10, 8))
plt.plot(ftse_data.index, ftse_data['Close'], label="FTSE 100 Closing Price", color='blue', linewidth = 0.8)

# Add Titles and Labels
plt.title("FTSE 100 Historical Closing Prices", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Closing Price (GBP)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# # Creating a new column for price difference

# In[155]:


ftse_data['PriceDiff'] = ftse_data['Close'].shift(-1) - ftse_data['Close']

print(ftse_data.head(20))


# # Creating a new column for daily return

# In[156]:


ftse_data['DailyReturn'] = ftse_data['PriceDiff']/ftse_data['Close']

print(ftse_data.head(20))


# # Creating a new column for Direction using List Comprehension

# In[157]:


ftse_data['Direction'] = [1 if ftse_data['PriceDiff'].loc[ei] > 0 else 0 for ei in ftse_data.index ]

print(ftse_data.head(20))


# # Create a new column in the DataFrame using Rolling Window calculation (.rolling()) - Moving average

# In[159]:


# Calculate Moving Averages
MA20 = 20  # Short-term moving average (e.g., 20 days)
MA50 = 50  # Long-term moving average (e.g., 50 days)

ftse_data['MA20'] = ftse_data['Close'].rolling(window= MA20).mean()
ftse_data['MA50'] = ftse_data['Close'].rolling(window= MA50).mean()

# Plot the Closing Price and Moving Averages
plt.figure(figsize=(10, 8))
plt.plot(ftse_data.index, ftse_data['Close'], label="FTSE 100 Closing Price", color='blue', linewidth=0.8)
plt.plot(ftse_data.index, ftse_data['MA20'], label=f"{MA20}-Day Moving Average", color='orange', linewidth=0.8)
plt.plot(ftse_data.index, ftse_data['MA50'], label=f"{MA50}-Day Moving Average", color='green', linewidth=0.8)

# Add Titles and Labels
plt.title("FTSE 100 Closing Price and Moving Averages", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (GBP)", fontsize=12)
plt.legend()

# Show the Plot
plt.show()


# # Building a Simple Trading Strategy

# In[161]:


# Calculate Moving Averages
ftse_data['MA20'] = ftse_data['Close'].rolling(window= MA20).mean()  # 20-day moving average
ftse_data['MA50'] = ftse_data['Close'].rolling(window= MA50).mean()  # 50-day moving average
ftse_data = ftse_data.dropna()

# Display the first few rows of the DataFrame
print(ftse_data.head(60))


# # Add a new column "Shares", if MA20 > MA50, denote as 1 (long one share of stock), otherwise, denote as 0 (do nothing)

# In[162]:


# Add Shares Column: 1 if MA10 > MA50, else 0
ftse_data['Shares'] = (ftse_data['MA20'] > ftse_data['MA50']).astype(int)

# Display the first few rows of the DataFrame
print(ftse_data[['Close', 'MA20', 'MA50', 'Shares']].head(60))  # Showing 60 rows to include MA50 computation


# # Adding a new column "Profit" using List Comprehension, for any rows in FTSE, if Shares = 1, the profit is calculated as " the close price of tomorrow - the close price of today". Otherwise the profit is 0.

# In[164]:


# Calculate Profit Column Using List Comprehension
ftse_data['Close1'] = ftse_data['Close'].shift(-1)
ftse_data['Profit'] = [ftse_data.loc[ei, 'Close1'] - ftse_data.loc[ei, 'Close'] if ftse_data.loc[ei, 'Shares']==1 else 0 for ei in ftse_data.index]

# Display the first few rows
print(ftse_data[['Close', 'MA20', 'MA50', 'Shares', 'Profit']].head(60))  # Show more rows for context


# # Calculating Cumulative Profit (Wealth Accmulated)

# In[168]:


ftse_data['wealth'] = ftse_data['Profit'].cumsum()
ftse_data.tail(10)


# In[173]:


# Plot the Cumulative Profit Over Time
plt.figure(figsize=(10, 6))
plt.plot(ftse_data.index, ftse_data['wealth'], label="Cumulative Profit", color='blue', linewidth=0.8)

# Add Titles and Labels
plt.title('Total money you win is {}'.format(ftse_data.loc[ftse_data.index[-2], 'wealth']))
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Profit (GBP)", fontsize=12)
plt.legend()

