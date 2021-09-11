# Joni Väärälä
# 11/9/2021

import pandas as pd
import pandas_datareader as web
from yahoo_fin import stock_info as si

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

start = datetime(2018, 1, 1)
end = datetime.today()

# Import stock symbols from csv-file
tickers_csv = pd.read_csv("tickers.csv")
tickers = tickers_csv["Symbol"].tolist()

# Fetch daily closing price from Yahoo Finance using tickers and pandas datareader
data = web.DataReader(tickers, "yahoo", start=start, end=end)
df_close = data["Adj Close"]
print(df_close.head(10))

# Fetch the needed valuation data
totalStockholderEquity = pd.DataFrame(columns=["Ticker", "totalStockholderEquity"])
valuation = pd.DataFrame(columns=["Ticker", "PE", "ForwardPE", "PB", "DividendRate", "DividendYield", "ROE"])

def get_data(ticker):
    global totalStockholderEquity
    global netIncome
    global valuation

    tempBalanceSheet = si.get_balance_sheet(ticker)
    years = tempBalanceSheet.columns
    while True:
        try:
            totalStockholderEquity = tempBalanceSheet[years[0]]["totalStockholderEquity"]
        except (KeyError, NameError):
            totalStockholderEquity = 0.00
        break

    tempNetIncome = si.get_income_statement(ticker)
    while True:
        try:
            netIncome = tempNetIncome[years[0]]["netIncome"]
        except (KeyError, NameError):
            netIncome = 0.00
        break

    while True:
        try:
            ROE = np.round((netIncome/totalStockholderEquity)*100,decimals = 2)
        except (KeyError, NameError):
            ROE = 0.00
        break
    
    tempValuation = si.get_quote_data(ticker)
    while True:
        try:
            PE = np.round(tempValuation["trailingPE"], decimals = 2)
        except (KeyError, NameError):
            PE = 0.00
        break

    while True:
        try:
            forwardPE = np.round(tempValuation["forwardPE"], decimals = 2)
        except (KeyError, NameError):
            forwardPE = 0.00
        break

    while True:
        try:
            PB = np.round(tempValuation["priceToBook"], decimals = 2)
        except (KeyError, NameError):
            PB = 0.00
        break

    while True:
        try:
            DivRate = np.round(tempValuation["trailingAnnualDividendRate"], decimals = 2)
        except (KeyError, NameError):
            DivRate = 0.00
        break

    while True:
        try:
            DivYield = np.round(tempValuation["trailingAnnualDividendYield"] * 100, decimals = 2)
        except (KeyError, NameError):
            DivYield = 0.00
        break

    valuation = valuation.append({"Ticker":ticker, "PE":PE, "ForwardPE":forwardPE, "PB":PB, "DividendRate":DivRate, "DividendYield":DivYield, "ROE":ROE}, ignore_index=True)

for ticker in tickers:
    get_data(ticker)

valuation.to_clipboard(excel=True, sep=",")

# Calculate linear regression for PB vs ROE scatter plot
x = valuation["ROE"]
y = valuation["PB"]

m, b = np.polyfit(x, y, 1)

# Pyplot visualisations

# Declare figure and marginals
fig = plt.figure(figsize=(14,8))
specs = gridspec.GridSpec(ncols=6, nrows=2, figure=fig) # Declare 6x2 figure
plt.subplots_adjust(left=0.03, bottom=0.135, right=0.98,top=0.93, wspace=0.25, hspace=0.4)

ax1 = fig.add_subplot(specs[0, :4]) # First row, slots 0,1,2,3
ax2 = fig.add_subplot(specs[1, 0]) # Second row first slot
ax3 = fig.add_subplot(specs[1, 1]) # Second row second slot
ax4 = fig.add_subplot(specs[1, 2]) # Second row third slot
ax5 = fig.add_subplot(specs[1, 3]) # Second row fourth slot
ax6 = fig.add_subplot(specs[1, 4]) # Second row fith slot
ax7 = fig.add_subplot(specs[1, 5]) # Second row sixth slot
ax8 = fig.add_subplot(specs[0, 4:]) # First row, slots 4 & 5

ax1.plot(df_close)
ax1.set_title("Closing price €", fontsize=9)
ax1.tick_params("x",labelrotation=90, labelsize=8)
ax1.tick_params("y", labelsize=8)
ax1.legend(tickers, loc="upper left", ncol=2, fontsize=7)

ax2.bar(valuation["Ticker"], valuation["PE"])
ax2.set_title("PE value", fontsize=9)
ax2.tick_params("x",labelrotation=90, labelsize=8)
ax2.tick_params("y", labelsize=8)

ax3.bar(valuation["Ticker"], valuation["ForwardPE"])
ax3.set_title("Forward PE", fontsize=9)
ax3.tick_params("x",labelrotation=90, labelsize=8)
ax3.tick_params("y", labelsize=8)

ax4.bar(valuation["Ticker"], valuation["PB"])
ax4.set_title("PB value", fontsize=9)
ax4.tick_params("x",labelrotation=90, labelsize=8)
ax4.tick_params("y", labelsize=8)

ax5.bar(valuation["Ticker"], valuation["DividendRate"])
ax5.set_title("Dividend rate €", fontsize=9)
ax5.tick_params("x",labelrotation=90, labelsize=8)
ax5.tick_params("y", labelsize=8)

ax6.bar(valuation["Ticker"], valuation["DividendYield"])
ax6.set_title("Dividend Yield %", fontsize=9)
ax6.tick_params("x",labelrotation=90, labelsize=8)
ax6.tick_params("y", labelsize=8)

ax7.bar(valuation["Ticker"], valuation["ROE"])
ax7.set_title("Return of Equity %", fontsize=9)
ax7.tick_params("x",labelrotation=90, labelsize=8)
ax7.tick_params("y", labelsize=8)

ax8.scatter(valuation["ROE"], valuation["PB"])
ax8.plot(x, m*x + b, "g--")
ax8.set_title("PB value compared to ROE (regression line added)", fontsize=9)
ax8.set_xlabel("Return of Equity %", fontsize=8)
ax8.set_ylabel("PB value", fontsize=8)
for i, txt in enumerate(tickers):
    ax8.annotate(txt, (valuation["ROE"][i], valuation["PB"][i]), fontsize=8)

plt.show()



