"""
Joni Väärälä
23/11/2022

The code is inspired by the article in Machine Learning +

https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/

Portfolio Optimization with Python using 
Efficient Frontier with Practical Examples

I have used the article's practical examples and created a dashboard that visualize
the portfolio optimization analysis with simple graphs. I have also created
a simple entry window that should be used to type in the needed variables.

"""

import pandas as pd
import pandas_datareader as web

from datetime import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tkinter import *
from tkinter.font import BOLD


"""
ENTRY WINDOW FOR NEEDED VARIABLES

    Create an entry window for needed variables, like stock symbols,
    start date, portfolio size and risk free rate

"""
window = Tk()
window.eval("tk::PlaceWindow . center") # place the window in the middle of the screen

#start_date = []
year = 0
month = 0
day = 0
portfolio_size = []
#rf = []
rf = 0
tickers = []

# create submit funtionality
def submit():
    global year
    global month
    global day
    global portfolio_size
    global rf
    
    year = int(year_entry.get())
    month = int(month_entry.get())
    day = int(day_entry.get())
    portfolio_size.append(int(port_entry.get()))
    #rf.append(float(risk_entry.get())/100)
    rf = float(risk_entry.get())/100

    window.destroy()

# create add functionality for tickers' list
def add():
    global tickers
    ticker = ticker_entry.get()

    tickers.append(ticker)

    ticker_entry.delete(0, END)

# create submit all button
submit_button = Button(window, text="Submit all", command=submit, font=("Calibre", 10, "bold"))
add_button = Button(window, text="Add", command=add, font=("Calibre", 10))

# create name labels
year_label = Label(window, text = "Start year", font=("Calibre", 10, BOLD))
month_label = Label(window, text = "Start month", font=("Calibre", 10, BOLD))
day_label = Label(window, text = "Start day", font=("Calibre", 10, BOLD))
port_label = Label(window, text = "Portfolio size", font=("Calibre", 10, BOLD))
risk_label = Label(window, text = "Risk free rate", font=("Calibre", 10, BOLD))
ticker_label = Label(window, text = "Stock symbol", font=("Calibre", 10, BOLD))
empty_label = Label(window, text = "")

# create an entry for input
year_entry = Entry(window, width=20, font=("Calibre", 10))
year_entry.insert(0, "yyyy")

month_entry = Entry(window, width=20, font=("Calibre", 10))
month_entry.insert(0, "mm")

day_entry = Entry(window, width=20, font=("Calibre", 10))
day_entry.insert(0, "dd")

port_entry = Entry(window, width=20, font=("Calibre", 10))
port_entry.insert(0, "€€€€")

risk_entry = Entry(window, width=20, font=("Calibre", 10))
risk_entry.insert(0, "1 = 1%")

ticker_entry = Entry(window, width=20, font=("Calibre", 10))
ticker_entry.insert(0, "Yahoo Finance")

window.title("Input data for the portfolio analysis")

# place the label and entry in required position
ticker_label.grid(row=0, column=0)
ticker_entry.grid(row=0, column=1, pady=10)
add_button.grid(row=0, column=2)

year_label.grid(row=2, column=0)
year_entry.grid(row=2,column=1)
month_label.grid(row=3,column=0)
month_entry.grid(row=3, column=1)
day_label.grid(row=4,column=0)
day_entry.grid(row=4, column=1)

port_label.grid(row=6, column=0)
port_entry.grid(row=6, column=1)

risk_label.grid(row=7, column=0)
risk_entry.grid(row=7, column=1)

submit_button.grid(row=7, column=2)

window.mainloop()

print(year)
print(month)
print(day)
print(portfolio_size)
print(rf)
print(tickers)

# Fetch daily closing price from Yahoo Finance *
data = web.DataReader(tickers, "yahoo", start=datetime(year,month,day), end=datetime.today())
df_close = data["Adj Close"]
print(df_close.head(10))

# Calcalate log of percentage change (log returns for calculations of covariance and correlation) *
log_returns = df_close.pct_change().apply(lambda x: np.log(1+x))

# Calculate variance (volatility) *
# How volatile the investment will be (how returns can fluctuate)
var_log_returns = log_returns.var()
var_log_returns = var_log_returns.sort_values(ascending=False)

# Calculate standard deviation (volatility) *
# How volatile the investment will be (how returns can fluctuate)
annual_stddev = df_close.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

# Calculate covariance matrix *
# To understand how different assets behave with respect to each other
cov_matrix = log_returns.cov()

# Calculate correlation matrix *
# To understand how different assets behave with respect to to each other
corr_matrix = log_returns.corr()


"""
PORTFOLIO OPTIMIZATION
    Building an optimal risky portfolio

    1. Calculate variance portfolio
    2. Calculate optimal risk portfolio and efficient frontier

"""
# Portfolio expected returns

# Yearly returns (expected returns)
yearly_returns = df_close.resample("Y").last().pct_change().mean()
print(yearly_returns)

# Table for visualising returns and volatility
assets = pd.concat([yearly_returns, annual_stddev], axis = 1)
assets.columns = ["Returns", "Volatility"]

# Check NaN values and drop them from Assets and Yearly Returns
is_nan = assets.isnull()
rows_with_nan = is_nan.any(axis=1)
rows_with_nan = assets[rows_with_nan].index.tolist()

# Stocks which can't be taken into account due to NaN values
print(rows_with_nan)

assets = assets.drop(index=rows_with_nan) # stock which will be taken into consideration
yearly_returns = yearly_returns.drop(index=rows_with_nan)
cov_matrix = cov_matrix.drop(index=rows_with_nan)
cov_matrix = cov_matrix.drop(columns=rows_with_nan)


"""
PLOT THE EFFICIENT FRONTIER

    To plot the graph of efficient frontier, we need to run a loop. In each iteration,
    the loop considers different weights for assets and calculates the return and
    volatility of that particular portfolio combination

"""
portfolio_returns = [] # Define an empty array for portfolio returns
portfolio_volatility = [] # Define an empty array for portfolio volatility
portfolio_weights = [] # Define an empty array for asset weigts

num_assets = len(assets.index) # Take into account only stocks for which Return and Volatility are calculated
num_portfolios = 10000 # Define how many times the loop will be run and random portfolio created
print(num_assets)

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets) # Create random weight for each asset
    weights = weights/np.sum(weights) # Sum of weights must be 1
    portfolio_weights.append(weights)
    returns = np.dot(weights, yearly_returns) # Returns are the product of individual expected returns of asset and its weights

    portfolio_returns.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum() # Portfolio variance
    std_dev = np.sqrt(var) # Daily standard deviation
    ann_stddev = std_dev * np.sqrt(250) # Annual standard deviation = volatility
    portfolio_volatility.append(ann_stddev)

data = {"Returns":portfolio_returns, "Volatility":portfolio_volatility}

#for counter, symbol in enumerate(df_close.columns.tolist()):
for counter, symbol in enumerate(assets.index.tolist()):    
    data[symbol + " weight"] = [w[counter] for w in portfolio_weights]

portfolios = pd.DataFrame(data)


"""
PLOT EFFICIENT FRONTIER
    Each point on the line (left edge) represents
    an optimal portfolio of stocks that maximises 
    the returns for any given level of risk.

You can see the combination of weights that will give
all possible combinations:
    1. Minimum volatility (left most point)
    2. Maximum returns (top most point)

MINIMUM VOLATILITY (THE LEAST RISK) PORTFOLIO
    The red start denotes the most efficient porfolio 
    with minimum volatility

"""
min_vol_port = portfolios.iloc[portfolios["Volatility"].idxmin()]
print(min_vol_port)


"""
OPTIMAL RISKY PORTFOLIO
    The optimal risky portfolio is the one with the highest Sharpe ratio.
    In this example the risk-free rate is defined to the 0.5% or 0.005

"""
optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]
print(optimal_risky_port)
# tsekkaa, että rf toimii, tulee olla int, ei lista


"""
MINIMUM AND OPTIMAL PORTFOLIOS IN CURRENCY

    Calculate the shares based on the portfolio size in currency
"""
# Minumum and optimal risk portfolios in currency
min_currency = []
opt_currency = []

for i in range(len(assets)):
    min_currency.append(round(min_vol_port[2+i] * portfolio_size[0]))
    opt_currency.append(round(optimal_risky_port[2+i] * portfolio_size[0]))

min_portfolio_currency = pd.DataFrame({"Stock":assets.index, "Share in currency": min_currency})
opt_portfolio_currency = pd.DataFrame({"Stock":assets.index, "Share in currency": opt_currency})

print("Minimum portfolio in currency")
print(min_portfolio_currency)

print("Optimal portfolio in currency")
print(opt_portfolio_currency)


"""
CREATE VISUALS
    
    Create dashboard

"""
# Declare figure 1 and marginals
fig1 = plt.figure(1, figsize = (14,8))
specs = gridspec.GridSpec(ncols=6, nrows=3, figure=fig1)
plt.subplots_adjust(left=0.03, bottom=0.135, right=0.98, top=0.93, wspace=0.45, hspace=0.45) #0.25 & 0.40

ax1 = fig1.add_subplot(specs[0 :4]) # First row, slots 0,1,2,4
ax2 = fig1.add_subplot(specs[0:2, 4:6]) # First row, slots 3,4,5
ax3 = fig1.add_subplot(specs[1, 0]) # Second row first slot
ax4 = fig1.add_subplot(specs[1, 1]) # Second row second slot
ax5 = fig1.add_subplot(specs[1, 2]) # Second row third slot
ax6 = fig1.add_subplot(specs[1, 3]) # Second row fourth slot
ax7 = fig1.add_subplot(specs[2, :4]) # Third row, slot 0,1,2,3
ax8 = fig1.add_subplot(specs[2, 4]) # Third row fifth slot
ax9 = fig1.add_subplot(specs[2, 5]) # Third row sixth slot

ax1.plot(df_close)
ax1.set_title("Closing price €", fontsize=9)
ax1.tick_params("x", labelrotation=45, labelsize=8)
ax1.tick_params("y", labelsize=8)
ax1.legend(tickers, loc="upper left", ncol=2, fontsize=7)

ax2.scatter(portfolios["Volatility"], portfolios["Returns"], marker="o", s=10, alpha=0.3, label="Portfolio")
ax2.scatter(min_vol_port[1], min_vol_port[0], color="r", marker="*", s=100, label="Minimum risk portfolio")
ax2.scatter(optimal_risky_port[1], optimal_risky_port[0], color="lime", marker="*", s=100, label="Optimal risk portfolio")
ax2.set_title("Efficient frontier", fontsize=9)
ax2.set_xlabel("Volatility", fontsize=8)
ax2.set_ylabel("Annual return", fontsize=8, loc="top")
ax2.tick_params("x", labelsize=8)
ax2.tick_params("y", labelsize=8)
ax2.legend(loc="upper left", fontsize=7)

ax3.pie(min_vol_port[2:], labels=assets.index, 
        autopct="%.0f%%",
        textprops={"size": "smaller"},
        shadow=False,
        radius=0.9, 
        startangle=90)
ax3.set_title("Minimum risk portfolio", fontsize=9)

ax4.axis("tight")
ax4.axis("off")
ax4.table(cellText=min_portfolio_currency.values, colLabels=min_portfolio_currency.columns, loc="upper center", cellLoc="center")
ax4.set_title("Minimum risk portfolio allocation for " + str(portfolio_size[0]) + "€", fontsize=9)

ax5.pie(optimal_risky_port[2:], labels=assets.index, 
        autopct="%.0f%%",
        textprops={"size": "smaller"},
        shadow=False,
        radius=0.9, 
        startangle=90)
ax5.set_title("Optimal risk portfolio", fontsize=9)

ax6.axis("tight")
ax6.axis("off")
ax6.table(cellText=opt_portfolio_currency.values, colLabels=opt_portfolio_currency.columns, loc="upper center", cellLoc="center")
ax6.set_title("Optimal risk portfolio allocation for " + str(portfolio_size[0]) + "€", fontsize=9)

ax7.axis("off")
ax7.set_title("OPTIMAL RISK PORTFOLIO ANALYSIS" + "\n\n" +
                "Volatility of minimum risk portfolio is " + str(round(min_vol_port[1], 2)) + " and return " + str(round(min_vol_port[0], 2)) + "\n" + 
                "Volatility of optimal risk porfolio is " + str(round(optimal_risky_port[1], 2)) + " and return " + str(round(optimal_risky_port[0], 2)) + "\n\n" +
                "When difference in risk between minimum volatility porfolio and optimal risk portfolio is " + str(round((optimal_risky_port[1] - min_vol_port[1])*100, 2)) + "%" + ",\n" +
                "the difference in returns is " + str(round((optimal_risky_port[0] - min_vol_port[0])*100, 2)) + "%. " +
                "If risk increase " + str(round((optimal_risky_port[1] - min_vol_port[1])*100, 2)) + "%" + ", the return icrease by " + str(round((optimal_risky_port[0] - min_vol_port[0])*100, 2)) + "%." + "\n\n" +
                "There was " + str(len(rows_with_nan)) + " stocks with NaN values. Stocks with NaN values are excluded from the portfolio analysis." + "\n" +
                "The list of exluded stock labels: " + str(rows_with_nan),
                loc = "left", fontsize="medium", fontfamily=("Calibri"), linespacing=2.0, verticalalignment="center", y=0.7, pad=-30)

ax8.bar(annual_stddev.index, annual_stddev)
ax8.set_title("Annual standard deviation (volatility)", fontsize=9)
ax8.tick_params("x", labelrotation=90, labelsize=8)
ax8.tick_params("y", labelsize=8)

yearly_returns_bar = round(yearly_returns * 100)

ax9.bar(yearly_returns.index, yearly_returns_bar)
ax9.set_title("Yearly returns %", fontsize=9)
ax9.tick_params("x", labelrotation=90, labelsize=8)
ax9.tick_params("y", labelsize=8)

plt.show()






