# stockAnalysis

## Summary
Simple fundamental analysis to identify undervalued stocks. Objective is to identify stock's real value by fetching the stock data from Yahoo Finance and visualizing metrics.

Please note that before making any investment, you should do your own research.

## What
Application fetch data for predetermined stocks and produce graphs from different metrics. Graphs are produced for:
1. Daily closing price
2. PE
3. Forward PE
4. PB
5. Dividend Rate
6. Dividen Yield
7. ROE
9. Scatter plot to observe relationship between ROE and PB (regression line added)

## Installation
1. Clone the repository to your local machine 
`$ git clone https://github.com/vaarajon/portfolio.git`

2. Run `cd stockAnalysis`

3. Make sure you have needed packages installed in your virtual environment
- pandas
- pandas_datareader
- yahoo_fin
- datetime
- numpy
- matplotlib.pyplot
- matplotlib.gridspec

## How does it work
1. Add predefined stock ticker symbols to csv-file
2. Run the code
3. Paste dataframe into a speadsheet for investigating metrics more closely




