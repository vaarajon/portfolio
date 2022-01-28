# portfolioAnalysis

## Summary
Portfolio analysis visualize the optimal portfolio allocation that optimize the risk and return ratio. The analysis is based on the efficient frontier which is the set of optimal portfolios that offer the highest expected return for a defined level of risk (Investopedia.com).

The optimal portfolio aims to balance securities with the greatest potential returns with an acceptable degree of risk or securities with the lowest degree of risk for a given level of potential return. (Investopedia.com)

Please note that before making any investment, you should do your own research.

## What
Application fetch data from Yahoo Finance and produce graphs using user pre-defined variables. The application takes into consideration:
1. Stock symbols from Yahoo Finance
2. Start year (from which point the application fetch the data)
3. Start month
4. Start day
5. Portfolio size (€, $...)
6. Risk free rate (rate of return of an investment with zero risk)

Putting together the variables, volatility (standard deviation) and expected returns, the application provides graphs for the efficient frontier, minumum risk portfolio, optimal risk portfolio, volatiliy and yearly returns.

The code is inspired by the article in Machine Learning +
`https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/`

## Installation
1. Clone the repository to your local machine 
`$ git clone https://github.com/vaarajon/portfolio.git`

2. Run `cd portfolioAnalysis`

3. Make sure you have needed packages installed in your virtual environment:
- pandas
- pandas_datareader
- datetime
- math
- numpy
- matplotlib.pyplot
- matplotlib.gridspec
- tkinter
- tkinter.font import BOLD

## How does it work
1. Run the code
2. Type in the needed variables
3. The application provides the image of the optimal portfolio allocation.

![image](https://user-images.githubusercontent.com/44286044/150795109-8568d2a0-bbcb-4a03-82f1-f9a7f736f63e.png)
