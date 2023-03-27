import quandl
import os
import matplotlib.pyplot as plt
from IPython.display import display

quandl.ApiConfig.api_key = os.environ.get('rF_LzzCor2SjS8CHEFTr')

def get_stock_data(ticker):
    stock_data = quandl.get('WIKI/' + ticker)
    return stock_data

def get_price_data(ticker):
    price_data = quandl.get('WIKI/' + ticker + '/PRICES')
    return price_data

if __name__ == '__main__':
    stock_data = get_stock_data('AAPL')
    display(stock_data)
    price_data = get_price_data('AAPL')
    display(price_data)
    
    plt.plot(price_data['Adj Close'])
    plt.show()
