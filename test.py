# write a python function for a call option price calculator
# import libraries
import numpy as np
from scipy.stats import norm

def call_option_price(spot_price, strike_price, risk_free_rate, time_to_maturity, volatility):
    

    
    # calculate d1 and d2
    d1 = (np.log(spot_price/strike_price) + (risk_free_rate + (volatility**2)/2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - (volatility * np.sqrt(time_to_maturity))
    
    # calculate call option price
    call_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    
    return call_price

# define the parameters and run the function

spot_price = 100
strike_price = 95
risk_free_rate = 0.05
time_to_maturity = 0.25
volatility = 0.2

call_option_price(spot_price, strike_price, risk_free_rate, time_to_maturity, volatility)
# Output: 6.461647767531209

# write code to plot this with a simulated market

import numpy as np
import matplotlib.pyplot as plt

# define parameters
spot_price = 100
strike_price = 95
risk_free_rate = 0.05
time_to_maturity = 0.25
volatility = 0.2

# create array of volatilities to simulate market
vol_arr = np.arange(0.1, 0.4, 0.1)

# calculate call option prices
call_prices = []
for vol in vol_arr:
    call_price = call_option_price(spot_price, strike_price, risk_free_rate, time_to_maturity, vol)
    call_prices.append(call_price)

# plot call option prices vs. volatilities
plt.plot(vol_arr, call_prices)
plt.xlabel('Volatility')
plt.ylabel('Call Option Price')
plt.title('Call Option Price vs. Volatility')
plt.show()