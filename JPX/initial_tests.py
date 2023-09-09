# import requests
# import datetime as dt

# ticker = 'TSLA'
# base_url = 'https://query1.finance.yahoo.com'
# url = "{}/v8/finance/chart/{}".format(base_url, ticker)
# params = {'interval': '1h', 'range': '7d', 'includePrePost': True}

# try:
#     response = requests.get(url=url, params=params)
#     response.raise_for_status() # raise an exception for 4xx or 5xx status codes
#     data = response.json()
#     epoch = data['chart']['result'][0]['timestamp']
#     prices = data['chart']['result'][0]['indicators']['quote'][0]['close']
#     count = 0
#     list_of_time_and_price = []
#     for entry in epoch:
#         date_and_time = dt.datetime.fromtimestamp(entry).strftime('%Y-%m-%d %H:%M:%S')
#         list_of_time_and_price.append([date_and_time, prices[count]])
#         count += 1
# except requests.exceptions.HTTPError as e:
#     print("HTTP error:", e)
# except requests.exceptions.RequestException as e:
#     print("Network error:", e)
# except Exception as e:
#     print("Unexpected error:", e)

import yfinance as yf


ticker = yf.Ticker("MSFT")
hist = ticker.history(period="max")
print(hist)