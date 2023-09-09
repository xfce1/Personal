import yfinance as yf
import glob
import pandas as pd
import numpy as np
import datetime

import umap
# import umap.plot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf

DEBUG = True
 
SPY =  yf.Ticker("SPY")
SPY_histo = SPY.history(start="2017-01-01")
SPY_histo.head()

 
plt.figure(figsize=(10,10))
plt.plot(SPY_histo.index, SPY_histo['Close'])
plt.xlabel("date")
plt.ylabel("$ price")
plt.title("Stock Price")


mpf.plot(SPY_histo[SPY_histo.index>'2022-01-01'], # the dataframe containing the OHLC (Open, High, Low and Close) data
         type='candle', # use candlesticks 
         volume=True, # also show the volume
         mav=(5,10,30), # use three different moving averages
         figratio=(16,9), # set the ratio of the figure
         style='yahoo',  # choose the yahoo style
         title='SPY price since 2022-01-01');


 
df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv',usecols=['SecuritiesCode','Date'])
tickers = [str(t)+'.T' for t in df.SecuritiesCode.unique()]

if DEBUG:
    tickers = tickers[:3]+tickers[-3:]

 
df_prices_all = yf.download(tickers, start='2012-01-01', interval='1d')
df_prices_all.tail()

 
plt.figure(figsize=(10,10))
plt.plot(df_prices_all.index, df_prices_all['Close'])
plt.xlabel("date")
plt.ylabel("$ price")
plt.title("Stock Price - standardised at 1$ in 2012")

 
# # log-prices

 
plt.figure(figsize=(10,10))
plt.plot(df_prices_all.index, np.log(df_prices_all['Close']))
plt.xlabel("date")
plt.ylabel("$ price")
plt.title("Stock Price (log-scale)")

 
# # Returns

 
plt.figure(figsize=(10,10))
plt.plot(df_prices_all.index, np.log(df_prices_all['Close']/df_prices_all['Close'].shift()))
plt.xlabel("date")
plt.ylabel("$ price")
plt.title("Stock returns");

 
# # Outliers

 
plt.figure(figsize=(10,10))
plt.plot(df_prices_all.index, np.clip(np.log(df_prices_all['Close']/df_prices_all['Close'].shift()),-1,1))
plt.xlabel("date")
plt.ylabel("$ price")
plt.title("Stock returns");

 
# # UMAP embedding

 
ret = np.log(df_prices_all['Close']/df_prices_all['Close'].shift()).fillna(0)

reducer = umap.UMAP(random_state=42)
emb = reducer.fit_transform(ret)

plt.scatter(emb[:, 0], emb[:, 1], c=ret.mean(axis=1), cmap='Spectral', s=10 , vmin=-0.025, vmax=0.025)
plt.colorbar()
plt.title('UMAP projection of 10 year returns', fontsize=12);

 
# # Correlation of returns

 
corr = np.log(df_prices_all['Close']/df_prices_all['Close'].shift()).fillna(0).corr()
plt.figure(figsize=(10,10))
plt.imshow(corr)

 
# # Basic clustering

 
import scipy
import scipy.cluster.hierarchy as sch

X = corr
d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [tickers[i] for i in list((np.argsort(ind)))]

plt.figure(figsize=(10,10))
plt.matshow(np.log(df_prices_all['Close']/df_prices_all['Close'].shift()).fillna(0)[columns].corr(), fignum=1, aspect='auto')

 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

plt.figure(figsize=(18,6))
dissimilarity = 1 - abs(corr)
Z = linkage(squareform(dissimilarity), 'complete')

dendrogram(Z, labels=columns, orientation='top', 
           leaf_rotation=90)
 
ticker_name = tickers[0]
t =  yf.Ticker(ticker_name)
dict_t = t.info
dict_t

 
# Lot of infos - rather unformated.

 
# # Format one ticker:
# 
# 
# 

 
def transform_list_to_df(list, test=False):
    formated_list = []
    for i in dict_t.items():
        if i[1] is None:
            formated_list.append((i[0],i[1]))
        elif type(i[1]) in [str,bool,float,int]:
            formated_list.append((i[0],i[1]))
        elif type(i[1]) in [dict]:    
            for j in i[1]:
                formated_list.append((i[0]+'_'+j,i[1][j]))
        else:
            if i[0]=='companyOfficers':
                if test:
                    print(i[1])
            elif i[0]=='sectorWeightings':
                for j in i[1]:
                    for k in j:
                        formated_list.append(('w_'+k,j[k]))
            elif i[0]=='holdings':
                if test:
                    print(i[1])
            elif i[0]=='bondRatings':
                for j in i[1]:
                    for k in j:
                        formated_list.append((i[0]+'_'+k,j[k]))
                         
    df_t = pd.DataFrame(formated_list)
    df_t = df_t.set_index(0)
    df_t.columns = [ticker_name]
    
    return(df_t.T)

 
transform_list_to_df(dict_t,test=True)

 
# Handled everything except: company officer (generally not available) and holdings for etf as it would require a lot of columns for storing each companies. It could be aggregated by sectors... but this would require significant works to request each company individualy and get its sector.

 
# # Get all ticker

df = pd.DataFrame()

for ticker_name in tickers:
    if DEBUG:
        print(ticker_name)
    t =  yf.Ticker(ticker_name)
    dict_t = t.info
    df = pd.concat((df,transform_list_to_df(dict_t).copy()))

 
# # Study Main columns

 
pd.options.display.max_columns = 200
df

 
# # Stock Info EDA

 
def labeler(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def piechart(col):
    sizes = col.value_counts().values # wedge sizes
    fig0, ax1 = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax1.pie(sizes,
                                       autopct=lambda pct: labeler(pct, sizes),
                                       radius=1,
                                       #colors=['#0066ff','#cc66ff'],
                                       startangle=90,
                                       textprops=dict(color="w"),
                                       wedgeprops=dict(width=0.7, edgecolor='w'))

    ax1.legend(wedges, col.value_counts().index,
               loc='center right',
               bbox_to_anchor=(1, 0, 0.5, 1))

    plt.text(0,0, 'TOTAL\n{}'.format(col.value_counts().values.sum()),
             weight='bold', size=12, color='#52527a',
             ha='center', va='center')

    plt.setp(autotexts, size=12, weight='bold')
    ax1.axis('equal')  # Equal aspect ratio
    plt.show()

 
# # Type
# 
# (100% Equity as expected)

 
piechart(df.quoteType)

 
# # Sectors / category

 
piechart(df.sector)

 
# # Recommendations and target prices
# Could be used as features

 
piechart(df.recommendationKey)

 
df.targetMedianPrice/df.currentPrice

 
# # Financial ratio
# 
# Some interesting data about performance (size, profit...).
# Note that this is current data and it would be a bit dangerous to use without history. But it can give us idea for deature engineering fundamental data.

 
cols = ['profitMargins',
'shortRatio',
'yield',
'beta',
'beta3Year',
'priceToBook',
'navPrice',
'fullTimeEmployees']

for c in cols:
    print(c)
    df[c].hist()
    plt.show()

 
# # Save Data

 
df_prices_all.to_parquet('JPX_10Y_daily.parquet')
df.to_parquet('JPX_yfinance_Fundamentals.parquet')


