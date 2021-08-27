from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
%matplotlib inline

##--------------------------------------------------------------------


start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)
##--------------------------------------------------------------------
# Bank of America
BAC = data.DataReader("BAC", 'google', start, end)

# CitiGroup
C = data.DataReader("C", 'google', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'google', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'google', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'google', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'google', start, end)
##--------------------------------------------------------------------------------------
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'google', start, end)
##---------------------------------------------------------------------------------------
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
##--------------------------------------------------------------------------------------
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers) 
##-----------------------------------------------------------------------------------------
bank_stocks.columns.names = ['Bank Ticker','Stock Info']

## Calling the First few Rows to check out my Data ##

bank_stocks.head()

## Some Data Analysis ## 
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()

returns = pd.DataFrame()

##--------------------------------------------------------------------------------------------------------
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()

## Now to see what Paticular Stock Stands out the most ##
#returns[1:]
import seaborn as sns
sns.pairplot(returns[1:])

## Interestingly if I code the worst drop for each stock, 4 of them were '2009-01-20'.
## After Looking up this date I was reminded that on this day Obama was Inaugerated! #
returns.idxmin()

# Best Single Day Gain
# citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()

returns.std() # Citigroup riskiest as their SD was the highest

----------------------------------
## Now to Visulaise the data ## 
sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
 ## Bins of 100 when aggregating the data ## 

sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100) 
## Citi Return ##

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

## To paint some aesthetically pleasing Visuals' 
import plotly
import cufflinks as cf
cf.go_offline()

for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


## Stock Market Crash in 2009 can really be seen here. 
##Citi Dropped significatnly and since then Goldman Sach has taken the lead ## 
bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()



# plotly
bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()

------------------

## Heatmap of correlation in close prices ## 
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

#Some Technical Analysis ##



## Rolling 30 day average on close price for Bank of American in 2008 up to the crash.
## The Moving Average might have suggested early signs ## 
plt.figure(figsize=(12,6))
BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend() 




## Candle PLot of BOA ##

BAC[['Open', 'High', 'Low', 'Close']].ix['2015-01-01':'2016-01-01'].iplot(kind='candle')

## SMA of Morgan Stanley ##
MS['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')

## Bollinger Band Plot for Bank of America for the year 2015.## 

BAC['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='boll')


--------------------
##RSI ##
------------



close =bank_stocks.xs(key='Close',axis=1,level='Stock Info')

# Window length for moving average
window_length = 14


# Get the difference in price from previous step
delta = close.diff()
# Get rid of the first row, which is NaN since it did not have a previous 
# row to calculate the differences
delta = delta[1:] 

# Make the positive gains (up) and negative gains (down) Series
up, down = delta.clip(lower=0), delta.clip(upper=0)

# Calculate the EWMA
roll_up1 = up.ewm(span=window_length).mean()
roll_down1 = down.abs().ewm(span=window_length).mean()

# Calculate the RSI based on EWMA
RS1 = roll_up1 / roll_down1
RSI1 = 100.0 - (100.0 / (1.0 + RS1))

# Calculate the SMA
roll_up2 = up.rolling(window_length).mean()
roll_down2 = down.abs().rolling(window_length).mean()

# Calculate the RSI based on SMA
RS2 = roll_up2 / roll_down2
RSI2 = 100.0 - (100.0 / (1.0 + RS2))

# Compare graphically
plt.figure(figsize=(8, 6))
RSI1.iplot(title = 'RSI based on EWMA')
RSI2.iplot(title = 'RSI based on SMA')





--------------------------
##Moving Average## 
------------------
plt.figure(figsize=(12,6))
BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()


------------------------

from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime


start = datetime.datetime(2019, 1, 1)
end = datetime.datetime.now()


# NIO
NIO = data.DataReader("NIO", 'yahoo', start, end)
NIO["Date"] = NIO.index
NIO["Stock"] = "NIO"
# TESLA
TSLA = data.DataReader("TSLA", 'yahoo', start, end)
TSLA["Date"] = TSLA.index
TSLA["Stock"] = "TSLA"
# AMD
AMD = data.DataReader("AMD", 'yahoo', start, end)
AMD["Date"] = AMD.index
AMD["Stock"] = "AMD"

tickers = ['NIO', 'TSLA', 'AMD']

stocks = pd.concat([NIO, TSLA, AMD],axis=1,keys=tickers)


stocks.columns.names = ['Bank Ticker','Stock Info']

returns = pd.DataFrame()
for tick in tickers:
    returns[tick+' Return'] = stocks[tick]['Close'].pct_change()
    returns["Date"] = returns.Date
returns.head()
returns.sort_values(by=['Date'], ascending=False)



