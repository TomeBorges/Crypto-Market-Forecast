#!/usr/bin/python3

import numpy as np
import pandas as pd
import time

from binance.client import Client # https://github.com/sammchardy/python-binance

#My api_key, api_secret
client = Client('api_key', 'api_secret')

#symbol = "TRXETH"
start = "1 Jan, 2016"
end = "31 Dec, 2018"

# Possible data ranges:
#interval = Client.KLINE_INTERVAL_1WEEK
#interval = Client.KLINE_INTERVAL_1HOUR
#interval = Client.KLINE_INTERVAL_30MINUTE
interval = Client.KLINE_INTERVAL_1MINUTE

count = 1
# The .txt file called ListBinancePairs.txt contains all the pairs (1 per line) to be downloaded from Binance's API.
with open('TestTest1.txt') as f:
	for line in f:
		symbol = str(line.strip('\n'))

		print(str(count) + ". Getting " + str(interval) + " " + symbol + " data, since " + start + '...')
		StartTime = time.time()

		klines = client.get_historical_klines(symbol, interval, start)#, end)

		#create new pandas dataframe containing Binance's klines 
		df = pd.DataFrame(klines)
		#columns are named this way
		df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'ignored']
		df["Open time"] = pd.to_datetime(df["Open time"], unit = 'ms')
		df["Close time"] = pd.to_datetime(df["Close time"], unit = 'ms')
		FileName = 'DataJanuary/' + symbol + '_' + str(interval) + '.csv'

		df.to_csv(FileName, index = False) #index columns is removed
		#print (df)		
		print (str(df.index[-1]) + " OHLC points retrieved from Binance in " + str(round(time.time()-StartTime, 1)) + 's (' + str(round((time.time()-StartTime)/60, 1)) + ' mins) saved to:\n    ' + str(FileName) + '\n')
		
		count = count + 1