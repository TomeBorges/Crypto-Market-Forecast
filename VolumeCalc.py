#!/usr/bin/python3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import time
import datetime

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, BoxAnnotation, Span, Range1d, DataRange1d, LinearAxis, CDSView, BooleanFilter, markers
from bokeh.layouts import column
DataDirectory = 'Data388'

VolumeTime = time.time()
pd.options.display.max_columns = 500

FileCount = 1
Volumedf = pd.DataFrame(columns=['Pair','Volume', 'FirstOpenTime'])

###############
output_file('HTMLs/' + 'VolumeCalc.html')

Fig = figure(x_axis_type="datetime", plot_height=500, plot_width=1000, y_axis_type="log", title = 'Current volume and introduction date of Binance pairs.')

df = pd.read_csv('OrderedVolume.csv', sep=',')
df["Date"] = pd.to_datetime(df["FirstOpenTime"])
df20=df[0:20]
df50=df[20:50]
df100=df[50:100]
dfrem=df[100:388]

source20=ColumnDataSource(ColumnDataSource.from_df(df20))
r1=Fig.circle(x='Date', y='Volume', color="blue", fill_alpha=0.75, size=13, legend='Top 20', source=source20)
source50=ColumnDataSource(ColumnDataSource.from_df(df50))
r2=Fig.triangle(x='Date', y='Volume', color="green", fill_alpha=0.75, size=13, legend='Top 20-50', source=source50)
source100=ColumnDataSource(ColumnDataSource.from_df(df100))
r3=Fig.square(x='Date', y='Volume', color="red", fill_alpha=0.75, size=13, legend='Top 50-100', source=source100)
sourcerem=ColumnDataSource(ColumnDataSource.from_df(dfrem))
r4=Fig.circle(x='Date', y='Volume', color="grey", fill_alpha=0.75, size=10, legend='Remaining', source=sourcerem)


Fig.legend.label_text_font_size = '15pt'

#Fig.text(x='Date', y='Volume', text='Pair', source=source20, text_font_size="9pt", text_align="left", text_baseline="middle")
#Fig.text(x='Date', y='Volume', text='Pair', source=source50, text_font_size="9pt", text_align="left", text_baseline="middle")
Fig.add_tools(HoverTool(
	tooltips=[
		("Date", "@Date{%F %T}"), #Format: %Y-%M-%D %h:%m:%s
		("Volume", "@{Volume}$"),
		("Pair", "@{Pair}"),],
    formatters={
    'Date': 'datetime', # use 'datetime' formatter for 'Date' field
    'ROI' : 'printf', # In order to write '%'' at the end
	},
	mode='mouse',
	renderers = [r1,r2,r3,r4],))
	
Fig.yaxis.axis_label = "Current sum of volume in USD"
Fig.xaxis.axis_label = "Date of first listing on Binance"

Fig.yaxis.axis_label_text_font_size = '15pt'
Fig.xaxis.axis_label_text_font_size = '15pt'
Fig.yaxis.major_label_text_font_size = '13pt'
Fig.xaxis.major_label_text_font_size = '13pt'

Fig.legend.location = "bottom_left"
Fig.legend.background_fill_alpha = 0.5
show(Fig)
exit()

###############

dfBTC = pd.read_csv('/home/balhelhe/Downloads/' + 'gemini_BTCUSD_1min.csv', sep=',')
dfETH = pd.read_csv('/home/balhelhe/Downloads/' + 'gemini_ETHUSD_1min.csv', sep=',')
dfBNB = pd.read_csv(DataDirectory + '/BNBBTC_1m.csv', sep=',')
	
dfBTC['avg'] = dfBTC[['Open', 'Close']].mean(axis=1) #Average BTCUSD
dfETH['avg'] = dfETH[['Open', 'Close']].mean(axis=1) #Average ETHUSD
dfBNB['avgTemp'] = dfBNB[['Open', 'Close']].mean(axis=1) #Average BNBBTC

dfBTC = dfBTC.drop(columns=['Unix Timestamp','Open','High', 'Low', 'Close'])
dfBTC = dfBTC.set_index('Date')

dfETH = dfETH.drop(columns=['Unix Timestamp','Open','High', 'Low', 'Close'])
dfETH = dfETH.set_index('Date')
dfBNB = dfBNB.drop(columns=['Open','High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'ignored'])
dfBNB = dfBNB.set_index('Open time')

dfBNB = dfBNB.merge(dfBTC, left_index=True, right_index=True, how='inner')
dfBNB['avg'] = dfBNB.avgTemp * dfBTC.avg #BNBBTC * BTCUSD = BNBUSD
dfBNB = dfBNB.drop(columns=['avgTemp'])

# .txt file called ListBinancePairs.txt has all the pairs downloaded (to file Data) from Binance's API
with open('ListBinancePairs388.txt') as f:
	for line in f:

		FileLocation = str(line.strip('\n')) + '_1m.csv'
		df = pd.read_csv(DataDirectory + '/' + FileLocation, sep=',')

		df = df.drop(columns=['Open','High', 'Low', 'Close', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'ignored'])
		df = df.set_index('Open time')
	
		# Volume conversion to same coin (BTC). Conversion done according to coinmarketcap.com at 21h30 of 10/Nov/2018
		#If ___BTC : 1 BTC = 1 BTC
		if FileLocation.strip('_1m.csv')[-3:]== 'BTC':
			dfTemp = dfBTC.merge(df, left_index=True, right_index=True, how='inner')
			TotalVolume = (df.Volume*dfTemp.avg).sum()
			del dfTemp
		#If ___ETH: 1 ETH = 0.03334171 BTC
		elif FileLocation.strip('_1m.csv')[-3:]== 'ETH':
			dfTemp = dfETH.merge(df, left_index=True, right_index=True, how='inner')
			TotalVolume = (df.Volume*dfTemp.avg).sum()
			del dfTemp
		#If ___BNB: 1 BNB = 0.00148597 BTC
		elif FileLocation.strip('_1m.csv')[-3:]== 'BNB':
			dfTemp = dfBNB.merge(df, left_index=True, right_index=True, how='inner')
			TotalVolume = (df.Volume*dfTemp.avg).sum()
			del dfTemp
		#If ___USDT: 1 USDT = 0.00015449 BTC  BTC
		elif FileLocation.strip('_1m.csv')[-4:]== 'USDT':
			TotalVolume = df.Volume.sum()*1
		else:
			print('Missing a type of Counter Currency: ' + str(FileLocation))
			exit()
		Volumedf = Volumedf.append({'Pair': FileLocation.strip('_1m.csv'), 'Volume': TotalVolume, 'FirstOpenTime': df.index[0]}, ignore_index=True)
		#print(TotalVolume)

		del df

		FileCount = FileCount+1
		if(FileCount%20 == 0):
			print(str(FileCount) + ' out of 388 (%.5f %%), %.5f seconds passed' % ((FileCount/388)*100, time.time() - VolumeTime) )
		
	

# Opening a .txt file to store the volume per pair in DESCENDING order.
VolumeTXT = open('OrderedVolume.csv','w')
#VolumeTXT.write('['+ str(datetime.datetime.now()) + ']\n File containing all pairs ordered by total volume in descending order:\n\n\n,')
VolumeTXT.write(str(Volumedf.sort_values('Volume', ascending=False).set_index('Pair').to_csv()))
#VolumeTXT.write('\n\nEnd of files.')
VolumeTXT.close()

print('\nTotal elapsed time: %.5f seconds (or %.5f minutes).\n' % (time.time() - VolumeTime, (time.time() - VolumeTime)/60))