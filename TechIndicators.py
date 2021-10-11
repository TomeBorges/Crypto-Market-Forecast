#!/usr/bin/python3

import pandas as pd
import numpy as np

from bokeh.plotting import show, output_file, reset_output
from bokeh.layouts import column

from BokehVisualization import plotStock, plotEntranceExit, plotZigZag, plotMA, plotMAderiv, plotEMA, plotRSI, plotMACD, plotOBV, plotROC, plotCCI, plotATR, plotStochOsc
from Indicators import ZigZagAlgorithm, MA, EMAi, PctDerivative, MACDi, RSIi, OBVi, ROCi, CCIi, ATRi, StochOscil, PercentVar, AmountVar

import os
import time
from operator import itemgetter

DrawIndicatorGraphs = False



#Function to calculate the technical indicators and store them in the received Dataframe.
def TechnicalIndCalc(df_ohlc, ResultTXT, FileLocation, OptionChosen):

	# Possible indicators:
	# "AmountVar_0.05_3", "AmountVar_0.1_5" #Currently commented (due to being slow to generate + not very good), but ready to be created
	# "VarPercent"
	# "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100", "EMA_200"
	# "MA_5, "MA_10", "MA_20", "MA_50", "MA_100", "MA_200"
	# "EMA_5_deriv", "EMA_10_deriv", "EMA_20_deriv", "EMA_50_deriv", "EMA_100_deriv", "EMA_200_deriv"
	# "MA_5_deriv", "MA_10_deriv", "MA_20_deriv", "MA_50_deriv", "MA_100_deriv", "MA_200_deriv"
	# "RSI", "RSI_deriv"
	# "MACD","MACDsign", "MACDhist", "MACD_deriv"
	# "OBV", "OBV_deriv"
	# "ROC", "ROC_deriv"
	# "CCI", "CCI_deriv" 
	# "ATR", "ATR_deriv"
	# "StochOscSlowK", "StochOscSlowD"

	IndicatorTime = time.time()
	features = []

	#Finding the pivots for plotting the Zig Zag Algorithm.
	ZigZagPercentage = 0.01
	df_ohlc = ZigZagAlgorithm(df_ohlc, ZigZagPercentage) # This function leaks future data, only use for data representation.

	MA_VALUES = [5,10,20,50,100,200]
	for n in MA_VALUES:
		# Moving Averages added to the DataFrame and plotted:
		df_ohlc = MA(df_ohlc, n)
		df_ohlc = PctDerivative(df_ohlc, 'MA_'+str(n))
		#MAderivatives(df_ohlc, n)
		#features.append('MA_'+str(n))
		#features.append('MA_'+str(n)+'_deriv')

		# Exponential Moving Averages added to the DataFrame and plotted:
		df_ohlc = EMAi(df_ohlc, n)
		df_ohlc = PctDerivative(df_ohlc, 'EMA_'+str(n))
		#df_ohlc = EMAderivatives(df_ohlc, n)
		features.append('EMA_'+str(n))
		#features.append('EMA_'+str(n)+'_deriv')
	# Relative Strength Index added to the DataFrame and plotted:

	RSIperiods = 14
	df_ohlc = RSIi(df_ohlc, RSIperiods)	
	df_ohlc = PctDerivative(df_ohlc, 'RSI')
	features.extend(['RSI'])#, 'RSI_deriv'])
	
	# Moving Average Convergence/Divergence Oscillator added to the DataFrame and plotted:
	MACDn_fast = 12
	MACDn_slow = 26
	df_ohlc = MACDi(df_ohlc, MACDn_fast, MACDn_slow)
	df_ohlc = PctDerivative(df_ohlc, 'MACD')
	#features.extend(['MACD', 'MACD_deriv', 'MACDsign', 'MACDhist'])
	features.extend(['MACDhist']) #adicoasdjasoifnadofndsfafsdfsfsdfsfs

	# On-balance Volume added to the DataFrame and plotted:
	#OBVperiods = 20 #20-period moving average of the OBV is often added
	df_ohlc = OBVi(df_ohlc)#, OBVperiods)	
	df_ohlc = PctDerivative(df_ohlc, 'OBV')
	#features.extend(['OBV', 'OBV_deriv'])
	features.extend(['OBV'])
	# Rate of Change added to the DataFrame and plotted:
	ROCperiods = 10
	df_ohlc = ROCi(df_ohlc, ROCperiods)
	#df_ohlc = PctDerivative(df_ohlc, 'ROC') #Derivadas do indicador ROC_deriv dão valores muito elevados e consequentemente erro.
	features.append('ROC')

	# Commodity Channel Index (CCI) added to DataFrame:
	CCIperiods = 14
	df_ohlc = CCIi(df_ohlc, CCIperiods)
	#df_ohlc = PctDerivative(df_ohlc, 'CCI') #Derivadas do indicador CCI_deriv dão valores muito elevados e consequentemente erro.
	features.append('CCI')

	# Average True Range added to DataFrame:
	ATRperiods = 14
	df_ohlc = ATRi(df_ohlc, ATRperiods)
	df_ohlc = PctDerivative(df_ohlc, 'ATR')
	#features.extend(['ATR', 'ATR_deriv'])
	features.extend(['ATR'])

	# Stochastic Oscillator:
	fastkPeriod = 14
	slowkPeriod = 3
	slowkMatype = 0
	slowdPeriod = 3
	slowdMatype = 0
	fastdperiod = 3
	fastdMatype = 0
	df_ohlc = StochOscil(df_ohlc, fastkPeriod, slowkPeriod, slowkMatype, slowdPeriod, slowdMatype, fastdperiod, fastdMatype)
	#features.extend(['StochOscSlowK', 'StochOscSlowD', 'StochSlowHist'])
	features.extend(['StochOscFastK','StochOscFastD', 'StochOscSlowD'])
	features.extend(['StochFastHist', 'StochSlowHist'])
	

	#features.extend(['Open','High','Low','Close'])

	"""
	# Indicator to show percentual variations over 'n' in one tick:
	for n in [0.01, 0.03]:
		df_ohlc = PercentVar(df_ohlc, n)

	# Indicator to show if an amount 'n' of POSITIVE percentual 'x' variations happens on last periods of time 't':
	for x, n, t in zip([0.05, 0.1], [1, 1], [3, 5]):
		df_ohlc = AmountVar(df_ohlc, x, n, t)
	"""
	if DrawIndicatorGraphs:		
		#Function to plot the Stock prices.
		StockFig = plotStock(df_ohlc, 'Stock Price, ZigZag Algorithm & Moving Averages of '+ FileLocation.strip('.csv') +' - ' + str(OptionChosen) + ' rearrangement.')
		#Function to plot the ZigZag algorithm.
		#StockFig = plotZigZag(df_ohlc, StockFig, ZigZagPercentage)

		#Plot Moving Average and Exponential MA
		i=0
		for n in MA_VALUES:
			StockFig = plotMA(df_ohlc, StockFig, i, n)
			#StockFig = plotMAderiv(df_ohlc, StockFig, i, n)
			#StockFig = plotEMA(df_ohlc, StockFig, i, n)
			i=i+1
		
		#Plot the remaining indicators
		RSIfig = plotRSI(df_ohlc, RSIperiods)		
		MACDfig = plotMACD(df_ohlc, MACDn_fast, MACDn_slow)
		OBVfig = plotOBV(df_ohlc)
		ROCfig = plotROC(df_ohlc, ROCperiods)
		CCIfig = plotCCI(df_ohlc, CCIperiods)
		ATRfig = plotATR(df_ohlc, ATRperiods)		
		Stochfig = plotStochOsc(df_ohlc, fastkPeriod, fastdperiod, slowdPeriod)
		# Pan or zooming actions across X axis is linked in these plots:
		RSIfig.x_range = StockFig.x_range
		MACDfig.x_range = StockFig.x_range
		OBVfig.x_range = StockFig.x_range
		ROCfig.x_range = StockFig.x_range
		CCIfig.x_range = StockFig.x_range
		ATRfig.x_range = StockFig.x_range
		Stochfig.x_range = StockFig.x_range
		#Check if directory to save htmls for this pair exists.
		if not os.path.exists('HTMLs/' + FileLocation.strip('.csv') + '/'):#If directory doesn't exist create it.
			os.makedirs('HTMLs/' + FileLocation.strip('.csv') + '/')

		output_file( 'HTMLs/' + FileLocation.strip('.csv') + '/Indicators-' + str(OptionChosen) + ".html", 
			title = FileLocation.strip('.csv') + '/' + str(OptionChosen) +'/Candlesticks')

		#show(column(StockFig, RSIfig, Stochfig, CCIfig, ATRfig)) # open HTML in browser
		
		show(column(StockFig, RSIfig, MACDfig, OBVfig, ROCfig, CCIfig, ATRfig, Stochfig)) # open HTML in browser
		reset_output()
		print('Elapsed time calculating and plotting Indicators: {:.5f} seconds.'.format(time.time() - IndicatorTime))
	else:
		print('Elapsed time calculating Indicators: {:.5f} seconds.'.format(time.time() - IndicatorTime))
	return df_ohlc, features
