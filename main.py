#!/usr/bin/python3

import pandas as pd
import numpy as np

from bokeh.plotting import show, output_file, reset_output
from bokeh.layouts import column

#from BokehVisualization import plotStock, plotEntranceExit, plotZigZag, plotMA, plotMAderiv, plotEMA, plotRSI, plotMACD, plotOBV, plotROC, plotCCI, plotATR, plotStochOsc
#from Indicators import ZigZagAlgorithm, MA, MAderivatives, EMAi, EMAderivatives, MACDi, RSIi, OBVi, ROCi, CCIi, ATRi, StochOscilSlow, PercentVar, AmountVar
from ohlcRearrangement import ohlcPercent, ohlcFixedVal, ohlcLogPrice, ohlcTime
from Predictors import PredictiveAlgorithms
from TechIndicators import TechnicalIndCalc
from Dumps import InitListROI, InitListParams, DumpROIList, DumpParamsList, RemoveROI, RemoveParams, CreateTotalROI
from Files import  WriteAlgStatsCSV, WriteGroupStatsCSV, WriteFeaturesCSV, ROIbeaten

from BokehVisualization import plotStock

import time
from operator import itemgetter
from itertools import groupby
import datetime

File = 'NoDerivs_VFinal/'
ListDirectory = File + 'Lists388'
#ListDirectory = 'Lists388'
TotalROIDirectory = File + 'TotalROIs'
DataDirectory = 'Data388'
ModelCVdir = File + 'ModelCVs388'

#from ta.trend import macd, macd_diff, macd_signal

def main():

	TotalTime = time.time()

	pd.options.display.max_columns = 50
	pd.options.display.max_rows = 100
	pd.options.display.precision = 20

	# Possible indicators:
	# "AmountVar_0.05_3", "AmountVar_0.1_5" #Currently commented (due to being slow to generate + not very good), but ready to be created
	# "VarPercent"
	# "MA_5, "MA_10", "MA_20", "MA_50", "MA_100", "MA_200"
	# "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100", "EMA_200"
	# "MA_5_deriv", "MA_10_deriv", "MA_20_deriv", "MA_50_deriv", "MA_100_deriv", "MA_200_deriv"
	# "EMA_5_deriv", "EMA_10_deriv", "EMA_20_deriv", "EMA_50_deriv", "EMA_100_deriv", "EMA_200_deriv"
	# "RSI", "RSI_deriv"
	# "MACD","MACDsign", "MACDhist", "MACD_deriv"
	# "OBV", "OBV_deriv"
	# "ROC", "ROC_deriv"
	# "CCI", "CCI_deriv" 
	# "ATR", "ATR_deriv" #Fica
	# "StochOscSlowK", "StochOscSlowD"

	# Opening a .txt file to store all details and results.
	ResultTXT = open(File + 'Results.txt','a')
	ResultTXT.write('['+ str(datetime.datetime.now()) + ']\n File containing all the detailed data for each pair:')

	(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV) = InitListROI(ListDirectory)
	(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC) = InitListParams(ListDirectory)

	TotalPROIdf,FirstPerTot = CreateTotalROI('Per')
	TotalAROIdf,FirstAmTot = CreateTotalROI('Am')
	TotalLAROIdf,FirstLATot = CreateTotalROI('LA')
	TotalTROIdf,FirstTimTot = CreateTotalROI('Tm')	

	FileCount = 1
	# .txt file called ListBinancePairs.txt has all the pairs downloaded (to file Data) from Binance's API
	with open('TestTest1.txt') as f: # TestTest1 Top100
		for line in f:	

			PairTime = time.time()

			#List to contain how many candles each grouping has in order to group by time at the end.
			#AmountOfOHLC = []

			FileLocation = str(line.strip('\n')) + '_1m.csv'
			print('\n' + str(FileCount) + '. Starting file ' + str(FileLocation) + '.')
			ResultTXT.write('\n\n\n ' + 54*'_' + ' File ' + str(FileCount) + ' - ' + str(FileLocation) + ' ' + 54*'_' + '\n\n')

			df = pd.read_csv(DataDirectory + '/' + FileLocation, sep=',')

			# If origin changes column names may differ! Copy columns with the following names:

			# Column names received from Binance. 
			new = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

			# Column names received from Kraken.
			#new = df[['Open time', 'Open', 'High', 'Low', 'Close', 'VolumeFrom', 'VolumeTo']].copy()
			#new["Open time"] = pd.to_datetime(new["Open time"],format = '%Y-%m-%d')

			# Columns renamed
			new.columns = ["Date","Open","High","Low","Close","Volume"]
			#new["Date"] = pd.to_datetime(new["Date"])

			# 12 hour format (AM/PM)
			#new["Open time"] = pd.to_datetime(new["Open time"],format = '%Y-%m-%d %I-%p') #(%Y-%m-%d %I:%M:%S %p') -> '2015-08-12 01:07:32 PM'
			
			# 24 hour format
			new["Date"] = pd.to_datetime(new["Date"],format = '%Y-%m-%d %H:%M:%S')
			
			# This way, first index (index 0) ALWAYS corresponds to oldest instant
			if new.at[0, 'Date'] > new.at[1, 'Date']:
				new = (new.sort_index(ascending=False)).reset_index(drop=True)

			# Indicator to show percentage of variation compared to previous entry added to the DataFrame:
			new = new.join(pd.Series(new.Close.pct_change(), name = 'VarPercent'))
			"""
			StockFig = plotStock(new, 'Original OHLC chart of '+str(FileLocation))
			show(StockFig)
			exit()
			"""
			#Expression to estimate the two-day corrected spread for a given period. Proposed by Abdi & Ranaldo in "A Simple Estimation of 
			#Bid-Ask Spreads from Daily Close, High, and Low Prices"
			"""
			new['Spread'] = (2*(( np.log(new.Close) - ((np.log(new.High)+np.log(new.Low))/2) )*( np.log(new.Close) - ((np.log(new.High).shift(-1)+np.log(new.Low).shift(-1))/2) )).clip(lower=0).apply(np.sqrt))
			if (new['Spread'].isnull().sum()!= 1): #If for some reason there are more than 1 NaN entries than something is wrong (only last entry should be NaN, as there is no following period). This is never supposed to happen.
				print('What!??? main.py line 105')
				exit()
			new['Spread'] = new['Spread'].fillna(0)
			"""
			
			# Verify if ROI or Params has been previously calculated and stored. If so, eliminate those entries.
			RemoveROI(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, FileLocation.strip('.csv'))
			RemoveParams(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, FileLocation.strip('.csv'))
									
			for PercentBar in [0.1]: # This value is \in[0,1], multiply by 100 to get percentage. #https://github.com/pandas-dev/pandas/issues/20752
				AmountOfOHLC = []
				# New DataFrame to have all candlesticks with same height :
				for OptionChosen in [0,1,2,3]:
				#OptionChosen = 0

					# 0-According to percentage:
					if OptionChosen == 0:						
						print("\n 0 - Percentage " + str(PercentBar) + ":")
						ResultTXT.write("\n\n" + 50*'_' + " 0 - Percentage "+str(PercentBar)+":\n\n")
						#PercentBar = 0.2 # -> Value is in percentage already.
						#Fig1=plotStock(new,'WADDDUP')
						####
						#dfEqualCandlGrouped = ohlcPercent(new, PercentBar)
						#AmountOfOHLC.append(dfEqualCandlGrouped.shape[0])
						#df_ohlc, features = TechnicalIndCalc(dfEqualCandlGrouped, ResultTXT, FileLocation, "No")
						#exit()
						####
						dfEqualCandlGrouped = ohlcPercent(new, PercentBar)
						AmountOfOHLC.append(dfEqualCandlGrouped.shape[0])
						df_ohlc, features = TechnicalIndCalc(dfEqualCandlGrouped, ResultTXT, FileLocation, "Percentage")
						#Fig2=plotStock(df_ohlc,'WADDDUPSSSSSSSSSs')						
						#show(column(Fig1,Fig2))		
						#exit()
						FirstPerTot = PredictiveAlgorithms(df_ohlc, '', "Percentage", FileLocation, ResultTXT, ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC,
							ListROIMV, ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, features, ModelCVdir, TotalPROIdf, FirstPerTot)
						#print(" 0 - DONE")

					# 1-According to a specific amount:
					if OptionChosen == 1:
						print("\n 1 - Specific amount:")
						ResultTXT.write("\n\n" + 50*'_' + " 1 - Specific amount:\n\n")
						# FixedVal -> Value calculated in ohlcFixedVal. Contrarily to a percentage, a value cannot be fixed for 2 different pairs.
						FixedVal, dfEqualCandlGrouped = ohlcFixedVal(new, AmountOfOHLC[0])
						AmountOfOHLC.append(dfEqualCandlGrouped.shape[0])
						df_ohlc, features = TechnicalIndCalc(dfEqualCandlGrouped, ResultTXT, FileLocation, "Amount")
						FirstAmTot = PredictiveAlgorithms(df_ohlc, '', "Amount", FileLocation, ResultTXT, ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC,
							ListROIMV, ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, features, ModelCVdir, TotalAROIdf, FirstAmTot)
						#print(" 1 - DONE")
					
					# 2-According to the difference of the logarithm over the value:
					if OptionChosen == 2:
						print("\n 2 - Difference of the logarithm:")
						ResultTXT.write("\n\n" + 50*'_' + " 2 - Difference of the logarithm:\n\n")
						#FixedVal -> Value calculated in ohlcLogPrice. Contrarily to a percentage, a value cannot be fixed for 2 different pairs.
						periods = 1 # 1 period is the best value
						FixedVal, dfEqualCandlGrouped = ohlcLogPrice(new, AmountOfOHLC[0], periods)
						AmountOfOHLC.append(dfEqualCandlGrouped.shape[0])
						df_ohlc, features = TechnicalIndCalc(dfEqualCandlGrouped, ResultTXT, FileLocation, "LogAmount")
						FirstLATot = PredictiveAlgorithms(df_ohlc, '', "LogAmount", FileLocation, ResultTXT, ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC,
							ListROIMV, ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, features, ModelCVdir, TotalLAROIdf, FirstLATot)
						#print(" 2 - DONE")

					# 3-Time rearrangement is done to the data. All options should have approximately the same amount of candles to have fair results.
					if OptionChosen == 3:
						print("\n 3 - Time Rearrangement:")
						ResultTXT.write("\n\n" + 50*'_' + " 3 - Time Rearrangement:\n\n")
						dfEqualCandlGrouped = ohlcTime(new, np.mean(AmountOfOHLC))
						df_ohlc, features = TechnicalIndCalc(dfEqualCandlGrouped, ResultTXT, FileLocation, "Time")
						FirstTimTot = PredictiveAlgorithms(df_ohlc, '', "Time", FileLocation, ResultTXT, ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, 
							ListROIMV, ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, features, ModelCVdir, TotalTROIdf, FirstTimTot)
						print('\n\n Number of Candles per grouping: ['+str(AmountOfOHLC[0])+', '+str(AmountOfOHLC[1])+', '+str(AmountOfOHLC[2])+', '+str(dfEqualCandlGrouped.shape[0])+']')
						#print(" 3 - DONE")
					
			#Store ROI and Param list in memory to possibilitate pausing the simulation and avoid losing all data
			DumpROIList(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, ListDirectory)
			DumpParamsList(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, ListDirectory)
			del new # memory release 'new' DataFrame
			
			FileCount +=1
			print('Elapsed time for ' + str(FileLocation.strip('.csv')) + ': %.5f seconds (or %.5f minutes).\n\n' % (time.time() - PairTime, (time.time() - PairTime)/60))
			

	# Opening a .csv file to store the results of each algorithm.
	AlgStatsCSV = open(File + 'AlgStatistics.csv','w')
	AlgStatsCSV.write('['+ str(datetime.datetime.now()) + ']\n' + 9*' ' + 'Algorithm statistics File:')
	#Buy & Hold Strategy Statistics
	if (ListROIBH != []):
		WriteAlgStatsCSV(AlgStatsCSV, ListROIBH, 0, 'BH', features)
	#Logistic Regression Statistics
	if (ListROILR != []):
		WriteAlgStatsCSV(AlgStatsCSV, ListROILR, ListParamsLR, 'LR', features)
	#Random Forest Statistics
	if (ListROIRF != []):
		WriteAlgStatsCSV(AlgStatsCSV, ListROIRF, ListParamsRF, 'RF', features)
	#XGBoost Statistics
	if (ListROIXG != []):
		WriteAlgStatsCSV(AlgStatsCSV, ListROIXG, ListParamsXG, 'XG', features)
	#Support Vector Classifier Statistics
	if (ListROISVC != []):
		WriteAlgStatsCSV(AlgStatsCSV, ListROISVC, ListParamsSVC, 'SVC', features)
	#Majority Voting Statistics
	if (ListROIMV != []):
		WriteAlgStatsCSV(AlgStatsCSV, ListROIMV, [], 'MV', features)
	#Total Statistics
	TotalListROI = ListROIBH + ListROILR + ListROIRF + ListROIXG + ListROISVC + ListROIMV
	WriteAlgStatsCSV(AlgStatsCSV, TotalListROI, [], 'Total', features)
	# Close algorithm statistics file
	AlgStatsCSV.write(5*'\n'+ "END of Algorithm statistics.\n\n")
	AlgStatsCSV.close()

	# Opening a .csv file to store the results of each grouping.
	GroupStatsCSV = open(File + 'GroupStatistics.csv','w')
	GroupStatsCSV.write('['+ str(datetime.datetime.now()) + ']\n' + 30*' ' + 'Grouping statistics File:')
	#Percentage Rearrangement Statistics
	if (any(e[2] == 'Percentage' for e in TotalListROI)):
		WriteGroupStatsCSV(GroupStatsCSV, TotalListROI, 'Percentage')
	#Specific Value Rearrangement Statistics
	if (any(e[2] == 'Amount' for e in TotalListROI)):
		WriteGroupStatsCSV(GroupStatsCSV, TotalListROI, 'Amount')
	#Log of Specific Value Rearrangement Statistics
	if (any(e[2] == 'LogAmount' for e in TotalListROI)):		
		WriteGroupStatsCSV(GroupStatsCSV, TotalListROI, 'LogAmount')
	#Time Rearrangement Statistics
	if (any(e[2] == 'Time' for e in TotalListROI)):
		WriteGroupStatsCSV(GroupStatsCSV, TotalListROI, 'Time')
	#Total Statistics
	WriteGroupStatsCSV(GroupStatsCSV, TotalListROI, 'Total')
	# Close grouping statistics file
	GroupStatsCSV.write(5*'\n'+ "END of Grouping statistics.\n\n")
	GroupStatsCSV.close()

	# Opening a .csv file to store the weights of each feature.
	FeaturesCSV = open(File + 'Features.csv','w')
	FeaturesCSV.write('['+ str(datetime.datetime.now()) + ']\n' + 9*' ' + 'Feature weight File:\n\n')
	ListParams = ListParamsLR + ListParamsRF + ListParamsXG + ListParamsSVC
	WriteFeaturesCSV(FeaturesCSV, ListParams, features)
	#FeaturesCSV.writelines(str(ListParams).replace('\n','\n,').replace("['","\n"))
	FeaturesCSV.write(5*'\n'+ "END of feature weight file.\n\n")
	FeaturesCSV.close()

	ResultTXT.write(5*'\n'+ "END of individual results.\n\n")
	ResultTXT.close()
	
	TotalPROIdf.loc[FirstPerTot:].to_csv(TotalROIDirectory + '/ListTotalROI_Percentage.csv')
	TotalAROIdf.loc[FirstAmTot:].to_csv(TotalROIDirectory + '/ListTotalROI_Amount.csv')
	TotalLAROIdf.loc[FirstLATot:].to_csv(TotalROIDirectory + '/ListTotalROI_LogAmount.csv')
	TotalTROIdf.loc[FirstTimTot:].to_csv(TotalROIDirectory + '/ListTotalROI_Time.csv')

	#ROIbeaten(ListROILR + ListROIRF + ListROIXG + ListROISVC + ListROIMV)	
	print('\nTotal elapsed time: %.5f seconds (or %.5f minutes).\n' % (time.time() - TotalTime, (time.time() - TotalTime)/60))
main()