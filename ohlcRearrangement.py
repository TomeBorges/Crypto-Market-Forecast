#!/usr/bin/python3

import pandas as pd
import numpy as np

import math
import time

from numba import njit, prange

def ohlcPercent(dfEqualCandl, PercentBar):
	StartTime = time.time()
	# Percentual variation between each consecutive entrance obtained already. 
	# Obtain it's module next:
	dfEqualCandl = dfEqualCandl.join(pd.Series(abs(dfEqualCandl.VarPercent), name = 'AbsVarPer'))
	# Obtain cumulative sum of the module:
	#dfEqualCandl = dfEqualCandl.join(pd.Series(dfEqualCandl.AbsVarPer.cumsum(), name = 'CumsumAbsVarPer'))
	# Obtain quocient of column CumsumAbsVarPercent divided by the wanted percentage for each bar:
	#PercentBar = 0.02
	
	#Dynamic Cumsum function assigns each original row to a new group through a dynamic restarting cumsum method.
	dfEqualCandl = dfEqualCandl.join(pd.Series(dynamic_cumsum(np.delete(dfEqualCandl.AbsVarPer.values,0), PercentBar), name = 'QuocCumsumAbs'))
	dfEqualCandl.QuocCumsumAbs=dfEqualCandl.QuocCumsumAbs.shift(1)
	# Group entrances by percentage variation:
	dfEqualCandlGrouped = grouping(dfEqualCandl, 'QuocCumsumAbs')

	#print(pd.concat([dfEqualCandl.Date, dfEqualCandl.Open, dfEqualCandl.High, dfEqualCandl.Low],axis=1))
	#print(pd.concat([dfEqualCandl.Close, dfEqualCandl.AbsVarPer, dfEqualCandl.QuocCumsumAbs],axis=1))
	#print(pd.concat([dfEqualCandlGrouped.Date, dfEqualCandlGrouped.Open, dfEqualCandlGrouped.High, dfEqualCandlGrouped.Low, dfEqualCandlGrouped.Close],axis=1))
	#print(pd.concat([dfEqualCandlGrouped.Date.shift(1),dfEqualCandlGrouped.Date],axis=1))
	#exit()
	print('Elapsed time on Percentage rearrangement: {:.5f} seconds.'.format(time.time() - StartTime))
	del dfEqualCandl
	return dfEqualCandlGrouped





def ohlcFixedVal(dfEqualCandl, NumOHLC):
	StartTime = time.time()

	# Fixed difference in module between each consecutive entrance must be obtained:
	dfEqualCandl = dfEqualCandl.join(pd.Series(abs(dfEqualCandl.Close.diff()), name = 'AbsFixedVar'))
	# Obtain cumulative sum of the module:
	dfEqualCandl = dfEqualCandl.join(pd.Series(dfEqualCandl.AbsFixedVar.cumsum(), name = 'CumsumAbsFixedVar'))
	#print(dfEqualCandl.CumsumAbsFixedVar.iloc[-1],NumOHLC)
	FixedVal = (dfEqualCandl.CumsumAbsFixedVar.iloc[-1]/NumOHLC)	
	#print(FixedVal)
	# Obtain quocient of column CumsumAbsVarPercent divided by the wanted fixed value for each bar:
	#dfEqualCandl = dfEqualCandl.join(pd.Series(dfEqualCandl.CumsumAbsFixedVar//FixedVal, name = 'QuocCumsumAbs'))
	
	#Dynamic Cumsum function assigns each original row to a new group through a dynamic restarting cumsum method.
	dfEqualCandl = dfEqualCandl.join(pd.Series(dynamic_cumsum(np.delete(dfEqualCandl.AbsFixedVar.values,0), FixedVal), name = 'QuocCumsumAbs'))
	dfEqualCandl.QuocCumsumAbs=dfEqualCandl.QuocCumsumAbs.shift(1)
	# Group entrances by Value variation:
	dfEqualCandlGrouped = grouping(dfEqualCandl, 'QuocCumsumAbs')


	#print(pd.concat([dfEqualCandl.Date, dfEqualCandl.Close, dfEqualCandl.CumsumAbsFixedVar],axis=1), FixedVal)
	#print(pd.concat([dfEqualCandl.Close, dfEqualCandl.AbsFixedVar, dfEqualCandl.QuocCumsumAbs],axis=1))
	#print(dfEqualCandlGrouped.Date)
	#print(pd.concat([dfEqualCandl.Close, dfEqualCandl.Close.diff(), dfEqualCandl.CumsumAbsFixedVar, dfEqualCandl.QuocCumsumAbs],axis=1))
	print('Elapsed time on Fixed Value rearrangement: {:.5f} seconds.'.format(time.time() - StartTime))
	del dfEqualCandl	
	return (FixedVal, dfEqualCandlGrouped)


def ohlcLogPrice(dfEqualCandl, NumOHLC, periods):
	StartTime = time.time()

	# Fixed difference in module between each consecutive entrance must be obtained:

	# Fixed difference in moduled log between each consecutive entrance must be obtained:
	dfEqualCandl = dfEqualCandl.join(pd.Series(abs(np.log(dfEqualCandl.Close).diff(periods=periods)), name = 'AbsFixedLog')) #log with base 'e': Natural Logarithm
	#dfEqualCandl = dfEqualCandl.join(pd.Series(abs(dfEqualCandl.Close.diff(periods=5)), name = 'AbsFixedLog')) #log with base 'e'
	
	# Obtain cumulative sum of the module:
	dfEqualCandl = dfEqualCandl.join(pd.Series(dfEqualCandl.AbsFixedLog.cumsum(), name = 'CumsumAbsFixedLog'))

	FixedLog = (dfEqualCandl.CumsumAbsFixedLog.iloc[-1]/NumOHLC)
	#print(FixedLog)
	# Obtain quocient of column CumsumAbsVarPercent divided by the wanted percentage for each bar:
	#dfEqualCandl = dfEqualCandl.join(pd.Series(dfEqualCandl.CumsumAbsFixedLog//FixedLog, name = 'QuocCumsumAbs'))
	
	#Dynamic Cumsum function assigns each original row to a new group through a dynamic restarting cumsum method.
	dfEqualCandl = dfEqualCandl.join(pd.Series(dynamic_cumsum(np.delete(dfEqualCandl.AbsFixedLog.values,0), FixedLog), name = 'QuocCumsumAbs'))
	dfEqualCandl.QuocCumsumAbs=dfEqualCandl.QuocCumsumAbs.shift(1)
	# Group entrances by fixed log value variation:
	dfEqualCandlGrouped = grouping(dfEqualCandl, 'QuocCumsumAbs')
	
	#print(pd.concat([dfEqualCandl.Close, dfEqualCandl.AbsFixedLog, dfEqualCandl.QuocCumsumAbs],axis=1))
	#print(pd.concat([dfEqualCandl.Date, dfEqualCandl.Close, dfEqualCandl.CumsumAbsFixedLog],axis=1), FixedLog)
	#print(dfEqualCandlGrouped.Date)

	print('Elapsed time on Log Diff rearrangement: {:.5f} seconds.'.format(time.time() - StartTime))
	del dfEqualCandl 
	return (FixedLog, dfEqualCandlGrouped)

# The dfEqualCandl dataframe is grouped in order to have around 'AverageValue' rows.
# Because the number of rows is an integer and in this type of grouping each row should be equally spaced, 
# the final number of rowsmay not coincide with the Average Value (Also, the remainder of the division will add 1 extra row.)
def ohlcTime(dfEqualCandl, AverageValue):
	StartTime = time.time()
	#TimeValue = int(math.floor(dfEqualCandl.shape[0] / AverageValue)) #Rounding down is done. To prove that Time Grouping even being at an advantage is worse.
	TimeValue = int(dfEqualCandl.shape[0] / AverageValue)
	#print(TimeValue)
	# Group entrances by number of rows:
	dfEqualCandlGrouped = grouping(dfEqualCandl, dfEqualCandl.index // TimeValue)

	#The remainder of the division corresponds to the number of rowsgrouped in the last iteration.	
	#print(pd.concat([dfEqualCandl.Date,dfEqualCandl.High, dfEqualCandlGrouped.Date, dfEqualCandlGrouped.High],axis=1))
	#print(TimeValue, dfEqualCandlGrouped.Date)
	print('Elapsed time on Time rearrangement: {:.5f} seconds.'.format(time.time() - StartTime))
	return dfEqualCandlGrouped

# Each entrance is grouped according to either the module of the Cumulative Sum's Quocient, or to a fixed amount of rows (only in ohlcTime's case)
def grouping(dfEqualCandl, string):
	#print pd.concat([dfEqualCandl.Date , dfEqualCandl.Open],axis=1)
	dfTemp = dfEqualCandl.groupby(string).last()
	dfEqualCandlGrouped = dfTemp[['Close','Date']].copy()
	dfTemp = dfEqualCandl.groupby(string).first()
	dfEqualCandlGrouped = dfEqualCandlGrouped.join(dfTemp.Open)
	dfTemp = dfEqualCandl.groupby(string).max()
	dfEqualCandlGrouped = dfEqualCandlGrouped.join(dfTemp.High)
	#dfTemp = dfEqualCandl.groupby(string).mean()
	#dfEqualCandlGrouped = dfEqualCandlGrouped.join(dfTemp.Spread)
	dfTemp = dfEqualCandl.groupby(string).min()
	dfEqualCandlGrouped = dfEqualCandlGrouped.join(dfTemp.Low)
	dfTemp = dfEqualCandl.groupby(string).sum()
	dfEqualCandlGrouped = dfEqualCandlGrouped.join(dfTemp.Volume)


	#dfEqualCandlGrouped = dfEqualCandlGrouped.join(dfTemp.VarPercent)
	#Add entrance 0:
	Line0 = pd.DataFrame({'Date': dfEqualCandl.Date[0], 'Close': dfEqualCandl.Close[0], 'Open': dfEqualCandl.Open[0], 'High': dfEqualCandl.High[0], 'Low': dfEqualCandl.Low[0], 'Volume': dfEqualCandl.Volume[0]}, index=[0])

	dfEqualCandlGrouped = pd.concat([Line0,dfEqualCandlGrouped.ix[:]],sort=False).reset_index(drop=True)

	dfEqualCandlGrouped = dfEqualCandlGrouped.join(pd.Series(dfEqualCandlGrouped.Close.pct_change(), name = 'VarPercent'))
	#print pd.concat([dfEqualCandlGrouped.Date , dfEqualCandlGrouped.Open],axis=1) 
	del dfTemp, Line0 # memory release dfTemp and first line
	return dfEqualCandlGrouped


#Dynamic cumsum: Used to identify entries with their respective final group number. Solves problems: 1.Single candles larger than a
# given threshold instantly create or close a group. 2.Avoids data leaking. Only after a threshold is surpassed can the group be closed.
# (e.g. if the threshold is 0.02, the group cannot be closed at 0.016, this would indicate next candle is larger than at least 0.004.)
@njit
def dynamic_cumsum(seq, max_value):
	FinalList = [0]
	cumsum = 0
	running = 0
	for i in prange(len(seq)):
		cumsum += seq[i]
		if cumsum >= max_value:
			cumsum = 0
			running += 1		
		FinalList.append(running)
	return FinalList
	
	"""#Slow alternative, 5 seconds execution time for GOBTC file
	FinalList = [0]
	index=0
	cumsum = 0
	i=1
	while(i!=dfEqualCandl.index[-1]):
		cumsum = cumsum + dfEqualCandl.AbsVarPer.iloc[i]
		if cumsum > PercentBar:				
			cumsum=0
			index+=1
		FinalList.append(index)
		i+=1
	FinalList = pd.Series(FinalList, name = 'fdc')
	#cumsum = pd.Series(cumsum, name = 'fdc')
	print(pd.concat([dfEqualCandl.AbsVarPer, dfEqualCandl.CumsumAbsVarPer, FinalList],axis=1))
	"""

# Example for a Defined Percentage = 1%
# 0.5 - 1
# 0.7 - 1 
# 0.9 - 1
# 1.1 - 1 Only after crossing the threshold do we know that the threshold has been crossed. If this candle were to be alone data leaking was happening.
# 2.0 - 2
# 3.1 - 4 (3 was skiped over) candle >1% stays alone automatically
# 3.2 - 5


# While only the close parameter is used to 'divide'. Both increases and decreases work together (due to abs) as the variation.
# For example, with a defined value of 1, a candle that seems flat may have increases 0.5 in the first half and decreased -0.5 in the second
# half. The total variation however is |0.5| + |-0.5| = 1. In this case, the High and/or Low sticks should be tall.
#
# This, however, may be changed to have increases working against decreases, however, the defined value must be decreased when compared to the 
# first case.


# In Time Grouping the following happens:
# For grouping of every 3 rows:
#	  Original			  Grouped
#0    2018-06-12 07:59:00 2018-06-12 07:59:00
#1    2018-09-19 04:00:00 2018-09-19 04:01:00
#2    2018-09-19 04:01:00 2018-09-19 04:04:00
#3    2018-09-19 04:02:00 2018-09-19 04:07:00
#4    2018-09-19 04:03:00 2018-09-19 04:10:00
#5    2018-09-19 04:04:00 2018-09-19 04:13:00
#6    2018-09-19 04:05:00 2018-09-19 04:16:00
#	  ...				  ...
#8336 2018-09-24 22:55:00 2018-09-24 22:46:00 (index 2776)
#8337 2018-09-24 22:56:00 2018-09-24 22:49:00 (index 2777)
#8338 2018-09-24 22:57:00 2018-09-24 22:52:00 (index 2778)
#8339 2018-09-24 22:58:00 2018-09-24 22:55:00 (index 2779)
#8340 2018-09-24 22:59:00 2018-09-24 22:58:00 (index 2780)
#8341 2018-09-24 23:00:00 2018-09-24 23:00:00 (index 2781)
# Last row had one less row grouped because of the remainder.