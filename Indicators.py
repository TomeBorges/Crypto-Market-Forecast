#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib as mpl

from talib import MACD, OBV, RSI, EMA, ROC, CCI, ATR, STOCH, STOCHF

from sklearn.preprocessing import maxabs_scale

from zigzag import peak_valley_pivots 

# Function peak_valley_pivots has future data leaking. Use only for graphics.
def ZigZagAlgorithm(df, n): # Price movement greater than a designated percentage amount: 'n'.
	# ZigZag Pivots
	dfZig = df['Close'].values
	pivots = peak_valley_pivots(dfZig, n, -n)
	
	# Dataframe containing Pivots and Close (value) synced with the Date for the ZigZag algorithm
	dfZig = pd.DataFrame({'Pivots_'+str(n):pivots})
	df = df.join(dfZig)
	return df

# No leaking, no shift required.
# Moving Average
def MA(df, n): 
	MA = pd.Series(df['Close'].rolling(window=n,center=False).mean(), name = 'MA_' + str(n)) # n is the number of observations.
	df = df.join(MA)
	return df

# No leaking, no shift required.
# Moving Average derivative.
def MAderivatives(df, n):
	MAderiv = pd.Series(df['MA_'+str(n)].pct_change(), name = 'MAderiv_' + str(n))#.shift(1) # Moving Averages derivates.
	#MAderiv = pd.Series(df['MA_'+str(n)].diff(), name = 'MAderiv_' + str(n))#.shift(1) # Moving Averages derivates.
	df = df.join(MAderiv)
	return df

# No leaking, no shift required.	
# Exponential Moving Average.
def EMAi(df, n): 
	EMAdf = pd.Series(EMA(np.array(df.Close), timeperiod=n), name = 'EMA_' + str(n))
	df = df.join(EMAdf)
	return df

# No leaking, no shift required.
# Exponential Moving Average derivative.
def EMAderivatives(df, n):
	EMAderiv = pd.Series(df['EMA_'+str(n)].pct_change(), name = 'EMAderiv_' + str(n))#.shift(1) # Moving Averages derivates.
	#MAderiv = pd.Series(df['MA_'+str(n)].diff(), name = 'MAderiv_' + str(n))#.shift(1) # Moving Averages derivates.
	df = df.join(EMAderiv)
	return df

def PctDerivative(df, name):
	Deriv = pd.Series(df[str(name)].pct_change(), name = str(name)+'_deriv')#.shift(1) # Moving Averages derivates.
	#MAderiv = pd.Series(df['MA_'+str(n)].diff(), name = 'MAderiv_' + str(n))#.shift(1) # Moving Averages derivates.
	df = df.join(Deriv)
	return df

# MACD, MACD Signal and MACD histogram
def MACDi(df, n_fast, n_slow):  
	EMAfast = pd.Series(df['Close'].ewm(ignore_na = False, span = n_fast, min_periods = n_slow - 1, adjust = True).mean()) #Exponentially Weighted Moving Average
	EMAslow = pd.Series(df['Close'].ewm(ignore_na = False, span = n_slow, min_periods = n_slow - 1, adjust = True).mean()) #Exponentially Weighted Moving Average
	MACD = pd.Series(EMAfast - EMAslow, name = 'MACD')# + str(n_fast) + '_' + str(n_slow))
	MACDsign = pd.Series(MACD.ewm(ignore_na = False, span = 9, min_periods = 8, adjust = True).mean(), name = 'MACDsign')# + str(n_fast) + '_' + str(n_slow))
	MACDhist = pd.Series(MACD - MACDsign, name = 'MACDhist')# + str(n_fast) + '_' + str(n_slow))
	df = df.join(MACD) #MACD is MACD line,
	df = df.join(MACDsign) #MACDsign is signal line,
	df = df.join(MACDhist) #MACDhist is MACD histogram.

	# MACD pela TA-Lib:
	#pd.set_option('display.max_rows', 200)
	#macd, macdsignal, macdhist = MACD(np.array(df.Close), fastperiod=12, slowperiod=26, signalperiod=9)
	#print(pd.concat([df.MACD, pd.Series(macd), df.MACDsign, pd.Series(macdsignal), df.MACDhist, pd.Series(macdhist)],axis=1))
	#print(pd.concat([df.MACD, macd(df.Close, n_fast=12, n_slow=26, fillna=False), df.MACDsign, macd_diff(df.Close, n_fast=12, n_slow=26, n_sign=9, fillna=False), df_ohlc.MACDhist, macd_signal(df_ohlc.Close, n_fast=12, n_slow=26, n_sign=9, fillna=False),df_ohlc.Close, df_ohlc.Date], axis=1))
	return df

# Relative Strength Index
def RSIi(df, n):  
	"""
	i = 0
	UpI = [0]  
	DownI = [0]
	# The order of indexes was inverted. The most recent has the highest index, and the last had the largest index
	while i + 1 <= df.index[-1]:
	    UpMove = df.at[i + 1, 'High'] - df.at[i, 'High']
	    DownMove = df.at[i, 'Low'] - df.at[i + 1, 'Low']
	    #Closing price HIGHER than previous one -> UpD = price difference and DownD = 0
	    if UpMove > DownMove and UpMove > 0:  
	        UpD = UpMove  
	    else: UpD = 0  
	    UpI.append(UpD)
	    #Closing price LOWER than previous one -> DownD = price difference and UpD = 0
	    if DownMove > UpMove and DownMove > 0:  
	        DownD = DownMove  
	    else: DownD = 0  
	    DownI.append(DownD)  
	    i = i + 1

	UpI = pd.Series(UpI)
	DownI = pd.Series(DownI)
	#Exponential weigthed moving averages
	PosDI = UpI.ewm(ignore_na = False, span = n,min_periods = n - 1, adjust = True).mean() # n upward closing days price differences
	NegDI = DownI.ewm(ignore_na = False, span = n,min_periods = n - 1, adjust = True).mean() # n downward closing differences
	RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI')  
	df = df.join(RSI)
	"""
	RSIdf = pd.Series(RSI(np.array(df.Close), timeperiod=n), name = 'RSI')
	df = df.join(RSIdf)
	#print(pd.concat([pd.Series(RSI(np.array(df.Close), timeperiod=14)), df["RSI"], df.Volume, df.Close],axis=1))
	return df

# On-balance Volume  
def OBVi(df):#, n): 
	"""
	# This has a rolling average of the previous 20 OBV's
	i = 0
	OBV = [0]
	while i  < df.index[-1]: 
	    if (df.at[i + 1, 'Close'] - df.at[i, 'Close']) > 0:  
	        OBV.append(df.at[i + 1, 'Volume'])  
	    if (df.at[i + 1, 'Close'] - df.at[i, 'Close']) == 0:  
	        OBV.append(0)  
	    if (df.at[i + 1, 'Close'] - df.at[i, 'Close']) < 0:  
	        OBV.append(-df.at[i + 1, 'Volume'])  
	    i = i + 1  

	OBV = pd.Series(OBV)
	OBV_ma = pd.Series(OBV.rolling(window=n, center=False).mean(), name = 'OBV_' + str(n)) # n is the number of observations.
	df = df.join(OBV_ma)
	"""
	OBVdf = pd.Series(OBV(np.array(df.Close), np.array(df.Volume)), name = 'OBV')
	df = df.join(OBVdf)
	#print(pd.concat([pd.Series(OBV(np.array(df.Close), np.array(df.Volume))), df["OBV_20"], df.Volume, df.Close],axis=1))

	return df

# Rate of Change (ROC):
def ROCi(df, n):
	ROCdf = pd.Series(ROC(np.array(df.Close), timeperiod=n), name = 'ROC')
	df = df.join(ROCdf)
	return df

# Commodity Channel Index (CCI):
def CCIi(df, n):
	CCIdf = pd.Series(CCI(np.array(df.High), np.array(df.Low), np.array(df.Close), timeperiod=n), name = 'CCI')
	df = df.join(CCIdf)
	return df

# Average True Range:
def ATRi(df, n):
	ATRdf = pd.Series(ATR(np.array(df.High), np.array(df.Low), np.array(df.Close), timeperiod=n), name = 'ATR')
	df = df.join(ATRdf)
	return df

# Stochastic Oscillator:
def StochOscil(df, fastkPeriod, slowkPeriod, slowkMatype, slowdPeriod, slowdMatype, fastdperiod, fastdMatype):
	fastk, fastd = STOCHF(np.array(df.High), np.array(df.Low), np.array(df.Close), fastk_period=fastkPeriod, 
		fastd_period=fastdperiod, fastd_matype=fastdMatype)
	StochOscFastHist = pd.Series(fastk - fastd, name = 'StochFastHist')
	df = df.join(pd.Series(fastk, name='StochOscFastK'))
	df = df.join(pd.Series(fastd, name='StochOscFastD'))
	df = df.join(pd.Series(StochOscFastHist, name='StochFastHist'))

	slowk, slowd = STOCH(np.array(df.High), np.array(df.Low), np.array(df.Close),
		fastk_period=fastkPeriod, slowk_period=slowkPeriod, slowk_matype=slowkMatype, slowd_period=slowdPeriod, slowd_matype=slowdMatype)
	StochOscSlowHist = pd.Series(fastk - slowd, name = 'StochSlowHist')
	df = df.join(pd.Series(slowk, name='StochOscSlowK'))
	df = df.join(pd.Series(slowd, name='StochOscSlowD'))
	df = df.join(pd.Series(StochOscSlowHist, name='StochSlowHist'))

	return df

# Percentage Variation: If the percentage variation compared to previous entry is superior to 'n'->bit 1. 
# If inferior to '-n'->bit -1. Otherwise->bit 0.
def PercentVar(df, n):
	i = 1
	Var = [] 
	while i  <= df.index[-1]: 
	    if (df.at[i, 'VarPercent']) > n:
	        Var.append(1)
	    if (df.at[i, 'VarPercent']) < (-n):
	    	#print(df.at[i, 'VarPercent'])
	        Var.append(-1)
	    if ( (df.at[i, 'VarPercent'] < n) and (df.at[i, 'VarPercent'] > -n) ):
	    	Var.append(0)	    
	    i = i + 1  
	Var = pd.Series(Var, name = 'Var_' + str(n)).shift(1) # No comparation to be made on first position, hence a shift is needed
	df = df.join(Var)
	return df


# Indicator to show if an amount 'n' of POSITIVE percentual 'x' variations happens on last periods of time 't':
def AmountVar(df, x, n, t):
	if n>t:
		print('Minimum number of variations larger than time interval ---> ERROR, n must be < than t')
		return df
	Amount =[0]
	i=1
	while i < (t-1):
		if (df.at[i, 'VarPercent'] >= x ):
			Amount.append(1)
		if (df.at[i, 'VarPercent'] < x ):
			Amount.append(0)
		i=i+1

	while i <= df.index[-1]: #Go through all entries
		if(sum(n > x for n in df["VarPercent"].iloc[i-t+1:i+1]) >= n): 
			Amount.append(1)
		else:
			Amount.append(0)
		i=i+1

	"""
	i = 1 
	Amount = [0]
	AmountTemp=0
	while i < t: #Go to position t-1
		if(Amount[i-1] == 1 or df.at[i, 'VarPercent']>x):
			AmountTemp = AmountTemp+1
			if(AmountTemp >= n):
				Amount.append(1)
			else:
				Amount.append(0)
		else:
			Amount.append(0)
		i=i+1
	
	while i < df.index[-1]: #Go through all entries
		print(i-t+1,i+1)
		if df.at[i, 'VarPercent'] > 0: # If actual value has increased

			#temp contains number of entries that are over 'x' in a 't' time interval
			temp=pd.Series((np.select([(df["VarPercent"].iloc[i-t+1:i+1] >= x), (df["VarPercent"].iloc[i-t+1:i+1] < x)], [1, 0], default=0))).value_counts()
			print (df["VarPercent"].iloc[i-t+1:i+1])
			print(temp)
			if((temp.index == 1).any()):
				#print('\nOKE\n')
				if temp[1] >= n: #If variation of percentage 'x' happens 'n' or more times -> bit 1
					Amount.append(1)
				if temp[1] < n: #If not-> bit 0
					Amount.append(0)
			else:
				Amount.append(0)
		else: # If actual value has decreased
			Amount.append(0)

		i = i + 1
	"""
	Amount = pd.Series(Amount, name = 'AmountVar_' + str(x) + '_' + str(t))
	print('For AmountVar_' + str(x) + '_' + str(t) + ':\n' + str(Amount.value_counts(normalize=False)))
	df = df.join(Amount)
	#print(pd.concat([Amount,df["VarPercent"], df.Close, df.Date],axis=1))
	return df