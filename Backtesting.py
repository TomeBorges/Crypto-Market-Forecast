#!/usr/bin/python3

import pandas as pd
import numpy as np

COMMISSION_FEE = 0.999 # Binance fee 0.999 (0.1%), BitMex 0.99925 (0.075%)
SLIPPAGE = 0

# Function to calculate the ideal entry and exit points of the market. Vector Y to the fit function.
# Price Variation: If the Closing price in the previous period is larger or equal than actual period, bit 1. Otherwise, bit 0. 
# (Comissions are included in prices).
def PriceVar(df, ResultTXT):
	i = 2
	y = []
	
	# First entry
	if ((df.at[0, 'Close']*(2-COMMISSION_FEE)) < (df.at[1, 'Close']*(COMMISSION_FEE))): #Price growing, go in
		y.append(1)
	else:
		y.append(0)

	#print (pd.concat([df.ATR, df.Close],axis=1))
	
	while i <= df.index[-1]:
		
		#if(df.at[i, 'ATR'] >  0.0000005): 
		#Previously inside market
		if(y[-1]==1):
			if ((df.at[i-1, 'Close']*(2-COMMISSION_FEE)) < (df.at[i, 'Close']*(COMMISSION_FEE)) ): #Price continues growing, keep inside
				y.append(1)
				#print('1 - '+str(i))

			elif ((df.at[i-1, 'Close']*(2-COMMISSION_FEE)) == (df.at[i, 'Close']*(COMMISSION_FEE)) ): #Price growth stagnated. 
				# CONFIRM WHAT IS THE BEST OPTION!!! 0, 1 or previous entry
				y.append(1)
				#y.append(y[-1])
				#y.append(0)
				#print('2 - '+str(i))

			elif ((df.at[i-1, 'Close']*(2-COMMISSION_FEE)) > (df.at[i, 'Close']*(COMMISSION_FEE)) ): #Price dropped, exit market
				y.append(0)
				#print('3 - '+str(i))

		#Previously outside market
		else:
			if ((df.at[i-1, 'Close']*(2-COMMISSION_FEE)) < (df.at[i, 'Close']*(COMMISSION_FEE)) ): #Price now growing, enter market
				y.append(1)
				#print('4 - '+str(i))

			elif ((df.at[i-1, 'Close']*(2-COMMISSION_FEE)) == (df.at[i, 'Close']*(COMMISSION_FEE)) ): #Price drop stagnated. CONFIRM WHAT IS THE BEST OPTION!!! 0, 1 or previous entry
				#y.append(1)
				#y.append(y[-1])
				y.append(0)
				#print('5 - '+str(i))

			elif ((df.at[i-1, 'Close']*(2-COMMISSION_FEE)) > (df.at[i, 'Close']*(COMMISSION_FEE)) ): #Price dropping, stay out of market
				y.append(0)
				#print('6 - '+str(i))

		i = i + 1

	y = pd.Series(y, name = 'Real_PriceVar')
	
	#print(pd.concat([y, df.Close],axis=1))
	"""
	i = 1
	while i <= df.index[-1]:
		if (df.at[i - 1, 'Close'] < df.at[i, 'Close']):
			y.append(1)
		if (df.at[i - 1, 'Close'] == df.at[i, 'Close']):
			y.append(1)
		if (df.at[i - 1, 'Close'] > df.at[i, 'Close']):
			y.append(0)	
		i = i + 1
	y = pd.Series(y, name = 'Real_PriceVar')
	"""
	print('Using ' + str(df.index[-1]) + ' total samples in Y (0:'+str(y.value_counts(normalize=False)[0])+' 1:'+str(y.value_counts(normalize=False)[1])+')\n')
	ResultTXT.write('Out of ' + str(df.index[-1]) + ' samples in Y (train + test), the class relative distribution is:\n' + str(y.value_counts(normalize=True)))
	return y


# The total earnings are calculated starting with 1 Unit of Counter Currency. All is invested when prediction is bullish and all is sold
# when prediction is bearish:
def ROI(df, pred, y, ResultTXT, StopLoss):

	real = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
	real.drop(real.index[:pred.index[0]], inplace=True)

	#This function is prepared to do ROI by iterations or all at once (df w/all predicted points, 'pred', +df w/real points, 'y', are provided)
	if 'ROI' in df.columns:
		# Money used on predicted stock price
		CounterCurrency2 = 1+df.ROI[df.ROI.last_valid_index()]/100 # Start with unit of second crypto in pair
		print(CounterCurrency2)
		# Money used on real stock price, MAXIMUM possible earnings!		
		realCounterCurrency2 = 1+df.ROI[df.ROI.last_valid_index()]/100 # Start with unit of second crypto in pair
	else:
		# Money used on predicted stock price
		CounterCurrency2 = 1 # Start with unit of second crypto in pair		
		# Money used on real stock price, MAXIMUM possible earnings!
		realCounterCurrency2 = 1 # Start with unit of second crypto in pair

	BaseCurrency1 = 0 # Start with 0 units of first crypto in the pair
	realBaseCurrency1 = 0 # Start with 0 units of first crypto in the pair

	# Variables to count the number of trades.
	NumTrades = 0 
	NumTradesReal = 0

	# Simulation starts out of the market.
	InMarketReal = False
	InMarket = False

	# Variables to count the number of profitable trades and Max profit and loss.
	NumProfitTradesPred = 0
	NumProfitTradesReal = 0

	BuyPrice = 0
	BuyPriceReal= 0 
	MAXProfit = 0
	MAXLoss = 0
	PrevCounterCurrency2 = CounterCurrency2 #Variable  to calculate max profit and max loss
	# Variables to define a stop-loss.
	#StopLoss = 0.2#*100%
	StopLossCntr = 0 #Variable to count the number of stop losses.
	# BuyPrice is used for stop-loss as well

	# Variables to define a stop-gain.
	StopGainPercn = StopLoss*2 #*100%
	# BuyPrice is used for stop-gain as well
	
	#ROI = (Gain from Investment - Cost of Investment) / Cost of Investment
	ROI = [0] #ROI starts one entry before 1st prediction entry. In case an immediate investment is done (changing the ROI), this serves an indication that the investor started with a 0% ROI.
	
	#Crappy way of saving in a series whether nothing is done-0, buy-1, sell-2 or stop-loss-3.
	Action = [] #Action follows the entries of prediction. With no prediction nothing can be done
	ActionUpdate = False
	
	i = pred.index[0]
	# If first entry is 1, entry immediately. Otherwise, 0, default (stay out)
	if (pred[i] == 1): #Buy everything
			PrevCounterCurrency2 = CounterCurrency2
			BaseCurrency1 = (CounterCurrency2*(COMMISSION_FEE))/(df.at[i, 'Close']) # Commission fee taken into account
			CounterCurrency2 = 0
			InMarket = True
			ROI.append((100*((CounterCurrency2 + BaseCurrency1*(df.at[i, 'Close']))-1)))
			NumTrades += 1
			BuyPrice = df.at[i, 'Close']
			Action.append(1)
			PeriodsInMarket = 1
	else:
		ROI.append(100*(CounterCurrency2-1))
		Action.append(0)
		PeriodsInMarket = 0

	#####IDEAL#####
	if (real[i] == 1): #Buy everything
			realBaseCurrency1 = (realCounterCurrency2*(COMMISSION_FEE))/(df.at[i, 'Close']) # Commission fee taken into account
			realCounterCurrency2 = 0
			InMarketReal = True
			NumTradesReal = NumTradesReal + 1
			BuyPriceReal = df.at[i, 'Close']
	###############

	i = i + 1
	while i <= pred.index[-1]: #First prediction (row) is done, now go through all remaining ones
		if i == 20:
			exit()
		# If outside market:
		if(InMarket == False):
			#Predicted market price will rise: Buy!
			if (pred[i-1] == 0 and pred[i] == 1): # Out of market, buy everything
				PrevCounterCurrency2 = CounterCurrency2
				BaseCurrency1 = (CounterCurrency2*(COMMISSION_FEE))/(df.at[i, 'Close']) # Commission fee taken into account
				CounterCurrency2 = 0
				InMarket = True
				NumTrades += 1
				BuyPrice = df.at[i, 'Close']
				ActionUpdate = True
				Action.append(1)

		# If inside market:
		if(InMarket == True):
			PeriodsInMarket += 1
			#Predicted market price will drop: Sell!
			if (pred[i-1] == 1 and pred[i] == 0): #In market, sell everything
				CounterCurrency2 = (BaseCurrency1*COMMISSION_FEE*(df.at[i, 'Close'])) # Commission fee taken into account	
				BaseCurrency1 = 0
				InMarket = False
				# Fees have been taken into account already.
				#Update Number of profitable Trades
				if CounterCurrency2 > PrevCounterCurrency2:
					NumProfitTradesPred = NumProfitTradesPred+1
				#Update Maximum profit
				if (CounterCurrency2 - PrevCounterCurrency2) > MAXProfit:
					#print('Profit - ' + str(MAXProfit))
					MAXProfit = (CounterCurrency2 - PrevCounterCurrency2)/PrevCounterCurrency2
				#Update Maximum loss
				if (CounterCurrency2 - PrevCounterCurrency2) < MAXLoss:
					#print('Loss - ' + str(MAXLoss))
					MAXLoss = (CounterCurrency2 - PrevCounterCurrency2)/PrevCounterCurrency2
				ActionUpdate = True
				Action.append(2)
			
			#Predicted market will continue rising: Stay, but verify whether any of the stops should be used.
			if ( (pred[i-1] == 1 and pred[i] == 1) and 
				( ((df.at[i, 'Close'])*(COMMISSION_FEE)) < (BuyPrice*(2-COMMISSION_FEE)*(1-StopLoss)) ) ): # ADD or !!!! # If accumulated loss is larger than a threshold, sell everything
				#( ((df.at[i, 'Close'])*(COMMISSION_FEE)) > (BuyPrice*(1+StopGainPercn)) ) ): # Stop-gain, sell everything
				CounterCurrency2 = (BaseCurrency1*COMMISSION_FEE*(df.at[i, 'Close'])) # Commission fee taken into account	
				BaseCurrency1 = 0
				InMarket = False
				# Fees have been taken into account already.
				#Update Number of profitable Trades
				if CounterCurrency2 > PrevCounterCurrency2:
					NumProfitTradesPred = NumProfitTradesPred+1
				#Update Maximum profit
				if (CounterCurrency2 - PrevCounterCurrency2) > MAXProfit:
					#print('Profit - ' + str(MAXProfit))
					MAXProfit = (CounterCurrency2 - PrevCounterCurrency2)/PrevCounterCurrency2
				#Update Maximum loss
				if (CounterCurrency2 - PrevCounterCurrency2) < MAXLoss:
					#print('Loss - ' + str(MAXLoss))
					MAXLoss = (CounterCurrency2 - PrevCounterCurrency2)/PrevCounterCurrency2

				ActionUpdate = True
				StopLossCntr = StopLossCntr + 1
				Action.append(3)	
				#print('Activate stop-loss @' + str(df.at[i, 'Date']))
		
		if ActionUpdate == False: # No buy, sell or stop-loss happened this iteration.
			Action.append(0)

		ROI.append(100*((CounterCurrency2 + BaseCurrency1*(df.at[i, 'Close']))-1))

		#####IDEAL##### Simpolified equivalent to the above. Ideal results just for statistical results.
		if (InMarketReal == False and real[i-1] == 0 and real[i] == 1): #Buy
			realBaseCurrency1 = (realCounterCurrency2*(COMMISSION_FEE))/(df.at[i, 'Close']) # Commission fee taken into account
			realCounterCurrency2 = 0
			InMarketReal = True
			NumTradesReal = NumTradesReal+1
			BuyPriceReal = df.at[i, 'Close']
			
		if (InMarketReal == True and real[i-1] == 1 and real[i] == 0):#Sell
			realCounterCurrency2 = (realBaseCurrency1*(df.at[i, 'Close'])) # Commission fee taken into account
			realBaseCurrency1 = 0
			InMarketReal = False
			if BuyPriceReal*(2-COMMISSION_FEE) < (df.at[i, 'Close']*(COMMISSION_FEE)):# Profit has been made
				NumProfitTradesReal = NumProfitTradesReal+1
			else: #Loss has been made
				print(BuyPriceReal, (2-COMMISSION_FEE), BuyPriceReal*(2-COMMISSION_FEE)) #Check it as it shouldn't happen
				print((df.at[i, 'Close']*(COMMISSION_FEE))) #Check it as it shouldn't happen
		###############
		i = i + 1
		ActionUpdate = False
	
	# If iteration ends in market, instead of forced selling, save the position so next iteration starts in market.
	# Unless it is the last entry of last iteration, in this case just sell for statistical results.
	if(InMarket == True): #If iteration ends in market, sell on last value.
		CounterCurrency2 = (BaseCurrency1*COMMISSION_FEE*(df.at[i, 'Close'])) # Commission fee taken into account	
		BaseCurrency1 = 0
		ROI[-1] = (100*((CounterCurrency2 + BaseCurrency1*(df.at[i, 'Close']))-1))
		# Fees have been taken into account already.
		#Update Number of profitable Trades
		if CounterCurrency2 > PrevCounterCurrency2:
			NumProfitTradesPred = NumProfitTradesPred+1
		#Update Maximum profit
		if (CounterCurrency2 - PrevCounterCurrency2) > MAXProfit:
			#print('Profit - ' + str(MAXProfit))
			MAXProfit = (CounterCurrency2 - PrevCounterCurrency2)/PrevCounterCurrency2
		#Update Maximum loss
		if (CounterCurrency2 - PrevCounterCurrency2) < MAXLoss:
			#print('Loss - ' + str(MAXLoss))
			MAXLoss = (CounterCurrency2 - PrevCounterCurrency2)/PrevCounterCurrency2
		Action[-1]=2
	
	#####IDEAL#####
	if(InMarketReal == True): # If in ideal market when data ends, add 1 to the profitable trades. Just for data's integrity sake
		NumProfitTradesReal = NumProfitTradesReal+1
	###############
	
	ROI = pd.Series(ROI, name = 'ROI')
	ROI.index = pd.RangeIndex(start = (pred.index[0]-1), stop = pred.index[-1]+1)
	if 'ROI' in df.columns:
		#print('BOI')
		df.ROI.update(ROI)
	else:
		#ROI.to_csv(path='ROI1.csv')
		df = df.join(ROI)
	
	Action = pd.Series(Action, name = 'Action')
	Action.index = pd.RangeIndex(start = (pred.index[0]), stop = pred.index[-1]+1)

	if (len(Action) != len(pred)):
		print(len(Action), len(pred))
		print('\n\n\nCrap.\n\n\n')
		exit()
	#print(len(Action), len(pred))
	#print (pd.concat([df.Close, pred, df.ROI, df.Action],axis=1))
	if 'Action' in df.columns:
		#print('BOI')
		df.Action.update(Action)
	else:
		#ROI.to_csv(path='ROI1.csv')
		df = df.join(Action)
	
	#print('\nThe used comission fee is %f.' % COMMISSION_FEE)
	print('Earned: %.5f%% CC, out of: %.5f%% CC' % (ROI.iloc[-1], 100*((realCounterCurrency2 + realBaseCurrency1*(df.Close[i]))-1) ))
	#print('Total number of trades: %.f' % (NumTrades))
	#print('Total number of profitable trades: %.f (Percentage of profitable trades: %.5f%%)' % (NumProfitTradesPred, (100*NumProfitTradesPred/NumTrades)))
	#print('Ideal Total number of profitable trades: %.f (Ideal percentage of profitable trades: %.5f%%)' % (NumProfitTradesReal, (100*NumProfitTradesReal/NumTradesReal)))
	#print('Maximum profit is: %.10f; Maximum Loss is %.10f' % (MAXProfit, MAXLoss))
	#print('Average profit per trade is: %.10f\n' % (((df['ROI'][df.index[-3]]/100)*1)/NumTrades))
	
	if NumTrades != 0:
		AvgProfit = (((ROI.iloc[-1]/100)*1)/NumTrades)

	if (ResultTXT != None):
		ResultTXT.write('\nComission fee: %f%%.' % (1-COMMISSION_FEE))
		ResultTXT.write('\nEarned: %.5f%% Counter Currency, out of: %.5f%% Counter Currency.' % (ROI.iloc[-1], 100*((realCounterCurrency2 + realBaseCurrency1*(df.Close[i]))-1) ))
		ResultTXT.write('\nTotal number of trades: %.f; ----- ' % (NumTrades))
		if NumTrades != 0:
			ResultTXT.write('Total number of profitable trades: %.f (Percentage of profitable trades: %.5f%%).' % (NumProfitTradesPred, (100*NumProfitTradesPred/NumTrades)))
		ResultTXT.write('\nIdeal Total number of profitable trades: %.f (Ideal percentage of profitable trades: %.5f%%).' % (NumProfitTradesReal, (100*NumProfitTradesReal/NumTradesReal)))
		
		MAXProfit = 'NONE ' if MAXProfit == 0 else round(MAXProfit, 7)
		MAXLoss = 'NONE ' if MAXLoss == 0 else round(MAXLoss, 7)
		ResultTXT.write('\nMaximum profit is: ' + str(MAXProfit*100) + ' %; Maximum Loss is: ' + str(MAXLoss*100) + ' %')# % (MAXProfit, MAXLoss))
		if NumTrades != 0:
			ResultTXT.write(' ----- Average profit per trade is: %.10f %% [UofCC];' % (round(AvgProfit*100, 7)))
		ResultTXT.write('\nA Stop-Loss of %.1f%% was used %d time(s).' % (StopLoss*100, StopLossCntr))
		ResultTXT.write('\n\n\n')
		#print(df['ROI'], df.index[-1])

	if NumTrades != 0:
		return df, NumTrades, PeriodsInMarket, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr
	else:
		return df, NumTrades, PeriodsInMarket, 0, MAXProfit, MAXLoss, 0, StopLossCntr


def SharpeSortinoRatio(df, pred):
	
	#dfDate = df.groupby(pd.Grouper(freq='D'))
	dfDailyROI = pd.DataFrame({'Date':df.Date.iloc[pred.index[0]:-1], 'ROI': df.ROI.iloc[pred.index[0]:-1]})
	dfDailyROI['ROIcorrect'] = ((dfDailyROI.ROI.copy())/100)+1	
	dfDailyROI.set_index(dfDailyROI["Date"],inplace=True)
	#print(dfDailyROI)
	dfDailyROI = dfDailyROI.groupby(pd.Grouper(freq='D')).resample('D').last() #Group ROI by day
	dfDailyROI['DailyReturn'] = dfDailyROI['ROIcorrect'].pct_change(1)
	#print(dfDailyROI['DailyReturn'].mean(), dfDailyROI['DailyReturn'].std())
	
	# Assume an average annual risk-free rate over the period of 3.5%
	dfDailyROI['DailyReturn'] = dfDailyROI['DailyReturn']-0.035/365
	Sharpe_Ratio = dfDailyROI['DailyReturn'].mean() /dfDailyROI['DailyReturn'].std()
	#print (dfDailyROI['ROI'].mean(), dfDailyROI['ROI'].std())
	#print(dfDailyROI.shape[0])

	# Annualised Sharpe ratio based on the excess daily returns
	Annual_Sharpe_Ratio = np.sqrt(365)  * Sharpe_Ratio
	#print('Sharpe Ratio: %.5f; Annual Sharpe Ratio: %.5f.' %(Sharpe_Ratio, Annual_Sharpe_Ratio))

	dfDailyROI['downside_returns'] = 0
	#print(pd.concat([dfDailyROI.DailyReturn, dfDailyROI.downside_returns],axis=1))
	dfDailyROI.loc[dfDailyROI['DailyReturn'] < 0, 'downside_returns'] = dfDailyROI['DailyReturn']**2
	expected_return = (dfDailyROI['DailyReturn']-0.035/365).mean()
	#The square root of the average is the downside deviation.
	down_stdev = np.sqrt(dfDailyROI['downside_returns'].mean())
	sortino_ratio = (expected_return)/down_stdev
	Annual_Sortino_Ratio = np.sqrt(365) * sortino_ratio

	return (Annual_Sharpe_Ratio, Annual_Sortino_Ratio)


# Maximum Drawdown calculation:
def MaxDrawdown(df, pred):
#DrawDownt = Max i∈(Start Date,t) (RORi) − RORt ,
#MaxDrawDown = Max t∈(Start Date,End Date) (DrawDownt),

	#print(df.ROI)
	#print(pred.index[0], pred.index[1])

	dfROIval = ((df.ROI.copy())/100)+1
	#print(pd.concat([df.ROI, dfROIval],axis=1))
	drawdown = 0
	max_seen = dfROIval[pred.index[0]]
	for val in dfROIval[pred.index[1]:pred.index[-2]]:
		#print(max_seen, val, drawdown)
		max_seen = max(max_seen, val)
		drawdown = max(drawdown, max_seen-val)
	#print(100*(drawdown/max_seen))
	#print('MaxDrawdown: %.5f (%.5f %%); Peak Value:%.5f; ' % (drawdown, 100*(drawdown/max_seen), max_seen) ) #Max drawdown[%] = Drawdown/Peak
	return (max_seen, drawdown, 100*(drawdown/max_seen))

#Simple Buy & Hold strategy:
def BuyHold(df, y, pred):

	#B&H market actions: Pandas DataFrame to contain only zeros except for first and last entries that are 1's.
	#Through copying existing predict vector, indexes come out synchronized.
	predBH = pred.copy().replace(0,1)
	predBH.iloc[-1]=0

	dfpredBH = pd.concat([df.Date, df.Close],axis=1)

	print('Buy & Hold', end= ' ')
	StopLoss = float("inf") #No stop-loss must be activated
	dfpredBH, _, PeriodsInMarket, NumProfitTradesPred, MAXProfit, MAXLoss, _, _ = ROI(dfpredBH, predBH, y, None, StopLoss)
	BHSharpeR, BHSortinoR = SharpeSortinoRatio(dfpredBH, predBH)
	BHMDDmax, BHMDDdraw, BHMDD = MaxDrawdown(dfpredBH, predBH)
	
	return dfpredBH, predBH, PeriodsInMarket, NumProfitTradesPred, MAXProfit, MAXLoss, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, BHSortinoR
