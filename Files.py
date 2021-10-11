#!/usr/bin/python3

import pandas as pd
import numpy as np

import os
import pickle

import time
from operator import itemgetter


# Function to write the final statistics of each algorithm in a .txt file. ANTIGO!!!!!!!
def WriteStatisticsTXT(StatisticsTXT, Lists, ListParams, string, features):

	if (string == 'LR'):
		StatisticsTXT.write("\n\n\n" + 32*'/' + " Logistic Regression:\n\n")
	if (string == 'RF'):
		StatisticsTXT.write("\n\n\n" + 32*'/' + " Random Forest Classifier:\n\n")
	if (string == 'XG'):
		StatisticsTXT.write("\n\n\n" + 32*'/' + " XG Boost:\n\n")
	if (string == 'SVC'):
		StatisticsTXT.write("\n\n\n" + 32*'/' + " Support Vector Classifier:\n\n")
	if (string == 'Total'):				
		StatisticsTXT.write('\n\n\n' + 132*'/' + '\n' + 57*'/' + ' Total Statistics ' + 57*'/' + '\n' + 132*'/' + '\n\n')
	
	AvgROIcolumn = np.mean([x[3] for x in Lists])
	StatisticsTXT.write('The average obtained ROI is: ' + str(AvgROIcolumn) + '%\n')
	AvgNegLogLosscolumn = np.mean([x[4] for x in Lists])
	StatisticsTXT.write('The average obtained Negative Log Loss for the Test is: ' + str(AvgNegLogLosscolumn) + '\n')
	AvgAccuracycolumn = np.mean([x[5] for x in Lists])
	StatisticsTXT.write('The average obtained Accuracy for the Test is: ' + str(AvgAccuracycolumn) + '%\n\n')

	StatisticsTXT.write('Obtained ROIs in descending order [Pair, Used Algorithm, Type of Candle, ROI, NegativeLogLoss, Accuracy]:\n')
	ListBestROI = sorted(Lists, key=itemgetter(3), reverse=True)
	StatisticsTXT.writelines(str(ListBestROI).replace('), (','\n'))
	"""
	if (string != 'Total'):	
		dfParams = pd.DataFrame(ListParams, columns=features)
		#StatisticsTXT.write('\n\nFeature importance:\n' + str(dfParams))
		print('Feature importance in ' + str(string) + ':\n' + str(dfParams))
		StatisticsTXT.write('\n\nMean value of each feature is: \n' + str(dfParams.mean().sort_values()))
	"""
	return

# Function to write the final statistics of each algorithm in a .csv file.
def WriteAlgStatsCSV(StatisticsCSV, Lists, ListParams, string, features):

	if (string == 'BH'):
		StatisticsCSV.write("\n\n\n" + "0- Buy & Hold:\n\n")
	if (string == 'LR'):
		StatisticsCSV.write("\n\n\n" + "1- Logistic Regression:\n\n")
	if (string == 'RF'):
		StatisticsCSV.write("\n\n\n" + "2- Random Forest Classifier:\n\n")
	if (string == 'XG'):
		StatisticsCSV.write("\n\n\n" + "3- XG Boost:\n\n")
	if (string == 'SVC'):
		StatisticsCSV.write("\n\n\n" + "4- Support Vector Classifier:\n\n")
	if (string == 'MV'):
		StatisticsCSV.write("\n\n\n" + "5- Majority Voting:\n\n")
	if (string == 'Total'):				
		StatisticsCSV.write("\n\n\n" + "6- Total Statistics:\n\n")

	# Position: 0-FileName, 1-Algorithm, 2-Rearrangement, 3-ROI, 4-TestLogLoss, 5-TestAccuracy, 6-NumTrades, 7-Periods In Market, 
	# 8-Num Profitable Trades, 9-Max Profit, 10-Max Loss, 11-Avg Profit, 12-Num of StopLoss Activated, 13-MDD max, 14-MDD draw, 15-MDD,
	# 16-AnnualSharpeRatio, 17-AnnualSortinoRatio, 18-Total Test Periods
	
	# OLDD
	# 18-B&H ROI, 19-B&H Periods In Market, 20-B&H num profit trades, 21-B&H MaxProfit, 
	# 22-B&H MaxLoss, 23-B&H MDD max, 24-B&H MDD draw, 25-B&H MDD, 26-B&H Annual Sharpe Ratio, 27-B&H Annual Sortino Ratio



	AvgROIcolumn = np.mean([x[3] for x in Lists])
	NumPositROI = len([x[3] for x in Lists if x[3] > 0])
	#AvgBHROIcolumn = np.mean([x[18] for x in Lists])
	#StatisticsCSV.write('Average obtained ROI is: %.10f %% (%d Positive ROIs out of %d: %.5f%%). B&H obtained %.10f %%\n' % (AvgROIcolumn, NumPositROI, len(Lists), 100*(NumPositROI/len(Lists)), AvgBHROIcolumn) )
	StatisticsCSV.write('Average obtained ROI is: %.10f %% (%d Positive ROIs out of %d: %.5f%%);\n' % (AvgROIcolumn, NumPositROI, len(Lists), 100*(NumPositROI/len(Lists))) )
	
	AvgAccuracycolumn = np.mean([x[5] for x in Lists])
	StatisticsCSV.write('Average obtained Accuracy for the Test is: %.10f;\n' % (AvgAccuracycolumn))
	AvgNegLogLosscolumn = np.mean([x[4] for x in Lists])
	StatisticsCSV.write('Average obtained Negative Log Loss for the Test is: %.10f;\n' % (AvgNegLogLosscolumn))	
	

	#print(np.mean([x[7] for x in Lists]))
	PercentProfitPosit = (np.mean([x[8] for x in Lists])/np.mean([x[6] for x in Lists]))
	#PercentProfirBH = len([x[18] for x in Lists if x[18] > 0]) / len([x[18] for x in Lists])
	#StatisticsCSV.write('Average percentage of profitable positions for the Test is: %.10f%%; Percentage of Profitable B&H ROIs is: %.10f%% \n' % (PercentProfitPosit*100, PercentProfirBH*100))
	StatisticsCSV.write('Average percentage of profitable positions for the Test is: %.10f%%;\n' % (PercentProfitPosit*100))
	
	#B&H is 100% of test period periods in market.
	AvgDaysMarket = np.mean([(x*1.0)/y for x, y in zip([x[7] for x in Lists], [x[18] for x in Lists])])
	StatisticsCSV.write('Average percentage of periods in market is %.10f%% (B&H is in market 100%% of all periods);\n'% (AvgDaysMarket*100))

	AvgProfitcolumn = np.mean([x[11] for x in Lists])
	StatisticsCSV.write('Average obtained Profit per Position for the Test is: %.10f%%;\n' % (AvgProfitcolumn*100))
	MaxPcolumn = np.mean([x[9] for x in Lists])
	MaxLcolumn = np.mean([x[10] for x in Lists])
	StatisticsCSV.write('Average Largest Gain is: %.10f%%. Average Largest Loss is: %.10f%%;\n'% (MaxPcolumn*100,MaxLcolumn*100))
	#MaxPBHcolumn = np.mean([x[21] for x in Lists])
	#MaxLBHcolumn = np.mean([x[22] for x in Lists])
	#StatisticsCSV.write('B&H\'s Average Largest Gain is: %.10f%%. B&H\'s Average Largest Loss is: %.10f%%\n'% (MaxPBHcolumn*100,MaxLBHcolumn*100))
	
	#print([x[17] for x in Lists if str(x[17]) != 'nan'])
	AvgMDDcolumn = np.mean([x[15] for x in Lists if (str(x[15]) != 'inf' and x[15]>0)])
	#AvgBHMDDcolumn = np.mean([x[24] for x in Lists if (str(x[24]) != 'inf' and x[24]>0)])
	StatisticsCSV.write('Average obtained MDD for the Test is: %.10f;\n' % (AvgMDDcolumn,))#, for B&H is: %.10f.\n' % (AvgMDDcolumn, AvgBHMDDcolumn))
	AvgSharpeRcolumn = np.mean([x[16] for x in Lists])# if str(x[16]) != 'nan']))
	#AvgBHSharpeRcolumn = np.mean(np.abs([x[26] for x in Lists if str(x[26]) != 'nan']))
	StatisticsCSV.write('Average obtained Annual Sharpe Ratio for the Test is: %.10f;\n'% (AvgSharpeRcolumn))#, for B&H is: %.10f.\n' % (AvgSharpeRcolumn, AvgBHSharpeRcolumn))
	AvgSortinoRcolumn = np.mean([x[17] for x in Lists])# if (str(x[17]) != 'inf' and str(x[17]) != 'nan')]))
	#AvgBHSortinoRcolumn = np.mean(np.abs([x[27] for x in Lists if (str(x[27]) != 'inf' and str(x[27]) != 'nan')]))
	StatisticsCSV.write('Average obtained Annual Sortino Ratio for the Test is: %.10f.\n'% (AvgSortinoRcolumn))#, for B&H is: %.10f.\n' % (AvgSortinoRcolumn, AvgBHSortinoRcolumn))
	#Acrescentar aqui o resto para fazer averages
		
	StatisticsCSV.write('Obtained ROIs in descending order:\n,Pair, Used Algorithm, Type of Candle, ROI, NegativeLogLoss, Accuracy, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeRatio, SortinoRatio, Total Test Periods\n,')
	ListBestROI = sorted(Lists, key=itemgetter(3), reverse=True)
	StatisticsCSV.writelines(str(ListBestROI).replace('), (','\n , '))
	"""
	if (string != 'Total' and string != 'MV'):	
		dfParams = pd.DataFrame(ListParams, columns=features)
		print('Feature importance in ' + str(string) + ':\n' + str(dfParams))
		StatisticsCSV.write('\n\nMean value of each feature is: \n,')
		StatisticsCSV.write(str(dfParams.mean().sort_values().to_csv()).replace('\n','\n,'))
	"""
	return

# Function to write the final statistics of each grouping in a .csv file.
def WriteGroupStatsCSV(StatisticsCSV, Lists, string):

	TempList = []

	if (string == 'Percentage'):
		StatisticsCSV.write("\n\n\n" + "1- Percentage Rearrangement:\n\n")
	if (string == 'Amount'):
		StatisticsCSV.write("\n\n\n" + "2- Specific Value Rearrangement:\n\n")
	if (string == 'LogAmount'):
		StatisticsCSV.write("\n\n\n" + "3- Log Of Specific Value Rearrangement:\n\n")
	if (string == 'Time'):
		StatisticsCSV.write("\n\n\n" + "4- Time Rearrangement:\n\n")
	if (string == 'Total'):				
		StatisticsCSV.write("\n\n\n" + "5- Total Statistics:\n\n")
	
	#Save all list entries that are apart of the actual grouping in a temporary list. 
	#This way results from different types of grouping are printed separately
	if (string != 'Total'):	
		for sublist in Lists:
			if sublist[2] == string:
				TempList.append(sublist)
	else:
		TempList = Lists
	# Position: 3-ROI, 4-TestLogLoss, 5-TestAccuracy
	
	# Position: 0-FileName, 1-Algorithm, 2-Rearrangement, 3-ROI, 4-TestLogLoss, 5-TestAccuracy, 6-NumTrades, 7-Periods In Market, 
	# 8-Num Profitable Trades, 9-Max Profit, 10-Max Loss, 11-Avg Profit, 12-Num of StopLoss Activated, 13-MDD max, 14-MDD draw, 15-MDD,
	# 16-AnnualSharpeRatio, 17-AnnualSortinoRatio, 18-Total Test Periods
	
	# OLDD
	# 18-B&H ROI, 19-B&H Periods In Market, 20-B&H num profit trades, 21-B&H MaxProfit, 
	# 22-B&H MaxLoss, 23-B&H MDD max, 24-B&H MDD draw, 25-B&H MDD, 26-B&H Annual Sharpe Ratio, 27-B&H Annual Sortino Ratio

	AvgROIcolumn = np.mean([x[3] for x in TempList])
	NumPositROI = len([x[3] for x in TempList if x[3] > 0])
	#AvgBHROIcolumn = np.mean([x[18] for x in TempList])
	StatisticsCSV.write('Average obtained ROI is: %.10f %% (%d Positive ROIs out of %d: %.5f%%);\n'% (AvgROIcolumn, NumPositROI, len(TempList), 100*(NumPositROI/len(TempList))))#. B&H obtained %.10f %%\n' % (AvgROIcolumn, NumPositROI, len(TempList), 100*(NumPositROI/len(TempList)), AvgBHROIcolumn) )
	
	AvgAccuracycolumn = np.mean([x[5] for x in TempList])
	StatisticsCSV.write('Average obtained Accuracy for the Test is: %.10f;\n' % (AvgAccuracycolumn))	
	AvgNegLogLosscolumn = np.mean([x[4] for x in TempList])
	StatisticsCSV.write('Average obtained Negative Log Loss for the Test is: %.10f;\n' % (AvgNegLogLosscolumn))	

	#B&H is 100% of test period periods in market.
	AvgDaysMarket = np.mean([(x*1.0)/y for x, y in zip([x[7] for x in TempList], [x[18] for x in TempList])])
	StatisticsCSV.write('Average percentage of periods in market is %.10f%% (B&H is in market 100%% of all periods);\n'% (AvgDaysMarket*100))

	#print(np.mean([x[7] for x in TempList]))
	PercentProfitPosit = (np.mean([x[8] for x in TempList])/np.mean([x[6] for x in TempList]))
	#PercentProfirBH = len([x[18] for x in TempList if x[18] > 0]) / len([x[18] for x in TempList])
	StatisticsCSV.write('Average percentage of profitable positions for the Test is: %.10f%%;\n'% (PercentProfitPosit*100))#; Percentage of Profitable B&H ROIs is: %.10f%% \n' % (PercentProfitPosit*100, PercentProfirBH*100))
	
	AvgProfitcolumn = np.mean([x[11] for x in TempList])
	StatisticsCSV.write('Average obtained Profit per Position for the Test is: %.10f%%;\n' % (AvgProfitcolumn*100))
	MaxPcolumn = np.mean([x[9] for x in TempList])
	MaxLcolumn = np.mean([x[10] for x in TempList])
	StatisticsCSV.write('Average Largest Gain is: %.10f%%. Average Largest Loss is: %.10f%%;\n'% (MaxPcolumn*100,MaxLcolumn*100))
	#MaxPBHcolumn = np.mean([x[21] for x in TempList])
	#MaxLBHcolumn = np.mean([x[22] for x in TempList])
	#StatisticsCSV.write('B&H\'s Average Largest Gain is: %.10f%%. B&H\'s Average Largest Loss is: %.10f%%\n'% (MaxPBHcolumn*100,MaxLBHcolumn*100))
	

	#print([x[17] for x in TempList if str(x[17]) != 'nan'])
	AvgMDDcolumn = np.mean([x[15] for x in TempList if (str(x[15]) != 'inf' and x[15]>0)])
	#AvgBHMDDcolumn = np.mean([x[24] for x in TempList if (str(x[24]) != 'inf' and x[24]>0)])
	StatisticsCSV.write('Average obtained MDD for the Test is: %.10f;\n'% (AvgMDDcolumn))#, for B&H is: %.10f.\n' % (AvgMDDcolumn, AvgBHMDDcolumn))
	AvgSharpeRcolumn = np.mean([x[16] for x in TempList])# if str(x[16]) != 'nan'])
	#AvgBHSharpeRcolumn = np.mean(np.abs([x[26] for x in TempList if str(x[26]) != 'nan']))
	StatisticsCSV.write('Average obtained Annual Sharpe Ratio for the Test is: %.10f;\n'% (AvgSharpeRcolumn))#, for B&H is: %.10f.\n' % (AvgSharpeRcolumn, AvgBHSharpeRcolumn))
	AvgSortinoRcolumn = np.mean([x[17] for x in TempList])# if (str(x[17]) != 'inf' and str(x[17]) != 'nan')]))
	#AvgBHSortinoRcolumn = np.mean(np.abs([x[27] for x in TempList if (str(x[27]) != 'inf' and str(x[27]) != 'nan')]))
	StatisticsCSV.write('Average obtained Annual Sortino Ratio for the Test is: %.10f.\n'% (AvgSortinoRcolumn))#, for B&H is: %.10f.\n' c, AvgBHSortinoRcolumn))
	
	StatisticsCSV.write('Obtained ROIs in descending order:\n,Pair, Used Algorithm, Type of Candle, ROI, NegativeLogLoss, Accuracy, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeRatio, SortinoRatio, Total Test Periods\n,')
	ListBestROI = sorted(TempList, key=itemgetter(3), reverse=True)
	StatisticsCSV.writelines(str(ListBestROI).replace('), (','\n , '))

	return
	
# Write a list in the form: ['Type of Grouping', 'Used Algorithm', [Unnamed List of feature weight]] into a .csv file.
def WriteFeaturesCSV(FeaturesCSV, ListParams, features):
	#print(ListParams)
	TempListP = []
	TempListA = []
	TempListL = []
	TempListT = []

	if (any(e[0] == 'Percentage' for e in ListParams)):
		for sublist in ListParams:
			if sublist[0] == 'Percentage':
				TempListP.append(sublist) #TempList containing only runs grouped in percentage, from all algorithms.
		FeaturesCSV.write('\n\nAverage feature importance in Percentage grouping is:\n\n')
		FeatureWriter(FeaturesCSV, TempListP, features)

	if (any(e[0] == 'Amount' for e in ListParams)):
		for sublist in ListParams:
			if sublist[0] == 'Amount':
				TempListA.append(sublist) #TempList containing only runs grouped in Amount, from all algorithms.
		FeaturesCSV.write('\n\nAverage feature importance in Amount grouping is:\n\n')
		FeatureWriter(FeaturesCSV, TempListA, features)

	if (any(e[0] == 'LogAmount' for e in ListParams)):
		for sublist in ListParams:
			if sublist[0] == 'LogAmount':
				TempListL.append(sublist) #TempList containing only runs grouped in LogAmount, from all algorithms.
		FeaturesCSV.write('\n\nAverage feature importance in LogAmount grouping is:\n\n')
		FeatureWriter(FeaturesCSV, TempListL, features)

	if (any(e[0] == 'Time' for e in ListParams)):
		for sublist in ListParams:
			if sublist[0] == 'Time':
				TempListT.append(sublist) #TempList containing only runs grouped in Time, from all algorithms.
		FeaturesCSV.write('\n\nAverage feature importance in Time grouping is:\n\n')
		FeatureWriter(FeaturesCSV, TempListT, features)

	FeaturesCSV.write('\n\nAverage feature importance of ALL groupings is:\n\n')
	FeatureWriter(FeaturesCSV, ListParams, features)


	return


def FeatureWriter(FeaturesCSV, TempList, features):
	
	TempDictLR = {}
	TempDictRF = {}
	TempDictXG = {}
	TempDictSVC = {}
	
	if (any(e[1] == 'LR' for e in TempList)):
		for sublist in TempList:
			if sublist[1] == 'LR':
				for i in range(len(sublist[3])):
					if sublist[3][i] in TempDictLR.keys():
						TempDictLR[sublist[3][i]].append([sublist[4][i]]) #Dictionary containing only 1 entry per feature with all weights.
					else:
						TempDictLR[sublist[3][i]] = [[sublist[4][i]]] #Dictionary containing only 1 entry per feature with all weights.
	
		dfParams = Dict2df(TempDictLR)

		FeaturesCSV.write('\nLogistic Regression - mean value of each feature is: \n,')
		FeaturesCSV.write(str(dfParams.set_index('Pair').to_csv()).replace('\n','\n,'))
	
	if (any(e[1] == 'RF' for e in TempList)):
		for sublist in TempList:
			if sublist[1] == 'RF':
				for i in range(len(sublist[3])):
					if sublist[3][i] in TempDictRF.keys():
						TempDictRF[sublist[3][i]].append([sublist[4][i]]) #Dictionary containing only 1 entry per feature with all weights.
					else:
						TempDictRF[sublist[3][i]] = [[sublist[4][i]]] #Dictionary containing only 1 entry per feature with all weights.
		dfParams = Dict2df(TempDictRF)

		FeaturesCSV.write('\nRandom Forest - mean value of each feature is: \n,')
		FeaturesCSV.write(str(dfParams.set_index('Pair').to_csv()).replace('\n','\n,'))
	
	if (any(e[1] == 'XG' for e in TempList)):
		for sublist in TempList:
			if sublist[1] == 'XG':
				for i in range(len(sublist[3])):
					if sublist[3][i] in TempDictXG.keys():
						TempDictXG[sublist[3][i]].append([sublist[4][i]]) #Dictionary containing only 1 entry per feature with all weights.
					else:
						TempDictXG[sublist[3][i]] = [[sublist[4][i]]] #Dictionary containing only 1 entry per feature with all weights.

		dfParams = Dict2df(TempDictXG)

		FeaturesCSV.write('\nXG Boost - mean value of each feature is: \n,')
		FeaturesCSV.write(str(dfParams.set_index('Pair').to_csv()).replace('\n','\n,'))

	if (any(e[1] == 'SVC' for e in TempList)):
		for sublist in TempList:
			if sublist[1] == 'SVC':
				for i in range(len(sublist[3])):
					if sublist[3][i] in TempDictSVC.keys():
						TempDictSVC[sublist[3][i]].append([sublist[4][i]]) #Dictionary containing only 1 entry per feature with all weights.
					else:
						TempDictSVC[sublist[3][i]] = [[sublist[4][i]]] #Dictionary containing only 1 entry per feature with all weights.
		
		dfParams = Dict2df(TempDictSVC)

		FeaturesCSV.write('\nSupport Vector Classifier - mean value of each feature is: \n,')
		FeaturesCSV.write(str(dfParams.set_index('Pair').to_csv()).replace('\n','\n,'))
	return

def Dict2df(TempDictSVC):

	avgList = []
	for k,v in TempDictSVC.items():
		avgList.append([k,len(v),np.mean(v)]) #Dictionary containing only 1 entry per feature and average of respective weights.

	dfParams = pd.DataFrame(avgList).sort_values(2)
	dfParams.columns = ["Pair","NumberUsedTimes","AverageWeight"]

	return dfParams

# Function to save the df's containing end results of the model, used to create plots without waiting 1 hour of modeling.
def SaveFitModel(FileLocation, OptionChosen, Algorithm, modelCV, Directory, LastEntry):
	if not os.path.exists(Directory + '/' + str(FileLocation).strip('.csv') + '_Pred/'): #If directory doesn't exist create it.
		os.makedirs(Directory + '/' + str(FileLocation).strip('.csv') + '_Pred/')

	with open (Directory+'/'+str(FileLocation).strip('.csv')+'_Pred/'+str(OptionChosen)+'_'+str(Algorithm)+'_'+str(LastEntry)+'.txt', 'wb') as fp:
		pickle.dump(modelCV, fp)
	
	return
	# Grouped data or indicators will not be stored as these take approximately 1 second to be calculated and would take lots of memory.
	# Predict function is also pretty fast, less than 1 second.
	# The problem is the fit function. Hence, the result of this function, the variable 'modelcv', will be stored. Only the variable 
	# 'modelCV' is stored in order to reduce exagerated memory usage.


# Function to retrieve the df's containing end results of the fit function ('modelCV' variable).
def ReturnFitModel(FileLocation, OptionChosen, Algorithm, Directory, LastEntry):
	if os.path.exists(Directory+'/'+str(FileLocation).strip('.csv')+'_Pred/'+str(OptionChosen)+'_'+str(Algorithm)+'_'+str(LastEntry)+'.txt'):
		with open (Directory+'/'+str(FileLocation).strip('.csv')+'_Pred/'+str(OptionChosen)+'_'+str(Algorithm)+'_'+str(LastEntry)+'.txt','rb') as fp:
			return pickle.load(fp) #The previous file contains a previousy fit 'modelCV'.
	else:
		return None # If 'None' is returned, no fit has been previously saved, hence fit has to be done.

def ROIbeaten(ListParams):
	BeatCounter = 0
	TotalCounter = 0

	for sublist in ListParams:
		if sublist[3] > sublist[18]:
			BeatCounter += 1;
		TotalCounter += 1

	print('B&H beaten %d out of %d times(%.5f%%).'% (BeatCounter, TotalCounter, 100*(BeatCounter/TotalCounter)))
	return
