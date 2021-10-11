#!/usr/bin/python3

import pandas as pd
import numpy as np

import datetime
import time
from operator import itemgetter

from Dumps import InitListROI, InitListParams
from Files import  WriteAlgStatsCSV, WriteGroupStatsCSV, WriteFeaturesCSV, ROIbeaten

File = 'NoDerivs_VFinal/'
ListDirectory = File+'Lists388'
pd.options.display.max_columns = 500

(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV) = InitListROI(ListDirectory)
(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC) = InitListParams(ListDirectory)

features = ["RSI", "RSI_deriv", "MACD_deriv", "OBV", "OBV_deriv", "ROC", "CCI", "ATR", "StochOscSlowK", "StochOscSlowD"]
features.extend(["EMA_5_deriv", "EMA_10_deriv", "EMA_20_deriv", "EMA_100_deriv", "EMA_200_deriv", "VarPercent"])

# Opening a .csv file to store the results of each algorithm.
AlgStatsCSV = open(File+'AlgStatistics.csv','w')
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

#ROIbeaten(ListROILR + ListROIRF + ListROIXG + ListROISVC + ListROIMV)

# Opening a .csv file to store the results of each grouping.
GroupStatsCSV = open(File+'GroupStatistics.csv','w')
GroupStatsCSV.write('['+ str(datetime.datetime.now()) + ']\n' + 30*' ' + 'Grouping statistics File:')
#Percentage Rearrangement Statistics
if (any(e[2] == 'Percentage' for e in ListROIBH)):
	WriteGroupStatsCSV(GroupStatsCSV, ListROIBH, 'Percentage')
#Specific Value Rearrangement Statistics
if (any(e[2] == 'Amount' for e in ListROIBH)):
	WriteGroupStatsCSV(GroupStatsCSV, ListROIBH, 'Amount')
#Log of Specific Value Rearrangement Statistics
if (any(e[2] == 'LogAmount' for e in ListROIBH)):		
	WriteGroupStatsCSV(GroupStatsCSV, ListROIBH, 'LogAmount')
#Time Rearrangement Statistics
if (any(e[2] == 'Time' for e in ListROIBH)):
	WriteGroupStatsCSV(GroupStatsCSV, ListROIBH, 'Time')
#Total Statistics
WriteGroupStatsCSV(GroupStatsCSV, TotalListROI, 'Total')
# Close grouping statistics file
GroupStatsCSV.write(5*'\n'+ "END of Grouping statistics.\n\n")
GroupStatsCSV.close()

# Opening a .csv file to store the weights of each feature.
FeaturesCSV = open(File+'Features.csv','w')
FeaturesCSV.write('['+ str(datetime.datetime.now()) + ']\n' + 9*' ' + 'Feature weight File:\n\n')
ListParams = ListParamsLR + ListParamsRF + ListParamsXG + ListParamsSVC
WriteFeaturesCSV(FeaturesCSV, ListParams, features)
#FeaturesCSV.writelines(str(ListParams).replace('\n','\n,').replace("['","\n"))
FeaturesCSV.write(5*'\n'+ "END of feature weight file.\n\n")
FeaturesCSV.close()
