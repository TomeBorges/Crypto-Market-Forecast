#!/usr/bin/python3

import pandas as pd
import numpy as np

from Dumps import InitListROI, InitListParams, DumpROIList, DumpParamsList, RemoveROI, RemoveParams

ListDirectory = 'Lists388'
DataDirectory = 'Data388'

#Retrieve the lists saved in the directory contained in ListDirectory
(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV) = InitListROI(ListDirectory)
(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC) = InitListParams(ListDirectory)

#Load File containing the names of the pairs to be removed from list.
with open('RemoveThesePairs.txt') as f:
	for line in f:
		FileLocation = str(line.strip('\n')) + '_1m.csv'
		RemoveROI(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, FileLocation.strip('.csv')) #Remove all ROI's from these pairs
		RemoveParams(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, FileLocation.strip('.csv')) #Remove all Params from these pairs

#Save List in ListDirectory without the unwanted parameters
DumpROIList(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, ListDirectory)
DumpParamsList(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, ListDirectory)

#In order to add the removed pairs back to the list. Simply re-run the pair on the main program.
#The modelCV is still saved, so it won't take too long.

#This program will be used to remove HOTBTC, NPXSBTC and DENTBTC, as the OLHC values in these pairs are always locked in the same values
# and these pairs do not have the liquidity to support the amount of transactions this algorithm wants.
