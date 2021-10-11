#!/usr/bin/python3

import pandas as pd
import numpy as np

import os
import pickle

import time
from operator import itemgetter

# Initiate lists to contain the ROI of each algorithm run
def InitListROI(Directory):
	#Buy & Hold
	if os.path.exists(Directory + '/ListROIBH.txt'):
		with open (Directory + '/ListROIBH.txt', 'rb') as fp:
			ListROIBH = pickle.load(fp)
	else:
		ListROIBH = []
		open(Directory + '/ListROIBH.txt','w')
	#Logistic Regression
	if os.path.exists(Directory + '/ListROILR.txt'): # Verify whether the .txt file containing Lists exists
		with open (Directory + '/ListROILR.txt', 'rb') as fp: # If so, open the file in read binary mode
			ListROILR = pickle.load(fp) # Copy the saved list into the list variable
	else: #If file is inexistent:
		ListROILR = [] # List variable is created empty
		open(Directory + '/ListROILR.txt','w') #List file is created empty as well.
	#Same for all the remaining cases.
	#Random Forest Classifier
	if os.path.exists(Directory + '/ListROIRF.txt'):
		with open (Directory + '/ListROIRF.txt', 'rb') as fp:
			ListROIRF = pickle.load(fp)
	else:
		ListROIRF = []
		open(Directory + '/ListROIRF.txt','w')
	#XG Boost
	if os.path.exists(Directory + '/ListROIXG.txt'):
		with open (Directory + '/ListROIXG.txt', 'rb') as fp:
			ListROIXG = pickle.load(fp)
	else:
		ListROIXG = []
		open(Directory + '/ListROIXG.txt','w')
	#Support Vector Classifier
	if os.path.exists(Directory + '/ListROISVC.txt'):
		with open (Directory + '/ListROISVC.txt', 'rb') as fp:
			ListROISVC = pickle.load(fp)
	else:
		ListROISVC = []
		open(Directory + '/ListROISVC.txt','w')
	#Majority Voting
	if os.path.exists(Directory + '/ListROIMV.txt'):
		with open (Directory + '/ListROIMV.txt', 'rb') as fp:
			ListROIMV = pickle.load(fp)
	else:
		ListROIMV = []
		open(Directory + '/ListROIMV.txt','w')
	# Lists containing all previously calculated ROI's from the different algorithms
	return ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV

# Initiate lists to contain the importance of each feature from the different algorithms
def InitListParams(Directory):
	#Logistic Regression
	if os.path.exists(Directory + '/ListParamsLR.txt'): # Verify whether the .txt file containing Lists exists
		with open (Directory + '/ListParamsLR.txt', 'rb') as fp: # If so, open the file in read binary mode
			ListParamsLR = pickle.load(fp) # Copy the saved list into the list variable
	else: #If file is inexistent:
		ListParamsLR = [] # List variable is created empty
		open(Directory + '/ListParamsLR.txt','w') #List file is created empty as well.
	#Same for all the remaining cases.
	#Random Forest Classifier
	if os.path.exists(Directory + '/ListParamsRF.txt'):
		with open (Directory + '/ListParamsRF.txt', 'rb') as fp:
			ListParamsRF = pickle.load(fp)
	else:
		ListParamsRF = []
		open(Directory + '/ListParamsRF.txt','w')
	#XG Boost
	if os.path.exists(Directory + '/ListParamsXG.txt'):
		with open (Directory + '/ListParamsXG.txt', 'rb') as fp:
			ListParamsXG = pickle.load(fp)
	else:
		ListParamsXG = []
		open(Directory + '/ListParamsXG.txt','w')
	#Support Vector Classifier
	if os.path.exists(Directory + '/ListParamsSVC.txt'):
		with open (Directory + '/ListParamsSVC.txt', 'rb') as fp:
			ListParamsSVC = pickle.load(fp)
	else:
		ListParamsSVC = []
		open(Directory + '/ListParamsSVC.txt','w')

	# Lists containing all previously calculated parameter weights from the different algorithms
	return ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC

def DumpROIList(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, Directory):
	with open(Directory + '/ListROIBH.txt', 'wb') as fp: #Buy & Hold
		pickle.dump(ListROIBH, fp)
	with open(Directory + '/ListROILR.txt', 'wb') as fp: #Logistic Regression
		pickle.dump(ListROILR, fp)
	with open(Directory + '/ListROIRF.txt', 'wb') as fp: #Random Forest Classifier
		pickle.dump(ListROIRF, fp)
	with open(Directory + '/ListROIXG.txt', 'wb') as fp: #XG Boost
		pickle.dump(ListROIXG, fp)
	with open(Directory + '/ListROISVC.txt', 'wb') as fp: #Support Vector Classifier
		pickle.dump(ListROISVC, fp)
	with open(Directory + '/ListROIMV.txt', 'wb') as fp: #Majority Voting
		pickle.dump(ListROIMV, fp)
	return

def DumpParamsList(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, Directory):
	with open(Directory + '/ListParamsLR.txt', 'wb') as fp: #Logistic Regression
		pickle.dump(ListParamsLR, fp)
	with open(Directory + '/ListParamsRF.txt', 'wb') as fp: #Random Forest Classifier
		pickle.dump(ListParamsRF, fp)
	with open(Directory + '/ListParamsXG.txt', 'wb') as fp: #XG Boost
		pickle.dump(ListParamsXG, fp)
	with open(Directory + '/ListParamsSVC.txt', 'wb') as fp: #Support Vector Classifier
		pickle.dump(ListParamsSVC, fp)
	return


# Function to find and remove all ROI's associated with a specific pair, this way no duplicated results will be used on final stats calculation.
def RemoveROI(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, FileLocation):
	
	ElimFlag = 0

	if (any(e[0] == FileLocation for e in ListROIBH)):
		for sublist in ListROIBH[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[0] == FileLocation:
				ListROIBH.remove(sublist) #Only entries from specific pair are removed from List
				#print('Eliminating:' + str(sublist))
				ElimFlag = 1
	if (any(e[0] == FileLocation for e in ListROILR)):
		for sublist in ListROILR[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[0] == FileLocation:
				ListROILR.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[0] == FileLocation for e in ListROIRF)):
		for sublist in ListROIRF[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[0] == FileLocation:
				ListROIRF.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[0] == FileLocation for e in ListROIXG)):
		for sublist in ListROIXG[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[0] == FileLocation:
				ListROIXG.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[0] == FileLocation for e in ListROISVC)):
		for sublist in ListROISVC[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[0] == FileLocation:
				ListROISVC.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[0] == FileLocation for e in ListROIMV)):
		for sublist in ListROIMV[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[0] == FileLocation:
				ListROIMV.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if ElimFlag == 1:
		print('\n\nROI Elimination Happened!\n')
	return
# Function to find and remove all param's associated with a specific pair, this way no duplicated results will be used on final stats calculation.
def RemoveParams(ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, FileLocation):

	ElimFlag = 0

	if (any(e[2] == FileLocation for e in ListParamsLR)):
		for sublist in ListParamsLR[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[2] == FileLocation:
				ListParamsLR.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[2] == FileLocation for e in ListParamsRF)):
		for sublist in ListParamsRF[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[2] == FileLocation:
				ListParamsRF.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[2] == FileLocation for e in ListParamsXG)):
		for sublist in ListParamsXG[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[2] == FileLocation:
				ListParamsXG.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1
	if (any(e[2] == FileLocation for e in ListParamsSVC)):
		for sublist in ListParamsSVC[:]:#Look for indexes containg data about specific pair (variable 'FileLocation')
			if sublist[2] == FileLocation:
				ListParamsSVC.remove(sublist) #Only entries from specific pair are removed from List
				ElimFlag = 1

	if ElimFlag == 1:
		print('\n\nParams Elimination Happened!\n')
	return

def CreateTotalROI(Option):
	# Create a vector of dates from "2017-07-14 04:00:00" until "2018-10-30 00:00:00" with an entry every other minute.
	TotalROIdf = pd.DataFrame(pd.date_range("2017-07-14 04:00:00", "2018-10-30 00:00:00", freq="1min"),columns=['Date'])
	# Make dates the index
	TotalROIdf.set_index(pd.DatetimeIndex(TotalROIdf['Date']), inplace=True)
	TotalROIdf['LR']=0
	TotalROIdf['RF']=0
	TotalROIdf['XG']=0
	TotalROIdf['SVC']=0
	TotalROIdf['MV']=0
	TotalROIdf['BH']=0
	FirstTot = TotalROIdf.Date[-1]

	return TotalROIdf.drop(columns='Date'), FirstTot