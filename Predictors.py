#!/usr/bin/python3
#!/usr/bin/env python -W ignore::DeprecationWarning

import pandas as pd
import numpy as np

from bokeh.plotting import figure, show, output_file, gridplot, reset_output
from bokeh.layouts import column

from Backtesting import PriceVar, ROI, SharpeSortinoRatio, MaxDrawdown, BuyHold
from BokehVisualization import plotStock, plotEntranceExit, PlotIntervalBox, plotROI
from Files import SaveFitModel, ReturnFitModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV#, PurgedKFold
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, log_loss, roc_curve, accuracy_score

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

import statsmodels.api as sm # Implementing the model @ https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

import matplotlib.pyplot as plt

import os
import time
from operator import itemgetter
import itertools

import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')

DrawPredictionGraph = True
REFITALWAYS = False
ModelCVdir = 'ModelCVs388'

StopLoss = 0.2

# Variables (X) are setup as well as the desired obtained results (y) and are sent to each machine learning algorithm.
def PredictiveAlgorithms(df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV,
	ListParamsLR, ListParamsRF, ListParamsXG, ListParamsSVC, cols, ModelCVdir, TotalROIdf, FirstTot):
	
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

	#List of indicators to be used:
	#cols = ["EMA_5_deriv", "EMA_10_deriv", "EMA_20_deriv", "EMA_100_deriv", "EMA_200_deriv", "VarPercent"]
	#cols.extend(["RSI_deriv", "MACD_deriv","OBV_deriv", "ROC", "CCI", "ATR"])#, "StochOscSlowK", "StochOscSlowD"])
	#cols.extend(["MACD_deriv"])
	"""
	i=1
	for L in range(6, len(cols)+1):
		for subset in itertools.combinations(cols, L):
			print(i, list(subset))
			i=i+1
			ResultTXT.write("\nThe used subset is: " + str(subset)+".\n")
	"""
	if DrawPredictionGraph:#Create directory to save html's if drawing prediction graph is required.
		if not os.path.exists('HTMLs/' + FileLocation.strip('.csv') + '/'):#If directory doesn't exist create it.
			os.makedirs('HTMLs/' + FileLocation.strip('.csv') + '/')

	#print(pd.concat([X.EMA_5, X.MACD_deriv],axis=1))

	X = df_ohlc[cols].fillna(value = 0)[:-1] # NaN values replaced with 0 + last line is eliminated, no real data for those values, so useless for predict function.
	y = PriceVar(df_ohlc, ResultTXT)
	
	#print(pd.concat([y, df_ohlc.Close],axis=1))

	if DrawPredictionGraph:
		#Draw all ROIs in same plot
		ROIfig = figure(x_axis_type="datetime", plot_width=800, plot_height=200, title="Return on investment (ROI)", 
			tools="box_zoom,pan,wheel_zoom,reset", toolbar_location='above')

		ResultTXT.write("\n\n\n" + 34* ' ' + "Logistic Regression:")
		LRpredProb, PredictFigLR, ROIfigLR, Limits, FirstTot = LogistRegr(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT,
			ListROIBH, ListROILR, ListParamsLR, cols, ModelCVdir, ROIfig, TotalROIdf, FirstTot)
		y_predList_0 = pd.DataFrame({'LR_pred_0': LRpredProb[0]})
		y_predList_1 = pd.DataFrame({'LR_pred_1': LRpredProb[1]})
		
		ResultTXT.write("\n" + 34*' ' + "Random Forest:")
		RFpredProb, PredictFigRF, ROIfigRF = RandomForest(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIRF,
			ListParamsRF, cols, ModelCVdir, ROIfig, TotalROIdf)
		y_predList_0['RF_pred_0'] = pd.Series(RFpredProb[0], index=y_predList_0.index)
		y_predList_1['RF_pred_1'] = pd.Series(RFpredProb[1], index=y_predList_1.index)
		
		ResultTXT.write("\n" + 34*' ' + "XG Boost:")
		XGpredProb, PredictFigXG, ROIfigXG = XGBoost(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIXG,
			ListParamsXG, cols, ModelCVdir, ROIfig, TotalROIdf)
		y_predList_0['XG_pred_0'] = pd.Series(XGpredProb[0], index=y_predList_0.index)
		y_predList_1['XG_pred_1'] = pd.Series(XGpredProb[1], index=y_predList_1.index)

		ResultTXT.write("\n" + 34*' ' + "Support Vector Classifier:")
		SVCpredProb, PredictFigSVC, ROIfigSVC = SupportVectorC(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, 
			ListROISVC, ListParamsSVC, cols, ModelCVdir, ROIfig, TotalROIdf)
		y_predList_0['SVC_pred_0'] = pd.Series(SVCpredProb[0], index=y_predList_0.index)
		y_predList_1['SVC_pred_1'] = pd.Series(SVCpredProb[1], index=y_predList_1.index)

		ResultTXT.write("\n" + 16*' ' + "Majority Voting of previous 4 algorithms:")
		MVpredProb, PredictFigMV, ROIfigMV = MajorityVoting(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIMV, 
			y_predList_0, y_predList_1, Limits, ROIfig, TotalROIdf)
		

		show(ROIfig)
		#show(column(PredictFigLR, ROIfigLR, PredictFigRF, ROIfigRF, PredictFigXG, ROIfigXG, PredictFigSVC, ROIfigSVC, PredictFigMV, ROIfigMV))
		reset_output() #Reset Bokeh output, otherwise graphs will be laid on top of each other in same html

	else:
		#for scoring in ['f1_weighted', 'f1', 'neg_log_loss', 'roc_auc' ]:
		
		ResultTXT.write("\n\n\n" + 34* ' ' + "Logistic Regression:")
		LRpredProb, FirstTot = LogistRegr(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIBH, ListROILR, ListParamsLR, cols, ModelCVdir,
			'', TotalROIdf, FirstTot)
		y_predList_0 = pd.DataFrame({'LR_pred_0': LRpredProb[0]})
		y_predList_1 = pd.DataFrame({'LR_pred_1': LRpredProb[1]})
		
		ResultTXT.write("\n" + 34*' ' + "Random Forest:")
		RFpredProb = RandomForest(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIRF, ListParamsRF, cols, ModelCVdir,
			'', TotalROIdf)
		y_predList_0['RF_pred_0'] = pd.Series(RFpredProb[0], index=y_predList_0.index)
		y_predList_1['RF_pred_1'] = pd.Series(RFpredProb[1], index=y_predList_1.index)
		
		ResultTXT.write("\n" + 34*' ' + "XG Boost:")
		XGpredProb = XGBoost(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIXG, ListParamsXG, cols, ModelCVdir,
			'',TotalROIdf)
		y_predList_0['XG_pred_0'] = pd.Series(XGpredProb[0], index=y_predList_0.index)
		y_predList_1['XG_pred_1'] = pd.Series(XGpredProb[1], index=y_predList_1.index)

		ResultTXT.write("\n" + 34*' ' + "Support Vector Classifier:")
		SVCpredProb = SupportVectorC(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROISVC, ListParamsSVC, cols, 
			ModelCVdir, '', TotalROIdf)
		y_predList_0['SVC_pred_0'] = pd.Series(SVCpredProb[0], index=y_predList_0.index)
		y_predList_1['SVC_pred_1'] = pd.Series(SVCpredProb[1], index=y_predList_1.index)
	
		ResultTXT.write("\n" + 16*' ' + "Majority Voting of previous 4 algorithms:")
		MVpredProb = MajorityVoting(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIMV, y_predList_0, y_predList_1,
			'','', TotalROIdf)

	return FirstTot


#######################################################################################################################################
####################################################LogistRegr#########################################################################
#######################################################################################################################################

def LogistRegr(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROIBH, ListROI, ListParams, features, ModelCVdir, ROIfig, TotalROIdf, FirstTot):

	# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
	# Se quiser juntar a tal lei do genero: subiu 1% logo entrar obrigatoriamente.

	#Variable to contain the train and test limits.
	Limits = []
	#Variable to contain all predicted outputs for plotting
	y_predTotal = pd.DataFrame()
	PredProbTot = pd.DataFrame()
	#Accuracy and LogLoss of each iteration is saved here:
	#AccList = []
	#LogList = []
	#ListCoefs = []

	RefitFlag = False

	print('Logistic Regression: ')
	LogisticStartTime = time.time()

	# Para serem feitas previsoes futuras:
	# https://stackoverflow.com/questions/48884782/how-to-forecast-future-dataframe-using-sklearn-python

	# Split into a training set and a test set using a stratified k fold
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle = False) # 30% of the data will be used as testing data.
	
	iterCount = 0
	#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
	tscv = TimeSeriesSplit(n_splits=4)
	#print(tscv)
	for train_index, test_index in tscv.split(X):
		iterCount = iterCount+1
		ResultTXT.write('\n\nIteration ' + str(iterCount)  + ' - ')
		ResultTXT.write("Training entries: " + str(train_index[0]) + " : "+ str(train_index[-1]))
		ResultTXT.write(" - Testing entries: " + str(test_index[0]) + " : "+ str(test_index[-1])+'\n')

		X_train, X_test = X.loc[train_index], X.loc[test_index]
		y_train, y_test = y.loc[train_index], y.loc[test_index]
		Limits.append(train_index[-1])
		
		#X_train.plot()
		#plt.show()
		#print pd.concat([df_ohlc.MAderiv_5, df_ohlc['Pivots_0.01'], df_ohlc.Date], axis=1)

		# Create pipeline
		estimators = []
		#estimators.append(('feature_union', feature_union))
		#estimators.append(('select_best',  SelectKBest(k=11)))
		#estimators.append(('pca', PCA(n_components=3)))
		estimators.append(('normalization', StandardScaler()))
		estimators.append(('logistic', LogisticRegression(class_weight = 'balanced', max_iter=1000, solver = 'saga', penalty = 'l2')))
		pipeline = Pipeline(estimators)


		# Construct a grid of all the combinations of hyperparameters
		parameters = {
		#'select_best__k':[10,11],
	    #'logistic__solver':['liblinear','sag','newton-cg','lbfgs','saga'],
	    #'logistic__solver':['saga'],
	    #'logistic__penalty':['l1', 'l2'],
	    'logistic__C':[0.1, 0.01, 0.001, 0.0001]#0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,] #Valores para L1
		}

		scoring = 'neg_log_loss'
		modelCV = ReturnFitModel(FileLocation, OptionChosen, 'LR', ModelCVdir, train_index[-1])
		if (modelCV == None) or (REFITALWAYS == True): # If 'None' was returned from ReturnFitPredictPoints(), no fit is saved. Model must be fitted now.
			print('Refit Happening: ' + str(iterCount), end='\r')
			RefitFlag = True
			# run randomized search, 'n_iter_search' times
			#n_iter_search = 20
			#modelCV = RandomizedSearchCV(pipeline, param_distributions=parameters, scoring=scoring, cv=20, n_iter=n_iter_search, return_train_score = True)
			modelCV = GridSearchCV(pipeline, param_grid=parameters, scoring=scoring, cv=TimeSeriesSplit(n_splits=10), return_train_score = False, iid=False, refit=True)#http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
			# GridSearch will split train data further into train and test to tune the hyper-parameters passed to it. And finally fit the model on the whole train data with best found parameters.	
			# The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.

			#1) hyperparameter search, on train data saved in: /usr/local/lib/python3.5/dist-packages/sklearn/model_selection/_split.py
			#X_time = pd.Series(pd.to_timedelta(df_ohlc.Date.head(X_train.index[-1]+1)).dt.total_seconds())
			#inner_cv=PurgedKFold(n_splits=5, t1=X_time, pctEmbargo=0.01*X_time.index[-1]) # purged
			#print (df_ohlc.Date, X_train)
			#modelCV=GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=scoring, cv=inner_cv, iid=False)
			#print(modelCV)
			#Cross validation ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			
			# Verify whether a modelcv has already been done and pickle saved in specific file. If so, use that one to save time. If not, fit now.

			
			#FitTime = time.time()
			modelCV = modelCV.fit(X_train, y_train) # Fit the pipeline
			#print('Fit time: {:.5f} seconds.'.format(time.time() - FitTime))
			#print(modelCV.best_estimator_.named_steps['normalization'].mean_)
			#print(modelCV.best_estimator_.named_steps['normalization'].scale_)
			#exit()
			SaveFitModel(FileLocation, OptionChosen, 'LR', modelCV, ModelCVdir, train_index[-1]) # Save fitted model for posterior plotting.
		#else: modelCV already exists. It has already been retrieved and stored in 'modelCV' variable
		

		#Cross validation ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		"""
		def clfHyperFit(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.], n_jobs=-1,pctEmbargo=0,**fit_params):
		if set(lbl.values)=={0,1}:
			scoring='f1' # f1 for meta-labeling
		else:
			scoring='neg_log_loss' # symmetric towards all cases

		#1) hyperparameter search, on train data

		
		#2) fit validated model on the entirety of the data
		if bagging[1]>0:
			gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),max_samples=float(bagging[1]),max_features=float(bagging[2]),n_jobs=n_jobs)
			gs=gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
			gs=Pipeline([('bag',gs)])
		return gs
		"""
		# Evaluate pipeline
		"""
		kfold = KFold(n_splits=10)#, random_state=7)
		scoring = 'neg_log_loss'
		results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
		print("\n10-fold cross validation average logloss: %.5f +/- %.5f" % (results.mean(),results.std()*2))
		#y=data_final['y']
		"""
		#PredTime = time.time()
		y_pred = pd.Series(modelCV.predict(X_test), name = 'Pred_PriceVar') #Apply transforms to the data, and predict with the final estimator
		y_pred.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print('Pred time: {:.5f} seconds.'.format(time.time() - PredTime))

		PredProb = pd.DataFrame(modelCV.predict_proba(X_test))
		PredProb.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print(pd.concat([PredProb, y_pred, y_test],axis=1))

		"""
		TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, y_test, y_pred, PredProb, 'LR','')
		AccList.append(TestAccuracy)
		LogList.append(TestLogLoss)
		"""

		#print ('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(modelCV.best_score_))
		ResultTXT.write('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(round(modelCV.best_score_,7)) + ' (+/-'+ str(round(modelCV.cv_results_['std_test_score'][modelCV.best_index_],7))+')\n')
		"""
		results = pd.DataFrame(modelCV.cv_results_)	
		results = results.sort_values(by='rank_test_score', ascending=True)
		#print (results[['rank_test_score','param_logistic__solver', 'param_logistic__C', 'param_logistic__penalty', 'mean_test_score']].head())
		#(results[['rank_test_score','param_logistic__solver', 'param_logistic__C', 'param_logistic__penalty', 'mean_test_score']].head()).to_csv(ResultTXT, sep = '	')
		(results[['rank_test_score', 'param_logistic__C', 'mean_test_score']].head()).to_csv(ResultTXT, sep = '	')
		#print(cols, modelCV.best_estimator_.named_steps['select_best'].pvalues_)
		"""
		#print modelCV.best_estimator_.named_steps['logistic'].class_weight

		"""
		kfold = KFold(n_splits=10)#, random_state=7)
		modelCV = LogisticRegression(C=0.001)
		scoring = 'neg_log_loss'
		results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
		print("\nNo Pipeline 10-fold cross validation average logloss: %.5f +/- %.5f" % (results.mean(),results.std()*2))
		#y=data_final['y']
		"""

		
		"""
		logit_roc_auc = roc_auc_score(y_test, y_pred)#logreg.predict(X_test))
		fpr, tpr, thresholds = roc_curve(y_test, modelCV.predict_proba(X_test)[:,1])
		plt.figure(figsize=(8,8))
		plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
		plt.plot([0, 1], [0, 1],'r--')
		plt.axis([-0.005, 1, 0, 1.005])
		plt.xlabel('False Positive Rate')
		plt.xticks(np.arange(0,1, 0.05), rotation=90)
		plt.ylabel('True Positive Rate(Recall)')
		plt.title('Receiver operating characteristic Curve')
		plt.legend(loc="lower right")
		plt.show()
		"""
		#beta = modelCV.named_steps['logistic'].coef_ # Without RandomizedSearchCV
		beta = modelCV.best_estimator_.named_steps['logistic'].coef_  #The weight of each feature is returned. The index of the weight corresponds to the index of the feature in the 'features' list
		Coefs = sorted(list(zip(beta.ravel(), X_train.columns.ravel())), key=itemgetter(0))
		#print('Logistic regression coefficients:\n [%s]' % Coefs)
		#ListCoefs.append(Coefs)
		ResultTXT.write('Logistic regression coefficients:\n')
		ResultTXT.writelines(str(Coefs).replace('), (','\n'))
		#print([str(OptionChosen), 'LR'] + [beta[0]])

		#In case selectkbest was used, this way it is verified which features were used:
		#UsedFeatures = np.array(features)[modelCV.best_estimator_.named_steps['select_best'].get_support()]		
		#ListParams.append([str(OptionChosen)+str(ValueChosen), 'LR', FileLocation.strip('.csv')] + [UsedFeatures] + [beta[0]]) #And written in a list.

		#However, selectkbest worsens results so all features are always used!
		ListParams.append([str(OptionChosen)+str(ValueChosen), 'LR', FileLocation.strip('.csv')] + [features] + [beta[0]])

		#Write df with predicted values from all iterations/fits/intervals
		if y_predTotal.empty:
			y_predTotal = y_pred
		else:
			y_predTotal = y_predTotal.append(y_pred)

		#Write df with predicted probabilities from all iterations/fits/intervals
		if PredProbTot.empty:
			PredProbTot = PredProb
		else:
			PredProbTot = PredProbTot.append(PredProb)
			#PredProbTot[0] = PredProbTot[0].update(PredProb[0])
			#PredProbTot[1] = PredProbTot[1].update(PredProb[1])

	if RefitFlag == True:
		print('Refiting completed.')

	real_y = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
	real_y.drop(real_y.index[:y_predTotal.index[0]], inplace=True)

	#Statistical parameters:
	TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, real_y, y_predTotal, PredProbTot, 'LR', 'Final')
	df_ohlc, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr = ROI(df_ohlc, y_predTotal, y, ResultTXT, StopLoss)
	SharpeR, SortinoR = SharpeSortinoRatio(df_ohlc, y_predTotal)
	MDDmax, MDDdraw, MDD = MaxDrawdown(df_ohlc, y_predTotal)	
	ListROI.append((FileLocation.strip('.csv'), 'LR', str(OptionChosen)+str(ValueChosen), df_ohlc.ROI.iloc[-2], TestLogLoss, TestAccuracy,
		NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeR, SortinoR,
		real_y.shape[0]))
	
	
	dfBHROI, y_predTotBH1, BHPeriodsInMrkt, BHProfitblTrad, BHMAXP, BHMAXL, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, BHSortinoR = BuyHold(df_ohlc, y, y_predTotal)
	#print(pd.concat([real_y, y_predTotal, PredProbTot],axis=1))
	y_predTotBH0 = y_predTotBH1.copy().replace(1,0)
	y_predTotBH0.iloc[-1]=1
	#print(pd.concat([real_y,y_predTotBH1, pd.concat([y_predTotBH0, y_predTotBH1],axis=1)],axis=1))
	#exit()

	BHTestAccuracy, BHTestLogLoss = CalcAccLLConfM(ResultTXT, real_y, y_predTotBH1, pd.concat([y_predTotBH0, y_predTotBH1],axis=1), 'BH', 'Final')
	ListROIBH.append((FileLocation.strip('.csv'), 'BH', str(OptionChosen)+str(ValueChosen), dfBHROI.ROI.iloc[-2], BHTestLogLoss, 
		BHTestAccuracy, 1, BHPeriodsInMrkt, BHProfitblTrad, BHMAXP, BHMAXL, dfBHROI.ROI.iloc[-2]/100, 0, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, 
		BHSortinoR, real_y.shape[0]))
	
	#Add LR ROI to total sum of LR ROI's
	dfAlt = df_ohlc[['Date', 'ROI']].copy() # Create alternate df to not damage the original one
	dfAlt.set_index(pd.DatetimeIndex(dfAlt['Date']), inplace=True) #Index must be a date in order to do 1min resampling
	dfAlt = dfAlt.drop(columns='Date').resample('1min').pad() #Drop Date column, unnecessary. Resample Alt df to 1 min, all new entries (previously non existent) obtain the next value.
	TotalROIdf.LR = pd.concat([TotalROIdf.LR, dfAlt.ROI], axis=1).sum(axis=1)
	#print(TotalROIdf)
	#print('Alternative'+str(dfAlt.ROI.first_valid_index()))
	FirstTot = min(FirstTot, dfAlt.first_valid_index())	
	#min(FirstTot, dfAlt.first_valid_index())
	#print('TotalROI'+str(TotalROIdf.LR.first_valid_index()))
	#TotalROIdf.LR = TotalROIdf.LR+dfAlt.ROI #Simple addition, index by index
	del dfAlt
	#print(TotalROIdf.LR.first_valid_index())
	#Add B&H ROI to total sum of B&H ROI's
	dfAlt = dfBHROI[['Date', 'ROI']].copy()
	dfAlt.set_index(pd.DatetimeIndex(dfAlt['Date']), inplace=True)
	dfAlt = dfAlt.drop(columns='Date').resample('1min').pad()
	TotalROIdf.BH = pd.concat([TotalROIdf.BH, dfAlt.ROI], axis=1).sum(axis=1)
	del dfAlt
	#print(TotalROIdf)
	
	if DrawPredictionGraph:
		PredictFig = plotStock(df_ohlc, str(FileLocation).strip('_1m.csv') + ' - Train and Test with entry and exit points. - Logistic Regression - ' + str(ValueChosen) + ' ' + str(OptionChosen))
		PredictFig = plotEntranceExit(df_ohlc, PredictFig, y_predTotal, y, True, 'Close')
		ROIfig.line(df_ohlc.Date, dfBHROI.ROI, line_width=2, color="#000000", legend='B&H')
		ROIfig = plotROI(df_ohlc, 'LR', TestAccuracy, TestLogLoss, ROIfig, Limits, "#E11F16")		
		#ROIfig = plotEntranceExit(df_ohlc, ROIfig, y_predTotal, y, False, 'ROI')
		ROIfig.x_range = PredictFig.x_range # Pan or zooming actions across X axis is linked
		PlotIntervalBox(df_ohlc, PredictFig, Limits)
		PlotIntervalBox(df_ohlc, ROIfig, Limits)
		output_file( 'HTMLs/' + FileLocation.strip('.csv') + '/Prediction-' + str(OptionChosen) + ".html", 
			title = FileLocation.strip('.csv') + '-Prediction-' + str(OptionChosen))
		#show(column(PredictFig, ROIfig))
		#show(PredictFig)
		print('Elapsed time on LR fit, predict and plot: {:.5f} seconds.\n'.format(time.time() - LogisticStartTime))
		return PredProbTot, PredictFig, ROIfig, Limits, FirstTot
	else:
		print('Elapsed time on LR fit and predict: {:.5f} seconds.\n'.format(time.time() - LogisticStartTime))
		return PredProbTot, FirstTot


#######################################################################################################################################
####################################################RandomForest#######################################################################
#######################################################################################################################################

def RandomForest(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROI, ListParams, features, ModelCVdir, ROIfig, TotalROIdf):
	# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
	# Se quiser juntar a tal lei do genero: subiu 1% logo entrar obrigatoriamente.

	#Variable to contain the train and test limits.
	Limits = []
	#Variable to contain all predicted outputs for plotting
	y_predTotal = pd.DataFrame()
	PredProbTot = pd.DataFrame()
	#Accuracy and LogLoss of each iteration is saved here:
	#AccList = []
	#LogList = []
	#ListCoefs = []

	RefitFlag = False

	print('Random Forest: ')
	RForestStartTime = time.time()

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle = False) # 30% of the data will be used as testing data.
	
	iterCount = 0
	#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
	tscv = TimeSeriesSplit(n_splits=4)
	#print(tscv)
	for train_index, test_index in tscv.split(X):
		iterCount = iterCount+1
		ResultTXT.write('\n\nIteration ' + str(iterCount)  + ' - ')
		ResultTXT.write("Training entries: " + str(train_index[0]) + " : "+ str(train_index[-1]))
		ResultTXT.write(" - Testing entries: " + str(test_index[0]) + " : "+ str(test_index[-1])+'\n')

		X_train, X_test = X.loc[train_index], X.loc[test_index]
		y_train, y_test = y.loc[train_index], y.loc[test_index]
		Limits.append(train_index[-1])

		# Create pipeline
		estimators = []
		#estimators.append(('feature_union', feature_union))
		#estimators.append(('pca', PCA(n_components=3)))
		#estimators.append(('select_best',  SelectKBest(k=11)))
		estimators.append(('normalization', StandardScaler()))
		estimators.append(('random_forest', RandomForestClassifier(class_weight = 'balanced', n_estimators=400)))
		pipeline = Pipeline(estimators)

		# Construct a grid of all the combinations of hyperparameters
		parameters = {
		#'random_forest__max_depth': [2, 4],
		#'select_best__k':[10,11],
		'random_forest__min_samples_leaf': [9],
		'random_forest__min_samples_split': [9],
		}
		scoring = 'neg_log_loss'

		modelCV = ReturnFitModel(FileLocation, OptionChosen, 'RF', ModelCVdir, train_index[-1])
		if (modelCV == None) or (REFITALWAYS == True): # If 'None' was returned from ReturnFitPredictPoints(), no fit is saved. Model must be fitted now.
			print('Refit Happening: ' + str(iterCount), end='\r')
			RefitFlag = True
			# run grid search cross validation
			modelCV = GridSearchCV(pipeline, param_grid=parameters, scoring=scoring, cv=TimeSeriesSplit(n_splits=10), return_train_score = False, iid=False)
			#FitTime = time.time()
			modelCV = modelCV.fit(X_train, y_train) # Fit the pipeline
			#print('Fit time: {:.5f} seconds.'.format(time.time() - FitTime))
			SaveFitModel(FileLocation, OptionChosen, 'RF', modelCV, ModelCVdir, train_index[-1]) # Save fitted model for posterior plotting.
		#else: modelCV already exists. It has now been retrieved and stored in 'modelCV' variable

		#PredTime = time.time()
		y_pred = pd.Series(modelCV.predict(X_test), name = 'Pred_PriceVar') #Apply transforms to the data, and predict with the final estimator
		y_pred.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print('Pred time: {:.5f} seconds.'.format(time.time() - PredTime))
		
		PredProb = pd.DataFrame(modelCV.predict_proba(X_test))
		PredProb.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print(pd.concat([PredProb, y_pred, y_test],axis=1))

		"""
		TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, y_test, y_pred, PredProb, 'RF','')
		AccList.append(TestAccuracy)
		LogList.append(TestLogLoss)
		"""

		#print ('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(modelCV.best_score_))
		ResultTXT.write('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(round(modelCV.best_score_,7)) + ' (+/-'+ str(round(modelCV.cv_results_['std_test_score'][modelCV.best_index_],7))+')\n')
		
		"""
		results = pd.DataFrame(modelCV.cv_results_)	
		results = results.sort_values(by='rank_test_score', ascending=True)
		#print(results)
		#print (results[['rank_test_score','param_logistic__solver', 'param_logistic__C', 'param_logistic__penalty', 'mean_test_score']].head())
		(results[['rank_test_score','params', 'mean_test_score']].head()).to_csv(ResultTXT, sep = '	')
		#print(cols, modelCV.best_estimator_.named_steps['select_best'].pvalues_)
		"""

		beta = modelCV.best_estimator_.named_steps['random_forest'].feature_importances_ # Para RandomForestClassifier c/ RandomizedSearchCV
		Coefs = sorted(list(zip(beta.ravel(), X_train.columns.ravel())), key=itemgetter(0))
		#print('Logistic regression coefficients:\n [%s]' % Coefs)
		ResultTXT.write('Random Forest coefficients:\n')
		ResultTXT.writelines(str(Coefs).replace('), (','\n'))

		ListParams.append([str(OptionChosen)+str(ValueChosen), 'RF', FileLocation.strip('.csv')] + [features] + [beta])
		
		#Write df with predicted values from all iterations/fits/intervals
		if y_predTotal.empty:
			y_predTotal = y_pred
		else:
			y_predTotal = y_predTotal.append(y_pred)

		#Write df with predicted probabilities from all iterations/fits/intervals
		if PredProbTot.empty:
			PredProbTot = PredProb
		else:
			PredProbTot = PredProbTot.append(PredProb)
			#PredProbTot[0] = PredProbTot[0].update(PredProb[0])
			#PredProbTot[1] = PredProbTot[1].update(PredProb[1])
	
	if RefitFlag == True:
		print('Refiting completed.')

	real_y = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
	real_y.drop(real_y.index[:y_predTotal.index[0]], inplace=True)

	#Statistical parameters:
	TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, real_y, y_predTotal, PredProbTot, 'RF', 'Final')
	df_ohlc, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr = ROI(df_ohlc, y_predTotal, y, ResultTXT, StopLoss)
	SharpeR, SortinoR = SharpeSortinoRatio(df_ohlc, y_predTotal)
	MDDmax, MDDdraw, MDD = MaxDrawdown(df_ohlc, y_predTotal)	
	#dfBHROI, BHPeriodsInMrkt, BHProfitblTrad, BHMAXP, BHMAXL, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, BHSortinoR = BuyHold(df_ohlc, y, y_predTotal)
	ListROI.append((FileLocation.strip('.csv'), 'RF', str(OptionChosen)+str(ValueChosen), df_ohlc.ROI.iloc[-2], TestLogLoss, TestAccuracy,
		NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeR, SortinoR,
		real_y.shape[0]))

	#Add RF ROI to total sum of RF ROI's
	dfAlt = df_ohlc[['Date', 'ROI']].copy()
	dfAlt.set_index(pd.DatetimeIndex(dfAlt['Date']), inplace=True)
	dfAlt = dfAlt.drop(columns='Date').resample('1min').pad()
	TotalROIdf.RF = pd.concat([TotalROIdf.RF, dfAlt.ROI], axis=1).sum(axis=1)
	del dfAlt

	if DrawPredictionGraph:
		PredictFig = plotStock(df_ohlc, str(FileLocation).strip('_1m.csv') + ' - Train and Test with entry and exit points. - Random Forest - ' + str(ValueChosen) + ' ' + str(OptionChosen))
		PredictFig = plotEntranceExit(df_ohlc, PredictFig, y_predTotal, y, True, 'Close')
		ROIfig = plotROI(df_ohlc, 'RF', TestAccuracy, TestLogLoss, ROIfig, Limits, "#1AAE03")
		#ROIfig = plotEntranceExit(df_ohlc, ROIfig, y_predTotal, y, False, 'ROI')
		ROIfig.x_range = PredictFig.x_range # Pan or zooming actions across X axis is linked
		#PlotIntervalBox(df_ohlc, PredictFig, Limits)
		#PlotIntervalBox(df_ohlc, ROIfig, Limits)
		output_file( 'HTMLs/' + FileLocation.strip('.csv') + '/Prediction-' + str(OptionChosen) + ".html", 
			title = FileLocation.strip('.csv') + '-Prediction-' + str(OptionChosen))
		#show(column(PredictFig, ROIfig))

		print('Elapsed time on RF fit, predict and plot: {:.5f} seconds.\n'.format(time.time() - RForestStartTime))
		return PredProbTot, PredictFig, ROIfig
	
	else:
		print('Elapsed time on RF fit and predict: {:.5f} seconds.\n'.format(time.time() - RForestStartTime))
		return PredProbTot


#######################################################################################################################################
####################################################XGBoost############################################################################
#######################################################################################################################################

def XGBoost(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROI, ListParams, features, ModelCVdir, ROIfig, TotalROIdf):
	#Variable to contain the train and test limits.
	Limits = []
	#Variable to contain all predicted outputs for plotting
	y_predTotal = pd.DataFrame()
	PredProbTot = pd.DataFrame()
	#Accuracy and LogLoss of each iteration is saved here:
	#AccList = []
	#LogList = []
	#ListCoefs = []

	RefitFlag = False

	print('XGBoost: ')
	XGBoostStartTime = time.time()

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle = False) # 30% of the data will be used as testing data.
	
	iterCount = 0
	#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
	tscv = TimeSeriesSplit(n_splits=4)
	#print(tscv)
	for train_index, test_index in tscv.split(X):
		iterCount = iterCount+1
		ResultTXT.write('\n\nIteration ' + str(iterCount)  + ' - ')
		ResultTXT.write("Training entries: " + str(train_index[0]) + " : "+ str(train_index[-1]))
		ResultTXT.write(" - Testing entries: " + str(test_index[0]) + " : "+ str(test_index[-1])+'\n')

		X_train, X_test = X.loc[train_index], X.loc[test_index]
		y_train, y_test = y.loc[train_index], y.loc[test_index]
		Limits.append(train_index[-1])

		#Specific for XGBoost. equivalent to class_weight to address class imbalance.
		ClassProportion = y_train.value_counts(normalize=False)[0]/y_train.value_counts(normalize=False)[1]
		
		# Create pipeline
		estimators = []
		#estimators.append(('feature_union', feature_union))
		#estimators.append(('pca', PCA(n_components=3)))
		#estimators.append(('select_best',  SelectKBest(k=11)))
		estimators.append(('normalization', StandardScaler()))
		estimators.append(('XGBoost', xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=100, scale_pos_weight=ClassProportion, learning_rate=0.01)))
		pipeline = Pipeline(estimators)

		# Construct a grid of all the combinations of hyperparameters
		parameters = {
		#'XGBoost__n_estimators': [100],
		#'XGBoost__max_depth':[3],
		'XGBoost__gamma': [0,1],
		'XGBoost__min_child_weight' : [1,2],
		'XGBoost__reg_lambda': [1],
		#'select_best__k':[10,11],
		#'XGBoost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
		
		
		}
		scoring = 'neg_log_loss'

		modelCV = ReturnFitModel(FileLocation, OptionChosen, 'XG', ModelCVdir, train_index[-1])
		if (modelCV == None) or (REFITALWAYS == True): # If 'None' was returned from ReturnFitPredictPoints(), no fit is saved. Model must be fitted now.
			print('Refit Happening: ' + str(iterCount), end='\r')
			RefitFlag = True
			# run grid search cross validation
			modelCV = GridSearchCV(pipeline, param_grid=parameters, scoring=scoring, cv=TimeSeriesSplit(n_splits=10), return_train_score = False, iid=False)
			#FitTime = time.time()
			modelCV = modelCV.fit(X_train, y_train) # Fit the pipeline
			#print('Fit time: {:.5f} seconds.'.format(time.time() - FitTime))
			SaveFitModel(FileLocation, OptionChosen, 'XG', modelCV, ModelCVdir, train_index[-1]) # Save fitted model for posterior plotting.
		#else: modelCV already exists. It has now been retrieved and stored in 'modelCV' variable

		#PredTime = time.time()
		y_pred = pd.Series(modelCV.predict(X_test), name = 'Pred_PriceVar') #Apply transforms to the data, and predict with the final estimator
		y_pred.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print('Pred time: {:.5f} seconds.'.format(time.time() - PredTime))

		PredProb = pd.DataFrame(modelCV.predict_proba(X_test))
		PredProb.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print(pd.concat([PredProb, y_pred, y_test],axis=1))

		"""
		TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, y_test, y_pred, PredProb, 'XG','')
		AccList.append(TestAccuracy)
		LogList.append(TestLogLoss)
		"""

		#print ('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(modelCV.best_score_))
		ResultTXT.write('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(round(modelCV.best_score_,7)) + ' (+/-'+ str(round(modelCV.cv_results_['std_test_score'][modelCV.best_index_],7))+')\n')
		"""
		results = pd.DataFrame(modelCV.cv_results_)	
		results = results.sort_values(by='rank_test_score', ascending=True)
		(results[['rank_test_score','params', 'mean_test_score']].head()).to_csv(ResultTXT, sep = '	')
		"""

		beta = modelCV.best_estimator_.named_steps['XGBoost'].feature_importances_ # Para RandomForestClassifier c/ RandomizedSearchCV
		Coefs = sorted(list(zip(beta.ravel(), X_train.columns.ravel())), key=itemgetter(0))
		#print('nXGBoost coefficients:\n [%s]' % Coefs)
		ResultTXT.write('XGBoost coefficients:\n')
		ResultTXT.writelines(str(Coefs).replace('), (','\n'))
		
		ListParams.append([str(OptionChosen)+str(ValueChosen), 'XG', FileLocation.strip('.csv')] + [features] + [beta])
		
		#Write df with predicted values from all iterations/fits/intervals
		if y_predTotal.empty:
			y_predTotal = y_pred
		else:
			y_predTotal = y_predTotal.append(y_pred)

		#Write df with predicted probabilities from all iterations/fits/intervals
		if PredProbTot.empty:
			PredProbTot = PredProb
		else:
			PredProbTot = PredProbTot.append(PredProb)
			#PredProbTot[0] = PredProbTot[0].update(PredProb[0])
			#PredProbTot[1] = PredProbTot[1].update(PredProb[1])
	
	if RefitFlag == True:
		print('Refiting completed.')

	real_y = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
	real_y.drop(real_y.index[:y_predTotal.index[0]], inplace=True)

	#Statistical parameters:
	TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, real_y, y_predTotal, PredProbTot, 'XG', 'Final')
	df_ohlc, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr = ROI(df_ohlc, y_predTotal, y, ResultTXT, StopLoss)
	SharpeR, SortinoR = SharpeSortinoRatio(df_ohlc, y_predTotal)
	MDDmax, MDDdraw, MDD = MaxDrawdown(df_ohlc, y_predTotal)	
	#dfBHROI, BHPeriodsInMrkt, BHProfitblTrad, BHMAXP, BHMAXL, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, BHSortinoR = BuyHold(df_ohlc, y, y_predTotal)
	ListROI.append((FileLocation.strip('.csv'), 'XG', str(OptionChosen)+str(ValueChosen), df_ohlc.ROI.iloc[-2], TestLogLoss, TestAccuracy,
		NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeR, SortinoR,
		real_y.shape[0]))

	#Add XG ROI to total sum of XG ROI's
	dfAlt = df_ohlc[['Date', 'ROI']].copy()
	dfAlt.set_index(pd.DatetimeIndex(dfAlt['Date']), inplace=True)
	dfAlt = dfAlt.drop(columns='Date').resample('1min').pad()
	TotalROIdf.XG = pd.concat([TotalROIdf.XG, dfAlt.ROI], axis=1).sum(axis=1)
	del dfAlt

	if DrawPredictionGraph:
		PredictFig = plotStock(df_ohlc, str(FileLocation).strip('_1m.csv') + ' - Train and Test with entry and exit points. - XGBoost - ' + str(ValueChosen) + ' ' + str(OptionChosen))
		PredictFig = plotEntranceExit(df_ohlc, PredictFig, y_predTotal, y, True, 'Close')
		ROIfig = plotROI(df_ohlc, 'GTB', TestAccuracy, TestLogLoss, ROIfig, Limits, "#FECD04")
		#ROIfig = plotEntranceExit(df_ohlc, ROIfig, y_predTotal, y, False, 'ROI')
		ROIfig.x_range = PredictFig.x_range # Pan or zooming actions across X axis is linked
		#PlotIntervalBox(df_ohlc, PredictFig, Limits)
		#PlotIntervalBox(df_ohlc, ROIfig, Limits)
		output_file( 'HTMLs/' + FileLocation.strip('.csv') + '/Prediction-' + str(OptionChosen) + ".html", 
			title = FileLocation.strip('.csv') + '-Prediction-' + str(OptionChosen))
		#show(column(PredictFig, ROIfig))

		print('Elapsed time on XGB fit, predict and plot: {:.5f} seconds.\n'.format(time.time() - XGBoostStartTime))
		return PredProbTot, PredictFig, ROIfig
	
	else:
		print('Elapsed time on XGB fit and predict: {:.5f} seconds.\n'.format(time.time() - XGBoostStartTime))
		return PredProbTot


#######################################################################################################################################
########################################################SVC############################################################################
#######################################################################################################################################

def SupportVectorC(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROI, ListParams, features, ModelCVdir, ROIfig, TotalROIdf):
	#Variable to contain the train and test limits.
	Limits = []
	#Variable to contain all predicted outputs for plotting
	y_predTotal = pd.DataFrame()
	PredProbTot = pd.DataFrame()
	#Accuracy and LogLoss of each iteration is saved here:
	#AccList = []
	#LogList = []
	#ListCoefs = []

	RefitFlag = False

	print('Support Vector Classifier: ')
	SVCStartTime = time.time()

	iterCount = 0
	#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
	tscv = TimeSeriesSplit(n_splits=4)
	#print(tscv)
	for train_index, test_index in tscv.split(X):
		iterCount = iterCount+1
		ResultTXT.write('\n\nIteration ' + str(iterCount)  + ' - ')
		ResultTXT.write("Training entries: " + str(train_index[0]) + " : "+ str(train_index[-1]))
		ResultTXT.write(" - Testing entries: " + str(test_index[0]) + " : "+ str(test_index[-1])+'\n')

		X_train, X_test = X.loc[train_index], X.loc[test_index]
		y_train, y_test = y.loc[train_index], y.loc[test_index]
		Limits.append(train_index[-1])

		# Create pipeline
		estimators = []
		#estimators.append(('feature_union', feature_union))
		#estimators.append(('pca', PCA(n_components=3)))
		#estimators.append(('select_best',  SelectKBest(k=11)))
		estimators.append(('normalization', StandardScaler()))
		estimators.append(('SVC', SVC(C=1, class_weight = 'balanced', probability=True, max_iter= 10000))) # para ver quantas iteracoes sao feitas adicionar verbose=2 aos parametros.
		pipeline = Pipeline(estimators)

		# Construct a grid of all the combinations of hyperparameters
		parameters = {
		'SVC__kernel': ['linear'],
		#'select_best__k':[10,11],
		#'SVC__C':[1],
		}
		scoring = 'neg_log_loss'

		modelCV = ReturnFitModel(FileLocation, OptionChosen, 'SVC', ModelCVdir, train_index[-1])
		if (modelCV == None) or (REFITALWAYS == True): # If 'None' was returned from ReturnFitPredictPoints(), no fit is saved. Model must be fitted now.
			print('Refit Happening: ' + str(iterCount), end='\r')
			RefitFlag = True
			# run grid search cross validation
			modelCV = GridSearchCV(pipeline, param_grid=parameters, scoring=scoring, cv=TimeSeriesSplit(n_splits=10), return_train_score = False, iid=False)
			#FitTime = time.time()
			modelCV = modelCV.fit(X_train, y_train) # Fit the pipeline
			#print('Fit time: {:.5f} seconds.'.format(time.time() - FitTime))
			SaveFitModel(FileLocation, OptionChosen, 'SVC', modelCV, ModelCVdir, train_index[-1]) # Save fitted model for posterior plotting.
		#else: modelCV already exists. It has now been retrieved and stored in 'modelCV' variable

		#PredTime = time.time()
		y_pred = pd.Series(modelCV.predict(X_test), name = 'Pred_PriceVar') #Apply transforms to the data, and predict with the final estimator
		y_pred.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print('Pred time: {:.5f} seconds.'.format(time.time() - PredTime))

		PredProb = pd.DataFrame(modelCV.predict_proba(X_test))
		PredProb.index = pd.RangeIndex(start = y_test.index[0], stop = y_test.index[-1]+1)
		#print(pd.concat([PredProb, y_pred, y_test],axis=1))

		"""
		TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, y_test, y_pred, PredProb, 'SVC','')
		AccList.append(TestAccuracy)
		LogList.append(TestLogLoss)
		"""

		#print ('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(modelCV.best_score_))
		ResultTXT.write('Best parameters are: ' + str(modelCV.best_params_) + '; With respective score:' + str(round(modelCV.best_score_,7)) + ' (+/-'+ str(round(modelCV.cv_results_['std_test_score'][modelCV.best_index_],7))+')\n')
		#results = pd.DataFrame(xgb_model.cv_results_)	
		#results = results.sort_values(by='rank_test_score', ascending=True)
		#print(results)
		#print (results[['rank_test_score','param_logistic__solver', 'param_logistic__C', 'param_logistic__penalty', 'mean_test_score']].head())
		#(results[['rank_test_score','params', 'mean_test_score']].head()).to_csv(ResultTXT, sep = '	')
		#print(cols, modelCV.best_estimator_.named_steps['select_best'].pvalues_)

		beta = modelCV.best_estimator_.named_steps['SVC'].coef_
		Coefs = sorted(list(zip(beta.ravel(), X_train.columns.ravel())), key=itemgetter(0))
		#print('SVC coefficients:\n [%s]' % Coefs)
		ResultTXT.write('\nSVC coefficients:\n')
		ResultTXT.writelines(str(Coefs).replace('), (','\n'))
		
		ListParams.append([str(OptionChosen)+str(ValueChosen), 'SVC', FileLocation.strip('.csv')] + [features] + [beta[0]])

		#Write df with predicted values from all iterations/fits/intervals
		if y_predTotal.empty:
			y_predTotal = y_pred
		else:
			y_predTotal = y_predTotal.append(y_pred)

		#Write df with predicted probabilities from all iterations/fits/intervals
		if PredProbTot.empty:
			PredProbTot = PredProb
		else:
			PredProbTot = PredProbTot.append(PredProb)
			#PredProbTot[0] = PredProbTot[0].update(PredProb[0])
			#PredProbTot[1] = PredProbTot[1].update(PredProb[1])
	
	if RefitFlag == True:
		print('Refiting completed.')

	real_y = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
	real_y.drop(real_y.index[:y_predTotal.index[0]], inplace=True)

	#Statistical parameters:
	TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, real_y, y_predTotal, PredProbTot, 'SVC', 'Final')
	df_ohlc, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr = ROI(df_ohlc, y_predTotal, y, ResultTXT, StopLoss)
	SharpeR, SortinoR = SharpeSortinoRatio(df_ohlc, y_predTotal)
	MDDmax, MDDdraw, MDD = MaxDrawdown(df_ohlc, y_predTotal)	
	#dfBHROI, BHPeriodsInMrkt, BHProfitblTrad, BHMAXP, BHMAXL, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, BHSortinoR = BuyHold(df_ohlc, y, y_predTotal)
	ListROI.append((FileLocation.strip('.csv'), 'SVC', str(OptionChosen)+str(ValueChosen), df_ohlc.ROI.iloc[-2], TestLogLoss, TestAccuracy,
		NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeR, SortinoR,
		real_y.shape[0]))

	#Add SVC ROI to total sum of SVC ROI's
	dfAlt = df_ohlc[['Date', 'ROI']].copy()
	dfAlt.set_index(pd.DatetimeIndex(dfAlt['Date']), inplace=True)
	dfAlt = dfAlt.drop(columns='Date').resample('1min').pad()
	TotalROIdf.SVC = pd.concat([TotalROIdf.SVC, dfAlt.ROI], axis=1).sum(axis=1)
	del dfAlt

	if DrawPredictionGraph:
		PredictFig = plotStock(df_ohlc, str(FileLocation).strip('_1m.csv') + ' - Train and Test with entry and exit points. - SVC - ' + str(ValueChosen) + ' ' + str(OptionChosen))
		PredictFig = plotEntranceExit(df_ohlc, PredictFig, y_predTotal, y, True, 'Close')
		ROIfig = plotROI(df_ohlc, 'SVC', TestAccuracy, TestLogLoss, ROIfig, Limits, "#4AE5DE")
		#ROIfig = plotEntranceExit(df_ohlc, ROIfig, y_predTotal, y, False, 'ROI')
		ROIfig.x_range = PredictFig.x_range # Pan or zooming actions across X axis is linked
		#PlotIntervalBox(df_ohlc, PredictFig, Limits)
		#PlotIntervalBox(df_ohlc, ROIfig, Limits)
		output_file( 'HTMLs/' + FileLocation.strip('.csv') + '/Prediction-' + str(OptionChosen) + ".html", 
			title = FileLocation.strip('.csv') + '-Prediction-' + str(OptionChosen))
		#show(column(PredictFig, ROIfig))

		print('Elapsed time on SVC fit, predict and plot: {:.5f} seconds.\n'.format(time.time() - SVCStartTime))
		return PredProbTot, PredictFig, ROIfig
	
	else:
		print('Elapsed time on SVC fit and predict: {:.5f} seconds.\n'.format(time.time() - SVCStartTime))
		return PredProbTot


#######################################################################################################################################
########################################################MajorityVoting#################################################################
#######################################################################################################################################

def MajorityVoting(X, y, df_ohlc, ValueChosen, OptionChosen, FileLocation, ResultTXT, ListROI, y_predList_0, y_predList_1, Limits, ROIfig, TotalROIdf):

	print('Majority Voting: ')
	MVStartTime = time.time()
	
	y_predListMean_0 = y_predList_0.apply(np.mean, axis=1) # Average probability of label 0
	y_predListMean_1 = y_predList_1.apply(np.mean, axis=1) # Average probability of label 1

	############ If average probability of being 0 is >0.5 then label 0 is assigned, else (<=0.5) label 1 is assigned.
	#print(pd.concat([y_predListSum/y_predList_0.shape[1], y_predList_0],axis=1))
	conditions = [y_predListMean_0 > 0.5, y_predListMean_0 ==0.5, y_predListMean_0 < 0.5]
	choices = [0, 1, 1]
	y_pred = pd.Series(np.select(conditions, choices, default=0), name='Pred_PriceVar')	#Predicted labels take into account average of all odds
	"""############ Majority Voting here. If at least half of the algotihms agree on a label, it is chosen.
	y_predListSum = y_predList.apply(np.sum, axis=1)
	#print(pd.concat([y_predListSum/y_predList.shape[1], y_predList],axis=1))	
	conditions = [y_predListSum > y_predList.shape[1]/2, y_predListSum == y_predList.shape[1]/2, y_predListSum < y_predList.shape[1]/2]
	choices = [1, 1, 0]
	y_pred = pd.Series(np.select(conditions, choices, default=0), name='Pred_PriceVar')
	"""############

	y_pred.index = pd.RangeIndex(start = y_predList_0.index[0], stop = y_predList_0.index[-1]+1)
	#print(pd.concat([y_predList_0, y_predListMean, y_pred, y_test, y_pred == y_test ],axis=1))

	PredProb = pd.concat([y_predListMean_0, y_predListMean_1], axis=1)
		
	real_y = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
	real_y.drop(real_y.index[:y_pred.index[0]], inplace=True)

	#Statistical parameters:
	TestAccuracy, TestLogLoss = CalcAccLLConfM(ResultTXT, real_y, y_pred, PredProb, 'MV', 'Final')
	df_ohlc, NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr = ROI(df_ohlc, y_pred, y, ResultTXT, StopLoss)
	SharpeR, SortinoR = SharpeSortinoRatio(df_ohlc, y_pred)
	MDDmax, MDDdraw, MDD = MaxDrawdown(df_ohlc, y_pred)
	#dfBHROI, BHPeriodsInMrkt, BHProfitblTrad, BHMAXP, BHMAXL, BHMDDmax, BHMDDdraw, BHMDD, BHSharpeR, BHSortinoR = BuyHold(df_ohlc, y, y_pred)
	ListROI.append((FileLocation.strip('.csv'), 'MV', str(OptionChosen)+str(ValueChosen), df_ohlc.ROI.iloc[-2], TestLogLoss, TestAccuracy,
		NumTrades, PeriodsInMrkt, NumProfitTradesPred, MAXProfit, MAXLoss, AvgProfit, StopLossCntr, MDDmax, MDDdraw, MDD, SharpeR, SortinoR,
		real_y.shape[0]))

	#Add MV ROI to total sum of MV ROI's
	dfAlt = df_ohlc[['Date', 'ROI']].copy()
	dfAlt.set_index(pd.DatetimeIndex(dfAlt['Date']), inplace=True)
	dfAlt = dfAlt.drop(columns='Date').resample('1min').pad()
	TotalROIdf.MV = pd.concat([TotalROIdf.MV, dfAlt.ROI], axis=1).sum(axis=1)
	del dfAlt

	if DrawPredictionGraph:
		PredictFig = plotStock(df_ohlc, str(FileLocation).strip('_1m.csv') + ' - Train and Test with entry and exit points. - Majority Voting - ' + str(ValueChosen) + ' ' + str(OptionChosen))
		PredictFig = plotEntranceExit(df_ohlc, PredictFig, y_pred, y, True, 'Close')
		ROIfig = plotROI(df_ohlc, 'EV', TestAccuracy, TestLogLoss, ROIfig, Limits, "#C200D5")
		#ROIfig = plotEntranceExit(df_ohlc, ROIfig, y_pred, y, False, 'ROI')
		ROIfig.x_range = PredictFig.x_range # Pan or zooming actions across X axis is linked
		PlotIntervalBox(df_ohlc, PredictFig, Limits)
		#PlotIntervalBox(df_ohlc, ROIfig, Limits)
		output_file( 'HTMLs/' + FileLocation.strip('.csv') + '/Prediction-' + str(OptionChosen) + ".html", 
			title = FileLocation.strip('.csv') + '-Prediction-' + str(OptionChosen))
		#show(PredictFig)
		#show(column(PredictFig, ROIfig))
		print('Elapsed time on MV fit, predict and plot: {:.5f} seconds.\n'.format(time.time() - MVStartTime))
		return PredProb, PredictFig, ROIfig
	
	else:
		print('Elapsed time on MV fit and predict: {:.5f} seconds.\n'.format(time.time() - MVStartTime))
		return PredProb


#Function to calculate and print accuracy, log loss, confusion matrix and classification report.
def CalcAccLLConfM(ResultTXT, y_test, y_pred, PredProb, algorithm, step):

	if step == 'Final':
		ResultTXT.write('\n\n---- All iterations together ----\n')

	###################################
	TestAccuracy = round(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None), 7)
	ResultTXT.write('Accuracy of ' + algorithm +' on test set: {:.5f}\n'.format(TestAccuracy))
	
	###################################	
	TestLogLoss = -round(log_loss(y_test, PredProb, normalize=True, sample_weight=None), 7) #y_pred must be the values returned from predict_proba
	ResultTXT.write('Neg LogLoss of ' + algorithm +' on test set: {:.5f}.\n'.format(TestLogLoss))

	if step == 'Final':
		print('Accuracy on test set: %.5f; Neg LogLoss on test set: %.5f.' % (TestAccuracy, TestLogLoss))	
		#print('\nConfusion matrix:')
		#print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Pred_Drop', 'Pred_Rise'], index=['Drop', 'Rise']))
		#print('\nClassification Report:\n' + str(classification_report(y_test, y_pred)))
		ResultTXT.write('Confusion matrix:\n')
		(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Pred_Drop', 'Pred_Rise'], index=['Drop', 'Rise'])).to_csv(ResultTXT)	
		ResultTXT.write('Classification Report:\n' + str(classification_report(y_test, y_pred)))

	return TestAccuracy, TestLogLoss 
