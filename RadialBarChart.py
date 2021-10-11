#!/usr/bin/python3

from collections import OrderedDict
from math import log, sqrt

import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file

from Dumps import InitListROI

from bokeh.palettes import Spectral5
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, LinearAxis, Range1d, Span, HoverTool


drug_color = OrderedDict([
	("Acc", "#003152"),
	("NLL", "#003152"),
	("ROI", "#003152"),
])
gram_color = OrderedDict([
		("0", "#e69584"), #negative
		("1", "#9dc183"), #positive
])

MotherFile = 'NoDerivs_VFinal/'
File = MotherFile+'Lists388'
SaveFile=MotherFile+'burtins/'
pd.options.display.max_rows = 100

def RadialPlotterROI(df, column, flag):
	
	if flag:
		df = df.sort_values(['gram', column]).reset_index(drop=True)	
	else:
		df = df.sort_values([column]).reset_index(drop=True)
	
	
	#df = df.sort_values([column,'gram']).reset_index()
	#print(df)
		
	inner_radius = 60
	outer_radius = 180 - 10
	minr = (log(100000))
	maxr = -(log(100))
	a = (outer_radius - inner_radius) / (minr - maxr)
	b = inner_radius - a * maxr

	big_angle = 2.0 * np.pi / (len(df) + 1)
	small_angle = big_angle / 7

	#print(rad(df[column]))
	width = 800
	height = 800
	p = figure(plot_width=width, plot_height=height, title="", x_axis_type=None, y_axis_type=None, x_range=(-420, 420), y_range=(-420, 420), 
		min_border=0, outline_line_color="black")#, background_fill_color="#f0e1d2")

	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None

	# annular wedges
	angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
	if flag:
		colors = [gram_color[gram] for gram in df.gram]
		p.annular_wedge(0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors)
	else:
		p.annular_wedge(0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color ="#ffffb5")
	# small wedges
	def rad(mic):
		return (a* (np.log((mic * 1)))+b)
	def radNeg(mic):		
		return inner_radius- (a * ((np.log(abs(mic * 1)))))

	#Because of log and sqrt, approximate [-1,0] to -1 and [0,1] to 1
	df[column] = np.where(df[column].between(0,1), 1, df[column])
	df[column] = np.where(df[column].between(-1,0), -1, df[column])

	conditions = [df[column]>=0, df[column]<0]
	choices = [inner_radius,radNeg(df[column])]	
	DrawSeriesLeft=pd.Series(np.select(conditions, choices, default=0), name='left')

	conditions = [df[column]>=0, df[column]<0]
	choices = [rad(df[column]), inner_radius]	
	DrawSeriesRight=pd.Series(np.select(conditions, choices, default=0), name='right')

	p.annular_wedge(x=0, y=0, inner_radius=DrawSeriesLeft, outer_radius=DrawSeriesRight, start_angle=(-big_angle+angles+3*small_angle), 
		end_angle=(-big_angle+angles+4*small_angle), color=drug_color[column])

	# circular axes and lables
	labels = np.power(10.0, np.arange(0, 6))
	radii = a * (np.log(labels * 1)) + b
	
	p.circle(0, 0, radius=radii, fill_color=None, line_color="#7d7d7d")
	p.text(0, radii[:-1], [str(r) for r in labels[:-1]], text_font_size="10pt", text_align="left", text_baseline="middle")
	labels = [inner_radius, inner_radius-(a*((np.log(abs(-10 * 1))))), inner_radius-(a*((np.log(abs(-100 * 1)))))]
	
	
	
	p.circle(0, 0, radius=labels, fill_color=None, line_color="#7d7d7d")
	p.text(5, 65, ['0'], text_font_size="10pt", text_align="center", text_baseline="middle")
	p.text(-10, 46, ['-10'], text_font_size="10pt", text_align="center", text_baseline="middle")
	p.text(0, 17, ['-100'], text_font_size="10pt", text_align="center", text_baseline="middle")
		
	p.circle(0, 0, radius=[inner_radius, inner_radius-(a*((np.log(abs(-100 * 1)))))], fill_color=None, line_color="black")
	
	# radial axes (line separating each value)
	#p.annular_wedge(0, 0, inner_radius-10, outer_radius+10, -big_angle+angles, -big_angle+angles, color="black")

	# bacteria labels
	"""
	xr = 5*radii[0]*np.cos(np.array(-big_angle/2 + angles))
	yr = 5*radii[0]*np.sin(np.array(-big_angle/2 + angles))
	label_angle=np.array(-big_angle/2+angles)
	label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
	p.text(xr, yr, df.Pair, angle=label_angle, text_font_size="5pt", text_align="center", text_baseline="middle")
	"""
	
	# OK, these hand drawn legends are pretty clunky, will be improved in future release
	p.circle([-150, 15], [-190, -210], color=list(gram_color.values()), radius=8)
	p.text([-140, 25], [-190, -210], text=["ROI worse than B&H strategy;", "ROI better than B&H strategy."], text_font_size="15pt", text_align="left", 
		text_baseline="middle")

	#p.rect([-40, -40, -40], [18, 0, -18], width=30, height=13, color=list(drug_color.values()))
	#p.text(-12, 0, text=["ROI"], text_font_size="9pt", text_align="left", text_baseline="middle")
	
	output_file(SaveFile+"burtinROI_"+df.Algorithm[0]+".html", title="burtin.py")
	
	show(p)
	return


def RadialPlotterNLL(df, column):

	df = df.sort_values(['gram', column]).reset_index(drop=True)

	inner_radius = 40
	outer_radius = 180 - 10
	minr = 1
	maxr = 0
	a = (outer_radius - inner_radius) / (minr - maxr)
	b = inner_radius - a * maxr

	big_angle = 2.0 * np.pi / (len(df) + 1)
	small_angle = big_angle / 7

	#print(rad(df[column]))
	width = 800
	height = 800
	p = figure(plot_width=width, plot_height=height, title="", x_axis_type=None, y_axis_type=None, x_range=(-420, 420), y_range=(-420, 420), 
		min_border=0, outline_line_color="black")#, background_fill_color="#f0e1d2")

	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None

	# annular wedges
	angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
	colors = [gram_color[gram] for gram in df.gram]
	p.annular_wedge(0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors)

	# small wedges
	def rad(mic):
		return (a*((mic * 1))+b)

	p.annular_wedge(x=0, y=0, inner_radius=inner_radius, outer_radius=rad(-df[column]), start_angle=(-big_angle+angles+3*small_angle), 
		end_angle=(-big_angle+angles+4*small_angle), color=drug_color[column])
	print(rad(-df[column]))
	
	# circular axes and lables
	p.circle(0, 0, radius=[inner_radius, a + b, a * (0.693 * 1) + b], fill_color=None, line_color="#7d7d7d")

	p.text(0, inner_radius, ['0'], text_font_size="8pt", text_align="center", text_baseline="middle", y_offset=10)
	p.text(-12, (a*0.693+ b), ['-0.693'], text_font_size="8pt", text_align="center", text_baseline="middle")
	p.text(-5, (a*1+ b), ['-1'], text_font_size="8pt", text_align="center", text_baseline="middle")	
	
	# radial axes (line separating each value)
	#p.annular_wedge(0, 0, inner_radius-10, outer_radius+10, -big_angle+angles, -big_angle+angles, color="black")

	# bacteria labels
	"""
	xr = 5*radii[0]*np.cos(np.array(-big_angle/2 + angles))
	yr = 5*radii[0]*np.sin(np.array(-big_angle/2 + angles))
	label_angle=np.array(-big_angle/2+angles)
	label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
	p.text(xr, yr, df.Pair, angle=label_angle, text_font_size="5pt", text_align="center", text_baseline="middle")
	"""
	
	# OK, these hand drawn legends are pretty clunky, will be improved in future release
	p.circle([-135, 15], [-190, -190], color=list(gram_color.values()), radius=5)
	p.text([-125, 25], [-190, -190], text=["NLL worse than random.", "NLL better than random."], text_font_size="15pt", text_align="left", 
		text_baseline="middle")

	#p.rect([-40, -40, -40], [18, 0, -18], width=30, height=13, color=list(drug_color.values()))
	#p.text(-15, 0, text=["NLL"], text_font_size="9pt", text_align="left", text_baseline="middle")

	output_file(SaveFile+"burtinNLL_"+df.Algorithm[0]+".html", title="burtin.py")
	
	show(p)
	return

def RadialPlotterAcc(df, column):

	df = df.sort_values(['gram', column]).reset_index(drop=True)
	inner_radius = 40
	outer_radius = 180 - 10
	minr = 1
	maxr = 0
	a = (outer_radius - inner_radius) / (minr - maxr)
	b = inner_radius - a * maxr

	big_angle = 2.0 * np.pi / (len(df) + 1)
	small_angle = big_angle / 7

	#print(rad(df[column]))
	width = 800
	height = 800
	p = figure(plot_width=width, plot_height=height, title="", x_axis_type=None, y_axis_type=None, x_range=(-420, 420), y_range=(-420, 420), 
		min_border=0, outline_line_color="black")#, background_fill_color="#f0e1d2")

	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None

	# annular wedges
	angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
	colors = [gram_color[gram] for gram in df.gram]
	p.annular_wedge(0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors)

	# small wedges
	def rad(mic):
		return (a*((mic * 1))+b)

	p.annular_wedge(x=0, y=0, inner_radius=inner_radius, outer_radius=rad(df[column]), start_angle=(-big_angle+angles+3*small_angle), 
		end_angle=(-big_angle+angles+4*small_angle), color=drug_color[column])
	#print(rad(-df[column]))
	
	# circular axes and lables
	p.circle(0, 0, radius=[inner_radius, a + b, a * (0.5 * 1) + b, a * (0.6 * 1) + b], fill_color=None, line_color="#7d7d7d")

	p.text(0, inner_radius, ['0'], text_font_size="10pt", text_align="center", text_baseline="middle", y_offset=10)
	p.text(10, (a*0.5+ b), ['0.5'], text_font_size="10pt", text_align="center", text_baseline="middle")
	p.text(10, (a*0.6+ b), ['0.6'], text_font_size="10pt", text_align="center", text_baseline="middle")
	p.text(7, (a*1+ b), ['1'], text_font_size="10pt", text_align="center", text_baseline="middle")	
	
	# radial axes (line separating each value)
	#p.annular_wedge(0, 0, inner_radius-10, outer_radius+10, -big_angle+angles, -big_angle+angles, color="black")

	# bacteria labels
	"""
	xr = 5*radii[0]*np.cos(np.array(-big_angle/2 + angles))
	yr = 5*radii[0]*np.sin(np.array(-big_angle/2 + angles))
	label_angle=np.array(-big_angle/2+angles)
	label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
	p.text(xr, yr, df.Pair, angle=label_angle, text_font_size="5pt", text_align="center", text_baseline="middle")
	"""
	
	# OK, these hand drawn legends are pretty clunky, will be improved in future release
	p.circle([-135, 15], [-190, -210], color=list(gram_color.values()), radius=5)
	p.text([-125, 25], [-190, -210], text=["Accuracy worse than 50%;", "Accuracy better than 50%."], text_font_size="15pt", text_align="left", 
		text_baseline="middle")

	#p.rect(-40, 0, width=30, height=13, color=drug_color[column])
	#print(column)
	#p.text(-27, 0, text=["Accuracy"], text_font_size="9pt", text_align="left", text_baseline="middle")
	
	output_file(SaveFile+"burtinACC_"+df.Algorithm[0]+".html", title="burtin.py")
	
	show(p)
	return


def PlotAcc(df, column):

	#PQP nao se pode repetir nomes no eixo x
	df = df.sort_values([column,'gram']).reset_index()

	df.plot.bar(x='Pair', y=column)
	df.Pair = df.Pair + df.Rearrang

	p = figure(x_range=df.Pair, plot_height=350,plot_width=1000, title="Fruit Counts",
	           toolbar_location=None, tools="")

	p.vbar(x=df.Pair, top=df[column], width=0.1)



	output_file("burtinAcc.html", title="burtin.py example")
	show(p)


def BHCase(crap):

	#crap = pd.DataFrame(ListBH, columns=['Pair', 'Algorithm', 'Rearrang', 'ROI', 'NLL', 'Acc', 'NumTrades', 'PeriodsInMrkt',
	#	'NumProfitTradesPred', 'MAXProfit', 'MAXLoss', 'AvgProfit', 'StopLossCntr', 'MDDmax', 'MDDdraw', 'MDD', 'ShR', 'SoR', 'TotalTestPer'])

	dfROI = crap[['Pair', 'Algorithm','Rearrang', 'ROI']].copy()
	RadialPlotterROI(dfROI, 'ROI', False)
	dfAcc = crap[['Pair', 'Algorithm','Rearrang', 'Acc']].copy()
	dfAcc['gram'] = np.where(dfAcc['Acc']>0.5, "1", "0")
	RadialPlotterAcc(dfAcc, 'Acc')
	return




def main():
	
	(ListROIBH, ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV) = InitListROI(File)
	dfcrapBH = pd.DataFrame(ListROIBH, columns=['Pair', 'Algorithm', 'Rearrang', 'ROI', 'NLL', 'Acc', 'NumTrades', 'PeriodsInMrkt',
		'NumProfitTradesPred', 'MAXProfit', 'MAXLoss', 'AvgProfit', 'StopLossCntr', 'MDDmax', 'MDDdraw', 'MDD', 'ShR', 'SoR', 'TotalTestPer'])

	BHCase(dfcrapBH)
	
	#print(dfcrapBH)
	dfcrapBH = dfcrapBH.sort_values(['Pair','Algorithm']).reset_index(drop=True)
	#print(dfcrapBH)
	for List in ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV:
		
		# Position: 0-FileName, 1-Algorithm, 2-Rearrangement, 3-ROI, 4-TestLogLoss, 5-TestAccuracy, 6-NumTrades, 7-Periods In Market, 
		# 8-Num Profitable Trades, 9-Max Profit, 10-Max Loss, 11-Avg Profit, 12-Num of StopLoss Activated, 13-MDD max, 14-MDD draw, 15-MDD,
		# 16-AnnualSharpeRatio, 17-AnnualSortinoRatio, 18-Total Test Periods
		crap = pd.DataFrame(List, columns=['Pair', 'Algorithm', 'Rearrang', 'ROI', 'NLL', 'Acc', 'NumTrades', 'PeriodsInMrkt',
			'NumProfitTradesPred', 'MAXProfit', 'MAXLoss', 'AvgProfit', 'StopLossCntr', 'MDDmax', 'MDDdraw', 'MDD', 'ShR', 'SoR', 'TotalTestPer'])

		#df___ always has 1.Pair, 2.Alg, 3.RearrngMethod, 4.ValueToPlot, 5.ValueFlag(Indicating whether value obtained is positive or negative)
		
		#print(pd.concat([dfROI, dfcrapBH],axis=1))
		dfROI = crap[['Pair', 'Algorithm','Rearrang', 'ROI']].copy()
		dfROI = dfROI.sort_values(['Pair','Algorithm']).reset_index(drop=True)
		dfROI['gram'] = np.where(dfROI['ROI']>dfcrapBH['ROI'], "1", "0")

		dfNLL = crap[['Pair', 'Algorithm','Rearrang', 'NLL']].copy()
		dfNLL['gram'] = np.where(dfNLL['NLL']>-0.693, "1", "0")

		dfAcc = crap[['Pair', 'Algorithm','Rearrang', 'Acc']].copy()
		dfAcc['gram'] = np.where(dfAcc['Acc']>0.5, "1", "0")
		del crap

		#For for ROI:

		RadialPlotterROI(dfROI, 'ROI', True)
		#exit()
		#For for NLL:
		#RadialPlotterNLL(dfNLL, 'NLL')
		
		#For for Accuracy:
		RadialPlotterAcc(dfAcc, 'Acc')
		#PlotAcc(dfAcc, 'Acc')
		
main()