#!/usr/bin/python3

import pandas as pd
import numpy as np

from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, LinearAxis, Range1d, Span, HoverTool
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import column

from Dumps import InitListROI, RemoveROI

#ListDirectory = 'Lists388Backup/8.ApenasHistsStochSlowKFastK+Hists'
ListDirectory = 'NoDerivs/Lists388'
TOOLS = "box_zoom,pan,wheel_zoom,reset"
output_file('HTMLs/' + 'ListBarChart.html')

(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV) = InitListROI(ListDirectory)
RemoveROI(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, 'HOTBTC_1m')
RemoveROI(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, 'NPXSBTC_1m')
RemoveROI(ListROILR, ListROIRF, ListROIXG, ListROISVC, ListROIMV, 'DENTBTC_1m')

GraphsList = []

for GroupingType in ['Percentage','Amount','LogAmount','Time']:

	pairs = []
	ROIList = []
	NegLogLossList = []
	ACCList = []

	# Position: 3-ROI, 4-TestLogLoss, 5-TestAccuracy
	for sublist in ListROIMV:
		if sublist[2] == GroupingType:
			pairs.append(sublist[0])
			ROIList.append(sublist[3])
			NegLogLossList.append(sublist[4])
			ACCList.append(sublist[5])

	data = {'Pair' : pairs,
	        'ROI'   : ROIList,
	        'NLL'   : NegLogLossList,
	        'Acc'   : ACCList}

	source = ColumnDataSource(data=data)
	# Seting the params for the first figure.
	p = figure(x_range=pairs, plot_height=600, tools=[TOOLS, HoverTool(tooltips=[('ROI', "@{ROI}%"),('NegLogLoss', "@NLL"), ('Accuracy', "@Acc")],mode='vline')], 
		plot_width=1200, title="MV stats for "+str(GroupingType)+" grouping. Average ROI is: "+ str(round(np.mean(ROIList),5)) + ". Average NegLogLoss is: " 
		+ str(round(np.mean(NegLogLossList),5)) + ". Average Accuracy is: " + str(round(np.mean(ACCList),5)) + ".", toolbar_location='above')

	#Middle point of the two ranges must coincide, otherwise unalignment happens
	p.yaxis.axis_label = "ROI[%]"
	p.y_range = Range1d(-80, 150)

	# Setting the second y axis range name and range
	p.extra_y_ranges = {"Prcnt": Range1d(start=-0.8, end=1.5)}
	# Adding the second axis to the plot.  
	p.add_layout(LinearAxis(y_range_name="Prcnt", axis_label="Accuracy and NegLogLoss"), 'right')

	# Horizontal line at 0
	hline1 = Span(location=0, dimension='width', line_color='black', line_width=0.5)
	# Horizontal line for Log Loss
	hline2 = Span(location=-0.693, y_range_name="Prcnt", dimension='width', line_color='black', line_width=0.5, line_dash='dashed')
	# Horizontal line for Accuracy
	hline3 = Span(location=0.5, y_range_name="Prcnt", dimension='width', line_color='black', line_width=0.5, line_dash='dashed')
	p.renderers.extend([hline1, hline2, hline3])

	# Average of ROI
	hline = Span(location=np.mean(ROIList), dimension='width', line_color='#C10032', line_width=0.5, line_dash='dotted')
	p.renderers.extend([hline])
	# Average of LogLoss
	hline = Span(location=np.mean(NegLogLossList), y_range_name="Prcnt", dimension='width', line_color='#002D62', line_width=0.5, line_dash='dotted')
	p.renderers.extend([hline])
	# Average of Accuracy
	hline = Span(location=np.mean(ACCList), y_range_name="Prcnt", dimension='width', line_color='#78C7EB', line_width=0.5, line_dash='dotted')
	p.renderers.extend([hline])

	# Using the default y range and y axis here.  
	p.vbar(x=dodge('Pair', -0.2, range=p.x_range), top='ROI', width=0.2, source=source, color="#C10032", legend=value("ROI"))
	# Using the aditional y range named "foo" and "right" y axis here. 
	p.vbar(x=dodge('Pair',  0.0,  range=p.x_range), top='NLL', y_range_name="Prcnt", width=0.2, source=source, color="#002D62", legend=value("Neg LogLoss"))
	# Using the aditional y range named "foo" and "right" y axis here. 
	p.vbar(x=dodge('Pair',  0.2, range=p.x_range), top='Acc', y_range_name="Prcnt", width=0.2, source=source, color="#78C7EB", legend=value("Accuracy"))

	p.x_range.range_padding = 0.1
	p.xgrid.grid_line_color = None
	p.legend.location = "top_left"
	p.legend.orientation = "horizontal"
	
	GraphsList.append(p)
	
show(column(GraphsList))