#!/usr/bin/python3

import pandas as pd
import numpy as np

from bokeh.plotting import figure, show, output_file
from bokeh.models import FixedTicker, ColumnDataSource, HoverTool, Span, Range1d, BasicTickFormatter
from bokeh.models.formatters import PrintfTickFormatter

from BokehVisualization import plotStock, plotEntranceExit, PlotIntervalBox, plotROI

#Function to group all consecutive ROI values (almost 1 million points makes it impossible for Bokeh to plot)
def Grouper(df, option):
	df = df.loc[df[option].shift(-1) != df[option]].reset_index()
	print(df.shape[0], end=' - ')
	return df[:-1] #Last entry is skipped because ROI remains zero (no trading happens in this instant), limitation of the code

def main():

	TotalROIDirectory = 'NoDerivs_VFinal/TotalROIs'
	for option in ['Percentage', 'Amount','LogAmount', 'Time']:
		#df = pd.read_csv(TotalROIDirectory + '/shitshit.csv', sep=',')
		df = pd.read_csv(TotalROIDirectory + '/ListTotalROI_'+option+'.csv', sep=',')
		#print(df.LR.first_valid_index())
		#print(df.iloc[232435:232486])
		#exit()
		df["Date"] = pd.to_datetime(df["Date"],format = '%Y-%m-%d %H:%M:%S')

		fig = figure(x_axis_type="datetime", plot_width=800, plot_height=200, title="Return on investment (ROI)", 
				tools="box_zoom,pan,wheel_zoom,reset", toolbar_location='above')

		dfAlt = Grouper(df[['Date','BH']], 'BH')

		#print(dfAlt)

		r1 = fig.line(dfAlt.Date, dfAlt.BH, line_width=2, color="#000000", legend='B&H')
		del dfAlt
		dfAlt = Grouper(df[['Date','LR']], 'LR')
		r1 = fig.line(dfAlt.Date, dfAlt.LR, line_width=2, color="#E11F16", legend='LR')
		del dfAlt
		dfAlt = Grouper(df[['Date','RF']], 'RF')
		r1 = fig.line(dfAlt.Date, dfAlt.RF, line_width=2, color="#1AAE03", legend='RF')
		del dfAlt
		dfAlt = Grouper(df[['Date','XG']], 'XG')
		r1 = fig.line(dfAlt.Date, dfAlt.XG, line_width=2, color="#FECD04", legend='GTB')
		del dfAlt
		dfAlt = Grouper(df[['Date','SVC']], 'SVC')
		r1 = fig.line(dfAlt.Date, dfAlt.SVC, line_width=2, color="#4AE5DE", legend='SVC')
		del dfAlt
		dfAlt = Grouper(df[['Date','MV']], 'MV')
		r1 = fig.line(dfAlt.Date, dfAlt.MV, line_width=2, color="#C200D5", legend='EV')
		del dfAlt
		
		# Y axis converted to percentage:
		#fig.yaxis.axis_label = "[%]"
		fig.yaxis.formatter = PrintfTickFormatter(format="%f%%")
		fig.grid.grid_line_color = 'navy'
		fig.ygrid.minor_grid_line_color = 'navy'
		fig.grid.grid_line_alpha = 0.1		
		fig.ygrid.minor_grid_line_alpha = 0.1
		#fig.yaxis.formatter = BasicTickFormatter(use_scientific=False)
		#fig.y_range = Range1d(-100, 1000000)
		#print(fig.y_range)
		# Horizontal line
		hline = Span(location=0, dimension='width', line_color='black', line_width=0.5, line_dash='dashed')
		fig.renderers.extend([hline])

		NewTime = pd.Series(pd.to_timedelta(df.Date).dt.total_seconds())
		fig.x_range = Range1d(NewTime[NewTime.index[0]]*1000, NewTime[NewTime.index[-1]]*1000) #Conversion from seconds to miliseconds
		del NewTime

		fig.yaxis.major_label_text_font_size = '13pt'
		fig.xaxis.major_label_text_font_size = '13pt'
		fig.legend.location = (0, -5)
		fig.legend.click_policy = "hide"
		fig.legend.border_line_alpha = 0
		fig.legend.background_fill_alpha = 0

		output_file(TotalROIDirectory+"/TotalROI_"+option+".html", title=option)
		show(fig)
		print('Option '+option+' done.')
		#exit()
		
main()