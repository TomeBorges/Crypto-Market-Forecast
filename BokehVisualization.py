#!/usr/bin/python3

import pandas as pd
import numpy as np

from math import pi
from bokeh.plotting import figure, show, output_file
from bokeh.models import FixedTicker,DatetimeTickFormatter, ColumnDataSource, HoverTool, BoxAnnotation, Span, Range1d, DataRange1d, LinearAxis, CDSView, BooleanFilter, markers
from bokeh.layouts import column
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.palettes import d3
WIDTH_PLOT = 800
HEIGHT_LARGE_PLOT = 200
HEIGHT_SMALL_PLOT = 200
TOOLS = "box_zoom,pan,wheel_zoom,reset"

# Plot of the candlesticks chart
def plotStock(df, Title):

	Fig = figure(x_axis_type="datetime", tools=TOOLS, toolbar_location='above', plot_width=WIDTH_PLOT, plot_height=HEIGHT_LARGE_PLOT, 
		title = Title)

	#Fig.xaxis.major_label_orientation = pi/4
	Fig.grid.grid_line_alpha = 0.4

	inc = df.Close > df.Open #Bullish entrances
	dec = df.Open > df.Close #Bearish entrances
	side = df.Open == df.Close #Precisely sideways entrances

	# Case where all candlesticks have the exact same width:
	"""
	#use ColumnDataSource to pass in data for tooltips (HoverTool)
	sourceInc=ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
	sourceDec=ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

	Fig = figure(x_axis_type="datetime", tools=TOOLS, plot_width=WIDTH_PLOT, title = 'Stock Price, ZigZag Algorithm & Moving Averages of '+ FileLocation.strip('.csv'))

	Fig.xaxis.major_label_orientation = pi/4
	Fig.grid.grid_line_alpha = 0.4

	# Up and down Tail
	width = (df["Date"][1] - df["Date"][0]).total_seconds() *1000 # Time difference between each entry on pandas DataFrame in ms --> *1000
	Fig.segment('Date', 'High', 'Date', 'Low', color="black", source=sourceInc)
	r1 = Fig.vbar('Date', width, 'Open', 'Close', fill_color="#7BE61D", line_color="black", source=sourceInc) #Bullish entrances
	r2 = Fig.vbar('Date', width, 'Open', 'Close', fill_color="#F2583E", line_color="black", source=sourceDec) #Bearish entrances
	"""

	#Case where candlesticks have different widths:	
	DateLeft = pd.Series(pd.to_timedelta(df.Date.shift(1)).dt.total_seconds(),name='DateLeft')
	DateRight = pd.Series(pd.to_timedelta(df.Date).dt.total_seconds(),name='DateRight')

	# Up and down Tail
	Center = pd.Series((DateLeft + DateRight)/2,name='Center')

	df = df.join(pd.to_datetime(Center,unit='s'))
	df = df.join(pd.to_datetime(DateLeft,unit='s'))
	df = df.join(pd.to_datetime(DateRight,unit='s'))

	df=df[1:]
	
	#use ColumnDataSource to pass in data for tooltips (HoverTool)
	sourceInc=ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
	sourceDec=ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))
	sourceSide=ColumnDataSource(ColumnDataSource.from_df(df.loc[side]))

	#Bullish entrances
	Fig.segment('Center', 'High', 'Center', 'Low', color="black", source=sourceInc)
	r1 = Fig.quad('DateLeft','DateRight', 'Open','Close', fill_color="#7BE61D", line_color="black", source=sourceInc)
	#Bearish entrances
	Fig.segment('Center', 'High', 'Center', 'Low', color="black", source=sourceDec)
	r2 = Fig.quad('DateLeft', 'DateRight', 'Open', 'Close', fill_color="#F2583E", line_color="black", source=sourceDec)
	#Sideways entrances
	r3 = Fig.segment('DateLeft', 'Open', 'DateRight', 'Close', color="#7BE61D", source=sourceSide) # Confirmar no programa backtesting qual o append para ==
	# Bokeh format types referenced in: https://bokeh.pydata.org/en/latest/docs/reference/models/formatters.html
	# the values for the tooltip come from ColumnDataSource

	Fig.add_tools(HoverTool(
		tooltips=[   
			("Close", "@Close"),
			("High", "@High"),
			("Low", "@Low"),
			("Open", "@Open"),
			("Volume", "@Volume{'0,0.0[000]'}"),
			("Date Start", "@DateLeft{%F %T}"), #Format: %Y-%M-%D %h:%m:%s
			("Date End", "@DateRight{%F %T}"), #Format: %Y-%M-%D %h:%m:%s
	    ],
	    formatters={
        'DateLeft': 'datetime', # use 'datetime' formatter for 'DateLeft' field
        'DateRight': 'datetime', # use 'datetime' formatter for 'DateRight' field
    	},
    	mode='vline',
    	renderers=[r1,r2, r3] # Added so hovertool only appears on top of candle-bar. Not on sticks or ZigZag's circles
	))
	#Fig.add_tools(hover)
	#Fig.legend.location = "top_left"
	#Fig.legend.click_policy = "hide" #If clicked on legend, all elements with the parameter {legend='ZigZag Algorithm'} will be hiden
	
	NewTime = pd.Series(pd.to_timedelta(df.Date).dt.total_seconds())
	Fig.x_range = Range1d(NewTime[NewTime.index[0]]*1000, NewTime[NewTime.index[-1]]*1000) #Conversion from seconds to miliseconds

	Fig.yaxis.major_label_text_font_size = '13pt'
	Fig.xaxis.major_label_text_font_size = '13pt'

	return Fig

# Zig Zag Algorithm plotting function
def plotZigZag(df, Fig, ZigZagPercentage):

	Fig.line(df.Date[df["Pivots_"+str(ZigZagPercentage)] != 0], df.Close[df["Pivots_"+str(ZigZagPercentage)] != 0], color='#0E96EE', legend=str(ZigZagPercentage*100)+'% ZigZag Algorithm')	
	Fig.circle((df.Date[df["Pivots_"+str(ZigZagPercentage)] == 1]).tolist(), df.Close[df["Pivots_"+str(ZigZagPercentage)] == 1], color="#7BE61D", fill_alpha=0.2, size=7, legend=str(ZigZagPercentage*100)+'% ZigZag Algorithm') #Top
	Fig.circle((df.Date[df["Pivots_"+str(ZigZagPercentage)] == -1]).tolist(), df.Close[df["Pivots_"+str(ZigZagPercentage)] == -1], color="#F2583E", fill_alpha=0.2, size=7, legend=str(ZigZagPercentage*100)+'% ZigZag Algorithm') #Bottom
	
	Fig.legend.location = "top_left"
	Fig.legend.click_policy = "hide" #If clicked on legend, all elements with the parameter {legend='ZigZag Algorithm'} will be hiden

	return Fig

# Plot Buys (Green triangle), Sells (Red inverted triangle) and Stop-losses (Black inverted triangle) in a desired 'Fig'
# Plot the predicted values from the algorithms (blue circle for correct prediction, cross for wrong prediction)
def plotEntranceExit(df1, Fig, y_predicted, y, WantPred, PlotVar):
	"""
	i = y_predicted.index[0]+1
	# The order of indexes was inverted. The most recent has the highest index, and the last had the largest index
	
	# If first entry is 1, entry immediately. 0 stay out(Default) 
	if (y_predicted[i-1] == 1):
			Fig.triangle((df1.Date[i-1]), df1[PlotVar][i-1], color='green', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='Entrance/Exit') #Top

	# If previous prediction (y_predicted.shift(1)) is 0 and present one (y_predicted) is 1, go long, BUY=1.
	# If previous prediction (y_predicted.shift(1)) is 1 and present one (y_predicted) is 0, go short, SELL=-1.
	# Else=0, by default.
	conditions = [(y_predicted.shift(1) == 0) & (y_predicted == 1), (y_predicted.shift(1) == 1) & (y_predicted == 0)]
	choices = [1, -1]
	InOutPlot = df1[['Date', PlotVar]].copy()

	InOutPoints=pd.Series(np.select(conditions, choices, default=0), name='InOutPoints')
	InOutPoints.index = pd.RangeIndex(start = y_predicted.index[0], stop = y_predicted.index[-1]+1)
	InOutPlot = InOutPlot.join(InOutPoints)

	#Plotting of the shorts and longs previously found:
	Fig.triangle((InOutPlot.Date[InOutPlot.InOutPoints == 1]).tolist(), InOutPlot[PlotVar][InOutPlot.InOutPoints == 1], color='green', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='Entrance/Exit') #Top
	Fig.inverted_triangle((InOutPlot.Date[InOutPlot.InOutPoints == -1]).tolist(), InOutPlot[PlotVar][InOutPlot.InOutPoints == -1], color='#B20000', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='Entrance/Exit') #Bottom
	"""
	
	#Fig.triangle((df1.Date[df1.Action == 1]).tolist(), df1[PlotVar][df1.Action == 1], color='#00FF44', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='Entrance/Exit') #Top
	#Fig.inverted_triangle((df1.Date[df1.Action == 2]).tolist(), df1[PlotVar][df1.Action == 2], color='#FF0000', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='Entrance/Exit') #Bottom
	#Fig.inverted_triangle((df1.Date[df1.Action == 3]).tolist(), df1[PlotVar][df1.Action == 3], color='black', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='Entrance/Exit') #Bottom
	

	#print pd.concat([y_predicted.shift(1), y_predicted, InOutPlot],axis=1)
	WantPred = False
	if (WantPred == True) :

		y_test = y.copy()#Drop inital y_test rows, this data was used for train. No test data to be compared there.
		y_test.drop(y_test.index[:y_predicted.index[0]], inplace=True)

		#To confirm if the predictions is correct, much like previously, the actual prediction is compared with the true actual value:
		conditions = [(y_predicted == y_test), (y_predicted != y_test)]
		choices = [1, -1]
		CompRealPred = df1[['Date', 'Close', 'Open']].copy()

		CompRealPredPoints=pd.Series(np.select(conditions, choices, default=0), name='CompRealPredPoints')
		CompRealPredPoints.index = pd.RangeIndex(start = y_predicted.index[0], stop = y_predicted.index[-1]+1)
		CompRealPred = CompRealPred.join(CompRealPredPoints)

		#Plotting of the shorts and longs previously found:
		Fig.circle((CompRealPred.Date[CompRealPred.CompRealPredPoints == 1]).tolist(), ((CompRealPred.Close+CompRealPred.Open)/2)[CompRealPred.CompRealPredPoints == 1], size=11, line_color='blue', muted_alpha=0.2, line_width=2, legend='Correct Prediction')
		Fig.x((CompRealPred.Date[CompRealPred.CompRealPredPoints == -1]).tolist(), ((CompRealPred.Close+CompRealPred.Open)/2)[CompRealPred.CompRealPredPoints == -1], size=11, line_color='black', muted_alpha=0.2, line_width=2, fill_color='black', legend='Wrong Prediction')
		
		NewTime = pd.Series(pd.to_timedelta(df1.Date).dt.total_seconds())
		Fig.x_range = Range1d(NewTime[0]*1000, NewTime[NewTime.index[-1]]*1000) #Conversion from seconds to miliseconds
		del NewTime
	"""
	i = y_test.index[0]+1
	while i  <= y_test.index[-1]: 
	    if (y_test.at[i-1] == 1 and y_test.at[i] == 0): #Entrance
	        #print "Enter\n"
	        Fig.triangle(df.at[i+1, 'Date'], df.at[i+1, 'Close'], color='#440154', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='REAL Entrance/Exit')
	    if (y_test.at[i-1] == 0 and y_test.at[i] == 1): #Exit
	    	#print "Exit\n"
	        Fig.inverted_triangle(df.at[i+1, 'Date'], df.at[i+1, 'Close'], color='#29788E', line_color='black', line_width=1, fill_alpha=1, muted_alpha=0.2, size=11, legend='REAL Entrance/Exit') 
	    i = i + 1
	    print i
	"""
	Fig.legend.orientation = "horizontal"
	Fig.legend.location = "top_right"
	Fig.legend.click_policy = "hide"
	Fig.legend.border_line_alpha = 0
	Fig.legend.background_fill_alpha = 0

	return Fig

def PlotIntervalBox(df, Fig, Limits):
	"""
	for i in range(0,len(Limits)-1):
		vLine = Span(location=time.mktime(df.Date[Limits[i]].timetuple()), dimension='height', line_color='black', line_dash='dashed', line_width=2)
		Fig.add_layout(vLine)

	"""
	Colors = d3['Category10'][len(Limits)+1]
	box = BoxAnnotation(right=df.Date[Limits[0]], fill_color=Colors[0], fill_alpha=0.2) #First Train, from 0 to end of first period of train.
	Fig.add_layout(box)
	for i in range(0,len(Limits)-1): #Intermediate trains and tests
		box = BoxAnnotation(left=df.Date[Limits[i]], right=df.Date[Limits[i+1]], fill_color=Colors[i+1], fill_alpha=0.2)
		Fig.add_layout(box)
	box = BoxAnnotation(left=df.Date[Limits[-1]], fill_color=Colors[len(Limits)], fill_alpha=0.2) #Final Test.
	Fig.add_layout(box)

	#Fig.xaxis.major_label_orientation = pi/4
	#print([df.Date.iloc[0], df.Date[Limits[0]], df.Date[Limits[1]], df.Date[Limits[2]], df.Date[Limits[3]], df.Date.iloc[-1]])
	tick_vals = pd.to_datetime([df.Date.iloc[0], df.Date[Limits[0]], df.Date[Limits[1]], df.Date[Limits[2]], df.Date[Limits[3]], df.Date.iloc[-1]]).astype(int) / 10**6
	
	Fig.xaxis.ticker = FixedTicker(ticks=list(tick_vals))
	#Fig.xaxis.formatter = DatetimeTickFormatter(format="%F %T") #Format: %Y-%M-%D %h:%m:%s
	#Fig.xaxis.formatter=DatetimeTickFormatter(minutes=["%m/%d %H:%M"])

	return Fig

# Return on investment (ROI):
def plotROI(df, Algorithm, Accuracy, LogLoss, fig, Limits, color):
	#fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, 
	#	title="Return on investment (ROI) - " + str(Algorithm) + " - Final ROI: " + str(round(df.ROI.iloc[-2],5)) + '% - Accuracy: '
	#	+ str(round(100*Accuracy,5)) + '% - Log Loss: '+ str(round(LogLoss,5)) + '.', tools=TOOLS, toolbar_location='above')
	
	#use ColumnDataSource to pass in data for tooltips (HoverTool)
	source=ColumnDataSource(ColumnDataSource.from_df(df))
	r1 = fig.line('Date', 'ROI', line_width=2, color=color, source=source, legend=Algorithm)
	"""
	fig.add_tools(HoverTool(
		tooltips=[
			("ROI", "@{ROI}%"),
			("Date", "@Date{%F %T}"), #Format: %Y-%M-%D %h:%m:%s
	    ],
	    formatters={
        'Date': 'datetime', # use 'datetime' formatter for 'Date' field
        'ROI' : 'printf', # In order to write '%'' at the end
    	},
    	mode='vline',
    	renderers = [r1],
	))
	"""
	# Y axis converted to percentage:
	fig.yaxis.axis_label = "[%]"
	fig.yaxis.formatter = PrintfTickFormatter(format="%f%%")
	fig.grid.grid_line_color = 'navy'
	fig.ygrid.minor_grid_line_color = 'navy'
	fig.grid.grid_line_alpha = 0.1		
	fig.ygrid.minor_grid_line_alpha = 0.1
	fig.legend.background_fill_alpha = 0
	#fig.y_range = Range1d(-100, 1000)
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

	return fig

# Moving Averages:
def plotMA(df, fig, counter, n):
	color = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
	fig.line(df["Date"], df["MA_" + str(n)], color=color[counter], line_width=2, legend = str(n) + ' Moving Average')
	fig.legend.location = "top_left"
	fig.legend.click_policy = "hide"
	return fig

# Moving Averages derivatives:
def plotMAderiv(df, fig, counter, n):
	color = ['#440164', '#404397', '#29789E', '#22A794', '#79D161', '#FDE734']
	fig.line(df["Date"], df["MAderiv_" + str(n)], color=color[counter], line_width=2, legend = str(n) + ' Moving Average Derivate')
	fig.legend.location = "top_left"
	fig.legend.click_policy = "hide"
	return fig

# Exponential Moving Averages:
def plotEMA(df, fig, counter, n):
	color = ['#00204C', '#31446B', '#666870', '#958F78', '#CAB969', '#FFE945']
	fig.line(df["Date"], df["EMA_" + str(n)], color=color[counter], line_width=2, legend = str(n) + ' Exponential Moving Average')
	fig.legend.location = "top_left"
	fig.legend.click_policy = "hide"
	return fig

# Relative Strength Index:
def plotRSI(df, Periods):
	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, title="Relative Strength Index (RSI) [%] - " + str(Periods) +" Periods.", tools=TOOLS, toolbar_location='above')
	fig.line(df["Date"], df["RSI"], color='#000001')#, legend='Relative Strength Index [%]')

	# Colored areas
	low_box = BoxAnnotation(top=30, fill_alpha=0.1, fill_color='#FF0001')
	fig.add_layout(low_box)
	high_box = BoxAnnotation(bottom=70, fill_alpha=0.1, fill_color='#0B6623')
	fig.add_layout(high_box)

	# Horizontal line
	hline = Span(location=50, dimension='width', line_color='black', line_width=0.5)
	fig.renderers.extend([hline])

	# Y axis converted to percentage
	fig.y_range = Range1d(0, 100)
	fig.yaxis.ticker = [0, 30, 50, 70, 100]
	fig.yaxis.axis_label = "RSI"
	fig.yaxis.formatter = PrintfTickFormatter(format="%f")
	fig.grid.grid_line_alpha = 0.3

	fig.xaxis.axis_label = "Date"
	fig.yaxis.axis_label_text_font_size = '13pt'
	fig.xaxis.axis_label_text_font_size = '13pt'

	return fig

# MACD (line + histogram)
def plotMACD(df, n_fast, n_slow):
	width = (df["Date"][1] - df["Date"][0]).total_seconds() *1000 # Time difference between each entry on pandas DataFrame in ms --> *1000
	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, 
		title="Moving Average Convergence/Divergence (MACD) (line + histogram) - " + str(n_fast) + " Periods for Fast EMA; "+ str(n_slow) + " Periods for Slow EMA.", 
		tools=TOOLS, toolbar_location='above')
	fig.vbar(x=df.Date, bottom=0, top=df.MACDhist, width=width, color="purple")
	#fig.vbar(x=(df.Date [df.MACDhist < 0]), bottom=0, top=df.MACDhist, width=20, color="#F2583E")

	fig.line(df.Date, 0, color='black')
	fig.line(df.Date, df.MACD, line_width=2, color='#111E6C', muted_alpha=0.2, legend='MACD',)
	fig.line(df.Date, df.MACDsign, line_width=2, color='#95C8D8', muted_alpha=0.2, legend='Signal Line')

	fig.legend.location = "top_left"
	fig.legend.border_line_alpha = 0.3
	fig.legend.background_fill_alpha = 0.3
	fig.legend.click_policy = "mute"

	fig.yaxis.axis_line_alpha = 0

	return fig


# On-Balance Volume (OBV)
def plotOBV(df):#, Periods):
	#fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, title="On-Balance Volume (OBV) " + str(Periods) + " Periods.", tools=TOOLS, toolbar_location='above')
	#fig.line(x=df.Date, y=df['OBV_'+str(Periods)], line_width=1, color="#000001")

	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, title="On-Balance Volume (OBV).", tools=TOOLS, toolbar_location='above')
	fig.line(x=df.Date, y=df['OBV'], line_width=1, color="#000001")

	fig.yaxis.axis_line_alpha = 0

	return fig

# Rate of Change (ROC):
def plotROC(df, Periods):

	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, title="Rate of Change (ROC) - " + str(Periods) + " Periods.", tools=TOOLS, toolbar_location='above')
	fig.line(x=df.Date, y=df['ROC'], line_width=1, color="#000001")

	# Horizontal line
	hline = Span(location=0, dimension='width', line_color='black', line_width=0.5)
	fig.renderers.extend([hline])

	fig.yaxis.axis_label = "[%]"
	fig.yaxis.formatter = PrintfTickFormatter(format="%f%%")
	fig.grid.grid_line_alpha = 0.3

	return fig

# Commodity Channel Index (CCI):
def plotCCI(df, Periods):

	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, title="Commodity Channel Index (CCI) - " + str(Periods) + " Periods.", tools=TOOLS, toolbar_location='above')
	fig.line(df["Date"], df["CCI"], color='#000001')

	# Colored areas
	low_box = BoxAnnotation(top=-100, fill_alpha=0.1, fill_color='#FF0001')
	fig.add_layout(low_box)
	high_box = BoxAnnotation(bottom=100, fill_alpha=0.1, fill_color='#0B6623')
	fig.add_layout(high_box)

	# Horizontal line
	hline = Span(location=0, dimension='width', line_color='black', line_width=0.5)
	hline = Span(location=100, dimension='width', line_color='#0B6623', line_width=0.5)
	hline = Span(location=-100, dimension='width', line_color='#FF0001', line_width=0.5)
	fig.renderers.extend([hline])

	fig.yaxis.axis_label = "CCI"
	fig.xaxis.axis_label = "Date"
	

	return fig

# Average True Range:
def plotATR(df, Periods):

	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, title="Average True Range (ATR) - " + str(Periods) + " Periods.", tools=TOOLS, toolbar_location='above')
	fig.line(x=df.Date, y=df['ATR'], line_width=1, color="#000001")

	fig.yaxis.axis_line_alpha = 0

	fig.yaxis.axis_label = "ATR"
	fig.xaxis.axis_label = "Date"

	return fig

# Stochastic Oscillator:
def plotStochOsc(df, fastkPeriod, fastdperiod, slowdPeriod):
	width = (df["Date"][1] - df["Date"][0]).total_seconds() *1000 # Time difference between each entry on pandas DataFrame in ms --> *1000
	fig = figure(x_axis_type="datetime", plot_width=WIDTH_PLOT, plot_height=HEIGHT_SMALL_PLOT, 
		title=" Stochastic Oscillator - " + str(fastkPeriod) + " Periods for %K. " + str(fastdperiod) + " Periods for fast %D. " + 
		str(slowdPeriod) + " Periods for slow %K. ", tools=TOOLS, toolbar_location='above')
	fig.line(x=df.Date, y=df['StochOscSlowD'], line_width=1, color='#30678D', line_alpha=1, muted_alpha=0.2)#, legend='SlowD')
	fig.line(x=df.Date, y=df['StochOscFastK'], line_width=1, color='#35B778', line_alpha=0.8, muted_alpha=0.2)#, legend='FastK')
	fig.line(x=df.Date, y=df['StochOscFastD'], line_width=1, color='#FDE724', line_alpha=1, muted_alpha=0.2)#, legend='FastD')
	#fig.vbar(x=df.Date, bottom=0, top=df.StochFastHist, width=width, color="purple")
	#fig.vbar(x=df.Date, bottom=0, top=df.StochSlowHist, width=width, color="purple")

	# Horizontal line
	hline = Span(location=0, dimension='width', line_color='black', line_width=0.5)
	fig.renderers.extend([hline])

	# Y axis converted to percentage
	fig.yaxis.axis_label = "Stochastic Oscillator"
	fig.xaxis.axis_label = "Date"
	fig.yaxis.formatter = PrintfTickFormatter(format="%f")
	fig.grid.grid_line_alpha = 0.3

	# Colored areas
	low_box = BoxAnnotation(top=20, fill_alpha=0.1, fill_color='#FF0001')
	fig.add_layout(low_box)
	high_box = BoxAnnotation(bottom=80, fill_alpha=0.1, fill_color='#0B6623')
	fig.add_layout(high_box)

	fig.legend.location = "top_left"
	fig.legend.border_line_alpha = 0.3
	fig.legend.background_fill_alpha = 1
	fig.legend.click_policy = "mute"

	return fig
