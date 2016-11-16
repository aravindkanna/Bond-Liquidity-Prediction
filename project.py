import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import re

def getDay(date):
	return int(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", date)[0])

f = ['./input/ML_Bond_metadata_corrected_dates.csv', './input/dataset.csv', 
'./input/dataset.csv', './input/data_sample.csv', './input/final_sell_sample.csv', 
'./input/f_buy.csv', './input/f_sell.csv', './input/d.csv']


"""
	These 2 methods are used to hash the isin numbers and dates so that we can 
	store them in a 2D array index by integers
"""
def hash_id(isin):
	return getDay(isin)

def hash_date(date):
	day = getDay(date)
	if date[2] == 'A' and date[3] == 'p' and date[4] == 'r' :
		return day - 1
	elif date[2] == 'M' and date[3] == 'a' and date[4] == 'y' :
		return 30 + day - 1
	elif date[2] == 'J' and date[3] == 'u' and date[4] == 'n' :
		return 61 + day - 1
	else:
		return -1

"""
	This method creates a dataset of required sizes with lookback
 	look back implies the number of items we took in each row
"""
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)



dp_buy = []
dp_sell = []

for i in range(17329):
	dp_buy.append([])
	dp_sell.append([])

for i in range(17329):
	for j in range(91):
		dp_buy[i].append(0)
		dp_sell[i].append(0)


file = open(f[2], 'rb')
r_buy = csv.DictReader( file )

valid = [False] * 17329

"""
	storing the volumes per day in a 2D array of size 17328 x 90
	that means volume on a day can be retreived as dp[isin][date]
"""
for row in r_buy:
	vol = int(row['volume'])
	h_id = hash_id(row['isin'])
	h_date = hash_date(row['date'])
	sd = row['side']
	if sd is 'B' and h_date is not -1:
		dp_buy[h_id][h_date] = dp_buy[h_id][h_date] + vol
	elif sd is 'S' and h_date is not -1: 
		dp_sell[h_id][h_date] = dp_sell[h_id][h_date] + vol
	if valid[h_id] is False:
		valid[h_id] = True


#fix random seed for reproducability
numpy.random.seed(7)


# This is for Buy volumes
for i in dp_buy:

	#loading the dataset
	dataset = numpy.array(i)
	dataset = dataset.astype(int)

	#Normalize the dataset
	scaler = MinMaxScaler(feature_range = (0, 1))
	dataset = scaler.fit_transform(dataset)

	#split into train and test datasets
	ln = len(dataset)
	train_size = int(ln * 0.67)
	test_size = ln - train_size
	train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

	#reshape into x = t and y = t + 1
	look_back = 10
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	

	#reshape input to be [samples, timesteps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_dim=look_back))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, nb_epoch=20, batch_size=1, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)


	dataset1 = numpy.array(i[len(i) - 14 : ])
	dataset1.astype(int)

	dataset1 = scaler.fit_transform(dataset1)
	

	trainX1, trainY1 = create_dataset(dataset1, 10)
	trainX1 = numpy.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
	
	data1 = model.predict(trainX1)
	#data




	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])


	data_predict = scaler.inverse_transform(data1)
	print "Next 3 days volumes"
	print data_predict[0] + data_predict[1] + data_predict[2]




	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))


"""
# This is for Sell Volumes
for i in dp_sell:

	#loading the dataset
	dataset = numpy.array(i)
	dataset = dataset.astype(int)

	#Normalize the dataset
	scaler = MinMaxScaler(feature_range = (0, 1))
	dataset = scaler.fit_transform(dataset)

	#split into train and test datasets
	ln = len(dataset)
	train_size = int(ln * 0.67)
	test_size = ln - train_size
	train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

	#reshape into x = t and y = t + 1
	look_back = 20
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	

	#reshape input to be [samples, timesteps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_dim=look_back))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, nb_epoch=20, batch_size=1, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	dataset1 = numpy.array(i[len(i) - 14 : ])
	dataset1.astype(int)

	dataset1 = scaler.fit_transform(dataset1)
	

	trainX1, trainY1 = create_dataset(dataset1, 10)
	trainX1 = numpy.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
	
	data1 = model.predict(trainX1)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	data_predict = scaler.inverse_transform(data1)
	print "Next 3 days volumes"
	print data_predict

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

"""

"""
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()
"""
