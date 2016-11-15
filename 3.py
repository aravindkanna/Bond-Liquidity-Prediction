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

def getDay(date):
	return int(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", date)[0])

f = ['./input/ML_Bond_metadata_corrected_dates.csv', './input/dataset.csv', 
'./input/dataset.csv', './input/data_sample.csv', './input/final_sell_sample.csv', 
'./input/f_buy.csv', './input/f_sell.csv']

file = open(f[2], 'rb')
r_buy = csv.DictReader( file )

rdr_buy = []
for i in r_buy:
	rdr_buy.append(i)

"""
file = open(f[4], 'rb')
r_sell = csv.DictReader( file )

rdr_sell = []
for i in r_sell:
	rdr_sell.append(i)"""

"""
data = pd.read_csv(f[3], usecols=['price'], engine = 'python')
plt.plot(data)
plt.show()
"""

## isin0 ----> isin17328
## 1Apr2016 ----> 30Jun2016

l = ['isin', 'date', 'volume']

with open(f[5], 'a') as w_file:
	writer = csv.writer(w_file)
	writer.writerow(l)

with open(f[6], 'a') as w_file:
	writer = csv.writer(w_file)
	writer.writerow(l)

ID = range(17329)
for ids in ID:
	_id = "isin" + str(ids)
	for apr in range(1, 31):
		#print [ids, apr]
		if apr < 10:
			dat = " " + str(apr) + "Apr2016"
		else:
			dat = str(apr) + "Apr2016"
		l_buy = []
		l_sell = []

		l_buy.append(_id)
		l_buy.append(dat)

		l_sell.append(_id)
		l_sell.append(dat)

		volume_buy = 0
		volume_sell = 0
		for row in rdr_buy:
			if row['isin'] == _id and row['date'] == dat:
				if(row['side'] == 'B'):
					volume_buy = volume_buy + int(row['volume'])
				else:
					volume_sell = volume_sell + int(row['volume'])

		l_buy.append(str(volume_buy))
		l_sell.append(str(volume_sell))

		with open(f[5], 'a') as w_file:
			writer = csv.writer(w_file)
			writer.writerow(l_buy)

		with open(f[6], 'a') as w_file:
			writer = csv.writer(w_file)
			writer.writerow(l_sell)


	for apr in range(1, 32):
		if apr < 10:
			dat = " " + str(apr) + "May2016"
		else:
			dat = str(apr) + "May2016"
		l_buy = []
		l_sell = []

		l_buy.append(_id)
		l_buy.append(dat)

		l_sell.append(_id)
		l_sell.append(dat)

		volume_buy = 0
		volume_sell = 0
		for row in rdr_buy:
			if row['isin'] == _id and row['date'] == dat:
				if(row['side'] == 'B'):
					volume_buy = volume_buy + int(row['volume'])
				else:
					volume_sell = volume_sell + int(row['volume'])

		l_buy.append(str(volume_buy))
		l_sell.append(str(volume_sell))

		with open(f[5], 'a') as w_file:
			writer = csv.writer(w_file)
			writer.writerow(l_buy)

		with open(f[6], 'a') as w_file:
			writer = csv.writer(w_file)
			writer.writerow(l_sell)

	for apr in range(1, 31):
		if apr < 10:
			dat = " " + str(apr) + "Jun2016"
		else:
			dat = str(apr) + "Jun2016"
		l_buy = []
		l_sell = []

		l_buy.append(_id)
		l_buy.append(dat)

		l_sell.append(_id)
		l_sell.append(dat)

		volume_buy = 0
		volume_sell = 0
		for row in rdr_buy:
			if row['isin'] == _id and row['date'] == dat:
				if(row['side'] == 'B'):
					volume_buy = volume_buy + int(row['volume'])
				else:
					volume_sell = volume_sell + int(row['volume'])

		l_buy.append(str(volume_buy))
		l_sell.append(str(volume_sell))

		with open(f[5], 'a') as w_file:
			writer = csv.writer(w_file)
			writer.writerow(l_buy)

		with open(f[6], 'a') as w_file:
			writer = csv.writer(w_file)
			writer.writerow(l_sell)