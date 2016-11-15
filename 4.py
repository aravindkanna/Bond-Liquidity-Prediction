#for sorting the files
import csv
import pandas as pd

f = ['./input/ML_Bond_metadata_corrected_dates.csv', './input/dataset.csv', 
'./input/final_sample.csv', './input/final_buy_sample.csv', './input/final_sell_sample.csv', 
'./input/sorted_buy.csv', './input/sorted_sell.csv']


file1= open( f[3], "rb" )
rdr= csv.DictReader( file1 )

price = 0;
for i in rdr:
	date = str(16) + "May2016"
	_id = "isin" + str(10033)
	if i['date'] == date and i['isin'] == _id:
		price = price + int(i['volume'])

print price