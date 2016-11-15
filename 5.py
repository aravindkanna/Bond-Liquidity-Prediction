import csv
import pandas as pd

f = ['./input/ML_Bond_metadata_corrected_dates.csv', './input/dataset.csv', 
'./input/final.csv', './input/final_buy.csv', './input/final_sell.csv', 
'./input/sorted_buy.csv', './input/sorted_sell.csv']

file = open(f[5], 'rb')
rdr= csv.DictReader( file )

for i in rdr:
	if i['isin'] == "isin3":
		print i['isin']