#To divide the data set based on whether it is buy or sell
import csv
import pandas as pd

f = ['./input/final.csv', './input/final_buy.csv', './input/final_sell.csv']

file = open(f[0], 'rb')
rdr= csv.DictReader( file )
flag1 = 0
flag2 = 0
for x in rdr:
	if x['side'] == 'B':
		if flag1 == 0:
			with open(f[1], 'a') as w_file:
				writer = csv.DictWriter(w_file, x.keys())
				writer.writeheader()
				writer.writerow(x)
				flag1 = 1
		else:
			with open(f[1], "a") as w_file:
				writer = csv.writer(w_file)
				writer.writerows([x.values()])
	else:
		if flag2 == 0:
			with open(f[2], 'a') as w_file:
				writer = csv.DictWriter(w_file, x.keys())
				writer.writeheader()
				writer.writerow(x)
				flag2 = 1
		else:
			with open(f[2], "a") as w_file:
				writer = csv.writer(w_file)
				writer.writerows([x.values()])