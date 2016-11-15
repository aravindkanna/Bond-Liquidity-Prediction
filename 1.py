#To convert the initial two data sets into a single data set
import csv
import pandas as pd

f = ['./input/ML_Bond_metadata_corrected_dates.csv', './input/dataset.csv', './input/final.csv']

"""
for file in f:
	with open(file, 'r') as csvfile:
		sales = csv.reader(csvfile)
		for row in sales:
			print row
"""


"""
f1 = pd.read_csv(f[0], header=None)
f2 = pd.read_csv(f[1], header=None)

merged = pd.concat(f1, f2)
merged.to_csv('merged.csv', index=None, header=None)
"""

"""
file1 = []
file2 = []

with open(f[0], 'r') as csvfile1:
	s1 = csv.DictReader(csvfile1)
	for row1 in s1:
		file1.append(row1)

for r in file1:
	print r['isin'] 


with open(f[1], 'r') as  csvfile2:
	s2 = csv.DictReader(csvfile2)
	for row2 in s2:
		file2.append(row2)

"""

import csv
import collections

index = collections.defaultdict(list)

file1= open( f[0], "rb" )
rdr= csv.DictReader( file1 )
for row in rdr:
    index[row['isin']].append( row )
file1.close()



file2= open( f[1], "rb" )
rdr= csv.DictReader( file2 )
flag = 0
for row in rdr:
	x = index[row['isin']][0].copy()
	x.update(row)
	if flag == 0:
		with open(f[2], 'a') as w_file:
			writer = csv.DictWriter(w_file, x.keys())
			writer.writeheader()
			writer.writerow(x)
			flag = 1
	else:
		with open(f[2], "a") as w_file:
			writer = csv.writer(w_file)
			writer.writerows([x.values()])		


file2.close()