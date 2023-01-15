import numpy as np
import pandas as pd
import csv

a = np.load("mm-fit/w20/w20_sw_l_gyr.npy")
timestamps = a[:, 0].astype('int').tolist()

features = {}

#print(timestamps.shape)
#gt = pd.read_csv('w20/w20_labels.csv')
with open("mm-fit/w20/w20_labels.csv", 'r') as file:
	csvreader = csv.reader(file)
	for row in csvreader:
		start, end, label = int(row[0]), int(row[1]), row[3]
		#itemindex = np.nonzero( timestamps == start)
		i_start = timestamps.index(start)
		i_end = timestamps.index(end)
		print(start, end, a[i_start: i_end, 2].shape, label)

		if label in features:
			features[label].append(a[i_start: i_end, 2:])
		else:
			features[label] = [a[i_start: i_end, 2:]]

print(len(features['squats']))

#print(gt)



