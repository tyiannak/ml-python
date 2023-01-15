import numpy as np
import pandas as pd
import csv
import os

dir_list = os.listdir('mm-fit')

features = {}


for d in dir_list:
    print(d)
    cur_numpy_file = os.path.join('mm-fit', d, d + '_sw_l_gyr.npy') 
    if os.path.isfile(cur_numpy_file):
        a = np.load(cur_numpy_file)
        timestamps = a[:, 0].astype('int').tolist()

        #print(timestamps.shape)
        #gt = pd.read_csv('w20/w20_labels.csv')
        with open(os.path.join('mm-fit', d, d + '_labels.csv'), 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                start, end, label = int(row[0]), int(row[1]), row[3]
                #itemindex = np.nonzero( timestamps == start)
                if start in timestamps and end in timestamps:
                    i_start = timestamps.index(start)
                    i_end = timestamps.index(end)
                    if label in features:
                        features[label].append(a[i_start: i_end, 2:])
                    else:
                        features[label] = [a[i_start: i_end, 2:]]
    print(len(features['squats']))

X = []
y = []
for class_name in features:
    for f in features[class_name]:
        X.append(np.mean(f, axis=0).tolist() + np.std(f, axis=0).tolist())
        y.append(class_name)
X = np.array(X)
y = np.array(y)
print(X)
print(y)

X_train = X[0:int(len(y) * 0.8), :]
y_train = y[0:int(len(y) * 0.8)]
X_test = X[int(len(y) * 0.8):, :]
y_test = y[int(len(y) * 0.8):]

print(X_train.shape, X_test.shape)

#print(gt)



