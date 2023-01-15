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
                    print(start, end, a[i_start: i_end, 2].shape, label)

                    if label in features:
                        features[label].append(a[i_start: i_end, 2:])
                    else:
                        features[label] = [a[i_start: i_end, 2:]]

    print(len(features['squats']))


#print(gt)



