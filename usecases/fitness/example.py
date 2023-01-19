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

# create the dataset:
X = []
y = []
for class_name in features:
    for f in features[class_name]:
        X.append(np.mean(f, axis=0).tolist() + np.std(f, axis=0).tolist() + np.min(f, axis=0).tolist() + np.max(f, axis=0).tolist())
        y.append(class_name)
X = np.array(X)
y = np.array(y)

X_train = X[1::2, :]
y_train = y[1::2]
X_test = X[0::2, :]
y_test = y[0::2]


# train the model:
from sklearn import svm
clf = svm.SVC(probability=True)     # initialize the classifier with probabilistic output!
clf.fit(X_train, y_train)           # train the classifier 
y_pred = clf.predict(X_test)


# test the model:
from sklearn import metrics
class_names = [k for k in features]
print(metrics.confusion_matrix(y_test, y_pred, labels=class_names))
print(class_names)
#print(gt)



