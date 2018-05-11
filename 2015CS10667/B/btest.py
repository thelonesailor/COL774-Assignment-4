
import numpy as np
import operator
from random import shuffle
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import glob
from sklearn.model_selection import cross_val_score

label = ['bulldozer',
         'banana',
         'violin',
         'nose',
         'harp',
         'trombone',
         'laptop',
         'penguin',
         'snowman',
         'hat',
         'hand',
         'foot',
         'flashlight',
         'chair',
         'skyscraper',
         'keyboard',
         'spider',
         'eyeglasses',
         'pig',
         'parrot']


classes = 20
with open('svm3', 'rb') as output:
	model = pickle.load(output)

with open('pca3', 'rb') as output:
	lst = pickle.load(output)
trainX = lst[1]
test_X = lst[3]


#
m=len(test_X)
# m = 1000
#


testY=model.predict(test_X[0:m])
print("Prediciton done")

f = open('svm_testLabels_btest.csv', 'w')
f.write('ID,CATEGORY\n')
for i in range(m):
	f.write(str(i) + ',' + label[testY[i]] + '\n')
f.close()
