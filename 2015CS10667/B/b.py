import numpy as np
import operator
from random import shuffle
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import glob
from sklearn.model_selection import cross_val_score

##############################################
#					PCA

numpy_vars = {}
for np_name in glob.glob('./train/*.np[yz]'):
	start = np_name.find('n/') + 2
	end = np_name.find('.npy')
	numpy_vars[np_name[start:end]] = np.load(np_name)


label=['bulldozer',
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

i=0
number={}
for a in label:
	number[a]=i
	i+=1

classes = 20
xy = []
s = 0
for np_name in label:

	for a in numpy_vars[np_name]:
		c = np.array(list(a))

		# c = c-127.5
		# c = c/127.5

		# c=c/255

		xy.append((c, number[np_name]))

	s += len(numpy_vars[np_name])
	print(np_name)

shuffle(xy)
shuffle(xy)

trainX = []
trainY = []
for a, b in xy:
	trainX.append(a)
	trainY.append(b)

print("PCA started")

#
m=len(trainX)
# m=1000
#

testX = np.load('./test/test.npy')
# for i in range(len(testX)):

	# testX[i] = testX[i]-127.5
	# testX[i] = testX[i]/127.5

	# testX[i] = testX[i]/255


pca = PCA(n_components=50)
pca.fit(trainX[0:m])
train_X = pca.transform(trainX[0:m])
test_X = pca.transform(testX[0:m])


print("PCA done")

lst = []
lst.append(train_X)
lst.append(trainX)
lst.append(trainY)
lst.append(test_X)

with open('pca3', 'wb') as output:
	pickle.dump(lst, output)


# ##############################################
# #				TRAINING

print("Training started")

with open('pca3', 'rb') as output:
	lst = pickle.load(output)
train_X = (lst[0]).tolist()
trainX = lst[1]
trainY = lst[2]



# for C in [0.000001, 0.000002, 0.00001, 0.0001, 0.001]:
C=0.000001
model = SVC(C=C, kernel='linear').fit(train_X[0:m], trainY[0:m])
with open('svm3', 'wb') as output:
	pickle.dump(model, output)


print("Training done C={}".format(C))
trainscore=model.score(train_X[0:m], trainY[0:m])
print("Training score={}".format(trainscore))

scores = cross_val_score(model, train_X[0:m], trainY[0:m], cv=5)
print("Cross validation scores={}\n".format(scores))

