import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import glob
import math
import numpy as np
from skorch.net import NeuralNetClassifier
from random import shuffle


import argparse
from torch.autograd import Variable

numpy_vars = {}
for np_name in glob.glob('./train/*.np[yz]'):
	start = np_name.find('n/') + 2
	end = np_name.find('.npy')
	numpy_vars[np_name[start:end]] = np.load(np_name)

i = 0
number = {}
label = []
for np_name in numpy_vars:
	number[np_name] = i
	label.append(np_name)
	i += 1

print("training data read")
print(len(numpy_vars))

classes = 20
xy = []
s = 0
for np_name in numpy_vars:

	for a in numpy_vars[np_name]:
		c = np.array(list(a))
		c=c-127.5
		c = c/127.5

		# c=c/255

		xy.append((c.astype(np.float32), number[np_name]))

	s += len(numpy_vars[np_name])
	print(np_name)


shuffle(xy)
shuffle(xy)

def convert(a):
	a2=[[0 for j in range(28)] for i in range(28)]
	for i in range(28):
		for j in range(28):
			a2[i][j]=a[i*28+j]
	return a2

X = []
Ytrain = []
for a, b in xy:
	a2=convert(a)
	X.append([a2])
	# X.append(a)

	Ytrain.append(b)

print(len(X))
print(s)


X = np.array(X)
Ytrain = np.array(Ytrain)

filters=32
k=5
print(filters)
print(k)

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, filters, kernel_size=k)
		self.mp = nn.MaxPool2d(2)
		
		self.fc = nn.Linear(4608, 20)
		# self.fc = nn.Linear(3872, 20)
	def forward(self, x):
		in_size = x.size(0)
		x = (self.conv1(x))
		x = F.relu(self.mp(x))

		x = x.view(in_size, -1)  
		x = self.fc(x)
		return F.softmax(x)


nnet = NeuralNetClassifier(
    module=Net,
    max_epochs=10,
    lr=0.05,
   	criterion=nn.NLLLoss,
   	optimizer=torch.optim.SGD,
	optimizer_momentum=0.9,   
)

# print(X)
# print(Ytrain)

X = X.astype(np.float32)
Ytrain = Ytrain.astype(np.int64)


# print(type(X))
# print(type(Ytrain))
# print(type(X[0]))
# print(type(X[0][0]))
# print(type(Ytrain[0]))

print("Training started")
assert(len(X) == len(Ytrain))
nnet.fit(X, Ytrain)

from sklearn.model_selection import cross_val_predict

# ytrain_pred = cross_val_predict(nnet, X, Ytrain,cv=5)


print("Training done")

Ytrain_pred = nnet.predict(X)
correct = 0
for i in range(len(X)):
	if(Ytrain_pred[i] == Ytrain[i]):
		correct += 1

print(Ytrain_pred)
print("Training Accuracy={}%".format(correct/len(X)*100))

tX = np.load("./test/test.npy")
testX = []
for a in tX:
	c = np.array(list(a))
	c = c-127.5
	c = c/127.5

	# c=c/255

	c2=np.array(convert(c))
	testX.append([c2.astype(np.float32)])
	# testX.append(c.astype(np.float32))

testX = np.array(testX)
testY = nnet.predict(testX)

s = "ID,CATEGORY\n"
i = 0
for d in testY:
	s += str(i)+","
	s += label[d]
	s += "\n"
	i += 1


f = open("testlabels_nn.csv", 'w')
f.write(s)
f.close()
