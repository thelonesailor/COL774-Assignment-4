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

numpy_vars = {}
for np_name in glob.glob('./train/*.np[yz]'):
	start = np_name.find('n/') + 2
	end = np_name.find('.npy')
	numpy_vars[np_name[start:end]] = np.load(np_name)

i=0
number={}
label=[]
for np_name in numpy_vars:
	number[np_name]=i
	label.append(np_name)
	i+=1

print("training data read")
print(len(numpy_vars))

classes=20
xy=[]
s = 0
i = 0
for np_name in numpy_vars:

	for a in numpy_vars[np_name]:
		c=np.array(list(a))
		xy.append((c.astype(np.float32), number[np_name]))
		i += 1

	s += len(numpy_vars[np_name])
	print(np_name)


shuffle(xy)
shuffle(xy)

X=[]
Ytrain=[]
for a,b in xy:
	X.append(a)
	Ytrain.append(b)

print(len(X))
print(s)


X = np.array(X)
Ytrain = np.array(Ytrain)

num=1400
print(num)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28 * 28,num)
		self.fc2 = nn.Linear(num, 20)

	def forward(self, X, **kwargs):
		X = F.sigmoid(self.fc1(X))
		X = F.softmax(self.fc2(X))
		return X


nnet=NeuralNetClassifier(
    Net,
    max_epochs=20,
    lr=0.0165,
   	criterion=nn.NLLLoss,
   	optimizer=torch.optim.SGD,
)

print(X)
print(Ytrain)

X = X.astype(np.float32)
Ytrain = Ytrain.astype(np.int64)


print(type(X))
print(type(Ytrain))

print(type(X[0]))
print(type(X[0][0]))
print(type(Ytrain[0]))

assert(len(X) == len(Ytrain))
nnet.fit(X, Ytrain)

from sklearn.model_selection import cross_val_predict

# ytrain_pred = cross_val_predict(nnet, X, Ytrain,cv=5)


print("Training done")

Ytrain_pred = nnet.predict(X)
correct=0
for i in range(len(X)):
	if(Ytrain_pred[i]==Ytrain[i]):
		correct+=1

print("Training Accuracy={}%".format(correct/len(X)*100))

tX = np.load("./test/test.npy")
testX=[]
for a in tX:

	c=np.array(list(a))	
	testX.append(c.astype(np.float32))

testX=np.array(testX)
testY=nnet.predict(testX)

s = "ID,CATEGORY\n"
i=0
for d in testY:
	s+=str(i)+","
	s+=label[d]
	s+="\n"
	i+=1


f = open("testlabels_nn.csv", 'w')
f.write(s)
f.close()
