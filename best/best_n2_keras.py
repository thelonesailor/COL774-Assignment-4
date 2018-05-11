import glob
import math
import numpy as np
import numpy
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
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
		# c = c-127.5
		# c = c/127.5

		c=c/255

		xy.append((c.astype(np.float32), number[np_name]))

	s += len(numpy_vars[np_name])
	print(np_name)


shuffle(xy)
shuffle(xy)


def convert(a):
	a2 = [[0 for j in range(28)] for i in range(28)]
	for i in range(28):
		for j in range(28):
			a2[i][j] = [a[i*28+j]]
	return a2


X = []
Ytrain = []
for a, b in xy:
	a2 = convert(a)
	X.append(a2)
	# X.append(a)

	Ytrain.append(b)

print(len(X))
print(s)


X = np.array(X)
Ytrain = np.array(Ytrain)

print("making model")

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
print(model.summary())

print("model made")

# print(X)
# print(Ytrain)

X = X.astype(np.float32)
Ytrain = Ytrain.astype(np.int64)


print(type(X))
print(type(Ytrain))

print(type(X[0]))
print(type(X[0][0]))
print(type(Ytrain[0]))

assert(len(X) == len(Ytrain))
print("fitting started")

# Fit the model
model.fit(X, Ytrain, validation_data=(X, Ytrain), epochs=10, batch_size=64)
print("Training done")

# Final evaluation of the model
scores = model.evaluate(X, Ytrain, verbose=0)



print("Training Accuracy={}%".format(scores[1]*100))

tX = np.load("./test/test.npy")
testX = []
for a in tX:
	# c = []
	# for b in a:
	# 	c.append(np.float32(b))
	c = np.array(list(a))
	# c = c-127.5
	# c = c/127.5

	c=c/255

	c2 = np.array(convert(c))
	testX.append(c2.astype(np.float32))
	# testX.append(c.astype(np.float32))

testX = np.array(testX)
testY=model.predict(testX)

s = "ID,CATEGORY\n"
i = 0
for d in testY:
	s += str(i)+","
	s += label[d]
	s += "\n"
	i += 1


f = open("testlabels_n2_keras.csv", 'w')
f.write(s)
f.close()

# model=nnet
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# def train(epoch):
# 	model.train()
# 	for data, target in enumerate(xy):
# 		data, target = Variable(data), Variable(target)
# 		optimizer.zero_grad()
# 		output = model(data)
# 		loss = F.nll_loss(output, target)
# 		loss.backward()
# 		optimizer.step()
# 		# if batch_idx % 10 == 0:
# 		print('Train Epoch: {}\tLoss: {:.6f}'.format(
# 			epoch, loss.data[0]))


# for epoch in range(1, 10):
#     train(epoch)
