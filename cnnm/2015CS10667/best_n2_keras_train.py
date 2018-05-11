import numpy as np
import random
import operator
from statistics import mean
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import cross_val_score
from keras import backend as K
from keras.models import model_from_json

labels = ['banana',
		  'bulldozer',
		  'chair',
		  'eyeglasses',
		  'flashlight',
		  'foot',
		  'hand',
		  'harp',
		  'hat',
		  'keyboard',
		  'laptop',
		  'nose',
		  'parrot',
		  'penguin',
		  'pig',
		  'skyscraper',
		  'snowman',
		  'spider',
		  'trombone',
		  'violin']

classes=20
X = []
trainY = []
for i in range(classes):
	x = np.load('./train/' + labels[i] + '.npy')
	size = len(x)
	X += x.tolist()
	for j in range(size):
		trainY.append(i)

X = np.array(X)
trainY = np.array(trainY)
X = X.astype('float32')
X = X/255.0

X = X.reshape(len(X), 28, 28, 1)

trainY = to_categorical(trainY, classes)

# model = None
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
m=len(X)
# m=100
model.fit(X[0:m], trainY[0:m], batch_size=256, epochs=30, verbose=1)

model_json = model.to_json()
with open("model_n2keras.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_n2keras.h5")
print("Model saved")


print("Predicting on train data")
prob = model.predict(X)
trainY_predicted = np.argmax(prob, axis=1)

correct = 0
for i in range(m):
	j2=0
	for j in range(0, len(trainY[i])):
		if trainY[i][j] == 1:
			j2=j
			break
	if trainY_predicted[i] == j2:
		correct += 1
accuracy = correct * 100 / len(trainY)
print ("Traininng accuracy={}%".format(accuracy))

print("Now run on testing file")