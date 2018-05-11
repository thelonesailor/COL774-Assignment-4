import numpy as np
from statistics import mean
# import random
# import pickle
# import operator
# from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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


json_file = open('model_n2keras.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_n2keras.h5")
model=loaded_model

testX = np.load('test/test.npy')
m=len(testX)
# m=100
testX = testX.astype('float32')
testX = testX/255

testX = testX[0:m].reshape(m, 28, 28, 1)

print("Predicting on test data")
predicted = model.predict(testX)
testY = np.argmax(predicted, axis=1)

f = open('n2_keras.csv','w')
f.write('ID,CATEGORY\n')
for i in range(m):
	f.write(str(i) + "," + labels[testY[i]] + '\n')
f.close()	
