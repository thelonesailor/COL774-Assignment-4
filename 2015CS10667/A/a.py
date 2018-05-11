import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import glob
import math

numpy_vars = {}
for np_name in glob.glob('./train/*.np[yz]'):
	start = np_name.find('n/') + 2
	end = np_name.find('.npy')
	numpy_vars[np_name[start:end]] = np.load(np_name)


print("training data read")
print(len(numpy_vars))

X=[]
s=0
i=0
mp={}
Ytrain={}
for np_name in numpy_vars:

	for a in numpy_vars[np_name]:
		X.append(a)
		mp[i]=a	
		Ytrain[i]=np_name
		i+=1

	s += len(numpy_vars[np_name])
	print(np_name)


# print(len(X))
# print(s)

X=np.array(X)

i=0
for a in X:
	mp[i]=a
	i+=1

Y=[]
kmeans = KMeans(n_clusters=20)
kmeans.fit(X)
Y=kmeans.labels_

classes=20
cl=[[] for i in range(classes)]
j=0
for i in Y:
	cl[i].append(j)
	j+=1

# centers=[]
centers=kmeans.cluster_centers_


newcl = [0 for i in range(classes)]
correct=0
for i in range(20):
	c={}
	for j in cl[i]:
		
		if(Ytrain[j] not in c):
			c[Ytrain[j]] = 1
		else:	
			c[Ytrain[j]]+=1

	nplabel=max(c, key=c.get)
	newcl[i]=nplabel
	correct+=c[nplabel]

print("{}/{} Accuracy={}%".format(correct, len(X), correct/len(X)*100))

testX = np.load("./test/test.npy")


testY=kmeans.predict(testX)

s="ID,CATEGORY\n"
i=0
for a in testY:
	s+=str(i)+","
	s+=newcl[a]
	s+="\n"
	i+=1

print(len(testX))

f=open("testlabels_kmeans.csv",'w')
f.write(s)
f.close()			