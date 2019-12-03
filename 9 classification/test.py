import numpy as np
from scipy.io import arff
from io import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def read(a):
	if a.endswith('.arff'):
		f = open(a,"r")
		if f.mode == 'r':
			content = f.read()
		c = StringIO(content)
		data, meta = arff.loadarff(c)
#	print(meta)
	return data
	
def remove_field(data, name):
	names = list(data.dtype.names)
	if name in names:
		names.remove(name)
	return data[names]
	

data = read('9.arff')
 
y = data[data.dtype.names[-1]]
X = remove_field(data, data.dtype.names[-1])

#print(X.dtype)
r = np.core.records.fromrecords(X)
print(r[0])
