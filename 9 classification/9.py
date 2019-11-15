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
	return np.asarray(data.tolist())

def KNN(X, y):
	neigh = KNeighborsClassifier(n_neighbors = 3)
	neigh.fit(X[:420],y[:420])
	return neigh.score(X[420:],y[420:])

# def SVM(X, y):


# ------------------> Read dataset file

data = read('9.arff')

# ------------------> Separate target variable
 
y = data[::,-1]
X = np.delete(data, -1, axis = 1)

# -------------------> Preprocessing

le = preprocessing.LabelEncoder()

for i in range(X.shape[1]):
	if X.T[i].dtype != np.float64:
		X[:,i] = le.fit_transform(X[:,i])
		
X = X.astype(np.float64)

# --------------------> KNN

print(KNN(X,y))

# --------------------> SVM


