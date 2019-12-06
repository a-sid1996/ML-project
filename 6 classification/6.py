import numpy as np
from io import StringIO
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def randomCV(clf, X, y, param_grid, n_iter, cv):
	random_search = RandomizedSearchCV(clf, param_distributions = param_grid,
					n_iter = n_iter, cv = cv, iid = 					False)
	random_search.fit(X, y)
	report(random_search.cv_results_)
	
#	neigh.fit(X[:420], y[:420])
#	return neigh.score(X[420:], y[420:])
	

def KNN(X, y):
	
	neigh = KNeighborsClassifier()
	param_grid = {
		"n_neighbors" : np.arange(1,20),
		"algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute'],
		"weights" : ['uniform', 'distance'],
		"leaf_size" : np.arange(1,60)
	}
	randomCV(neigh, X, y, param_grid, 400, 6)

def SVM(X, y):

        C_grid = [0.1, 1, 10]
        gamma_grid = np.logspace(-2, 1, 4)[0:3]
        svm_C = svm.SVC(kernel='poly')
        param_grid = { 'C' : C_grid, 'gamma' : gamma_grid, "kernel" : ['rbf', 'sigmoid']}
        gridcv = GridSearchCV(svm_C, param_grid,  n_jobs = -1, verbose=1, cv=3)
        gridcv.fit(X, y)
        print("best parameters:", gridcv.best_params_)
        print("%.1f%% accuracy on validation sets (average)" % (gridcv.best_score_*100))

#	svm_C = svm.SVC()
#	param_grid = {
#		"kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
#		"gamma" : ['scale', 'auto'],
#		"degree" : np.arange(10),
#		"coef0" : np.random.rand(60)*10,
#		"shrinking" : [False, True],
#		"decision_function_shape" : ['ovo','ovr'],
#	}
#	randomCV(svm_C, X, y, param_grid, 4, 6)

def DT(X, y):
	dt = DecisionTreeClassifier()
	param_grid = {
		"criterion" : ['gini', 'entropy'],
		"splitter" : ['best', 'random'],
		"min_samples_split" : np.random.random_sample((100,)),
		"max_features" : ['auto', 'sqrt', 'log2', None],
		"class_weight" : [None, 'balanced'],
		"presort" : [True, False],
		"min_samples_leaf" : np.arange(1,6)
	}
	randomCV(dt, X, y, param_grid, 400, 6)

def RF(X, y):
	rf = RandomForestClassifier()
	param_grid = {
		"n_estimators" : [10*x for x in np.arange(1,50)],
		"criterion" : ['gini', 'entropy'],
		"min_samples_split" : np.random.random_sample((100,)),
		"max_features" : ['auto', 'sqrt', 'log2', None],
#		"class_weight" : [None, 'balanced'],
		"min_samples_leaf" : np.arange(1,6),
#		"bootstrap" : [True, False],
#		"oob_score" : [True, False],
		"warm_start" : [True, False],
	}
	randomCV(rf, X, y, param_grid, 40, 6)

def Ada(X, y):
	ada = AdaBoostClassifier(algorithm = "SAMME")
	param_grid = {
#		"base_estimator" : ['classes', 'n_classes_', None],
		"n_estimators" : [10*x for x in np.arange(1,50)]
#		"learning_rate" : [10*x for x in np.random.random_sample((100,))]
#		"algorithm" : ['SAMME']
	}
	randomCV(ada, X, y, param_grid, 40, 6)

def LR(X, y):
	lr = LogisticRegression()
	param_grid = {
		"penalty" : ['l1', 'l2'],
#		"dual" : [True, False],
		"C" : np.random.rand(60),
		"fit_intercept" : [True, False],
		"warm_start" : [True, False],
		"multi_class" : ['ovr', 'auto'],
		"solver" : [ 'liblinear']
	}
	randomCV(lr, X, y, param_grid, 400, 6)

def GNB(X, y):
	gnb = GaussianNB()
	param_grid = {
		"var_smoothing" : np.random.random_sample((100,))
	}
	randomCV(gnb, X, y, param_grid, 100, 6)

def NN(X, y):
	nn = MLPClassifier()
	param_grid = {
		"activation" : ['identity', 'logistic', 'tanh', 'relu'],
		"solver" : ['lbfgs', 'sgd', 'adam'],
#		"verbose" : [True, False],
		"warm_start" : [False, True]
	}
	randomCV(nn, X, y, param_grid, 400, 6)


def read(a):
	if a.endswith('.NNA'):
		f = open(a,"r")
		if f.mode == 'r':
			content = f.read()
		c = StringIO(content)
		return np.loadtxt(c)


# ---------------> Read dataset file
data = read('Faults.NNA')
y = data[:,-1]
x = data[:,0:data.shape[1] -1]
# --------------------> KNN

#KNN(x,y)

## --------------------> SVM

SVM(x,y)

## --------------------> Decision Tree

#DT(x,y)

## --------------------> Random Forest

#RF(x, y)

## --------------------> Adaboost

#Ada(x, y)


## ---------------------> Logistic regression

#LR(x, y)

## ---------------------> Gaussian NB

#GNB(x, y)


## ---------------------> Neural Network

#NN(x, y)

