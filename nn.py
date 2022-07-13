# import all necessary libraries
import pandas
from pandas.plotting import scatter_matrix
import sklearn.model_selection as cross_validation
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# load the dataset (local path)
url = "data.csv"
# feature names
features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
dataset = pandas.read_csv(url, names = features)

# store the dataset as an array for easier processing
array = dataset.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(array)
# X stores feature values
X = scaled[:,0:22]
# Y stores "answers", the flower species / class (every row, 4th column)
Y = scaled[:,22]
validation_size = 0.25
# randomize which part of the data is training and which part is validation
seed = 7
# split dataset into training set (80%) and validation set (20%)
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size = validation_size, random_state = seed)

# 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
splits = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'accuracy'

results = []
clf = MLPClassifier(solver='lbfgs')
kfold = cross_validation.KFold(n_splits = splits, random_state = seed)
cv_results = cross_validation.cross_val_score(clf, X_train, Y_train, cv = kfold, scoring = scoring)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_validation)
print("NN")
print("Accuracy",accuracy_score(Y_validation, predictions)*100)
print("MCC",matthews_corrcoef(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
