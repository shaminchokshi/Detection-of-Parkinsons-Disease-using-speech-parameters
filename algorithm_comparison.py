# import all necessary libraries
import pandas
from sklearn import cross_validation
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plot


url = "data.csv"
features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
dataset = pandas.read_csv(url, names = features)
array = dataset.values

# X stores feature values
X = array[:,0:22]
# Y stores "answers", the flower species / class (every row, 4th column)
Y = array[:,22]
# split dataset into training set (70%) and validation set (30%)
validation_size = 0.3
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size = validation_size, random_state = seed)
num_folds = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'accuracy'
# algorithms / models
models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NN', MLPClassifier(solver='lbfgs')))
models.append(('NB', GaussianNB()))
models.append(('GB', GradientBoostingClassifier(n_estimators=10000)))

# evaluate each algorithm / model
results = []
names = []
accuracy = []
mat_coef= []
print("Scores for each algorithm:")
for name, model in models:
    kfold = cross_validation.KFold(n = num_instances, n_folds = num_folds, random_state = seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    accuracy.append(accuracy_score(Y_validation, predictions)*100)
    mat_coef.append(matthews_corrcoef(Y_validation, predictions))
    if name =="DT":
        tree.export_graphviz(model, out_file="tree.dot")
    print('*******************',name,'*******************')
    print("Confusion matrix for",name,'\n',confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print('Accuracy Score:',accuracy_score(Y_validation, predictions)*100)
    print('Matthews Correlation Coeficient:',matthews_corrcoef(Y_validation, predictions),'\n')


plot.figure(figsize=(10,5))
plot.subplot(121)
plot.bar(names,accuracy)
plot.xlabel('Algorithm names')
plot.ylabel('Accuracy in %')
plot.subplot(122)
plot.bar(names,mat_coef)
plot.xlabel('Algorithm names')
plot.ylabel('Matthews Correlation Coeficient')
plot.suptitle("Comparison of the Algorithm used")
plot.show()
