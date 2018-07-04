import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

def main():

    #Loading data
    url = "cancer.csv"
    names = ["id","clump thickness","uniformity of cell size","uniformity of cell shape","marginal adhesion","single epithelial cell size","bare nuclei","bland chromatin","normal nucleoli","mitoses","class"]
    dataset = pandas.read_csv(url,names=names)
    dataset = dataset.convert_objects(convert_numeric=True)
    dataset = dataset.fillna(method='ffill')
    #print(dataset.isnull().any())
    #wait = input("PRESS ENTER TO CONTINUE.")

    #Summarizing data
    print("Summarizing Data")
    print("Shape of dataset"+str(dataset.shape))
    print("....")
    print("Few instances of data")
    print(dataset.head(5))
    print("....")
    print("Statistical summary of data")
    print(dataset.describe())
    print("....")

    #Visualizing the data
    dataset.plot(kind='box',subplots=True,layout=(11,1),sharex=False,sharey=False)
    plt.show()
    dataset.hist()
    plt.show()

    #Split out the validation set
    array = dataset.values
    X = array[:,0:10]
    y = array[:,10]
    seed= 7
    validation_size = 0.20
    scoring = "accuracy"
    X_train,X_validation,y_train,y_validation = model_selection.train_test_split(X,y,test_size=validation_size,random_state=seed)

    #Algorithms
    models =[]
    models.append(('LR',LogisticRegression()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('LDA',LinearDiscriminantAnalysis()))
    models.append(('CART',DecisionTreeClassifier()))
    models.append(('NB',GaussianNB()))
    models.append(('SVM',SVC()))
    results = []
    names = []


    for name,model in models:
        kfold = model_selection.KFold(n_splits=10,random_state=seed)
        cv_results = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
        print(msg)




if __name__=="__main__":
    main()
