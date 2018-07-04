import pandas
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

def main():

    #Loading data
    url = "cancer.csv"
    names = ["id","clump thickness","uniformity of cell size","uniformity of cell shape","marginal adhesion","single epithelial cell size","bare nuclei","bland chromatin","normal nucleoli","mitoses","class"]
    dataset = pandas.read_csv(url,names=names)

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
    dataset.plot(kind='box',subplots=True,layout=(10,1),sharex=False,sharey=False)
    plt.show()
    dataset.hist()
    plt.show()






if __name__=="__main__":
    main()
