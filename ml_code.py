from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    cancer = load_breast_cancer
    X_train,X_test,y_train,y_test = train_test_split(cancer["data"],cancer["target"],random_state=0)
    clf = LogisticRegression().fit(X_train,y_train)
    print("Training score data: %f" % clf.score(X_train,y_train))
    print("Test score data: %f" % clf.score(X_test,y_test))


if __name__ == "__main__":
  main()
