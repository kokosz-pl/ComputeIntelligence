import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def classify_iris(sl, sw, pl, pw):
    if pw <= 0.7:
        return("Setosa")
    elif pw >= 1.8:
        return("Virginica")
    elif 0.9 < pw < 1.8:
        return("Versicolor")

def iris_classification(dataset):
    (train_set, test_set) = train_test_split(dataset.values, train_size=0.7, random_state=53)
    
    good_predictions = 0
    length = test_set.shape[0]
    
    
    for i in range(length):
        if classify_iris(*test_set[i, :4]) == test_set[i, 4]:
            good_predictions += 1
    print(f"Iris calssification: {good_predictions / length * 100} %\n")


def DivideDataset(dataset):
    all_inputs = dataset[["sepal.length","sepal.width","petal.length","petal.width"]].values
    all_classes = dataset['variety'].values
    return train_test_split(all_inputs, all_classes, train_size=0.7, random_state=53)


def DecisionTree(dataset):

    (train_inputs, test_inputs, train_classes, test_classes) = DivideDataset(dataset)
    
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(train_inputs, train_classes)
    print(f"Decision Tree Classifier: {dtc.score(test_inputs, test_classes) * 100} %")
    
    test_pred = dtc.predict(test_inputs)
    print(f"Confusion Matrix:\n {confusion_matrix(test_classes, test_pred)}\n")

def KNN(dataset):


    (train_inputs, test_inputs, train_classes, test_classes) = DivideDataset(dataset)

    k = [3, 5, 11]
    for neighbours in k:
        knn = KNeighborsClassifier(n_neighbors=neighbours)
        knn.fit(train_inputs, train_classes)
        knn_predict = knn.predict(test_inputs)
        score = accuracy_score(test_classes, knn_predict)
        print(f"{neighbours}NN score: {score * 100} %")
        print(f"{neighbours}NN Confusion Matrix:\n {confusion_matrix(test_classes, knn_predict)}\n")


def Naive_Bayes(dataset):

    (train_inputs, test_inputs, train_classes, test_classes) = DivideDataset(dataset)
    
    bayes = GaussianNB()
    bayes.fit(train_inputs, train_classes)
    bayes_predict = bayes.predict(test_inputs)
    score = accuracy_score(test_classes, bayes_predict)
    print(f"Bayes score: {score * 100} %")
    print(f"Bayes Confusion Matrix:\n {confusion_matrix(test_classes, bayes_predict)}")


if __name__ == '__main__':
    df = pd.read_csv("./iris.csv")
    iris_classification(df)
    DecisionTree(df)
    KNN(df)
    Naive_Bayes(df)    