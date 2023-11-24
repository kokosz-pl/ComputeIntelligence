import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model


def activation_func(x):
    return 1 / (1 + np.exp(-x))


def voleyball_NN():

    input_vec = np.array([[23, 25, 28, 22, 46, 50, 48],
                          [75, 67, 120, 65, 70, 68, 97],
                          [176, 180, 175, 165, 187, 180, 178]])

    l1_weight = np.array([[-0.46122, 0.78548],
                          [0.97314, 2.10584],
                          [-0.39203, -0.57847]])

    l2_weight = np.array([-0.81546, 1.03775])

    b1 = np.array([0.80109, 0.43529])
    b2 = -0.2368

    input_layer = input_vec[:, 0].reshape(3, 1) * l1_weight
    hidden_layer1 = np.array(
        [input_layer[:, 0].sum(), input_layer[:, 1].sum()]) + b1
    hidden_layer1_activ = activation_func(hidden_layer1)
    hidden_layer2 = (hidden_layer1_activ * l2_weight).sum() + b2
    hidden_layer2_activ = activation_func(hidden_layer2)
    output = hidden_layer2_activ
    print(f"\nNeural network result: {output * 100} %\n")


def IrisClassification():
    df = pd.read_csv("./iris.csv")
    df_norm = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    target = df[['variety']].replace(
        ['Setosa', 'Versicolor', 'Virginica'], [0, 1, 2])
    df = pd.concat([df_norm, target], axis=1)

    train, test = train_test_split(df, train_size=0.7)

    train_data = train[['sepal.length',
                        'sepal.width', 'petal.length', 'petal.width']]

    test_data = test[['sepal.length', 'sepal.width',
                      'petal.length', 'petal.width']]

    train_labels = train.variety  # output of our training data
    test_labels = test.variety   # output value of test data

    clf_2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                          hidden_layer_sizes=(2), random_state=1)
    clf_2.fit(train_data, train_labels)
    prediction_2 = clf_2.predict(test_data)
    print(
        f"The accuracy of the One-Layer model with 2 neurons is: {accuracy_score(prediction_2, test_labels) * 100} %")

    clf_3 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                          hidden_layer_sizes=(3), random_state=1, max_iter=1000)
    clf_3.fit(train_data, train_labels)
    prediction_3 = clf_3.predict(test_data)
    print(
        f"The accuracy of the One-Layer model with 3 neurons is: {accuracy_score(prediction_3, test_labels) * 100} %")

    clf_3_3 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(3, 3), random_state=1, max_iter=1000)
    clf_3_3.fit(train_data, train_labels)
    prediction_3_3 = clf_3_3.predict(test_data)
    print(
        f"The accuracy of the Two-Layer model with 3 neurons in each layer is: {accuracy_score(prediction_3_3, test_labels) * 100} %\n")


def data_split():
    df = pd.read_csv('./diabetes.csv')

    df['class'] = df[['class']].replace(
        ['tested_negative', 'tested_positive'], [0, 1])

    target_column = ['class']
    predictors = list(set(list(df.columns)) - set(target_column))
    df[predictors] = df[predictors] / df[predictors].max()

    x = df[predictors].values
    y = df[target_column].values
    y = np.ravel(y)

    return train_test_split(x, y, train_size=0.7, random_state=40)


def DiabetesClassification():
    train_data, test_data, train_label, test_label = data_split()

    mlp = MLPClassifier(hidden_layer_sizes=(
        6, 3), activation='relu', solver='adam', max_iter=500)
    mlp.fit(train_data, train_label)

    predict_test = mlp.predict(test_data)
    print("\nDiabetes Classification:")
    print(f"Model Accuracy: {accuracy_score(predict_test, test_label) * 100} %")
    print(f"Confusion Matrix:\n{confusion_matrix(test_label, predict_test)}")

def DiabetesKeras():
    train_data, test_data, train_label, test_label = data_split()

    model = Sequential()
    model.add(Dense(6, input_shape=(8,), activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=500, batch_size=10, verbose=0)
    _, accuracy = model.evaluate(test_data, test_label)
    print(f"Accuracy: {(accuracy * 100):.2f}")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig("./Adamax_relu_sigmoid.png")

    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



if __name__ == '__main__':
    voleyball_NN()
    IrisClassification()
    DiabetesClassification()
    # DiabetesKeras()
