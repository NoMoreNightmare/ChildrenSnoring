from sklearn.svm import OneClassSVM
from test.test1 import model_create


def attempt():
    X_train, x_test, Y_train, y_test = model_create("property.cfg", "split_audio")
    clf = OneClassSVM(gamma='auto').fit(X_train)
    print(clf.predict(x_test))


if __name__ == "__main__":
    attempt()
