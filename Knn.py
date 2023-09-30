import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class Knn:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X):
        # X -> is a list of test data points that we want to predict
        # x -> is one of the elements of X
        # y_pred -> we are going to return this as the list of predictions

        # Step 1: set up a loop for each of the test data points and predict each of their labels individually
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        # Step 2: compute the distance between the test data point and the training data points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # in the above statement, to get the distances I have use euclidean distance, however, we can use any other distance metric such as manhattan distance, etc.
        
        # Step 3: sort the distances and get the indices of the first k neighbors, we can use argsort() to get the indices
        closest_k_neighbors = np.argsort(distances)[:self.k]

        # Step 3: get the respective labels of the selected closest k neightbors
        closest_k_labels = [self.y_train[i] for i in closest_k_neighbors]

        # Step 4: get the most common class label out the closest k labels
        predicted_label = Counter(closest_k_labels).most_common(1)

        # Step 5: return the predicted label, the Counter module returns the tuple, in the form of [(label, count)]
        return predicted_label[0][0]
    
# pulling the data from the sklearn datasets to test the KNN model
data = datasets.load_iris()
X, y = data.data, data.target
print(X.shape, y.shape)

# splitting the data into train and test sets, with the ratio of 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
# The random_state parameter is to ensure that we get the same split everytime we run the code, this helps with test consistency

# Now, it's time to test the model and see how it hold
knn_model = Knn(k=3)
knn_model.fit(X_train, y_train)
predictions = knn_model.predict(X_test)

# Now, let's test the accuracy of the model
acc = np.sum(predictions == y_test) / len(y_test)
print("The accuracy of Knn from scratch model accuracy: ", acc)


# This seems to work well, but it is not often that we need to write the model from scractch, so let's see how the sklearn KNN model works
sklearn_knn_model = KNeighborsClassifier(n_neighbors=3)
sklearn_knn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_knn_model.predict(X_test)

# Now, let's test the accuracy of the sklearn model
sklearn_acc = accuracy_score(y_test, sklearn_predictions)
print("The accuracy of Knn from sklearn model accuracy: ", sklearn_acc)

# As we can see, the accuracy of the sklearn model is the same as the model that we wrote from scratch, which is a good sign that our model is working well