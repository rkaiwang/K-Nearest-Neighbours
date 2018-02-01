# Each row represents a different flower
# Sepal length, sepal width, petal length, petal width
# Given the variables above, we want to predict the species of flower
# Three species of iris flower - versicolor, virginica, setosa
# We are going to build a KNN-classifier to use on the data

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



plt.style.use('ggplot')
knn = KNeighborsClassifier(n_neighbors=6)

iris = datasets.load_iris()

#print(type(iris)) # the type is a Bunch
#print(iris.keys()) # keys are data, target, target_names, DESCR, feature_names
#print(type(iris.data)) # the data is in a numpy array
#print(iris.data.shape) # the shape is 150 rows and 4 columns

X = iris.data
y = iris.target # usually we set up the target variable as y, similar to how the dependent variable is always on the y-axis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
# test_size determines which proportion of the data is to be set as test data.
# random_state seeds a seed for the RNG that determines test/train split.
# Using the same number lets you recreate the same RNG so you can get the same results
# stratify keeps the data distributed in the train and test sets in the same way. y is the list/array containing the data labels

#we can build a dataframe out of the feature data (things like sepal length, sepal width)
df = pd.DataFrame(X, columns=iris.feature_names) #for DataFrame, the capital letters are important
#print(df.head()) #this way we can view the first few rows of our data

pd.plotting.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker='D') #we have set our target variable y as an argument to the parameter c which stands for colour.
#plt.show() #renders the scatter matrix

knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=6, p=2, weights='uniform')



# Remember that the scikit-learn API requires that the data is a numpy array or pandas dataframe.
# The features (dependent variables) must also be continuous
# Also requires that there are no missing data entries
# It is required that each feature is in an array where each column is a feature
# And each row must be a different observation/data point
# The target must have the same number of rows as the feature data

# An example of how to print predictions
new_prediction = knn.predict(X_test)
#print("Prediction: {}".format(new_prediction))
# Prints the predictions of the test set


test_score = knn.score(X_test, y_test)
print(test_score)

