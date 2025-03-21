# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets , linear_model
# from sklearn.metrics import mean_squared_error
# diabetes = datasets.load_diabetes()
# '''# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.target)'''
# diabetes_x = diabetes.data

# diabetes_x_train = diabetes_x[:-30]
# diabetes_x_test = diabetes_x[-30:]

# diabetes_y_train = diabetes.target[:-30]
# diabetes_y_test = diabetes.target[-30:]

# model = linear_model.LinearRegression()

# model.fit(diabetes_x_train,diabetes_y_train)
# diabetes_y_pridected = model.predict(diabetes_x_test)
# print("\nmean squared error is :-  " ,mean_squared_error(diabetes_y_test,diabetes_y_pridected))
# print("\nweights: " , model.coef_ )
# print ("\nintersept: ", model.intercept_)
# # plt.scatter(diabetes_x_test , diabetes_y_test)
# # plt.show()    use to plot graph 

# # mean squared error is :-   3035.0601152912686

# # weights:  [941.43097333]

# # intersept:  153.39713623331644
# load the iris dataset as an example
from sklearn.datasets import load_iris
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# printing the shapes of the new X objects
print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)

# printing the shapes of the new y objects
print("Y_train Shape:", y_train.shape)
print("Y_test Shape: ",y_test.shape)
