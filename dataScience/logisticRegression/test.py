""" ########################

LOGISTIC REGRESSION USING SKLEARN

    Classify products into two categories based on the sales forecast accuracy.

    Questions:
    To which products is have to focus on when updating the sales forecast?
    On which product I should double check the sales forecast?

    Steps for logistic regression
    1. Create x and y
    2. Create Train and Test set
    3. Train the model
    4. Make prediction
    5. Evaluate the model

########################### """

from turtle import color, position
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sn

data = pd.DataFrame(pd.read_csv("/Users/Joni/Desktop/Python/GitHub/portfolio/dataScience/logisticRegression/logReg_testData.csv", sep="\;", decimal=",", engine="python"))

# Cleaning the dataset
# Check null values/missing values
data.isnull().sum()

# Calculate metrics for evaluate planning accuracy
# MAD (Mean Absolute Deviation)
# MAPE (Mean Absolute Percent Error)
data["MAD"] = np.absolute(data["salesActual"]-data["salesForecast"])
data["MAPE"] = np.absolute(data["MAD"]/data["salesActual"]*100)

# Select the product (A, B, C, D, E)
product = input("Type in the product. Select from: A, B, C, D or E ")
data = data[data["product"] == product]
print(data)

# Feature variables (independent variable)
x = data[["salesForecast"]].values


# Target variable (dependent variable)
y = data["isAccepted"].values

# Create train and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,
                                                            test_size=0.3,
                                                            random_state=100)

# Instantiate the model
logreg = LogisticRegression()

# Fit the model with data
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

# Evaluate performance of the model
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("Model accuracy:", logreg.score(x_test, y_test))

fig = plt.figure(figsize=(8,6))
specs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

ax1 = fig.add_subplot(specs[0,0])
ax2 = fig.add_subplot(specs[0,1])
ax3 = fig.add_subplot(specs[1,0])
ax4 = fig.add_subplot(specs[1,1])

ax1.matshow(conf_matrix, cmap="Blues")
ax1.set_xlabel("True labels")
ax1.set_ylabel("Predicted labels")
ax1.set_xticks([0,1],["True", "False"])
ax1.set_yticks([0,1],["True", "False"])
ax1.xaxis.set_ticks_position("bottom")

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax1.text(x=j, y=i, s=conf_matrix[i,j], va="center", ha="center")




plt.show()
