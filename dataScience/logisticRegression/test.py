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

from turtle import color
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sn

data = pd.DataFrame(pd.read_csv("/Users/Joni/Desktop/Python/GitHub/Portfolio/dataScience/logisticRegression/logReg_testData.csv", sep="\;", decimal=",", engine="python"))

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

# Visualise the results
fig = plt.figure(1, figsize = (12, 6))
specs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
plt.subplots_adjust(left=0.03, bottom=0.135, right=0.98, wspace=0.93, hspace=0.45)

ax1 = fig.add_subplot(specs[0, 0]) # First row, first slot
ax1 = sn.heatmap(conf_matrix, annot=True, cmap="Blues")
ax1.set_ylabel("Actual label")
ax1.set_xlabel("Predicted label")
sampleTitle = "Accuracy Score (test set): {0}".format(logreg.score(x_test, y_test))
ax1.set_title(sampleTitle, size=15)

ax2 = fig.add_subplot(specs[0, 1]) # First row, second slot

# Remove borders
#ax2.spines["top"].set_visible(False)
#ax2.spines("right").set_visible(False)
#ax2.spines("left").set_visible(False)
#ax2.axis("off")


# Add gridlines and colors
ax2.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
train_colors = ["#4786D1" if target <= 0 else "#F28627" for target in y_train]
pred_colors = ["#4786D1" if target <= 0 else "#F28627" for target in y_pred]

# Scatter plot
y_train_len = len(y_train) 
y_pred_len = len(y_pred)
ax2 = plt.scatter(np.arange(0, y_train_len), y_train, color=train_colors, marker="o", s=[15*y_train_len], edgecolors="Black", linewidth=0.5)
ax2 = plt.scatter(np.arange(0, y_pred_len), y_pred, color=pred_colors, marker="^", s=[15*y_pred_len], edgecolors="Black", linewidth=0.5)




# Customize the legend





ax3 = fig.add_subplot(specs[1, 0]) # Second row, first slot
ax3 = plt.scatter(y_test, y_pred)
ax3.set

ax4 = fig.add_subplot(specs[1, 1]) # Second row, second slot




plt.show()

"""
    fig, ax = plt.subplots(figsize=(15, 10))
    # removing all borders except bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # adding major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    training_colors = ['#4786D1' if target <= 0 else '' for target in train_targets]
    prediction_colors = ['#4786D1' if target <= 0 else '#F28627' for target in predictions]
    train_set_len = len(train_targets)
    predictions_len = len(predictions)
    plt.scatter(np.arange(0, train_set_len), train_targets, color=training_colors, marker='o', s=[5 * train_set_len])
    plt.scatter(np.arange(0, predictions_len), predictions, color=prediction_colors, marker='^', s=[20 * predictions_len])
    ax.set_xlabel('Observation')
    ax.set_ylabel('Target value (sales forecast)')

    # Customizing symbols in the legend
    legend_items = [Line2D([0], [0], color='#4786D1', markersize=10), 
        Line2D([0], [0], color='#F28627', markersize=10),
        Line2D([0], [0], color='w', marker='o', markerfacecolor='#979797', markeredgecolor='#979797', markersize=10),
        Line2D([0], [0], color='w', marker='^', markerfacecolor='#979797', markeredgecolor='#979797', markersize=10)]
    # Adding some spacing between each legend row and padding
    ax.legend(handles=legend_items,
    labels=['Class 0: Not accepted', 'Class 1: Accepted', 'Training set', 'Predictions'],labelspacing=1.5, borderpad=1)

"""