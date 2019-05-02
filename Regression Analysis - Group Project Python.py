##Regression Analysis - HSG Project Python##


##About##

#This is a regression model using a downloaded dataset from kaggle (https://www.kaggle.com/spscientist/students-performance-in-exams) related to students performance in exams. First, we created simple scatter plots comparing two pieces of data at a time. Second, we performaned a simple linear regression analysis with a training and a test set. Third, we performed a multiple regression using selected data. The reason for only using selected data is, that using more or different data tended to decrease model quality. Last, we created a prediction tool based on a different multiple linear regression.


#import necessary packages
#Not all packages are imported right away but will be added once they are needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().magic(u'matplotlib inline')


#import dataset and read data
#StudentsPerformance.csv is imported and displayed.
df = pd.read_csv(r"C:\Users\yizhe\Anaconda3\Library\StudentsPerformance.csv")
df.head()



##Basic Data Exploration##

# summarize the data; see statistical details of the dataset
df.describe()

# Check number of rows and columns
df.shape

# check the column names
df.columns

#changing spaces in column heading to underscore
df.columns = ["gender","race/ethnicity","parental_level_of_education","lunch","test_preparation_course","math_score","reading_score","writing_score"]

# New dataframe contains selected columns from old dataframe
cdf=df[['gender','test_preparation_course','math_score','reading_score','writing_score']]
cdf.head()

# creatre data frame VIZ with selected columns from existing cdf dataframe
viz= cdf[['gender','test_preparation_course','math_score','reading_score','writing_score']]
viz.hist

plt.show()



##Scatter Plots##

#writing score in relation to math score
plt.scatter(cdf.math_score, cdf.writing_score,  color='red')
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()

#Math score in relation to reading score
plt.scatter(cdf.math_score,cdf.reading_score, color ='blue')
plt.xlabel("math_score")
plt.ylabel("reading_score")
plt.show()

#Reading score in relation to Writing score
plt.scatter(cdf.writing_score,cdf.reading_score, color ='green')
plt.xlabel("writing_score")
plt.ylabel("reading_score")
plt.show()

#Gender in relation to math score
plt.scatter(cdf.gender,cdf.math_score, color ='pink')
plt.xlabel("gender")
plt.ylabel("math_score")
plt.show()

#Math score in relation to test preparation
plt.scatter(cdf.test_preparation_course,cdf.math_score, color ='yellow')
plt.xlabel("test_preparation_course")
plt.ylabel("math_score")
plt.show()



##Simple Linear Regression Model##

#Splitting data into a train and test dataset, a simple linear regression model minimize the residual sum of squares between the independent x, in this case math score, in the dataset, and the dependent y, in this case writing score, by the linear approximation.
#create train and test dataset
msk = np.random.rand(len(df)) < 0.70
train = cdf[msk]
test = cdf[~msk]

#Train data distribution
plt.scatter(train.math_score,train.writing_score, color = 'red')
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()

#Modeling
#Using sklearn package to model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['math_score']])
train_y = np.asanyarray(train[['writing_score']])
regr.fit(train_x,train_y)

# The coefficients
print ('Coefficients : ',regr.coef_)
print ('Intercept : ',regr.intercept_)

#Plot Outputs
train_y_ = regr.predict(train_x)
plt.scatter(train.math_score,train.writing_score, color ='red')
plt.plot(train_x, train_y_, color='black', linewidth =3)
plt.xlabel("math_score")
plt.ylabel("writing_score")

#Evaluation
# Evaluate the model with the Test data
test_x = np.asanyarray(test[['math_score']])
test_y = np.asanyarray(test[['writing_score']])
test_y_ = regr.predict(test_x)

print ("Residual Sum of Squares : %.2f"
      % np.mean((test_y_ - test_y)**2))

# Explained variance score: 1 is perfect prediction
print('Variance Score : %.2f' % regr.score(test_x,test_y))

#Plot Outputs of evaluation
plt.scatter(test_x,test_y, color ='red')
plt.plot(test_x, test_y_, color ='black', linewidth =3)
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()



##Multiple Linear Regression Model##
#Example of a multiple linear regression model
#Using entire dataset to create the multiple regression model

# converting "none" and "completed" in test_preparation_course to 0 and 1 as dummy variables
# converting "female" and "male" in gender to 0 and 1 as dummy variables

df["test_preparation_completed"] = df.test_preparation_course.map({"none":0.0, "completed":1.0})
df["gender_male"] = df.gender.map({"female":0.0, "male":1.0})

df.head()

#Using statsmodel and rating dummy variables as categorical variables, a multiple linear regression analysis is conducted.
import statsmodels.api as sm
from statsmodels.formula.api import ols
fit = ols ("math_score ~ C(test_preparation_completed) + reading_score + writing_score + C(gender_male)",data=df).fit()
fit.summary()



##Prediction Tool##
#Based on a multiple linear regression, we created a tool thats allows prediction of the math score based variable inputs of the gender, writing score and reading score.
#Note: a seperate window might open to display the tool.

from pandas import DataFrame
import tkinter as tk 

X = df[["gender_male","writing_score","reading_score"]]
Y = df["math_score"]
model = sm.OLS(Y, X).fit ()

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

# tkinter GUI
root= tk.Tk() 

canvas1 = tk.Canvas(root, width = 1200, height = 450)
canvas1.pack()

# with statsmodels
print_model = model.summary()
label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='LightSkyBlue1')
canvas1.create_window(800, 220, window=label_model)

# New_Gender label and input box
label1 = tk.Label(root, text='Type gender(0 for female, 1 for male): ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Writing_score label and input box
label2 = tk.Label(root, text=' Type writing score: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

# New_reading_score label and input box
label3 = tk.Label(root, text=' Type reading score: ')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry (root) # create 3nd entry box
canvas1.create_window(270, 140, window=entry3)


def values(): 
    global New_Gender #our 1st input variable
    New_Gender = float(entry1.get()) 
    
    global New_Writing_score #our 2nd input variable
    New_Writing_score = float(entry2.get()) 
    
    global New_Reading_score #our 3rd input variable
    New_Reading_score = float(entry3.get())
    
    Prediction_result  = ('Predicted math score: ', regr.predict([[New_Gender ,New_Reading_score, New_Writing_score]]))
    
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict Math score',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 180, window=button1)
 

root.mainloop()

