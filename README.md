Group ID: 1107

# Regression-Analysis-Python
A regression analysis of the different factors influencing performance of students during their exams. 

## Table of contents
* [Introduction](#Introduction)
* [General-Content](#General-Content)
* [Repository](#Repository)
* [Installations](#Installations)
* [SetUp](#SetUp)
  
## Introduction
This is a mandatory group project  of the courses “FS19-6,781,1.00 Skills: Programming: Introduction Level“ and “FS19-8,789,1.00 Programming with Advanced Computer Languages” supervised by Dr. Mario Silic in the spring semester, 2019, at the University of St. Gallen. The project was done in a group of five people: Maï Billharz, Lea Buffenoir, Julia Hartweg, Muriel Strasser and Yi Zheng. 

## General-Content
This is a regression model using a downloaded dataset from kaggle (https://www.kaggle.com/spscientist/students-performance-in-exams) related to students performance in exams. First, we created simple scatter plots comparing two pieces of data at a time. Second, we performed a simple linear regression analysis with a training and a test set. Third, we performed a multiple regression using selected data. The reason for only using selected data is, that using more or different data tended to decrease model quality. Last, we created a prediction tool based on a different multiple linear regression.

## Repository
There are five files distributed within this repository. "README.md" is used for a project description as well as instructions. "Regression Analysis - Group Project Python.ipynb" and "Regression Analysis - Group Project Python.py" contain code. The file "StudentsPerformance.csv" is a dataset downloaded from kaggle (https://www.kaggle.com/spscientist/students-performance-in-exams), which this project is based on. The file "Prediction Tool Output - Screenshots.docx" shows the output of the prediction tool, which pops up in a new window. It predicts the math score using user input.

## Installations 
To do our analysis we use these programs: 
* Python 3.7.3
* Anaconda 3
* Jupyter Notebook

To run our code you will need these packages and datasets: 
* Packages: numpy, pandas, matplotlib.pyplot, pylab, sklearn, statsmodels.api, tkinter 
* Dataset from kaggle (https://www.kaggle.com/spscientist/students-performance-in-exams) 

## SetUp 
To run the code you have to save the kaggle dataset as CSV on your computer. 
