# House-Price-Prediction-Model
A supervised machine learning regression model designed to predict residential property values by analyzing core structural features, including total lot area, room configurations, furnishing status, parking, and essential home amenities.
# House Price Prediction Model

A supervised machine learning regression pipeline designed to predict residential property prices based on key housing attributes, structural dimensions, room configurations, and amenities.

# House Price Prediction Project

This project predicts residential property prices using supervised machine learning regression techniques.

## Project Summary
This repository contains an end-to-end machine learning workflow that analyzes residential housing data and estimates market values based on property characteristics.

## Dataset Overview
The dataset Housing.csv contains five hundred forty-five property records evaluated across thirteen distinct structural attributes. The target variable is the continuous selling price of the property. Input features include total lot area, bedroom count, bathroom count, number of stories, and dedicated parking spaces. Categorical indicators capture access to a main road, presence of a guest room, existence of a basement, hot water heating, air conditioning, preferred neighborhood location, and overall furnishing status.

## Step-by-Step Workflow
First, the program imports the Housing.csv dataset using the pandas data analysis library.

Second, the script identifies all categorical text columns and converts them into numeric dummy variables using one-hot encoding.

Third, the dataset is split into independent input features and the dependent target column representing house prices.

Fourth, the data is partitioned into an eighty percent training subset and a twenty percent testing subset to allow unbiased evaluation.

Fifth, a standard scaler normalizes all feature values to ensure uniform scale across measurements such as square footage and room counts.

Sixth, a linear regression algorithm fits the model by learning the statistical relationships between the standardized housing attributes and property sale prices.

Seventh, the fitted regression model generates predicted market values for all records within the hold-out test set.

Eighth, model accuracy is evaluated against true sale prices using R-squared score, Mean Absolute Error, and Root Mean Squared Error metrics.

## Installation and Execution
First, ensure that Python version 3.8 or higher is installed on your machine.

Second, install the required packages by running pip install pandas scikit-learn.

Third, verify that Housing.csv is located within the project working directory.

Fourth, execute the program by running python main.py in your terminal or command line interface.

## Model Evaluation Results
The linear regression model achieves an R-squared score of approximately point six five two nine on the test dataset. The Mean Absolute Error indicates an average price deviation of roughly nine hundred seventy thousand units. The Root Mean Squared Error sits at approximately one million three hundred twenty-four thousand units.

## Project File Layout
The Housing.csv file stores the raw tabular property dataset. The main.py script executes the full data preprocessing, training, and evaluation pipeline. The README.md file provides detailed documentation explaining the project structure and execution steps.
