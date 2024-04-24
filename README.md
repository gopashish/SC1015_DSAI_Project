# SC1015_DSAI_Project

Project Title: Stroke Prediction Using Machine Learning 
Objective 
This project is designed to predict the likelihood of stroke incidents by analyzing medical data. 
It serves as a vital analytical tool for healthcare providers, helping them identify high-risk patients. By utilizing predictive analytics, this project aids in making informed decisions about patient care management, preventive measures, and resource allocation in healthcare settings. 


Data Extraction 
The data for this project was sourced from a reputable medical dataset, likely containing patient demographics, health records, and stroke occurrence. The dataset was partitioned into 'train' and 'test' sets to train the predictive model and assess its performance. 
Data Cleaning 
The data cleaning phase addressed inconsistencies and missing values in critical columns such as 'smoking_status', 'bmi', and 'glucose_level'. These were rectified by inputting appropriate values or implementing imputation techniques to preserve data integrity. We also transformed complex columns with nested information into a workable format for the subsequent analysis. 


Data Preparation 
During the data preparation stage, we undertook several steps to transform the data. This included encoding categorical variables, normalizing numerical features, and engineering new attributes that may influence the prediction of stroke events. Techniques like one-hot encoding and feature scaling were employed to ensure the data was suitable for machine learning algorithms. 


Data Exploration 
In the exploratory phase, we examined the relationships between different variables and stroke risk. Visual tools such as histograms, box plots, and correlation matrices were utilized to discern patterns and identify significant predictive factors. This stage was instrumental in shaping the feature selection process. 
Machine Learning 
 
Machine Learning Models 
Our project harnesses a suite of diverse and powerful machine learning algorithms to predict stroke likelihood with increased accuracy and reliability. Each model brings its own strengths and is evaluated to determine the best performer in our specific context. 

Decision Tree Classifier 
The Decision Tree is a non-linear model that operates by splitting the data into branches at decision nodes, which are based on the feature values. It is akin to a flowchart, leading to different outcomes for different paths taken. This model is intuitive and easy to interpret, making it an excellent tool for preliminary insights. 

Random Forest Classifier 
Random Forest is an ensemble learning method that operates by constructing many decision trees at training time and outputting the mode of classes for classification tasks. It is known for its high accuracy, robustness to overfitting, and ability to handle imbalanced datasets by virtue of its ensemble nature. 

Logistic  Regression 
In our project, Logistic Regression is adapted for a classification task by thresholding the predicted continuous output. It is typically used for estimating real values based on continuous variable(s), relying on the assumption that there is a linear relationship between the input variables and the single output variable. 

Naive Bayes Classifier 
The Naive Bayes Classifier applies Bayes’ theorem with the “naive” assumption of independence between every pair of features. Despite its simplicity, this model can achieve highly competitive results, particularly in cases where the assumption of feature independence holds true. 

XGBoost Classifier 
XGBoost stands for eXtreme Gradient Boosting and is an optimized distributed gradient boosting library. It provides a highly efficient implementation of the gradient boosting framework. This model is known for its speed and performance, especially in structured data where feature relationships are complex and nonlinear. 
K-Nearest Neighbors Classifier


The K-Nearest Neighbors (KNN) Classifier is a simple, yet effective, algorithm that classifies new data points based on the majority vote of its 'k' nearest neighbors. 
It is a type of instance-based learning where the function is only approximated locally, and all computation is deferred until function evaluation. 
 
 
 
 
Usage 
Before diving into the notebook, ensure that all necessary libraries and dependencies are installed as listed in the initial setup cells.
Commence by loading the dataset, proceed through the notebook for data processing, exploratory analysis, model training, and evaluation, and culminate by deploying the model to predict stroke risk. 
 
 
References - 
 
 
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
https://keras.io/ 
https://plotly.com/python/ 
https://scikit-learn.org/stable/modules/svm.html 
https://towardsdatascience.com/mac 
hine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761 
https://scikit-learn.org/stable/modules/naive_bayes.html 
https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/ 
https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/ 
https://machinelearningmastery.com/rfe-feature-selection-in-python/ 
 
 
Authors 
 
Harineesh Reddy 
Gautham Krishna 
Gopashish Harikrishnan
