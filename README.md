Introduction

Lung cancer remains to be one of the most common forms of cancer with a fairly high mortality rate that increases the later the cancer is found. Early detection and predictive measures in order to find those at risk of lung cancer is imperative to ensure effective treatment and increase patients’ survival rates. In an attempt to address the issue, a predictive machine learning model using a sample dataset from Kaggle is developed using several factors related to each patient in order to predict the outcome of cancer.

Through using the aforementioned dataset, the aim is to make use of machine learning algorithms to find complicated patterns and different relationships in the data which contains several types of variables typically related to the patients.

Several steps must be taken in order to effectively develop the machine learning model, data exploration, preprocessing, feature selection, model selection and training until finally the evaluation and the validation of the machine learning model’s predictive performance.

Research Goal

The goal of this research is to find the correlating relationships between the variables and the target variable and detect patterns to develop a predictive machine learning model that allows to find patients at most risk of lung cancer based on several factors related to the patient’s history. Early detection of cancer is critical to ensure that patients have a higher survival rate and with early diagnosis allows for more varied options for patients to consider in order to treat their illness.

Objectives

The objective of the development of the machine learning model is to possibly detect lung cancer in its early stages or perhaps predict the possibility of its occurrence to thereby provide preventive measures rather than treatments as prevention is a far more effective option in comparison to the latter.

Data Preprocessing

The dataset used for the training and testing of the machine learning model was retrieved from Kaggle, with a 10/10 usability rating indicating that it is a reliable dataset to be used for the model. After searching for possible null values to find missing records in the dataset, there were a total of 0 missing records, therefore the dataset is considered complete.

In order to standardize the variables in the dataset, the column LUNG_CANCER was encoded from object values “Yes” and “No” into “1s” and “0s”, as for the GENDER column, F for females was encoded in “1s” and M for males was encoded into “0s”. As there were only two categorical values for both columns, binary encoding was done instead of one-hot encoding as it is the more appropriate approach to encoding categorical values with only two possible values.

Feature Engineering

	To further improve the model’s performance, feature engineering was performed through introducing a new feature named “Symptoms” which culminates the patient’s different symptoms of lung cancer using the following columns 'YELLOW_FINGERS', 'ANXIETY','CHRONIC DISEASE', 'FATIGUE ', 'WHEEZING','COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN' and another feature added was the “RISK_INDEX” which combines the following columns of 'SMOKING', 'CHRONIC DISEASE', 'ALCOHOL CONSUMING' into a singular column.

Data Splitting

	The dataset was split into the typical standard for developing machine learning models which is 80/20, where 80% of the dataset is used for training and the remaining 20% is used for testing the model. The separation was done carefully in order to ensure that the data is presented adequately. For cases where there was imbalance found, stratifying the dataset was considered.

Models

	In this study, a total of three different models were used to process the dataset which are as follows: Linear Regression model,  Decision Tree model and KNN (K-Nearest Neighbours) model. 

	Linear Regression models are simple for interpretation and understanding as well as known to be easier to train and test the models. But Linear Regression models assume a linear relationship between the variables and are not suitable for complex relationships and are sensitive to possible outliers in the dataset.

	Decision Tree models are able to interpret and understand complex relationships and similarly to Linear Regression models, it is easy to analyze the decisions made by the model. Due to its ability to capture both categorical and numerical variables, it is adaptable as well. But an issue with Decision Tree models is the possibility of overfitting and may not be able to take into consideration unseen data in the dataset.

	KNN (K-Nearest Neighbours) is able to interpret non-linear relationships and is easy to understand as well as implement. One of the drawbacks of the KNN model is its sensitivity to the number of neighbors (K) which may be able to impact the results of the model significantly.

Evaluating the Models

	In order to provide an assessment for these models, the typical usage of metrics such as precision, f-1 score, recall, support and the confusion matrix was used. These metrics allow for an in-depth review of the models and provide a simple interpretation of the results and reliability of the models in training and testing.

Summary

	As indicated in the research goals for this study, the aim is to develop a predictive model to accurately indicate the presence of lung cancer in a patient through the inclusion of symptoms and risk factors related to the patient. The dataset “survey lung cancer” obtained from Kaggle with a total of 309 rows and 16 columns is the foundation of the study. After conducting data preprocessing, a total of three different models were chosen for this study which were K-Nearest Neighbors, Decision Tree and Linear Regression models. Each of these models come with their own advantages and disadvantages, therefore it is imperative to experiment with each one in order to produce the best result.


Conducting the Exploratory Data Analysis


	The dataset comprises 309 rows and 16 columns related to respondents’ risk of lung cancer and their symptoms, as well as whether cancer is present for each respondent.  

	Describing the data presents a mathematical summary of the dataset.

	Using df.info and df.isnull().sum() was used to check the data types in each column and find any null values present in the dataset. The output indicates the presence of zero null values in the dataset.



In order to ensure the model doesn't misinterpret the dataset as having an ordinal relationship as 1 = No and 2 = Yes and it is standard practice to have these values as 0s and 1s, the dataset will be adjusted to address this by reducing the values by 1 (e.g. 2 becomes 1 and 1 becomes 0.)

Converting Gender and Lung Cancer columns into binary through the usage of Binary Encoding as these columns only have two categorical values where M = 0 and F = 1 and NO = 0 and YES = 1

Further EDA


	df['LUNG_CANCER'].value_counts() was used to find the data distribution of patients with lung cancer.


	The above is the age distribution of the patients in the dataset. As the model is for lung cancer prediction which can be correlated with age, it was critical to understand the age distribution in the dataset.


	The histogram is the gender distribution of the patients in the dataset where 0 = Males and 1 = Females which was shown as such due to binary encoding that was done.





Pairplot is used in order to visualize the relationships found in the dataset to understand the correlation relationships between the values using the graphs.



Through pairplot, the shape of the graph and the correlation between the values can be viewed but due to the somewhat complex nature of the dataset, the process of visualizing the dataset through this method took time for computing.




Correlation matrix heatmap was also used as another method to better understand the relationships between each variable where the correlation values indicate the correlation strength between the variables.






Modeling

	Before beginning the process of modeling, the separation of the datasets into two parts was needed. 



	A separation with a distribution of 80% and 20% was done where 80% will be used to train the model and the remaining 20% will be used for testing the model. To ensure the distribution is equal, printing the value counts for both distributions was necessary.

Decision Tree

	This section will focus on the development of the Decision Tree model and the output given by the model.

	
	A max depth of three was used in order to train the model.



	Based on the results of the training, an accuracy of 94% was achieved meaning that of all instances for categorizing cases as class 0 and 1s, the model predicted 94% of the cases correctly. The precision of the model is 84 for class 0 and 95 for class 1, which equates to the model predicting class 0s correctly 84% of the time and predicting class 1s correctly 95% of the time. As for the f1-score, 0.75 and 0.97 is considered a relatively good score, which suggests that the classifier is performing well. 




	Based on the results of the training, an accuracy of 89% was achieved meaning that of all instances for categorizing cases as class 0 and 1s, the model predicted 89% of the cases correctly. The precision of the model is 57 for class 0 and 93 for class 1, which equates to the model predicting class 0s correctly 57% of the time and predicting class 1s correctly 93% of the time. As for the f1-score, 0.53 and 0.94, whilst the f1-score for the class 1 is exceptional, the f1-score for class 0 is a cause for concern.

Hyperparameter Tuning

In order to conduct Hyperparameter Tuning, a grid search was done for the parameters max depth, minimum sample split, minimum samples leaf, max features and criterion.



	But after the hyperparameter tuning process, it was found that the accuracy of the model went down. Therefore the previous result is deemed more favourable over the results of the hyperparameter tuning.




K-Nearest Neighbour



	For K-Nearest Neighbour, the initial training set the K (number of neighbors) to 5.



	Based on the results of the training, an accuracy of 91% was achieved meaning that of all instances for categorizing cases as class 0 and 1s, the model predicted 91% of the cases correctly. The precision of the model is 80 for class 0 and 92 for class 1, which equates to the model predicting class 0s correctly 80% of the time and predicting class 1s correctly 92% of the time. As for the f1-score, 0.52 and 0.95 is once again a cause for concern as 0.52 is still middling.



Based on the results of the training, an accuracy of 94% was achieved meaning that of all instances for categorizing cases as class 0 and 1s, the model predicted 94% of the cases correctly. The precision of the model is 100 for class 0 and 93 for class 1, which equates to the model predicting class 0s correctly 100% of the time and predicting class 1s correctly 93% of the time. As for the f1-score, 0.67 and 0.96 is considered to be a relatively good score.

Hyperparameter Tuning

	

	In order to conduct the Hyperparameter Tuning, a grid search for the parameters metric and K (number of neighbors) was conducted. The result was to set the K at 7 and the metric at manhattan.

	

	The process resulted in an accuracy rating that was lower before the hyperparameter tuning was done, resulting in an output that is less favorable than the previous result.


	After a deep analysis on the results of the models, it was with clarity that hyperparameter tuning will not always result in an improvement towards the performance of the predictive models. It is observed that simply through binary encoding and standard inputs for the development of the predictive models may result in better performance. 

	Overall, it can be seen that while both models KNN (K-Nearest Neighbors) and Decision Tree have the same accuracy, in the context of the predictive model’s objective as a lung cancer predictive model, KNN’s 100% accuracy of predicting true negatives highlights an overall stronger result in comparison to the Decision Tree model’s more balanced result.


Decision Tree
N/A Best Score in Related Work
89% Best Score from Experiment

K-Nearest Neighbor
78% Best Score in Related Work
94% Best Score from Experiment

Logistic Regression
85% Best Score in Related Work
N/A Best Score from Experiment

Random Forest
88% Best Score in Related Work
N/A Best Score from Experiment

SVM
98% Best Score in Related Work
N/A Best Score from Experiment

Gradient Boosting
88% Best Score in Related Work
N/A Best Score from Experiment

LGBM
90% Best Score in Related Work
N/A Best Score from Experiment




In conclusion, the aim of this study was to develop a machine learning model for predicting the risk of lung cancer to be found in a patient based on the symptoms and risk factors surrounding the patient. After analyzing the results of the model, there were several noticeable key points to focus on.

	It has been made clear that certain attempts to improve the model’s performance, such as hyperparameter tuning and advanced pre-processing methods may not always produce the desired result, even going as far as to be a detriment to the model’s performance. 

	Though there were related works with higher accuracy rates in comparison to the study’s best result which was 94% compared to another work’s accuracy of 98%, the usage of K-Nearest Neighbors to reach that accuracy and surpass other work is a noteworthy achievement. 

	In addition, the aspects of data pre-processing showcased the importance of feature scaling, feature engineering, hyperparameter tuning and standardization during the process of developing the model. Despite the tediousness and time-consuming nature of such work, it presents itself as a critical aspect of improving and creating a model with a strong foundation. 

	Conclusively, this study has been invaluable in providing deep insights into machine learning and the nuances of processing data as well as tuning parameters in order to achieve desired results when possible, whilst also providing and understanding of the inner workings for such projects. 


Bhat, M. S. (2021) Lung Cancer. https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer?rvi=1  (Accessed: 15 November 2023). 


Gupta, H.  (2022)  Lung Cancer Prediction. https://www.kaggle.com/code/casper6290/lung-cancer-prediction-98 (Accessed: 15 November 2023). 


Bryant, M. (2021) Lung Cancer Classification. https://www.kaggle.com/code/michaelbryantds/lung-cancer-classification (Accessed: 15 November 2023). 

Nageswaran, S. et al. (2022) Lung Cancer Classification and prediction using machine learning and image processing, BioMed research international. Available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9424001/  (Accessed: 17 November 2023).
