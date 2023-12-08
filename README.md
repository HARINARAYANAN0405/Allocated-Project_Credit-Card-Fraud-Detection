# Credit Card Fraud Detection

This project is dedicated to developing a robust credit card fraud detection system using advanced machine learning techniques. The primary goal is to effectively differentiate between legitimate and fraudulent transactions, considering various transaction details such as amount, location, and timestamp.

The dataset employed for this project is derived from Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data. It encompasses details such as:

Transactions flagged as fraudulent or not (Fraud column)
Transaction features including amount, location, and timestamp.
Transaction-specific information such as merchant category, transaction type, and currency.
Customer-related details such as account age, credit limit, and spending behavior.
Please note that the features in the credit card fraud detection dataset are distinct from those in the Telco Customer Churn dataset, reflecting the specific context of fraudulent transaction detection.

## Methodology
The dataset is strategically divided into training and testing sets to ensure a comprehensive evaluation of the model's performance.

## Data Cleaning
Thorough data cleaning procedures are implemented, including handling missing values and addressing any outliers or anomalies present in the dataset. The focus is on preparing a pristine and reliable dataset for subsequent analysis.

## Exploratory Data Analysis
In-depth exploratory data analysis techniques are employed to gain valuable insights into the distribution of fraud and non-fraud cases. Visualization tools are utilized to unravel patterns, correlations, and anomalies in key features, contributing to a deeper understanding of the data.

## Feature Engineering
To enhance model performance, feature engineering strategies are implemented. This may involve creating new features, transforming existing ones, or extracting meaningful information to enrich the dataset.

## Feature Scaling
Certain machine learning models necessitate feature scaling for optimal performance. Techniques such as scaling and normalization are applied to ensure consistency and effectiveness in the modeling process.

## Data Imbalance
To mitigate potential class imbalance, the SMOTE (Synthetic Minority Oversampling Technique) library was employed. This technique synthetically increased the representation of the minority class ('fraudulent transactions').

## Preprocessing Function
A dedicated Python function, credit_card_prep(dataframe), was crafted to amalgamate and execute all preceding preprocessing steps on the test data. This function adeptly handles missing values by imputing them with the mean value derived from the training set.

## Models Training
State-of-the-art machine learning models are employed for classification tasks. Rigorous training and evaluation processes are carried out to select the most effective model, considering factors such as accuracy, precision, recall, and F1-score.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assuming 'X_train' is the feature matrix and 'y_train' is the target variable

# Instantiate the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))

Repeat similar code structure for other models mentioned in the project.
