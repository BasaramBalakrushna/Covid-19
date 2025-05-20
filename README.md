# Covid-19
COVID-19 Patient Classification Using Machine Learning
-------------------------------------------------------
This project applies machine learning techniques to classify patients as COVID-positive or COVID-negative based on clinical data. The classification is based on the CLASIFFICATION_FINAL column, which originally contains values from 1 to 7. We engineer this target feature into a binary classification problem to support effective predictive modeling.


**Project Workflow:**

1. Library Imports
Essential libraries such as pandas, NumPy, scikit-learn, and matplotlib are imported for data manipulation, machine learning, and visualization.

2. Data Preprocessing
Loaded the dataset and inspected its structure.

Handled missing values and ensured proper data types.

3. Feature Engineering
The CLASIFFICATION_FINAL column contains class labels from 1 to 7.

Values [1, 2, 3] indicate COVID-positive cases.

These were mapped to 1, and all other values to 0 using the .apply(...) method for binary classification.
df["CLASIFFICATION_FINAL"] = df["CLASIFFICATION_FINAL"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
4. Feature Selection
Separated the feature variables (X) and target variable (y).

5. Train-Test Split
Split the dataset into training and testing sets using an 80/20 ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
6. Feature Scaling
Standardized the feature set using StandardScaler to normalize the input data.

7. Modeling with Decision Tree
Trained a DecisionTreeClassifier and evaluated its performance using:

Confusion Matrix

Classification Report

Accuracy Score
8. Modeling with Random Forest
Implemented a RandomForestClassifier to improve classification performance.

Evaluated accuracy on the test set.

9. Results Visualization
Plotted the distribution of COVID-positive vs Non-COVID cases using matplotlib.

Used value_counts() to show class balance in the target variable.


**Results Summary**

Both models were evaluated on test data using classification metrics.

The Random Forest model demonstrated stronger accuracy and robustness.

The class distribution chart helped confirm the dataset was not overly imbalanced.

**Conclusion**

This project demonstrates the full pipeline of a machine learning classification problem:

Data cleaning

Feature engineering

Binary classification

Model training and evaluation

Visualization

It shows how real-world medical data can be transformed and used to support rapid COVID-19 detection through machine learning techniques.



