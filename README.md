
# AI Generated Text Detection

AI-generated texts have become increasingly prevalent across diverse industries, offering innovative solutions in areas such as Content Generation, Personalized Marketing, Virtual Assistants, and Creative Writing. However, with these advancements come challenges that must be addressed to ensure responsible and ethical use.


## Table of Contents
1. Import Libraries
2. EDA
3. Data Preparation
4. Feature Engineering
5. Modeling
6. Evaluation
7. Conclusion
8. Other Best Models
### Import libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# model evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# import other functions we'll need for classification modeling
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
```
## EDA
### 1. Target Variable Distribution Analysis
Examined the distribution of values (0s and 1s) in the target variable 'ind' to understand the class balance or imbalance within the dataset.

### 2. Top Feature Spread Exploration
Investigated the distribution of the top 10 features exhibiting the highest variability or spread across the dataset.

### 3. Punctuation Number Distribution
Employed a histogram to visually illustrate the contrast in punctuation usage between AI-generated and human-generated text.

### 4. Word Count Distribution
Created a histogram to compare human-generated vs AI-generated text based on word count.

### 5. Spread of Top Features by RFC (Random Forest Classifier)
Explored the spread and distribution patterns of the top 8 features (other than word count and punctuation count) identified by the Random Forest Classifier (RFC).

### 6. Correlated Feature Visualization
Constructed scatter plots to visualize the relationships and correlations between features, particularly focusing on those features displaying strong correlations. This aided in understanding how certain features interrelate within the dataset.

### Data Cleaning
The data is a cleaned dataset and has zero null values.

#### Missing Values Analysis
```python
# missing values by column
df.isnull().sum()

# missing values in the entire dataframe
df.isnull().sum().sum()
```
### Target Variable Distribution
```python
# Plotting a pie chart for target variable distribution
target_distribution.plot(kind='pie', title='ind Distribution', autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.ylabel('')  # Remove the label on the y-axis
plt.show()
```

###Distribution of Variables with Maximum Spread
```python
# Plotting the top 10 features with maximum spread
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[top_10_feature_spreads['Feature']])
plt.title('Box Plot of Top 10 Features')
plt.ylabel('Feature Values')
plt.xticks(rotation=45)
plt.show()

df['feature_497'].describe()
```

### Punctuation Count Histogram
```python
# Plotting the scaled frequency histograms for punctuation count
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(human_punc_bin_midpoints, human_scaled_punc_freq, width=(human_punc_bins[1] - human_punc_bins[0]), alpha=0.5, label='Human Generated', color='blue')
ax.bar(ai_punc_bin_midpoints, ai_scaled_punc_freq, width=(ai_punc_bins[1] - ai_punc_bins[0]), alpha=0.5, label='AI Generated', color='red')
ax.set_xlabel('Punctuation Count')
ax.set_ylabel('Scaled Frequency')
ax.legend()
plt.tight_layout()
plt.show()

```

### Word Count Histogram
```python
# Plotting the scaled frequency histograms for word count
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(human_bin_midpoints, human_scaled_freq, width=(human_bins[1] - human_bins[0]), alpha=0.5, label='Human Generated', color='blue')
ax.bar(ai_bin_midpoints, ai_scaled_freq, width=(ai_bins[1] - ai_bins[0]), alpha=0.5, label='AI Generated', color='red')
ax.set_xlabel('Word Count')
ax.set_ylabel('Scaled Frequency')
ax.legend()
plt.tight_layout()
plt.show()
```



## Data Preperation


### Define X and Y variables
```python
# Set the target variable (y) = 'ind'
# Set the predictor variable (X) = to the remaining features after dropping 'ind'
# Additionally, 'ID' was excluded
X = df.drop('ind', axis=1)
y = df['ind']
X = X.drop('ID', axis=1)  # Removing the 'ID' column from X
```
### Split the Data
```python
# Splitting the dataset into training and testing sets
X_train_plain, X_test_plain, y_train_plain, y_test_plain = train_test_split(X, y, test_size=0.1, random_state=42)
# MinMax Scaling
scaler = MinMaxScaler()
X_train_plain = scaler.fit_transform(X_train_plain)
X_test_plain = scaler.transform(X_test_plain)
```

### Top 10 Features
```python
# Fit a RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_plain, y_train_plain)
# Get feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train_plain.columns)
# Sort features based on importance
top_features = feature_importances.sort_values(ascending=False).head(10)
```

### Boxplot of Top 8 Variables
```python
# Boxplot of Top 8 variables on RFC
plt.figure(figsize=(10, 6))
X[top_8_columns].boxplot()
plt.title('Boxplots of Top 8 Variables')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()
```

### Correlation Matrix
```python
# Set the threshold value of 0.75 for correlation
threshold = 0.75
# Calculating the correlation matrix after scaling
correlation_matrix = X_train_plain.corr().abs()
# Finding columns with correlations above the threshold
...
# Scatter plots of highly correlated features
...
# Dropping highly correlated columns from the scaled data
X_train_plain = X_train_plain.drop(columns=features_to_drop, axis=1)
X_test_plain = X_test_plain.drop(columns=features_to_drop, axis=1)
```


## Feature Engineering
### PCA (Principal Component Analysis)

```python
# Implementing PCA to retain 95% variance
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_plain)
# Converting the transformed PCA training data (X_train_pca) to a Pandas DataFrame
X_train_pca = pd.DataFrame(X_train_pca)

# Applying PCA coordinate system to the test data
X_test_pca = pca.transform(X_test_plain)
# Converting the transformed PCA test data (X_test_pca) to a Pandas DataFrame
X_test_pca = pd.DataFrame(X_test_pca)

# The y variables are the same as before
y_train_pca = y_train_plain
y_test_pca = y_test_plain
```
## Modeling

### Stacked Model

#### Ensemble Stacked Model Explanation:

For model construction, an ensemble technique known as the Stacking Classifier was employed to achieve optimal performance. This Stacked Model amalgamates various base models, including RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, and SVC. These models were imported into pipelines along with StandardScaler for effective preprocessing.

The Stacking Classifier's architecture involves combining these base models using their predictions as features in a meta-estimator, Logistic Regression in this case. The stacking approach enhances predictive accuracy by leveraging the collective wisdom of diverse models.

Upon training, the Stacking Classifier learns on the PCA-transformed training data, X_train_pca, and their corresponding labels, y_train_pca. Subsequently, predictions are generated on the PCA-transformed test data, X_test_pca, and these predictions are stored as 'preds'.

```python
# Importing the important libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier

# Defining the estimators with different models using pipelines with random state = 42
estimators=[
    ( "RandomForestClassifier", make_pipeline(StandardScaler(),
                                          RandomForestClassifier(random_state=42))),
    ( "GradientBoostingClassifier", make_pipeline(StandardScaler(),
                                          GradientBoostingClassifier(random_state=42))),
    ( "AdaBoostClassifier", make_pipeline(StandardScaler(),
                                          AdaBoostClassifier(random_state=42))),
    ( "SVC", make_pipeline(StandardScaler(),
                                          SVC(random_state=42, probability=True)))
]

# Defining stackingClassifier with the above estimators and with final estimator as LogisticRegression
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))

clf.fit(X_train_pca, y_train_pca) # Fitting the model on X_train_pca and y_train_pca
preds = clf.predict(X_test_pca) # predicting the values of the X_test_pca and storing the predictions preds
# Evaluate the stacked model's performance (F1-score)
f1_rf = f1_score(y_test_pca, preds)
print(f"F1 Score: {f1_rf:.2f}")

# Get the classification report
report_rf = classification_report(y_test_pca, preds)
print("Classification Report for Stacked Model with Random Forest:\n", report_rf)
```

## Evaluation

### Confusion Matrix

```python
# import confusion matrix library
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_pca, preds)
print("Confusion Matrix:\n", conf_matrix)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test_pca, preds).ravel()
print("TP:", tp) # print True Positive
print("TN:", tn) # print True negative
print("FP:", fp) # print False Positive
print("FN:", fn) # print False Negative
```
Classification Report
```python
# Get the classification report
report_rf = classification_report(y_test_pca, preds)
print("Classification Report for Stacked Model with Random Forest:\n", report_rf)
```

### Permutation Importance
```python
# Import library for permutation importance
from sklearn.inspection import permutation_importance

# Calculating permutation importance using the classifier (clf),
# the transformed PCA test data (X_test_pca), and corresponding labels (y_test_pca).
# n_repeats define the number of times to permute the feature
result1 = permutation_importance(clf, X_test_pca, y_test_pca, n_repeats=15,
                                random_state=42)

# Sorting the indices of mean importances obtained from permutation importance
perm_sorted_idx = result1.importances_mean.argsort()

# Get the indices of the top 5 features based on mean importance scores
top_n = 5
top_indices = np.argsort(result1.importances_mean)[-top_n:]

# Get the names of top 5 features
top_feature_names = X_test_pca.columns[top_indices]

print("Top 5 PCA Features:")
print(top_feature_names)
```
### Box Plots
```python
# Get the indices of the top 5 features based on mean importance scores
top_n = 5
top_indices = np.argsort(result1.importances_mean)[-top_n:]

# Get the names of top 5 features
top_feature_names = X_test_pca.columns[top_indices]

# Extract permutation importance scores for top features
top_feature_importances = result1.importances[top_indices]

# Create a boxplot for the top 5 features
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(top_feature_importances.T, vert=False, labels=top_feature_names)
ax.set_title('Permutation Importance of Top 5 PCA Features')
ax.set_xlabel('Importance Score')
plt.show()
```
### Loadings in PCA
```python
# to retrieve the loadings from pca variables
loadings = pca.components_

# Creating a DataFrame to store the loadings with column names as original feature names
loadings_df = pd.DataFrame(loadings, columns=X_train_plain.columns)

# Display the loadings
print(loadings_df)

loadings[5]
```

## Conclusion


### **Understanding the Models Predictions Patterns (3)**

**Stacked Model:**

* Stacked Models allow for increased model performance. The process follows two steps known as level 0 and level 1.
  * At level 0, different models work together, making predictions about the data.
  * Next, the level 1 model is trained using the predictions made by the level 0 models as inputs.

* This technique is effective because each model at level 0 uniquely interprets the data. By combining their predictions, we provide additional insights to the top-level model, enabling it to learn more comprehensively about the patterns and connections in the data.

**PCA-Based Stacked Model:**

* Composition: This model employs Principal Components Analysis (PCA), followed by RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, and SupportVectorClassifier as estimators at level 0.
* Level 1 Model: Logistic Regression is used as the meta-model at level 1.
  * Performance: The model achieved an F1-score of 0.72

 After identifying the stacked models with the highest F1 scores, we initially implemented permutation importance on 770 variables. Additionally, the stacked model that includes Principal Components Analysis (PCA) also yielded high F1 scores. We then applied permutation importance to the PCA variables, setting 'n_repeats' to 15, to determine their importance.

 To better understand the original features that most contribute to the PCA variables with high importance, we employed loading analysis. This technique reveals how each original feature influences all the principal components. This analysis enabled us to identify the top five original features that significantly contribute to each respective principal component.

