
# AI Generated Text Detection

![Screenshot 2024-06-15 151143](https://github.com/Kritika97Gaikwad/AI-Generated-Text-Detection/assets/151272622/1e598bd6-c2fa-4f31-8a77-1044a12f7c9f)


AI-generated texts have become increasingly prevalent across diverse industries, offering innovative solutions in areas such as Content Generation, Personalized Marketing, Virtual Assistants, and Creative Writing. However, with these advancements come challenges that must be addressed to ensure responsible and ethical use.


## Project Overview

Developed a machine learning model using scikit-learn, implementing ensemble techniques, PCA, correlation analysis, and extensive feature engineering. The goal was to classify documents as either human-generated (0) or AI-generated (1) based on document embeddings, word count, and punctuation.


## Exploratory Data Analysis (EDA)
In the EDA phase, we analyze the dataset using the following visualizations and statistics:

Distribution of the target variable (ind): Understand the imbalance in the dataset.
Distribution of word counts: Analyze the length of the documents.
Frequency of punctuation marks: Examine the usage of punctuation in the documents.
Correlation heatmap of document embeddings: Identify relationships between different embedding dimensions.
PCA and t-SNE visualizations of document embeddings: Reduce dimensions to visualize the embeddings in 2D space.


## Data Preparation
During data preparation:

- Feature Engineering: Create additional features such as average word length and number of unique words.
- Train-Test Split: Split the data into training and testing sets (90/10 split) with a fixed random seed for reproducibility.
- Class Imbalance Handling: Use techniques like SMOTE to balance the classes in the training set.


## Model Training and Evaluation
### We train the following models:

- Logistic Regression
- Random Forest
- AdaBoost
- SVC
- Gradient Boosting
- AutoML/ TPOT


### For evaluation, we:

- Generate learning curves for accuracy and loss.
- Create confusion matrices.
- Produce classification reports.
- Calculate F1 scores, precision, and recall.
- Generate Permutation Importance
- Create Partial Dependence Plots

## Results
The results section in AI_Generated_Text_Detection_Project.ipynb provides a detailed analysis of model performance, highlighting the strengths and weaknesses of each model.

## Contributing
Contributions are welcome! If you have any improvements or bug fixes, please open an issue or submit a pull request.

