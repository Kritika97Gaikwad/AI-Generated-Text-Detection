
# AI Generated Text Detection

AI-generated texts have become increasingly prevalent across diverse industries, offering innovative solutions in areas such as Content Generation, Personalized Marketing, Virtual Assistants, and Creative Writing. However, with these advancements come challenges that must be addressed to ensure responsible and ethical use.


## Project Overview

Developed a machine learning model using scikit-learn, implementing ensemble techniques, PCA, correlation analysis, and extensive feature engineering. The goal was to classify documents as either human-generated (0) or AI-generated (1) based on document embeddings, word count, and punctuation.



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

