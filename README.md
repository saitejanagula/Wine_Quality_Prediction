# Wine Quality Prediction

## Project Overview

The Wine Quality Prediction project aims to classify wine samples as high-quality or low-quality based on their chemical properties. By applying machine learning techniques, this project helps in predicting wine quality more accurately, which can be valuable for quality control in the wine industry.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
  - [Classification Algorithms](#classification-algorithms)
  - [Ensemble Methods](#ensemble-methods)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Project Description

The goal of this project is to build and evaluate machine learning models that can predict the quality of wine based on various chemical features. The dataset includes information on red and white wines and is used to train and test various classification models to determine which provides the best performance.

## Dataset

The dataset consists of two CSV files:
- `redwine.csv`: Contains data on red wine samples.
- `whitewine.csv`: Contains data on white wine samples.

### Dataset Features

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (target variable)

### Data Source

The dataset is available from the UCI Machine Learning Repository: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

## Prerequisites

To run this project, you need to have the following Python packages installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- xgboost

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
```

## Data Preprocessing

1. **Load the Data**: Read the CSV files and combine them into a single DataFrame.
2. **Handle Missing Values**: Check for and handle any missing values.
3. **Feature Engineering**: Convert the `quality` variable into binary classes (high-quality vs. low-quality).
4. **Feature Scaling**: Standardize the feature variables.

## Exploratory Data Analysis (EDA)

Perform EDA to understand the data distribution and relationships between features:
- **Visualize Distributions**: Use histograms and box plots to examine feature distributions.
- **Correlation Analysis**: Generate a correlation matrix to understand feature relationships.
- **Feature Relationships**: Explore relationships between features and the target variable using scatter plots.
- **Pair Plots**: Analyze pairwise relationships between features using pair plots to understand correlations and distributions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='quality')
plt.show()
```

## Modeling

### Classification Algorithms

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine (SVM)**

### Ensemble Methods

1. **XGBoost**: A powerful gradient boosting framework that improves model performance through boosting tree-based algorithms. It is known for its efficiency and effectiveness in handling large datasets.
2. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms. It is designed for distributed and efficient training.

**Model Training and Evaluation**:

- **Train-Test Split**: Divide the dataset into training and testing sets.
- **Model Training**: Train each classification model on the training data.
- **Model Evaluation**: Evaluate models using accuracy, precision, recall, and F1-score.
- **ROC Curves**: Plot ROC curves for LightGBM and Random Forest to evaluate model performance.

```python
from sklearn.metrics import roc_curve, auc

# For LightGBM
import lightgbm as lgb

# Initialize and train the model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

# ROC curve
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_proba_lgb)
roc_auc_lgb = auc(fpr_lgb, tpr_lgb)

plt.figure()
plt.plot(fpr_lgb, tpr_lgb, color='blue', lw=2, label='LGBM ROC curve (area = %0.2f)' % roc_auc_lgb)

# For Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='RF ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

### Feature Importance

- **Feature Importance Graph**: Plot feature importance to identify the most significant features influencing the predictions.

```python
# Feature importance for Random Forest
importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

plt.figure()
plt.title("Feature importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances_rf[indices_rf], align="center")
plt.xticks(range(X_train.shape[1]), df.columns[indices_rf], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Feature importance for LightGBM
importances_lgb = lgb_model.feature_importances_
indices_lgb = np.argsort(importances_lgb)[::-1]

plt.figure()
plt.title("Feature importances (LightGBM)")
plt.bar(range(X_train.shape[1]), importances_lgb[indices_lgb], align="center")
plt.xticks(range(X_train.shape[1]), df.columns[indices_lgb], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
```

## Results

The Random Forest and LightGBM models showed superior performance in terms of accuracy compared to other classification algorithms. Detailed performance metrics for each model are as follows:

- **Random Forest**:
  - Accuracy: 0.9875
  - Precision: 0.954
  - Recall: 0.954
  - F1-score: 0.954

- **LightGBM**:
  - Accuracy: 0.99375
  - Precision: 0.977
  - Recall: 0.977
  - F1-score: 0.977

## Usage

To use the trained models for predicting wine quality, load the trained model and use it to predict the quality of new samples. Refer to the provided code examples for detailed instructions on how to use the models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the specific values and details as per your actual results and findings. This updated README provides a comprehensive overview, including the additional analyses and visualizations you've implemented.
