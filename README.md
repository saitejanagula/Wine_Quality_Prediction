# Wine Quality Prediction

##Project Overview

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

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
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

## Modeling

### Classification Algorithms

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine (SVM)**

### Ensemble Methods

1. **Random Forest**: An ensemble of decision trees that improves classification accuracy through voting.
2. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms. It is designed for distributed and efficient training.

**Model Training and Evaluation**:

- **Train-Test Split**: Divide the dataset into training and testing sets.
- **Model Training**: Train each classification model on the training data.
- **Model Evaluation**: Evaluate models using accuracy, precision, recall, and F1-score.

```python
# Example of training Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Results

The Random Forest and LightGBM models showed superior performance in terms of accuracy compared to other classification algorithms. Detailed performance metrics for each model are as follows:

- **Random Forest**:
  -Accuracy: 0.9875
  -Precision: 0.9540229885057471
  -Recall: 0.9540229885057471
  -F1-score: 0.9540229885057472

- **LightGBM**:
  -Accuracy: 0.99375
  -precision:  0.9770114942528736
  -recall:  0.9770114942528736
  -f1-score:  0.9770114942528736

## Usage

To use the trained models for predicting wine quality, load the trained model and use it to predict the quality of new samples. Refer to the provided code examples for detailed instructions on how to use the models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the specific values and details as per your actual results and findings. This template should give a comprehensive overview of your project, including methodology, results, and how to use the provided code.
