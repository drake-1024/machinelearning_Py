# Introduction
This project focuses on building a predictive model to estimate housing prices using the California Housing Dataset. The dataset, derived from the 1990 U.S. Census, contains information about various features of housing blocks in California. Using this data, we aim to predict the median house value for a block group, which is the smallest geographical unit defined by the U.S. Census Bureau.

This is a supervised learning task, specifically a univariate multiple regression task, where the goal is to predict a single target variable (`MedHouseVal`) using multiple input features. The project demonstrates the full machine learning workflow, including data exploration, preprocessing, model training, evaluation, model selection, and model fine-tuning.

# The Dataset
The California Housing Dataset contains:

### Instances
20,640 housing block groups
### Features
- `MedInc`: Median income in the block group (in tens of thousands of dollars)
- `HouseAge`: Median age of houses in the block group (in years)
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Total population of the block group
- `AveOccup`: Average number of household members
- `Latitude`: Latitude of the block group (geographic location)
- `Longitude`: Longitude of the block group (geographic location)

### Target Variable
`MedHouseVal`: Median house value in the block group (in hundreds of thousands of dollars, capped at $500,000)

### Key Points
- Data is organized at the level of block groups, each containing 600-3,000 people.
- House values are capped at $500,000, introducing a slight bias for high-value properties.

# Loading the Data
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Load the data
housing = fetch_california_housing(as_frame=True)
df = housing.frame
```

# Data Exploration
```
# Display the first few rows
print(df.head())

# Display quick description of the data
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# find correlation coefficients
df.corr()["MedHouseVal"].sort_values(ascending=False)
```

The median house value generally increases as the median income rises. Additionally, there is a slight negative correlation with latitude, where the median house value tends to decrease as you move north.

# Training Models
In this section, we will explore several regression models to identify the most promising one based on prediction error. The models we will use are:
- Linear Regression
- Random Forest
- 

## Linear Regression Model
Before using multiple features to predict `MedHouseVal`, let's begin with a single feature. Based on the correlation data, `MedInc` appears to be a promising attribute. Let's examine the scatterplot between `MedInc` and `MedHouseVal`.

```
# scatter plot
df.plot(kind='scatter', 
             x="MedInc", y="MedHouseVal", 
             alpha=0.1)
```

The diagram above clearly shows a strong correlation, with an upward trend and minimal dispersion.

### Data Preprocessing
To prepare the training and testing datasets for the machine learning model, we can use functions provided by Scikit-learn to split the data into subsets. The simplest method is `train_test_split()`.

However, the `train_test_split()` function uses random sampling, which works well for large datasets. Since our housing dataset is small, this random sampling may introduce sampling bias. Therefore, we need to use a stratified sampling method to ensure a more representative split.

Stratified sampling ensures that each subset (training and test sets) contains a proportional representation of the different categories (or strata) within the data. In this case, since median income is an important feature for predicting housing prices, we want both the training and test sets to accurately reflect the distribution of income across the entire dataset. This helps avoid situations where the training or test set might over-represent or under-represent certain income groups, which could lead to biased or inaccurate predictions.

The **StratifiedShuffleSplit** cross-validator combines two methods: **StratifiedKFold** and **ShuffleSplit**. It randomly splits the data into training and test sets while ensuring that the proportion of each class (or category) is preserved in both sets. This helps maintain a consistent distribution of classes across the splits, similar to how **StratifiedKFold** works, but with the added randomness of **ShuffleSplit**.

Now, let's create a histogram of median income for a more detailed analysis.

```
df.hist(column='MedInc', 
             bins=50, figsize=(9,6))
```

The histogram above shows that most median income values are concentrated between 1.5 and 6.

#### Income Categorization
To perform stratified sampling based on income, we need to create an income category attribute. We'll divide the income range into five categories (labeled 1 to 5): Category 1 will include incomes from 0 to 1.5 (less than $15,000), Category 2 will cover incomes from 1.5 to 3, and so on.


```
df["income_cat"] = pd.cut(df["MedInc"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
df["income_cat"].hist()
```

Note: It’s important not to create too many strata, and each stratum should contain a sufficient number of instances to ensure meaningful sampling and avoid bias in the results.

Let's predict the `MedHouseVal` using `MedInc` as the feature. First, we'll apply stratified sampling to create training and testing datasets, ensuring that both sets represent the different income categories. Then, we'll use a linear regression model to find the best fit and predict the housing price.


#### Stratified Sampling
Since the data is categorized by income, stratified sampling will ensure that both the training and test sets include samples from each income category.

```
# stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
housing_split = split.split(df, df["income_cat"])

# get train and test dataset
for train_index, test_index in housing_split:
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

# drop MedHouseVal from training and testing data
x_train = train_set.drop("MedHouseVal", axis=1)
x_test = test_set.drop("MedHouseVal", axis=1)

# create a new dataframe with MedHouseVal
y_train = train_set["MedHouseVal"].copy()
y_test = test_set["MedHouseVal"].copy()
```

#### Feature Scaling
We need to scale the features to ensure that all features are on a similar scale. This helps improve the performance of the machine learning model by ensuring that no feature dominates due to its larger range, leading to more balanced and accurate predictions.

```
scaler = StandardScaler().fit(x_train.iloc[:, :8])

def preprocessor(X):
    A = X.copy()
    A.iloc[:, :8] = scaler.transform(A.iloc[:, :8])
    return A

# Fit the scaler to the data and transform it
x_train_scaled, x_test_scaled = preprocessor(x_train), preprocessor(x_test)
```

### Model Training and Evaluation
```
# linear regression model for best fit
lin_reg = LinearRegression().fit(x_train_scaled, y_train)

# predict the median_house_value
y_hat = lin_reg.predict(x_train_scaled)

# Compute the RMSE
lin_rmse = mean_squared_error(y_hat, y_train, squared=False)
lin_rmse
```
The RMSE is 0.724.

## Decision Tree Model

The Decision Tree model has the ability to capture non-linear relationships. Unlike linear regression, it splits data into subsets, modeling complex patterns effectively. This makes it well-suited for this dataset, where factors like income, house age, and location interact non-linearly.

### Model Training and Evaluation
```
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Initialize the Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)

# Train the model on the training set
dt_reg.fit(x_train_scaled, y_train)

# Predict on the training set to evaluate performance
y_hat_train = dt_reg.predict(x_train_scaled)
train_rmse = mean_squared_error(y_train, y_hat_train, squared=False)
train_rmse
```
The RMSE of 2.783e-16 essentially zero, which suggests that the model has perfectly fit the training data. This is a classic sign of overfitting, where the model memorizes the training data instead of generalizing from it.

### Fine-tuning Model

To reduce overfitting in the Decision Tree, we’ll limit tree depth and adjust parameters like minimum samples for splits. Hyperparameter tuning with GridSearchCV will find optimal settings.

```
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(x_train_scaled, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best Parameters:", best_params)

# Evaluate best model
best_train_rmse = mean_squared_error(y_train, best_model.predict(x_train_scaled), squared=False)
best_cv_rmse = np.sqrt(-grid_search.best_score_)
print("Best Model Training RMSE:", best_train_rmse)
print("Best Model Cross-validated RMSE:", best_cv_rmse)
```
- Best Parameters: {'max_depth': 15, 'min_samples_leaf': 10, 'min_samples_split': 2}
- Best Model Training RMSE: 0.449
- Best Model Cross-validated RMSE: 0.628

The model performs well on the training set, but the higher cross-validated RMSE suggests some overfitting. The model might be capturing noise or specific patterns in the training data that do not generalize as well to unseen data. Further steps could include exploring ensemble methods like Random Forest or Gradient Boosting to reduce variance and improve generalization.

## Random Forest Model

By combining multiple Decision Trees into an ensemble, Random Forests reduce overfitting and improve generalization through averaging. This approach can better capture complex relationships. Additionally, Random Forests provide insights into feature importance while being less sensitive to outliers and noise, making them a powerful and reliable choice.


### Model Training and Evaluation
```
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rfr_reg = RandomForestRegressor(max_depth=10, random_state=42)

# Train the model on the training set
rfr_reg.fit(x_train_scaled, y_train)

# Predict on the training set to evaluate performance
y_hat_train = rfr_reg.predict(x_train_scaled)
train_rmse = mean_squared_error(y_train, y_hat_train, squared=False)
train_rmse
```
The RMSE is 0.420.

### Fine-tuning Model
Hyperparameter tuning with GridSearchCV will find optimal settings.
```
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]}, 
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features':[2, 3, 4]},
]

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(x_train_scaled, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best Parameters:", best_params)

# Evaluate best model
best_train_rmse = mean_squared_error(y_train, best_model.predict(x_train_scaled), squared=False)
best_cv_rmse = np.sqrt(-grid_search.best_score_)
print("Best Model Training RMSE:", best_train_rmse)
print("Best Model Cross-validated RMSE:", best_cv_rmse)
```
- Best Parameters: {'max_features': 8, 'n_estimators': 30}
- Best Model Training RMSE: 0.200
- Best Model Cross-validated RMSE: 0.520

# Model Selection
We select the Random Forest model with the optimal parameters: 8 features and 30 estimators. We consider this as our final model and evaluate its performance on the test set.

# Model Evaluation on Test Set
```
y_hat = best_model.predict(x_test_scaled)

test_rmse = mean_squared_error(y_test, y_hat, squared=False)
print("Test RMSE:", test_rmse)
```
The RMSE is 0.502.
