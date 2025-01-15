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
- Decision Tree
- Random Forest

## Linear Regression Model 1
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

To perform stratified sampling based on income, we need to create an income category attribute. We'll divide the income range into five categories (labeled 1 to 5): Category 1 will include incomes from 0 to 1.5 (less than $15,000), Category 2 will cover incomes from 1.5 to 3, and so on.


```
df["income_cat"] = pd.cut(df["MedInc"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
df["income_cat"].hist()
```

Note: It’s important not to create too many strata, and each stratum should contain a sufficient number of instances to ensure meaningful sampling and avoid bias in the results.

Let's predict the `MedHouseVal` using `MedInc` as the feature. First, we'll apply stratified sampling to create training and testing datasets, ensuring that both sets represent the different income categories. Then, we'll use a linear regression model to find the best fit and predict the housing price.

Since the data is categorized by income, stratified sampling will ensure that both the training and test sets include samples from each income category.

```
# stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
housing_split = split.split(df, df["income_cat"])

# get training and testing dataset
for train_index, test_index in housing_split:
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

# drop MedHouseVal from training data
housing_tr = train_set.drop("MedHouseVal", axis=1)

# create a new dataframe with MedHouseVal
housing_labels = train_set["MedHouseVal"].copy()
```

### Model Training and Evaluation
```
# linear regression model for best fit
lin_reg = LinearRegression()
lin_reg.fit(housing_tr, housing_labels)

# predict the median_house_value
predicted_data = lin_reg.predict(housing_tr)

# Compute the RMSE
lin_rmse = np.sqrt(mean_squared_error(housing_labels, predicted_data))
print("root mean square error:", lin_rmse)
```
The prediction error is 0.724.

Next, we can try to apply a linear regression model to predict the `MedHouseVal` using the other features. Hopefully, this gives us a lower prediction error.

## Linear Regression Model 2
The other features are on different scales, so we need to apply **feature scaling**. **Standardization** is a common method, which adjusts the data to have a mean of 0 and a standard deviation of 1. We can use Scikit-learn's **StandardScaler** to perform this transformation.








