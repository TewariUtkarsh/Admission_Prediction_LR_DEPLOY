import pandas as pd
import numpy as np
import json


from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import matplotlib as plt
import seaborn as sns

import pickle

df = pd.read_csv('Admission_Prediction.csv')

## Creating a Pandas Profile Report
pf = ProfileReport(df, minimal=True)
pf.to_file("report.html")

## Handling Missing Values
# 1. GRE Score is having Nan/null value
df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())

# 2. TOEFL Score is having NaN/null values
df['TOEFL Score'].fillna(value=df['TOEFL Score'].mean(), inplace=True)

# 3. University Rating is having NaN/null values:
df['University Rating'].fillna(df['University Rating'].mean(), inplace=True)

## Dropping Serial No. col as it is irrelevant for our model
df.drop('Serial No.', axis=1, inplace=True)

## Creating features and labels for our model
y = df[['Chance of Admit']]
x = df.drop(columns=['Chance of Admit'])

## Standardizing our Features using Standard Scaler(z-score) to improve model accuracy and understanding
scaler = StandardScaler()
std_x = scaler.fit_transform(x)
pickle.dump(scaler, open("scaler_obj.py", 'wb'))

## Checking for Multi-Collinearity using VIF(Variance Inflation Factor):
vif_df = pd.DataFrame()
vif_df['vif'] = [variance_inflation_factor(std_x, i) for i in range(std_x.shape[1])]
vif_df['features'] = x.columns
## As VIF < 10, so need to handle multi-collinearity


## Train Test Split on our data
# np.random.seed equivalent to random_state
x_train, x_test, y_train, y_test = train_test_split(std_x, y, test_size=0.15, random_state=100)


## Creating a Linear Regression Model
lr = LinearRegression()

## Training our LR model
lr.fit(x_train, y_train)

r_square_lr = lr.score(x_test, y_test)

def adj_r2(x,y):
    r2 = lr.score(x,y)
    n = x.shape[0]
    p = y.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

adj_r_square_lr = adj_r2(x_test, y_test)


## Saving Our Model
file = "lr_model.pickle"
pickle.dump(lr, open(file, 'wb'))






# linear Regression model but using Lasso

# Creating Lasso CV object with the required parameters
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=20000000, normalize=True)

# Training CV obj with our training data
lasso_cv.fit(x_train, y_train)

# Getting the Lambda value(Shrinkage factor) which is going to be involved implementign in Lasso Regression
lasso_cv_alpha = lasso_cv.alpha_

# Creating a Lasso Regression object with the previously determined lambda value(shrinkage factor)
lasso = Lasso(alpha=lasso_cv_alpha)

# Training our Lasso Regression obj
lasso.fit(x_train, y_train)

# Calculating lasso reg score
lasso_score = lasso.score(x_test, y_test)









# Creating Ridge CV obj with the required Paramenter
ridge_CV = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=50, normalize=True)

# Here the alpha selection range is too limited, so we can give our custom range for better results in CV
custom_alpha = np.random.uniform(0,10, 50)

ridge_cv = RidgeCV(alphas=custom_alpha, cv=50, normalize=True)

# Training Ridge CV object with our training data to find the hyper-parameter lambda(shrinkage factor)
ridge_cv.fit(x_train, y_train)

# Getting the lambda(shrinkage factor)
ridge_cv_alpha = ridge_cv.alpha_

# Creating a Ridge Regression obj
ridge = Ridge(alpha=ridge_cv_alpha)

# Training Ridge Regression obj
ridge.fit(x_train, y_train)

ridge_score = ridge.score(x_test, y_test)















# Creating Elastic Net CV obj with req params
elasticnet_CV = ElasticNetCV(alphas=None, cv=10)

# Training our Elastic Net obj with our train dataset
elasticnet_CV.fit(x_train, y_train)

# Getting the alpha(lambda value)
elastic_net_alpha = elasticnet_CV.alpha_

# Getting the l1_ratio(alpha value)
elastic_net_l1_ratio = elasticnet_CV.l1_ratio

# Creating elastic net reg model
elastic_net = ElasticNet(alpha=elastic_net_alpha, l1_ratio=elastic_net_l1_ratio)

# Training our elastic net model
elastic_net.fit(x_train, y_train)

# Getting the score/accuracy
elastic_net_score = elastic_net.score(x_test, y_test)


scores = {
    "R-Square Score(LR)": r_square_lr,
    "Adjusted R-Square Score(LR)":adj_r_square_lr,
    "R-Square Score(L1)":lasso_score,
    "R-Square Score(L2)":ridge_score,
    "R-Square Score(Elastic Net)":elastic_net_score
}

json_file = "score.json"
json.dump(scores, open(json_file,'w') )





