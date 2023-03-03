import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import metrics 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import ipywidgets as widgets
import warnings
from IPython.display import display
from time import strptime
warnings.filterwarnings('ignore')
#######This is an interface for the AI model###
PATH='./'
FILENAME='kenya_climate_data.csv'
data=pd.read_csv(PATH+FILENAME)
data.shape
data.info
data.describe().T
data.isnull().sum()
data.rename(str.strip,axis='columns',inplace=True)
data=data.rename(columns={'Rainfall - (MM)':'Rainfall','Month Average':'Month'})
data.columns
months=data.columns['Month']
print(months)
# for month in data():
#     month={
#         'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
#         'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
#     }
# return month
#data.Month=month

#pd.to_datetime(data.Year.astype(str) + '/' + data.Month.astype(str) + '/01')

# data.columns


# features = data.drop(['Rainfall'], axis=1)
# target=data['Rainfall']
# print(features)
# X=features
# y=target

# X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train)

# tree=DecisionTreeClassifier()
# rf=RandomForestClassifier()
# tree.fit(X_train,y_train)
# rf.fit(X_train,y_train)

# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#     print("\n")


# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(tree, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = numpy.sqrt(-scores)

# scores = cross_val_score(rf, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rf_rmse_scores = numpy.sqrt(-scores)

# # As the data was highly imbalanced we will
# # balance it by adding repetitive rows of minority class.

# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,
# # random_state=2)
# # mmscaler=MinMaxScaler()
# # X_train=mmscaler.fit_transform(X_train)
# # X_test=mmscaler.transform(X_test)
# # y_train=LabelEncoder().fit_transform(np.asarray(y_train).ravel())
# # y_test=LabelEncoder().fit_transform(np.asarray(y_test).ravel())
# # for ii, col in enumerate(features):
# #   print('{} (min,max): \t \t {:.2f} {:.2f}'.format(col,X_train[:,ii].min(),X_train[:,ii].max()))
