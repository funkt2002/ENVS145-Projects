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
warnings.filterwarnings('ignore')

PATH='./'
FILENAME='Crop_recommendation.csv'
data=pd.read_csv(PATH+FILENAME)

crop_names=data['label'].unique()
data['label'].value_counts()
data.rename(columns={'N':'nitrogen','P':'phosphorus','K':'potassium','label':'crop'}, inplace=True)
data.head()

features = ['nitrogen','phosphorus','potassium','temperature','humidity','ph','rainfall']
target = ['crop']
X = data[features]
y = data[target]

print(crop_names)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.33)
mmscaler=MinMaxScaler()
X_train=mmscaler.fit_transform(X_train)
X_test=mmscaler.transform(X_test)
y_train = LabelEncoder().fit_transform(np.asarray(y_train).ravel())
y_test = LabelEncoder().fit_transform(np.asarray(y_test).ravel())
for ii, col in enumerate(features):
  print('{} (min,max): \t \t {:.2f} {:.2f}'.format(col,X_train[:,ii].min(),X_train[:,ii].max()))

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Training Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_train,model.predict(X_train))*100,metrics.accuracy_score(y_test,model.predict(X_test))*100))

def get_predictions(x1,x2,x3,x4,x5,x6,x7):
    feature = mmscaler.transform(np.asarray([x1,x2,x3,x4,x5,x6,x7]).reshape((1,-1)))
    croptoplant = crop_names[model.predict(feature).item()]
    print('{} should grow very well under these conditions'.format(croptoplant.upper()))
    

N=56
P=67
K=111
temp=25
hum=21
ph=5
rain=32
get_predictions(x1=N,x2=P,x3=K,x4=temp,x5=hum,x6=ph,x7=rain)

