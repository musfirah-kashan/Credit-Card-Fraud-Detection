import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_auc_score,roc_curve

df=pd.read_csv('creditcard.csv')
# print(df.columns.tolist())
print(df['Class'].value_counts())
print(df.shape) 
fraud_pct=df['Class'].value_counts(normalize=True)[1]*100
print(f"{fraud_pct:.3f}")

plt.figure()
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0 = Not Fraud, 1 = Fraud)')
plt.show()

standard=StandardScaler()
df['Amount_scaled']=standard.fit_transform(df[['Amount']])
df['Time_scaled']=standard.fit_transform(df[['Time']])
df=df.drop(['Amount','Time'],axis=1)
# print(df.columns.tolist())

X=df.drop('Class',axis=1)
y=df['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

models={
    "LogisticRegression":{
        "model":LogisticRegression(class_weight='balanced',max_iter=1000),
        "params":{'C':[0.01,0.1,1,10]}
    }, 
    "KNN":{
        "model":KNeighborsClassifier(weights='distance'),
        "params":{'n_neighbours':[3,5,7]}
    },
    "DecisionTree":{
        "model":DecisionTreeClassifier(class_weight='balanced',random_state=42),
        "params":{'max_depth':[5,10,15],'min_samples_split':[2,10]}
    },
    "RandomForest":{
        "model":RandomForestClassifier(class_weight='balanced',random_state=42),
        "params":{'n_estimators':[100,200],'max_depth':[8,12]}
    },
    "GradientBoosting":{
        "model":GradientBoostingClassifier(random_state=42),
        "params":{'n_estimators':[100,150],'learning_rate':[0.05,0.1],'max_depth':[3,5]}
    },
    "NaiveBayes":{
        "model":GaussianNB(),
        "params":{'var_smoothing':[1e9,1e8]}
    }
}

