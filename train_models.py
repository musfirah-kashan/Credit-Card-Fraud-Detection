import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV

df=pd.read_csv('creditcard.csv')
# print(df.columns.tolist())
print(df['Class'].value_counts())
print(df.shape) 
fraus_pct=df['Class'].value_counts(normalize=True)[1]*100
print(f"{fraus_pct:.3f}")

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
