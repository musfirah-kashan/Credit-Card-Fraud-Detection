import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
import joblib

df=pd.read_csv('data/creditcard.csv')
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
        "model":LogisticRegression(class_weight='balanced',max_iter=1000,n_jobs=-1,solver='lbfgs'),
        "params":{'C':[0.01,0.1,1,10]}
    }, 
    "SGD":{
        "model":SGDClassifier(loss='log_loss',class_weight='balanced',random_state=42),
        "params":{'alpha':[0.0001,0.001]}
    },
    "DecisionTree":{
        "model":DecisionTreeClassifier(class_weight='balanced',random_state=42),
        "params":{'max_depth':[5,10,15],'min_samples_split':[2,10]}
    },
    "RandomForest":{
        "model":RandomForestClassifier(class_weight='balanced',random_state=42, n_jobs=-1),
        "params":{'n_estimators':[100,200],'max_depth':[8,12]}
    },
     "HistGradientBoosting":{
        "model":HistGradientBoostingClassifier(random_state=42),
        "params":{'max_iter':[100],'learning_rate':[0.05,0.1]}
    },
    "NaiveBayes":{
        "model":GaussianNB(),
        "params":{'var_smoothing':[1e-9,1e-8]}
    }
}

res=[]
for name,config in models.items():
    print(f"\nTraining {name}")
    grid=GridSearchCV(config['model'],config['params'],cv=3,scoring='f1',n_jobs=-1)
    grid.fit(X_train,y_train)
    best_model=grid.best_estimator_
    print(best_model)
    y_pred=best_model.predict(X_test)
    y_proba=best_model.predict_proba(X_test)[:,1]
    acc=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    roc_auc=roc_auc_score(y_test,y_proba)
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test,y_pred,target_names=['Not Fraud','Fraud']))

    cm=confusion_matrix(y_test,y_pred)
    plt.figure()
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Not Fraud','Fraud'],yticklabels=['Not Fraud','Fraud'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'graphs/confusion_matrices {name}', bbox_inches='tight')
    plt.show()

    res.append({
        'model':name,
        'best_params':grid.best_params_,
        'accuracy':acc,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'roc_auc':roc_auc
    })
print(res)
res_df=pd.DataFrame(res)
print(res_df)

res_df.to_csv("models/results.csv",index=False)
 
plt.figure(figsize=(10,5))
plt.bar(res_df['model'],res_df['f1'])
plt.title('F1 Score Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


best_row=res_df.sort_values(by='f1',ascending=False).iloc[0]
print(best_row)
best_model_name=best_row['model']
print(best_model_name)
best_model_config=models[best_model_name]
print(best_model_config)
final_model=best_model_config['model'].set_params(**best_row['best_params'])
print(final_model)
final_model.fit(X,y)
 
joblib.dump(final_model,"models/best_model.pkl")
joblib.dump(standard,"models/scaler.pkl")
feature_names=X.columns.tolist()
if hasattr(final_model,'feature_importances_'):
    importance=final_model.feature_importances_
elif hasattr(final_model,'coef_'):
    importance=abs(final_model.coef_[0])
else:
    importance=[0]*len(feature_names)
 
importance_df=pd.DataFrame({'feature':feature_names,'importance':importance})
importance_df=importance_df.sort_values(by='importance',ascending=False)
importance_df.to_csv("models/feature_importance.csv",index=False)
print(importance_df.head(10))

df['Class']=y
fraud_samples=df[df['Class']==1].sample(5,random_state=42)
safe_samples=df[df['Class']==0].sample(5,random_state=42)
samples=pd.concat([fraud_samples,safe_samples])
samples.to_csv("models/sample_transactions.csv",index=False)
 
predictions=joblib.load('models/best_model.pkl').predict(X_test)
print(predictions)



