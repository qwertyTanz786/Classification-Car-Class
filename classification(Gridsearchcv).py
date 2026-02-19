import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df=pd.read_csv(r"C:\Users\panch\Downloads\archive (5)\cars.csv")
X=df.drop("class", axis=1)
Y=df["class"]
cv_rf=GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        'n_estimators': [100,200,500],
        'max_depth': [10,20,30],
        'min_samples_split': [2,5,7]
    },
    cv=3
)
for cols in X.columns:
    if X[cols].dtype == "object":
        le=LabelEncoder()
        X[cols]=le.fit_transform(X[cols])
le_Y=LabelEncoder()
Y=le_Y.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
cv_rf.fit(X_train,Y_train)
Y_pred=cv_rf.predict(X_test)
print(classification_report(Y_test,Y_pred))