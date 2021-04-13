from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd

sample_csv = '/Users/theo/data/mega_razzle_2.csv'


input_df = pd.read_csv(sample_csv)


input_df.drop(columns=['3_uid', '4_uid'], inplace=True)
input_df.fillna(0, inplace=True)


X = input_df.iloc[:,0:11].values
y = input_df.iloc[:,0:12].values

print(list(input_df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestClassifier(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))