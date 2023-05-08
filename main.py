import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load datas from files
data = pd.read_csv('./datasets/train.csv')
data_test = pd.read_csv('./datasets/test.csv')

# Combine these two dataframes in one
Data = data.append(data_test, ignore_index=True)

# Delete education, employee_id columns and is_promoted
Data.drop(['employee_id', 'education', 'is_promoted'], axis=1, inplace=True)

# One-hot encoding of categorical features, because scikit learn's random forest classifier can't handle categorical features
df_OHE = pd.get_dummies(Data)

# Normalization of data
df_OHE_scaled = (df_OHE - df_OHE.min()) / (df_OHE.max() - df_OHE.min())

# Keep previous_year_ratings as they originally were for the classification
df_OHE_scaled['previous_year_rating'] = data['previous_year_rating'].append(data_test['previous_year_rating'], ignore_index=True)

# Creates a copy of the previous dataframe but deletes all rows containing NA
df_OHE_scaled2 = df_OHE_scaled.dropna(axis=0)

# Training the random forest classifier to predict the previous_year_rating and checking its performance
y=df_OHE_scaled2['previous_year_rating'].astype(int)
X=df_OHE_scaled2.drop(['previous_year_rating'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = RandomForestClassifier(max_depth=30, n_estimators=300)
clf.fit(X, y)
y_pred1 = clf.predict(X_train)
y_pred = clf.predict(X_test)
print(f1_score(y_train, y_pred1, average='weighted'))
print(f1_score(y_test, y_pred, average='weighted'))
print(metrics.confusion_matrix(y_test, y_pred))

# Completes the missing values in previous_year_rating column
for i in range(0, df_OHE_scaled.shape[0]):
    if pd.isna(df_OHE_scaled.loc[i, 'previous_year_rating']) == True:
        rowSeries = df_OHE_scaled.iloc[i].drop('previous_year_rating')
        prediction = clf.predict([rowSeries])
        df_OHE_scaled.at[i, 'previous_year_rating'] = prediction

# Training lightgbm classifier to predict whether an employee is_promoted
# after oversampling the minority class (is_promoted=1)
X = df_OHE_scaled[0:len(data)]
y = data['is_promoted'].astype(int)
sm1 = SMOTE(random_state=42)
X_res, y_res = sm1.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
print(model)

# Make predictions
expected_y = y_test
predicted_y = model.predict(X_test)

# Summarizes the fit of the model
print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))

# Final prediction of testing dataset and saving of result in csv file
is_promoted_pred = model.predict(df_OHE_scaled[len(data):])
is_promoted_pred = pd.DataFrame(is_promoted_pred)
is_promoted_pred.to_csv('prediction.csv', index=False, header=False)
