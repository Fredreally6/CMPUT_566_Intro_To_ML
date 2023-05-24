import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

# loading data
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
train_data = pd.read_csv('data/adult.data', names = columns)
test_data = pd.read_csv('data/adult.test', names = columns, skiprows=1)

# data preprocessing
for i in train_data.columns:        # replace the '?' by 'unknown'
    train_data[i].replace('?', 'unknown', inplace = True)
    test_data[i].replace('?', 'unknown', inplace = True)

non_int_columns = [1,3,5,6,7,8,9,13,14]
for j in non_int_columns:           # replace the blank before non-int data
    train_data.iloc[:,j] = train_data.iloc[:,j].map(lambda x: x.strip())
    train_data.iloc[:,j] = train_data.iloc[:,j].map(lambda x: x.replace(".", ""))
    test_data.iloc[:,j] = test_data.iloc[:,j].map(lambda x: x.strip())
    test_data.iloc[:,j] = test_data.iloc[:,j].map(lambda x: x.replace(".", ""))

train_data.drop('fnlwgt', axis = 1, inplace = True)         # drop column "fnlwgt", "education" and "native-country"
train_data.drop('education', axis = 1, inplace = True)
train_data.drop('native-country', axis = 1, inplace = True)     
test_data.drop('fnlwgt', axis = 1, inplace = True)
test_data.drop('education', axis = 1, inplace = True)
test_data.drop('native-country', axis = 1, inplace = True)

    # Label-encoded non-numeric variables into continuous numeric variables
mapper = DataFrameMapper([('age', LabelEncoder()),('workclass', LabelEncoder()), ('education-num', LabelEncoder()),
                            ('marital-status', LabelEncoder()), ('occupation', LabelEncoder()), ('relationship', LabelEncoder()),
                            ('race', LabelEncoder()), ('sex', LabelEncoder()), ('income', LabelEncoder())], df_out = True, default = None)

cols = list(train_data.columns)
cols.remove('income')
cols = cols[:-3] + ['income'] + cols[-3:]

train = mapper.fit_transform(train_data.copy())     # fit the data and standardize
train.columns = cols
test = mapper.fit_transform(test_data.copy())
test.columns = cols
cols.remove('income')

x, y = train[cols].values, train['income'].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state=1)    # split the validation set from training set
x_test, y_test = test[cols].values, test['income'].values

# end of data preprocessing

# Logistic Regression
# baseline
lgr = linear_model.LogisticRegression(max_iter=10000)       #default max_iter=100, which will cause "lbfgs failed to converge", so set max_iter=10000
lgr.fit(x_train, y_train)
lgr_val_acc = lgr.score(x_val, y_val)
lgr_test_acc = lgr.score(x_test, y_test)

# tuning
paras_lgr = {'solver': ('lbfgs', 'newton-cg', 'saga'),
        'penalty': ('l1', 'l2', 'none')}
lgr_tune = GridSearchCV(lgr, paras_lgr, n_jobs=-1, verbose=3, cv=5)
lgr_tune.fit(x_train, y_train)
lgr_best_acc = lgr_tune.best_score_ 
lgr_tune_val_acc = lgr_tune.score(x_val, y_val)
lgr_final_acc = lgr_tune.score(x_test, y_test)
lgr_best_paras = lgr_tune.best_params_


# Decision Tree
# baseline
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_val_acc = dt.score(x_val, y_val)
dt_test_acc = dt.score(x_test, y_test)

# tuning
paras_dt = {'criterion':('gini', 'entropy'),
            'max_depth':(None, 10, 15, 20),
            'min_samples_split':(2, 6, 9, 12),
            'min_samples_leaf':(1, 3, 6, 9),}

dt_tune = GridSearchCV(dt, paras_dt, verbose=3, cv=5)
dt_tune.fit(x_train, y_train)
dt_best_acc = dt_tune.best_score_ 
dt_tune_val_acc = dt_tune.score(x_val, y_val)
dt_final_acc = dt_tune.score(x_test, y_test)
dt_best_paras = dt_tune.best_params_


# Naive Bayes
# baseline
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_val_acc = nb.score(x_val, y_val)
nb_test_acc = nb.score(x_test, y_test)

# Cross-validation
k_fold = 10
nb_val_acc_cv = cross_val_score(nb, x_val, y_val, scoring = 'accuracy', cv = k_fold)
nb_test_acc_cv = cross_val_score(nb, x_test, y_test, scoring = 'accuracy', cv = k_fold)


# Linear Regression
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
lr_acc = lr.score(x_test, y_test)

print("********************************************************************************")
print("Linear Regression accuracy =",lr_acc)
print("Three models are used for this classification problem. The results are as follows:")
print("----------Logistic Regression---------")
print("Validation accuracy of Logistic Regression =", lgr_val_acc)
print("Test accuracy of Logistic Regression =", lgr_test_acc)
print("LR's best training accuracy after tuning =", lgr_best_acc)
print("LR's validation accuracy after tuning =", lgr_tune_val_acc)
print("LR's best parameters after tuning:", lgr_best_paras)
print("LR's final accuracy = ", lgr_final_acc)
print("-------------Decision Tree------------")
print("Validation accuracy of Decision Tree =", dt_val_acc)
print("Test accuracy of Decision Tree =", dt_test_acc)
print("DT's best training accuracy after tuning =", dt_best_acc)
print("DT's validation accuracy after tuning =", dt_tune_val_acc)
print("DT's best parameters after tuning:", dt_best_paras)
print("DT's final accuracy = ", dt_final_acc)
print("--------------Naive Bayes-------------")
print("Validation accuracy of Naive Bayes =", nb_val_acc)
print("Test accuracy of Naive Bayes =", nb_test_acc)
print("Average validation accuracy of Naive Bayes with 10-fold cross-validation =", np.mean(nb_val_acc_cv))
print("Average test accuracy of Naive Bayes with 10-fold cross-validation =", np.mean(nb_test_acc_cv))
