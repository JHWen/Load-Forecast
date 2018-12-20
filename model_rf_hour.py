import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,make_scorer

# 忽略掉Value值为0的时刻
data = pd.read_csv("/Users/fandebao/Desktop/比赛/数据/HourLoadSet.csv")
index_zero_value = []
for i in range(data.shape[0]):
    if data['Value'][i] == 0:
        index_zero_value.append(i)
df = data.loc[:]
for i in index_zero_value:
    df.loc[i,'Value'] = None
df = df.dropna()
# end

df = df[168:]
dfy = pd.DataFrame({'Value':df['Value']})
dfX = pd.DataFrame({'dayOfWeek':df['dayOfWeek'],
'isWorkday':df['isWorkday'],'isHoliday':df['isHoliday'],
'Season':df['Season'],'Tem':(df['Tem'] - np.mean(df['Tem']))/(np.max(df['Tem']) - np.min(df['Tem'])),
'RH':(df['RH'] - np.mean(df['RH']))/(np.max(df['RH']) - np.min(df['RH'])),
'value_oneweek_before':(df['value_oneweek_before'] - np.mean(df['value_oneweek_before']))/(np.max(df['value_oneweek_before']) - np.min(df['value_oneweek_before'])),
# 'value_oneday_before':(df['value_oneday_before'] - np.mean(df['value_oneday_before']))/(np.max(df['value_oneday_before']) - np.min(df['value_oneday_before'])),
'value_onedayavg_before':(df['value_onedayavg_before'] - np.mean(df['value_onedayavg_before']))/(np.max(df['value_onedayavg_before']) - np.min(df['value_onedayavg_before']))
})

df_X = np.array(dfX)
df_y = np.array(dfy)

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

# 最后两个月数据作测试
X_train = df_X[:16051]
X_test = df_X[16051:]
y_train = df_y[:16051]
y_test = df_y[16051:]
# X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

def fit_model_k_fold(X, y):
    k_fold = KFold(n_splits=10)
    regressor = RandomForestRegressor()
    params = {'n_estimators':range(10,61,10),'min_samples_split':range(2,103,10),
              'max_depth': range(3, 90, 10),'min_samples_leaf':range(2,63,10),
              'max_features': range(2, 8, 2)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=k_fold)
    grid = grid.fit(X, y)
    return grid.best_estimator_

reg = fit_model_k_fold(X_train, y_train)

print(performance_metric(y_test, reg.predict(X_test)))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y = []
y11 = np.sum(y_test[:720])
y12 = np.sum(y_test[720:])
y.append(y11)
y.append(y12)
y = np.array(y).reshape(-1,1)

y_pre = []
y_pred_11 = np.sum(reg.predict(X_test)[:720])
y_pred_12 = np.sum(reg.predict(X_test)[720:])
y_pre.append(y_pred_11)
y_pre.append(y_pred_12)
y_pre = np.array(y_pre).reshape(-1,1)

print("MAPE:", mean_absolute_percentage_error(y,y_pre),"%")