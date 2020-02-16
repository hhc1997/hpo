import nni
import logging
#數據分析庫
import pandas as pd
#科學計算庫
import numpy as np
import xgboost
from sklearn import model_selection
from sklearn.model_selection import cross_val_score,train_test_split
LOG = logging.getLogger('sklearn_randomForest')

def load_data():
    '''Load dataset'''
    import pandas as pd
    import numpy as np
    data = pd.read_csv('res_mean_norm_day.csv', encoding = 'UTF-8').reset_index()
    sampleNum = data.shape[0]
    dimNum = data.shape[1]
    #取得列名 不要时间取进去 时间只需要取一遍  所有从2到16
    columns = list(data)[2:16]
    ##新列名：当天接诊的一天前的环境特征
    new_columns_1 = ['ave_C_1',
     'min_C_1',
     'max_C_1',
     'ave_ws_1',
     'ave_rh_1',
     'ave_hpa_1',
     'daily_precipitation_1',
     'SO2_1',
     'NO2_1',
     'CO_1',
     'PM2_5_1',
     'PM10_1',
     'O38h_1',
     'AQI_1'] 
    ##新列名：当天接诊的两天前的环境特征
    new_columns_2 = ['ave_C_2',
     'min_C_2',
     'max_C_2',
     'ave_ws_2',
     'ave_rh_2',
     'ave_hpa_2',
     'daily_precipitation_2',
     'SO2_2',
     'NO2_2',
     'CO_2',
     'PM2_5_2',
     'PM10_2',
     'O38h_2',
     'AQI_2'] 
    #生成新的data 每个接诊人数的特征是前2天的环境因素+当天的时间特征
    new_data = pd.DataFrame()
    #生成 要预测的y influenza_num
    k = 7
    new_data['influenza_num'] = data['influenza_num'][k:].reset_index(drop=True)
    ## 生成 当天接诊的一天前的环境特征 
    for i in range(len(new_columns_1)):
        new_data[new_columns_1[i]] = data[columns[i]][k-1:].reset_index(drop=True)
    ## 生成 当天接诊的两天前的环境特征 
    for i in range(len(new_columns_2)):
        new_data[new_columns_2[i]] = data[columns[i]][k-2:].reset_index(drop=True)
    ## 再加上接诊当天的时间特征 day week month
    columns_time_feature = ['month', 'day', 'week','last_day_num']
    columns_time = list(data)[16:]
    for i in range(len(columns_time)):
        new_data[columns_time_feature[i]] = data[columns_time[i]][k:].reset_index(drop=True)
    ##新增特征 前一天的接诊人数
    new_data['last_day_num'] = data['influenza_num'][k-1:].reset_index(drop=True)
    ##新增特征 两天前的接诊人数
    new_data['2_ago_num'] = data['influenza_num'][k-2:].reset_index(drop=True)
    ##新增特征 三天前的接诊人数
    new_data['3_day_ago_num'] = data['influenza_num'][k-3:].reset_index(drop=True)
    ##新增特征 四天前的接诊人数
    new_data['4_day_ago_num'] = data['influenza_num'][k-4:].reset_index(drop=True)
    ##新增特征 5天前的接诊人数
    new_data['5_day_ago_num'] = data['influenza_num'][k-5:].reset_index(drop=True)
    ##新增特征 6天前的接诊人数
    new_data['6_day_ago_num'] = data['influenza_num'][k-6:].reset_index(drop=True)
    ##新增特征 7天前的接诊人数
    new_data['7_day_ago_num'] = data['influenza_num'][k-7:].reset_index(drop=True)
    X = np.array(new_data)[:,1:] # (2189, 32) 
    Y = np.array(new_data)[:,0] 
    X_train, X_validation, y_train, y_validation = train_test_split(X,Y,test_size=0.3,random_state=123)
    return X_train, y_train, X_validation, y_validation

def get_default_parameters():
    '''get default parameters'''
    params = {
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'learning_rate': 0.1,
        'seed':5,
        'n_estimators':200
    }
    return params

from sklearn.metrics import explained_variance_score,mean_absolute_error, mean_squared_error, r2_score

def measures(predict, test, type):
    # type = EVS解释方差分, MAE平均绝对误差, MSE均方误差, R2R方
    if type == 'mae': 
        return mean_absolute_error(test, predict)
    elif type == 'mse':
        return mean_squared_error(test, predict)
    elif type == 'r2':
        return r2_score(test, predict)


def run(X_train, y_train, X_validation, y_validation ,params):
    xgb = xgboost.XGBRegressor(nthread=10,  # 进程数
                                    max_depth=params["max_depth"],  # 最大深度
                                    n_estimators=params["n_estimators"],  # 树的数量
                                    learning_rate=params["learning_rate"],  # 学习率
                                    subsample=params["subsample"],  # 采样数
                                    min_child_weight=params["min_child_weight"], 
                                    objective='reg:linear',
                                    seed=params["seed"],
                                    colsample_bytree=0.5,
                                    colsample_bylevel = 1,
                                    reg_alpha=1e0,
                                    reg_lambda=0
                                    )
    '''Train model and predict result'''
    eval_set = [(X_validation, y_validation)]
    xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds = 100, eval_set = eval_set)
    Y_pred = xgb.predict(X_validation)
    ####eval
    mse = measures(Y_pred, y_validation, 'mse')
    print('The mse of prediction is:', mse)
    nni.report_final_result(mse)

if __name__ == '__main__':
    X_train, y_train, X_validation, y_validation= load_data()
    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = get_default_parameters()
    PARAMS.update(RECEIVED_PARAMS)
    PARAMS.update(RECEIVED_PARAMS)
    # train
    run(X_train, y_train, X_validation, y_validation ,PARAMS)