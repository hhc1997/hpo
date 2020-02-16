######TUNE & XGBOOST
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch
 
import argparse, time, numpy as np, pandas as pd
from hyperopt import hp
 
import xgboost
from sklearn.preprocessing import MinMaxScaler
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import cross_val_score,train_test_split #默认情况下，cross_val_score不会随机化数据。旧版sklearn.cross_validation包已废弃
####读取数据集
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
###训练集 测试集
X = np.array(new_data)[:,1:] # (2189, 32) 
Y = np.array(new_data)[:,0] 
X_train, X_validation, y_train, y_validation = train_test_split(X,Y,test_size=0.3,random_state=123)
from sklearn.metrics import explained_variance_score,mean_absolute_error, mean_squared_error, r2_score

def measures(predict, test, type):
    # type = EVS解释方差分, MAE平均绝对误差, MSE均方误差, R2R方
    if type == 'mae': 
        return mean_absolute_error(test, predict)
    elif type == 'mse':
        return mean_squared_error(test, predict)
    elif type == 'r2':
        return r2_score(test, predict)

import matplotlib.pyplot as plt
def show_data(Y_pred, Y_test, title=''):
    plt.title(title)
    plt.xlabel('days')
    plt.ylabel('number')  
    l1, = plt.plot(np.arange(0, len(Y_test)), Y_test, color='r')
    l2, = plt.plot(np.arange(0, len(Y_pred)), Y_pred, color='b')
    plt.legend([l1, l2], ['label', 'predict'], loc='upper right')
    plt.show()

def train_objective_tune(config, reporter ):
    '''
    Tune将通过在计算机上安排多个试验来自动化和分发您的超参数搜索。每个试验都运行一个用户定义的Python函数，其中包含一组采样的超参数。
    :param config(dict)：Parameters provided from the search algorithm or variant generation.
    :param reporter(Reporter): Handle to report intermediate metrics to Tune.
    :return: 
    '''
    max_depth = config["max_depth"] + 5
    n_estimators = config['n_estimators'] 
    learning_rate = config["learning_rate"] 
    subsample = config["subsample"] 
    min_child_weight = config["min_child_weight"]
    seed = config['seed']
    min_child_weight = config['min_child_weight']
    print("max_depth:" + str(max_depth))
    print('hahahahahhahahahaha')
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))
    global X_train, y_train
    
 
    xgb = xgboost.XGBRegressor(nthread=10,  # 进程数
                                max_depth=max_depth,  # 最大深度
                                n_estimators=n_estimators,  # 树的数量
                                learning_rate=learning_rate,  # 学习率
                                subsample=subsample,  # 采样数
                                min_child_weight=min_child_weight, 
                                objective='reg:linear',
                                seed=seed,
                                colsample_bytree=0.5,
                                colsample_bylevel = 1,
                                reg_alpha=1e0,
                                reg_lambda=0
                                
                                )
    eval_set = [(X_validation, y_validation)]
    xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds = 100, eval_set = eval_set)
    Y_pred = xgb.predict(X_validation)
    #metric = cross_val_score(xgb, X_validation, y_validation, cv=5, scoring="neg_mean_squared_error").mean()
    metric = measures(Y_pred, y_validation, 'mse')
    for i in range(100):
        ## reward_attr_mean_accuracy="neg_mean_loss" 是因为我们需要它想准确度一样越来越高
        # timesteps_total="training_iteration"
        reporter(timesteps_total=i, mean_accuracy=metric,mode=max) #reporter(mean_loss=test_loss, mean_accuracy=accuracy, metric2=1, metric3=0.3, ...)
        time.sleep(0.02)
    #return -metric

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    #parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint_dir", help="Size of first kernel")
    #args, _ = parser.parse_known_args()
    ray.shutdown()
    ray.init(num_gpus=2,num_cpus=24) #用户希望配置所有本地调度程序的cpu和GPU个数
 
    tune.register_trainable("train_objective_tune", train_objective_tune)
 
    space = {"max_depth": hp.randint("max_depth", 10)+5,
             "n_estimators": 3000,#hp.randint("n_estimators", 10)* 50 + 1000,  
             "learning_rate": hp.randint("learning_rate", 6)* 0.0002 + 0.005,  
             "subsample": hp.randint("subsample",3)*0.02+0.85,  
             "min_child_weight": hp.randint("min_child_weight", 10)+10,
             'seed':hp.randint("seed", 5)+30,
             }
 
    train_spec = {
        "byz_ahbs_xgb_exp": {
            "run": "train_objective_tune",
        "resources_per_trial": {
            "cpu": 24,
            "gpu": 2,
        },
            "num_samples": 1000, # 这指定了您计划运行的试验次数,从超参数空间中采样的次数，而不是批次的大小。
  
        }
    }
 

    algo = HyperOptSearch(space, max_concurrent=4, reward_attr="mean_accuracy")
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode='max'
  )
    #tune.run_experiments({"pbt_cifar10": train_spec}, scheduler=scheduler)
 
    tune.run_experiments(train_spec, search_alg=algo, scheduler=scheduler)


