import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import time

start_time = time.time()
from featureengineering import FeatureEngineering
#import traindata


path = 'F:/dataf/data/'
train = pd.read_csv(path+'train_all.csv')
test = pd.read_csv(path+'republish_test.csv')
offline = False

p=FeatureEngineering()
train = p.pre_processing(train)
test = p.pre_processing(test)




from sklearn.preprocessing import LabelEncoder
train['current_service'] = train.current_service.astype(int)
service_label = LabelEncoder().fit(train.current_service.astype(str))
train['current_service'] = service_label.transform(train['current_service'].astype(str))

train.drop('user_id',axis = 1, inplace = True)

print('data ready finished')


# 数据拆分(训练集+验证集+测试集)
print('拆分数据集')
train_xy, offline_test = train_test_split(train, test_size=0.2, random_state=21)
train, val = train_test_split(train_xy, test_size=0.2, random_state=21)

# 训练集
y_train = train.current_service                    # 训练集标签
X_train = train.drop(['current_service'], axis=1)   # 训练集特征矩阵

# 验证集
y_val = val.current_service                        # 验证集标签
X_val = val.drop(['current_service'], axis=1)       # 验证集特征矩阵

#测试集
offline_test_X = offline_test.drop(['current_service'], axis=1)
online_test_X = test

# 数据转换
print('数据转换')
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)

# 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
        
        'boosting_type': 'gbdt', # 训练方式
        'objective':  'multiclass',   # 目标  多分类
        'metric': {'multi_logloss'},  # 损失函数
        'num_class': 11,       
        }

# 交叉验证(调参)
print('交叉验证')
# 无限小浮点数
min_merror = float('Inf')
best_params = {}
# 准确率
print("调参1：提高准确率")

for num_leaves in range(20,200,5):
    for max_depth in range(3, 8, 1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=2018,
            nfold=5,
            early_stopping_rounds=15,
            metrics=['multi_logloss'],
            
        )

        mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()
        boost_rounds = pd.Series(cv_results['multi_logloss-mean']).argmin()

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
            print(best_params)

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

# 过拟合
print("调参2：降低过拟合")
for max_bin in range(1,255,5):
    for min_data_in_leaf in range(10,200,5):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=42,
                                nfold=3,
                                metrics=['multi_logloss'],
                                early_stopping_rounds=3,
                                verbose_eval=True
                                )
                    
            mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).argmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['max_bin']= max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf

params['min_data_in_leaf'] = best_params['min_data_in_leaf']
params['max_bin'] = best_params['max_bin']
# 参数不能为0
print("调参3：降低过拟合")
for feature_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 50, 5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq

            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=42,
                                nfold=3,
                                metrics=['multi_logloss'],
                                )

            mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).argmin()

            if mean_merror <min_merror:
                min_merror = mean_merror
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

params['feature_fraction'] = best_params['feature_fraction']
params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']

print("调参4：降低过拟合")
for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            params['min_split_gain'] = min_split_gain

            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=42,
                                nfold=3,
                                early_stopping_rounds=3,
                                metrics=['multi_logloss'],
                                )

            mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).argmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
                best_params['min_split_gain'] = min_split_gain

params['lambda_l1'] = best_params['lambda_l1']
params['lambda_l2'] = best_params['lambda_l2']
params['min_split_gain'] = best_params['min_split_gain']

print(best_params)

### 训练
params['learning_rate'] = 0.01
gbm = lgb.train(
          params,                     # 参数字典
          lgb_train,                  # 训练集
          valid_sets=lgb_eval,        # 验证集
          num_boost_round=950,       # 迭代次数
          early_stopping_rounds=50    # 早停次数
          )

# 线下预测
print("线下预测")
preds_offline = gbm.predict(offline_test_X,num_iteration=gbm.best_iteration) 
offline_lgb_predict = np.argmax(preds_offline, axis=1)

offline_lgb_f1 = metrics.f1_score(offline_test.current_service, offline_lgb_predict, average=None)
print('lightgbm f1 value:',np.mean(offline_lgb_f1))

# 线上预测
print("线上预测")

lgb_pred_value = gbm.predict(online_test_X,num_iteration=gbm.best_iteration) 
lgb_predict = np.argmax(lgb_pred_value, axis=1)
predict_label = service_label.inverse_transform(lgb_predict)
result = pd.DataFrame(data=test['user_id'].values, columns = ['user_id'])
result['current_service'] = predict_label
result.to_csv('../data/baseline1003csv',index=False)

end_time = time.time()
print('Runing time:', end_time-start_time)