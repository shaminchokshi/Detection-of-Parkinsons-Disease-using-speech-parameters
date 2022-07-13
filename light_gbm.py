
import lightgbm as lgb
import numpy as np
train_data = lgb.Dataset("data.csv")
data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(data, label=label)
validation_data = train_data.create_valid('validation.svm')
train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
w = np.random.rand(500, )
train_data = lgb.Dataset(data, label=label, weight=w)
param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = 'auc'
param['metric'] = ['auc', 'binary_logloss']
num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
bst.save_model('model.txt')
json_model = bst.dump_model()
bst = lgb.Booster(model_file='model.txt')  # init model
lgb.cv(param, train_data, num_round, nfold=5)
data = np.random.rand(7, 10)
ypred = bst.predict(data)
ypred = bst.predict(data, num_iteration=bst.best_iteration)
