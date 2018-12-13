import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler


train_data=pd.read_csv("./feature_data/train_data.csv",dtype=str)
train_data.drop(["user_id","merchant_id","coupon_id","discount_rate","use_data_k","date_received"],axis=1,inplace=True)
train_data=train_data.replace(np.nan,"-1")
train_data.label.replace("-1","0",inplace=True)


train_data.drop_duplicates(inplace=True)


test_data=pd.read_csv("./feature_data/test_data.csv",dtype=str)
# test_data.drop(["user_id","merchant_id","coupon_id","discount_rate","date_received"],axis=1,inplace=True)
test_data=test_data.replace(np.nan,"-1")


train_data_x=train_data[["distance","discount_rate_new","_is_man","weekday",'user_merchant_relation','merchant_relation']]
dataset_test_x=test_data[["distance","discount_rate_new","_is_man","weekday",'user_merchant_relation','merchant_relation']]
train_data_y=train_data[["label"]]


train_data_x=train_data_x.astype(float)
train_data_y=train_data_y.astype(float)
dataset_test_x=dataset_test_x.astype(float)


dataset=xgb.DMatrix(train_data_x,train_data_y)
dataset_test=xgb.DMatrix(dataset_test_x)

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

model = xgb.train(params,dataset,num_boost_round=3500,evals=[(dataset,'train')])

test_data['label'] = model.predict(dataset_test)
test_data.label = MinMaxScaler().fit_transform(test_data['label'].values.reshape(-1, 1))
test_data.to_csv("xgb_preds.csv",index=None,header=None)
