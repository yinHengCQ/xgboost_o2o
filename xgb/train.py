import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



# train_data=pd.read_csv("./feature_data/train_data7.csv",dtype=str)
# # train_data.drop(["user_id","merchant_id","coupon_id","discount_rate","u_m_k","date_received","date"],axis=1,inplace=True)
# train_data=train_data.replace(np.nan,"-1")
# train_data.label.replace("-1","0",inplace=True)
# train_data.drop_duplicates(inplace=True)
#
#
# test_data=pd.read_csv("./test_data/test_data6.csv",dtype=str)
# # test_data.drop(["user_id","merchant_id","coupon_id","discount_rate","u_m_k","date_received"],axis=1,inplace=True)
# test_data=test_data.replace(np.nan,"-1")
#
#
# train_data_x=train_data[['distance', 'discount_total_ratio', 'discount_use_ratio', 'is_man', 'discount', 'weekday', 'weekday_total_ratio', 'weekday_consume_ratio', 'u_total',
# 						  'u_consume', 'u_receive', 'u_consume_with_coupon', 'u_merchant_consume', 'u_merchant_receive', 'u_merchant_consume_with_coupon',
# 						  'u_receive_discount_type_count1', 'u_receive_discount_type_count2', 'u_consume_with_discount_type_count1',
# 						  'u_consume_with_discount_type_count2', 'm_total_ratio', 'm_consume_ratio', 'm_receive_ratio', 'm_consume_with_coupon_ratio', 'm_receive_discount_type_count1_ratio',
# 						  'm_receive_discount_type_count2_ratio', 'm_consume_with_discount_type_count1_ratio', 'm_consume_with_discount_type_count2_ratio', 'c_total_ratio', 'c_consume_ratio',
# 						  'd_total_ratio', 'd_consume_ratio', 'd_receive_ratio', 'd_consume_with_coupon_ratio']]
#
#
# dataset_test_x=test_data[['distance', 'discount_total_ratio', 'discount_use_ratio', 'is_man', 'discount', 'weekday', 'weekday_total_ratio', 'weekday_consume_ratio', 'u_total',
# 						  'u_consume', 'u_receive', 'u_consume_with_coupon', 'u_merchant_consume', 'u_merchant_receive', 'u_merchant_consume_with_coupon',
# 						  'u_receive_discount_type_count1', 'u_receive_discount_type_count2', 'u_consume_with_discount_type_count1',
# 						  'u_consume_with_discount_type_count2', 'm_total_ratio', 'm_consume_ratio', 'm_receive_ratio', 'm_consume_with_coupon_ratio', 'm_receive_discount_type_count1_ratio',
# 						  'm_receive_discount_type_count2_ratio', 'm_consume_with_discount_type_count1_ratio', 'm_consume_with_discount_type_count2_ratio', 'c_total_ratio', 'c_consume_ratio',
# 						  'd_total_ratio', 'd_consume_ratio', 'd_receive_ratio', 'd_consume_with_coupon_ratio']]
#
#
# train_data_y=train_data[["label"]]
#
#
# train_data_x=train_data_x.astype(float)
# train_data_y=train_data_y.astype(float)
# dataset_test_x=dataset_test_x.astype(float)
#
#
# dataset=xgb.DMatrix(train_data_x,train_data_y)
# dataset_test=xgb.DMatrix(dataset_test_x)
#
# params={'booster':'gbtree',
# 	    'objective': 'rank:pairwise',
# 	    'eval_metric':'auc',
# 	    'gamma':0.1,
# 	    'min_child_weight':1.1,
# 	    'max_depth':5,
# 	    'lambda':10,
# 	    'subsample':0.7,
# 	    'colsample_bytree':0.7,
# 	    'colsample_bylevel':0.7,
# 	    'eta': 0.01,
# 	    'tree_method':'exact',
# 	    'seed':0,
# 	    }
#
# model = xgb.train(params,dataset,num_boost_round=3500,evals=[(dataset,'train')])
#
# test_data['label'] = model.predict(dataset_test)
# test_data.label = MinMaxScaler().fit_transform(test_data['label'].values.reshape(-1, 1))
# test_data.to_csv("xgb_preds.csv",index=None,header=None)
#
# # save feature score
# feature_score = model.get_fscore()
# feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
# fs = []
# for (key, value) in feature_score:
# 	fs.append("{0},{1}\n".format(key, value))
#
# with open('xgb_feature_score.csv', 'w') as f:
# 	f.writelines("feature,score\n")
# 	f.writelines(fs)


train_data=pd.read_csv("./feature_data/train_data7.csv",dtype=str)
train_data=train_data.drop(['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'date_received', 'discount_total', 'discount_use',
				   'weekday_total', 'weekday_consume', 'u_m_k', 'm_total', 'm_consume', 'm_receive', 'm_consume_with_coupon',
				   'm_receive_discount_type_count1', 'm_receive_discount_type_count2', 'm_consume_with_discount_type_count1',
				   'm_consume_with_discount_type_count2', 'c_total', 'c_consume', 'd_total', 'd_consume', 'd_receive',
				   'd_consume_with_coupon','date'],axis=1)
train_data=train_data.replace(np.nan,"-1")
train_data.label.replace("-1","0",inplace=True)
train_data.drop_duplicates(inplace=True)


test_data=pd.read_csv("./test_data/test_data6.csv",dtype=str)
test_data=test_data.drop(['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'date_received', 'discount_total', 'discount_use',
				   'weekday_total', 'weekday_consume', 'u_m_k', 'm_total', 'm_consume', 'm_receive', 'm_consume_with_coupon',
				   'm_receive_discount_type_count1', 'm_receive_discount_type_count2', 'm_consume_with_discount_type_count1',
				   'm_consume_with_discount_type_count2', 'c_total', 'c_consume', 'd_total', 'd_consume', 'd_receive',
				   'd_consume_with_coupon'],axis=1)
test_data=test_data.replace(np.nan,"-1")



train_data=train_data.astype(float)
test_data=test_data.astype(float)


train_xy,val = train_test_split(train_data, test_size = 0.3,random_state=1)
y = train_xy.label
X = train_xy.drop(['label'],axis=1)
val_y = val.label
val_X = val.drop(['label'],axis=1)

xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test_data)


params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    }

model = xgb.train(params,xgb_train,num_boost_round=3500,evals=[(xgb_train, 'train'),(xgb_val, 'val')])

test_data['label'] = model.predict(xgb_test)
test_data.label = MinMaxScaler().fit_transform(test_data['label'].values.reshape(-1, 1))
test_data.to_csv("xgb_preds.csv",index=None,header=None)

# save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []
for (key, value) in feature_score:
	fs.append("{0},{1}\n".format(key, value))

with open('xgb_feature_score.csv', 'w') as f:
	f.writelines("feature,score\n")
	f.writelines(fs)