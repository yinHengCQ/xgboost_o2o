import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




train_data=pd.read_csv("./feature_data/t6.csv",dtype=str)
train_data=train_data.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'Date'],axis=1)
train_data=train_data.replace(np.nan,"-1")
train_data.label.replace("-1","0",inplace=True)
train_data.drop_duplicates(inplace=True)


test_data_origin=pd.read_csv("./test_data/t5.csv",dtype=str)
test_data_origin=test_data_origin.replace(np.nan,"-1")
test_data=test_data_origin.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received'],axis=1)




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
	    'min_child_weight':1.1,
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

test_data_origin['label'] = model.predict(xgb_test)
test_data_origin.label = MinMaxScaler().fit_transform(test_data_origin['label'].values.reshape(-1, 1))
test_data_origin.to_csv("xgb_preds.csv",index=None,header=None)


# save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []
for (key, value) in feature_score:
	fs.append("{0},{1}\n".format(key, value))

with open('xgb_feature_score.csv', 'w') as f:
	f.writelines("feature,score\n")
	f.writelines(fs)