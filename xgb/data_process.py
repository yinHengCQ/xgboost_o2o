import pandas as pd
import json,os
from datetime import date


def _calc_discount_rate(discount_rate):
    if discount_rate.__contains__(":"):
        s = discount_rate.split(':')
        if len(s) == 1:
            return float(s[0])
        else:
            return 1.0 - float(s[1]) / float(s[0])
    else:
        return discount_rate


def _calc_weekday(a):
    return str(date(int(a[:4]), int(a[4:6]), int(a[6:])).weekday())


def _is_man(discount_rate):
    if discount_rate.__contains__(":"):
        return "1"
    else:
        return "0"


def _generrate_key(user_id,merchant_id):
    return "{}_{}".format(user_id,merchant_id)


def generate_user_json():
    off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv', dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    use_data = off_train[pd.notna(off_train.date)]

    user_action_dict = dict()
    for i in use_data.values:
        k = "{}_{}".format(i[0], i[1])
        if k not in user_action_dict:
            user_action_dict[k] = []
        user_action_dict[k].append("{}_{}_{}_{}_{}".format(i[2], i[3], i[4], i[5], i[6]))

    with open("./feature_data/user_action_dict.json", "w", encoding="utf-8") as f:
        json.dump(user_action_dict, f, ensure_ascii=False)


def generate_user_merchant_relation_json():
    off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv', dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    use_data = off_train[pd.notna(off_train.date)]

    user_merchant_relation_dict = dict()
    for i in use_data.values:
        k = "{}_{}".format(i[0], i[1])
        if k not in user_merchant_relation_dict:
            user_merchant_relation_dict[k] = 1
        user_merchant_relation_dict[k]+=1

    with open("./feature_data/user_merchant_relation_dict.json", "w", encoding="utf-8") as f:
        json.dump(user_merchant_relation_dict, f, ensure_ascii=False)


def build_train_data():
    with open("./feature_data/user_action_dict.json", "r", encoding="utf-8") as f:
        user_action_dict=json.load(f)

    with open("./feature_data/user_merchant_relation_dict.json", "r", encoding="utf-8") as f:
        user_merchant_relation_dict=json.load(f)

    def _cala_day(day1,day2):
        return (date(int(day1[:4]), int(day1[4:6]), int(day1[6:]))-date(int(day2[:4]), int(day2[4:6]), int(day2[6:]))).days

    def _get_label(coupon_id,date_received,use_data_k):
        if use_data_k not in user_action_dict:
            return -1
        vs=user_action_dict[use_data_k]
        for v in vs:
            temp=v.split("_")
            data_coupon_id=temp[0]
            data_date = temp[-1]
            day_span = _cala_day(data_date, date_received)
            if day_span <= 15 and day_span >= 0:
                if data_coupon_id != "nan":
                    return 1
                else:
                    return 0
        return -1

    def _evaluate_user_merchant_relation(use_data_k):
        if use_data_k not in user_merchant_relation_dict:
            return 0
        else:
            return user_merchant_relation_dict[use_data_k]

    if os.path.exists("./feature_data/label_data.csv"):
        label_data= pd.read_csv('./feature_data/label_data.csv', dtype=str)
        label_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date','use_data_k','label']
        label_data["discount_rate_new"]=label_data.apply(lambda row: _calc_discount_rate(row['discount_rate']),axis=1)
        label_data["_is_man"] = label_data.apply(lambda row: _is_man(row['discount_rate']),axis=1)
        label_data["weekday"]=label_data.apply(lambda row: _calc_weekday(row['date_received']),axis=1)
        label_data['user_merchant_relation']=label_data.apply(lambda row: _evaluate_user_merchant_relation(row['use_data_k']),axis=1)
        label_data.drop("date",axis=1,inplace=True)
        label_data.to_csv("./feature_data/train_data.csv",index=None)
    else:
        if os.path.exists("./feature_data/received_data.csv"):
            received_data=pd.read_csv("./feature_data/received_data.csv",dtype=str)
            received_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date','use_data_k']
            received_data["label"] = received_data.apply(lambda row: _get_label(row['coupon_id'], row['date_received'], row['use_data_k']), axis=1)
            received_data.to_csv("./feature_data/label_data.csv",index=None)

            label_data = pd.read_csv('./feature_data/label_data.csv', dtype=str)
            label_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date', 'use_data_k', 'label']
            label_data["discount_rate_new"] = label_data.apply(lambda row: _calc_discount_rate(row['discount_rate']),axis=1)
            label_data["_is_man"] = label_data.apply(lambda row: _is_man(row['discount_rate']), axis=1)
            label_data["weekday"] = label_data.apply(lambda row: _calc_weekday(row['date_received']), axis=1)
            label_data['user_merchant_relation'] = label_data.apply(lambda row: _evaluate_user_merchant_relation(row['use_data_k']), axis=1)
            label_data.drop("date", axis=1, inplace=True)
            label_data.to_csv("./feature_data/train_data.csv", index=None)
        else:
            off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv', dtype=str)
            off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date']
            received_data = off_train[pd.notna(off_train.date_received)]
            received_data["use_data_k"] = received_data.apply( lambda row: _generrate_key(row['user_id'], row['merchant_id']), axis=1)
            received_data.to_csv("./feature_data/received_data.csv", index=None)

            received_data=pd.read_csv("./feature_data/received_data.csv",dtype=str)
            received_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date','use_data_k']
            received_data["label"] = received_data.apply(lambda row: _get_label(row['coupon_id'], row['date_received'], row['use_data_k']), axis=1)
            received_data.to_csv("./feature_data/label_data.csv",index=None)

            label_data = pd.read_csv('./feature_data/label_data.csv', dtype=str)
            label_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date', 'use_data_k', 'label']
            label_data["discount_rate_new"] = label_data.apply(lambda row: _calc_discount_rate(row['discount_rate']),axis=1)
            label_data["_is_man"] = label_data.apply(lambda row: _is_man(row['discount_rate']), axis=1)
            label_data["weekday"] = label_data.apply(lambda row: _calc_weekday(row['date_received']), axis=1)
            label_data['user_merchant_relation'] = label_data.apply(lambda row: _evaluate_user_merchant_relation(row['use_data_k']), axis=1)
            label_data.drop("date", axis=1, inplace=True)
            label_data.to_csv("./feature_data/train_data.csv", index=None)


def build_test_data():
    with open("./feature_data/user_merchant_relation_dict.json", "r", encoding="utf-8") as f:
        user_merchant_relation_dict=json.load(f)

    def _evaluate_user_merchant_relation(use_data_k):
        if use_data_k not in user_merchant_relation_dict:
            return 0
        else:
            return user_merchant_relation_dict[use_data_k]

    off_test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv', dtype=str)
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

    off_test["discount_rate_new"] = off_test.apply(lambda row: _calc_discount_rate(row['discount_rate']), axis=1)
    off_test["_is_man"] = off_test.apply(lambda row: _is_man(row['discount_rate']), axis=1)
    off_test["weekday"] = off_test.apply(lambda row: _calc_weekday(row['date_received']), axis=1)
    off_test["use_data_k"] = off_test.apply(lambda row: _generrate_key(row['user_id'], row['merchant_id']),axis=1)
    off_test['user_merchant_relation'] = off_test.apply(lambda row: _evaluate_user_merchant_relation(row['use_data_k']), axis=1)

    off_test.to_csv("./feature_data/test_data.csv", index=None)


def xgb_result2submit():
    xgb_result=pd.read_csv("xgb_preds.csv",dtype=str)
    xgb_result.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received',"discount_rate_new", "_is_man", "weekday", "use_data_k", "user_merchant_relation", "label"]

    submit=xgb_result[["user_id","coupon_id","date_received","label"]]
    submit.to_csv("sample_submission.csv", index=None,header=None)


# generate_user_json()

# generate_user_merchant_relation_json()

# build_train_data()

# build_test_data()

xgb_result2submit()

