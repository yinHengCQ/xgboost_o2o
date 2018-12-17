import pandas as pd
import json,os
from datetime import date
import numpy as np


"""
根据测试集中的字段User_id,Merchant_id,Coupon_id,Discount_rate,Distance,Date_received建立特征数据：

一.Discount_rate
    discount_total：各discount_rate出现的总次数，
    discount_total_ratio：各discount_rate出现比例，
    discount_use：各discount_rate有消费的次数，
    discount_use_ratio：各discount_rate有消费比例，
    is_man：是否是满减，
    discount：换算成统一的折扣率表示。

二.Date_received
    weekday：date_received换算成星期数，
    weekday_total：各date_received星期数出现的总次数，
    weekday_total_ratio：各date_received星期数出现的比例，
    weekday_consume：各date_received星期数有被消费的次数，
    weekday_consume_ratio：各date_received星期数有被消费的比例。
    
三.User_id：
    u_total：各user_id出现的总次数，
    u_consume：各user_id有消费的次数，
    u_receive：各user_id领取优惠券的次数，
    u_consume_with_coupon：各user_id有使用优惠券消费的次数，
    # u_consume_distance(1-11)：各user_id消费时、11种Distance出现的总次数（稀疏矩阵表示），
    # u_receive_distance(1-11)：各user_id领取优惠券时、11种Distance出现的总次数（稀疏矩阵表示），
    u_merchant_consume：各user_id消费各种merchant_id的次数，
    u_merchant_receive：各user_id领取各种merchant_id优惠券的次数，
    u_merchant_consume_with_coupon：各user_id消费各种merchant_id并使用了优惠券的次数，
    u_receive_discount_type_count(1-2)：各user_id领取特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    u_consume_with_discount_type_count(1-2)：各user_id领取并使用特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    u_receive_weekday_count(1-7)：各user_id领取优惠券时间的7种星期数次数（稀疏矩阵表示），
    u_consume_with_coupon_weekday_count(1-7)：各user_id领取并使用优惠券时间的7种星期数次数（稀疏矩阵表示）。

四.Merchant_id:
    m_total：各merchant_id出现的总次数，
    m_total_ratio：各merchant_id出现的比例，
    m_consume：各merchant_id有被消费的次数，
    m_consume_ratio：各merchant_id有被消费的比例，
    m_receive：各merchant_id被领取优惠券的次数，
    m_receive_ratio：各merchant_id被领取优惠券的比例，
    m_consume_with_coupon：各merchant_id有被使用优惠券消费的次数，
    m_consume_with_coupon_ratio：各merchant_id有被使用优惠券消费的比例，
    m_consume_distance_count(1-11)：各merchant_id被消费时、11种Distance出现的总次数（稀疏矩阵表示），
    m_consume_distance_count_ratio(1-11)：各merchant_id被消费时、11种Distance出现的比例（稀疏矩阵表示），
    m_receive_discount_type_count(1-2)：各merchant_id被领取特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    m_receive_discount_type_count_ratio(1-2)：各merchant_id被领取特定折扣类型（2种）的比例（稀疏矩阵表示），
    m_consume_with_coupon_distance_type_count(1-2)各merchant_id被领取并使用特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    m_consume_with_coupon_distance_type_count_ratio(1-2)各merchant_id被领取并使用特定折扣类型（2种）的比例（稀疏矩阵表示）。
    
五.Coupon_id
    c_total：各coupon_id出现的总次数，
    c_total_ratio：各coupon_id出现的比例，
    c_consume：各coupon_id有被消费的次数，
    c_consume_ratio：各coupon_id有被消费的比例。
    
六.Distance
    d_total：各distance出现的总次数，
    d_total_ratio：各distance出现的比例，
    d_consume：各distance有消费的次数，
    d_consume_ratio：各distance有消费的比例，
    d_receive：各distance被领取优惠券的次数，
    d_receive_ratio：各distance被领取优惠券的比例，
    d_consume_with_coupon：各distance有消费并使用了优惠券的次数，
    d_consume_with_coupon_ratio：各distance有消费并使用了优惠券的比例。

"""


def __common_get_value_from_json_dict(json_dict, k):
    if k in json_dict:
        return json_dict[k]
    else:
        return 0


def _calc_weekday(date_str):
    date_str = str(date_str)
    if date_str=='nan':
        return "0"
    return str(int(date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])).weekday())+1)


def __common_generate_double_k(column1,column2):
    return "{}_{}".format(column1,column2)


def __common_generate_total_json(column_name,json_file,data_file="../data/ccf_offline_stage1_train.csv"):
    off_train = pd.read_csv(data_file, dtype=str)
    if data_file=="../data/ccf_offline_stage1_train.csv":
        off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train[column_name])]
    temp=off_train[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"]=str(off_train.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_consume_json(column_name,json_file,data_file="../data/ccf_offline_stage1_train.csv"):
    off_train = pd.read_csv(data_file, dtype=str)
    if data_file=="../data/ccf_offline_stage1_train.csv":
        off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train['date'])&pd.notna(off_train[column_name])]
    temp=off_train[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"]=str(off_train.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_receive_json(column_name,json_file):
    off_train = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train['date_received'])&pd.notna(off_train[column_name])]
    temp=off_train[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"]=str(off_train.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_consume_with_coupon_json(column_name,json_file):
    off_train = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train['date'])&pd.notna(off_train['coupon_id'])&pd.notna(off_train[column_name])]
    temp=off_train[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"]=str(off_train.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_double_consume_json(column_name1,column_name2,json_file):
    off_train = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train['date'])&pd.notna(off_train[column_name1])&pd.notna(off_train[column_name2])]
    off_train["double_k"]=off_train.apply(lambda row:__common_generate_double_k(row[column_name1],row[column_name2]),axis=1)
    temp=off_train[["double_k"]]
    temp["count"]=1
    out=temp.groupby("double_k").agg("sum").reset_index()
    out_dict=dict()
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_double_receive_json(column_name1,column_name2,json_file):
    off_train = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train['date_received'])&pd.notna(off_train[column_name1])&pd.notna(off_train[column_name2])]
    off_train["double_k"]=off_train.apply(lambda row:__common_generate_double_k(row[column_name1],row[column_name2]),axis=1)
    temp=off_train[["double_k"]]
    temp["count"]=1
    out=temp.groupby("double_k").agg("sum").reset_index()
    out_dict=dict()
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_double_consume_with_coupon_json(column_name1,column_name2,json_file):
    off_train = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_train=off_train[pd.notna(off_train['date'])&pd.notna(off_train['coupon_id'])&pd.notna(off_train[column_name1])&pd.notna(off_train[column_name2])]
    off_train["double_k"]=off_train.apply(lambda row:__common_generate_double_k(row[column_name1],row[column_name2]),axis=1)
    temp=off_train[["double_k"]]
    temp["count"]=1
    out=temp.groupby("double_k").agg("sum").reset_index()
    out_dict=dict()
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_receive_discount_man_type_count_json(column_name,json_file):
    train_data1 = pd.read_csv("./feature_data/train_data1.csv", dtype=str)
    train_data1['is_man'].astype(str)
    train_data1=train_data1[pd.notna(train_data1['date_received'])&(train_data1['is_man']=="1")&pd.notna(train_data1[column_name])]
    temp=train_data1[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"]=str(train_data1.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_receive_discount_kou_type_count_json(column_name,json_file):
    train_data1 = pd.read_csv("./feature_data/train_data1.csv", dtype=str)
    train_data1['is_man'].astype(str)
    train_data1=train_data1[pd.notna(train_data1['date_received'])&(train_data1['is_man']=="0")&pd.notna(train_data1[column_name])]
    temp=train_data1[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"] = str(train_data1.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_consume_with_discount_man_type_count_json(column_name,json_file):
    train_data1 = pd.read_csv("./feature_data/train_data1.csv", dtype=str)
    train_data1['is_man'].astype(str)
    train_data1=train_data1[pd.notna(train_data1['coupon_id'])&pd.notna(train_data1['date'])&(train_data1['is_man']=="1")&pd.notna(train_data1[column_name])]
    temp=train_data1[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"] = str(train_data1.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


def __common_generate_consume_with_discount_kou_type_count_json(column_name,json_file):
    train_data1 = pd.read_csv("./feature_data/train_data1.csv", dtype=str)
    train_data1['is_man'].astype(str)
    train_data1=train_data1[pd.notna(train_data1['coupon_id'])&pd.notna(train_data1['date'])&(train_data1['is_man']=="0")&pd.notna(train_data1[column_name])]
    temp=train_data1[[column_name]]
    temp["count"]=1
    out=temp.groupby(column_name).agg("sum").reset_index()
    out_dict=dict()
    out_dict["total_size"] = str(train_data1.size)
    for row in out.values:
        out_dict[row[0]]=row[1]
    with open(json_file,"w",encoding="utf-8") as f:
        json.dump(out_dict,f,ensure_ascii=False)


"""
检查或构造各json数据
"""
def __check_discount_rate_feature_json_file():
    if not os.path.exists("./json_data/discount_total.json"):
        __common_generate_total_json("discount_rate","./json_data/discount_total.json")
    if not os.path.exists("./json_data/discount_use.json"):
        __common_generate_consume_json("discount_rate", "./json_data/discount_use.json")


def __check_date_received_feature_json_file():
    if not os.path.exists("./json_data/weekday_total.json"):
        __common_generate_total_json("weekday","./json_data/weekday_total.json",data_file="./feature_data/train_data2_temp.csv")
    if not os.path.exists("./json_data/weekday_consume.json"):
        __common_generate_consume_json("weekday", "./json_data/weekday_consume.json",data_file="./feature_data/train_data2_temp.csv")


def __check_user_id_feature_json_file():
    if not os.path.exists("./json_data/u_total.json"):
        __common_generate_total_json("user_id","./json_data/u_total.json")
    if not os.path.exists("./json_data/u_consume.json"):
        __common_generate_consume_json("user_id", "./json_data/u_consume.json")
    if not os.path.exists("./json_data/u_receive.json"):
        __common_generate_receive_json("user_id", "./json_data/u_receive.json")
    if not os.path.exists("./json_data/u_consume_with_coupon.json"):
        __common_generate_consume_with_coupon_json("user_id", "./json_data/u_consume_with_coupon.json")
    if not os.path.exists("./json_data/u_merchant_consume.json"):
        __common_generate_double_consume_json("user_id", "merchant_id", "./json_data/u_merchant_consume.json")
    if not os.path.exists("./json_data/u_merchant_receive.json"):
        __common_generate_double_receive_json("user_id", "merchant_id", "./json_data/u_merchant_receive.json")
    if not os.path.exists("./json_data/u_merchant_consume_with_coupon.json"):
        __common_generate_double_consume_with_coupon_json("user_id", "merchant_id", "./json_data/u_merchant_consume_with_coupon.json")
    if not os.path.exists("./json_data/u_receive_discount_type_count1.json"):
        __common_generate_receive_discount_man_type_count_json("user_id", "./json_data/u_receive_discount_type_count1.json")
    if not os.path.exists("./json_data/u_receive_discount_type_count2.json"):
        __common_generate_receive_discount_kou_type_count_json("user_id", "./json_data/u_receive_discount_type_count2.json")
    if not os.path.exists("./json_data/u_consume_with_discount_type_count1.json"):
        __common_generate_consume_with_discount_man_type_count_json("user_id", "./json_data/u_consume_with_discount_type_count1.json")
    if not os.path.exists("./json_data/u_consume_with_discount_type_count2.json"):
        __common_generate_consume_with_discount_kou_type_count_json("user_id", "./json_data/u_consume_with_discount_type_count2.json")


def __check_merchant_id_feature_json_file():
    if not os.path.exists("./json_data/m_total.json"):
        __common_generate_total_json("merchant_id","./json_data/m_total.json")
    if not os.path.exists("./json_data/m_consume.json"):
        __common_generate_consume_json("merchant_id", "./json_data/m_consume.json")
    if not os.path.exists("./json_data/m_receive.json"):
        __common_generate_receive_json("merchant_id", "./json_data/m_receive.json")
    if not os.path.exists("./json_data/m_consume_with_coupon.json"):
        __common_generate_consume_with_coupon_json("merchant_id", "./json_data/m_consume_with_coupon.json")
    if not os.path.exists("./json_data/m_receive_discount_type_count1.json"):
        __common_generate_receive_discount_man_type_count_json("merchant_id", "./json_data/m_receive_discount_type_count1.json")
    if not os.path.exists("./json_data/m_receive_discount_type_count2.json"):
        __common_generate_receive_discount_kou_type_count_json("merchant_id", "./json_data/m_receive_discount_type_count2.json")
    if not os.path.exists("./json_data/m_consume_with_discount_type_count1.json"):
        __common_generate_consume_with_discount_man_type_count_json("merchant_id", "./json_data/m_consume_with_discount_type_count1.json")
    if not os.path.exists("./json_data/m_consume_with_discount_type_count2.json"):
        __common_generate_consume_with_discount_kou_type_count_json("merchant_id", "./json_data/m_consume_with_discount_type_count2.json")


def __check_coupon_id_feature_json_file():
    if not os.path.exists("./json_data/c_total.json"):
        __common_generate_total_json("coupon_id","./json_data/c_total.json")
    if not os.path.exists("./json_data/c_consume.json"):
        __common_generate_consume_json("coupon_id", "./json_data/c_consume.json")


def __check_distance_feature_json_file():
    if not os.path.exists("./json_data/d_total.json"):
        __common_generate_total_json("distance","./json_data/d_total.json")
    if not os.path.exists("./json_data/d_consume.json"):
        __common_generate_consume_json("distance", "./json_data/d_consume.json")
    if not os.path.exists("./json_data/d_receive.json"):
        __common_generate_receive_json("distance", "./json_data/d_receive.json")
    if not os.path.exists("./json_data/d_consume_with_coupon.json"):
        __common_generate_consume_with_coupon_json("distance", "./json_data/d_consume_with_coupon.json")


def __check_user_action_dict_json_file():
    if not os.path.exists("./json_data/user_action_dict.json"):
        off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv', dtype=str)
        off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received','date']
        use_data = off_train[pd.notna(off_train.date)]

        user_action_dict = dict()
        for i in use_data.values:
            k = "{}_{}".format(i[0], i[1])
            if k not in user_action_dict:
                user_action_dict[k] = []
            user_action_dict[k].append("{}_{}_{}_{}_{}".format(i[2], i[3], i[4], i[5], i[6]))

        with open("./json_data/user_action_dict.json", "w", encoding="utf-8") as f:
            json.dump(user_action_dict, f, ensure_ascii=False)


"""
生成特征数据
"""
def generate_discount_rate_feature(in_file="../data/ccf_offline_stage1_train.csv",out_file="./feature_data/train_data1.csv"):
    """
    discount_total：各discount_rate出现的总次数，
    discount_total_ratio：各discount_rate出现比例，
    discount_use：各discount_rate有消费的次数，
    discount_use_ratio：各discount_rate有消费比例，
    is_man：是否是满减，
    discount：换算成统一的折扣率表示。
    """
    if os.path.exists(out_file):
        return
    __check_discount_rate_feature_json_file()

    off_train = pd.read_csv(in_file, dtype=str)
    if in_file=="../data/ccf_offline_stage1_train.csv":
        off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    else:
        off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

    with open("./json_data/discount_total.json", "r", encoding="utf-8") as f:
        discount_total_dict=json.load(f)
        discount_total_dict_size=int(discount_total_dict["total_size"])
        off_train["discount_total"]=off_train.apply(lambda row:__common_get_value_from_json_dict(discount_total_dict,row["discount_rate"]),axis=1)
        off_train["discount_total_ratio"] = off_train.discount_total.apply(lambda row: int(row) / discount_total_dict_size)

    with open("./json_data/discount_use.json", "r", encoding="utf-8") as f:
        discount_use_dict=json.load(f)
        discount_use_dict_size=int(discount_use_dict["total_size"])
        off_train["discount_use"] = off_train.apply(lambda row: __common_get_value_from_json_dict(discount_use_dict, row["discount_rate"]), axis=1)
        off_train["discount_use_ratio"] = off_train.discount_use.apply(lambda row: int(row) / discount_use_dict_size)

    def __is_man(discount_rate):
        if str(discount_rate).__contains__(":"):
            return 1
        else:
            return 0
    off_train["is_man"] = off_train.apply(lambda row: __is_man(row["discount_rate"]), axis=1)

    def _calc_discount_rate(discount_rate):
        if str(discount_rate).__contains__(":"):
            s = discount_rate.split(':')
            if len(s) == 1:
                return float(s[0])
            else:
                return 1.0 - float(s[1]) / float(s[0])
        else:
            return discount_rate
    off_train["discount"] = off_train.apply(lambda row: _calc_discount_rate(row["discount_rate"]), axis=1)

    off_train.to_csv(out_file,index=None)


def generate_date_received_feature(in_file="./feature_data/train_data1.csv",out_file="./feature_data/train_data2.csv",temp_file="./feature_data/train_data2_temp.csv"):
    """
    weekday：date_received换算成星期数，
    weekday_total：各date_received星期数出现的总次数，
    weekday_total_ratio：各date_received星期数出现的比例，
    weekday_consume：各date_received星期数有被消费的次数，
    weekday_consume_ratio：各date_received星期数有被消费的比例。
    """
    if os.path.exists(out_file):
        return
    if not os.path.exists(temp_file):
        train_data1 = pd.read_csv(in_file, dtype=str)
        train_data1["weekday"]=train_data1.apply(lambda row:_calc_weekday(row["date_received"]),axis=1)
        train_data1.to_csv(temp_file,index=None)

    __check_date_received_feature_json_file()
    train_data2_temp = pd.read_csv(temp_file, dtype=str)
    with open("./json_data/weekday_total.json", "r", encoding="utf-8") as f:
        weekday_total_dict=json.load(f)
        weekday_total_dict_size=int(weekday_total_dict["total_size"])
        train_data2_temp["weekday_total"]=train_data2_temp.apply(lambda row:__common_get_value_from_json_dict(weekday_total_dict,row["weekday"]),axis=1)
        train_data2_temp["weekday_total_ratio"] = train_data2_temp.weekday_total.apply(lambda row: int(row) / weekday_total_dict_size)

    with open("./json_data/weekday_consume.json", "r", encoding="utf-8") as f:
        weekday_consume_dict=json.load(f)
        weekday_consume_dict_size=int(weekday_consume_dict["total_size"])
        train_data2_temp["weekday_consume"] = train_data2_temp.apply(lambda row: __common_get_value_from_json_dict(weekday_consume_dict, row["weekday"]), axis=1)
        train_data2_temp["weekday_consume_ratio"] = train_data2_temp.weekday_consume.apply(lambda row: int(row) / weekday_consume_dict_size)

    train_data2_temp.to_csv(out_file,index=None)


def generate_user_id_feature(in_file="./feature_data/train_data2.csv",out_file="./feature_data/train_data3.csv"):
    """
    u_total：各user_id出现的总次数，
    u_consume：各user_id有消费的次数，
    u_receive：各user_id领取优惠券的次数，
    u_consume_with_coupon：各user_id有使用优惠券消费的次数，
    # u_consume_distance(1-11)：各user_id消费时、11种Distance出现的总次数（稀疏矩阵表示），
    # u_receive_distance(1-11)：各user_id领取优惠券时、11种Distance出现的总次数（稀疏矩阵表示），
    u_merchant_consume：各user_id消费各种merchant_id的次数，
    u_merchant_receive：各user_id领取各种merchant_id优惠券的次数，
    u_merchant_consume_with_coupon：各user_id消费各种merchant_id并使用了优惠券的次数，
    u_receive_discount_type_count(1-2)：各user_id领取特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    u_consume_with_discount_type_count(1-2)：各user_id领取并使用特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    # u_receive_weekday_count(1-7)：各user_id领取优惠券时间的7种星期数次数（稀疏矩阵表示），
    # u_consume_with_coupon_weekday_count(1-7)：各user_id领取并使用优惠券时间的7种星期数次数（稀疏矩阵表示）。
    """
    if os.path.exists(out_file):
        return
    __check_user_id_feature_json_file()

    train_data2 = pd.read_csv(in_file, dtype=str)
    # off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    train_data2["u_m_k"] = train_data2.apply(lambda row: __common_generate_double_k(row["user_id"], row["merchant_id"]),axis=1)

    with open("./json_data/u_total.json", "r", encoding="utf-8") as f:
        u_total_dict=json.load(f)
        train_data2["u_total"]=train_data2.apply(lambda row:__common_get_value_from_json_dict(u_total_dict,row["user_id"]),axis=1)

    with open("./json_data/u_consume.json", "r", encoding="utf-8") as f:
        u_consume_dict=json.load(f)
        train_data2["u_consume"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_consume_dict, row["user_id"]), axis=1)

    with open("./json_data/u_receive.json", "r", encoding="utf-8") as f:
        u_receive_dict = json.load(f)
        train_data2["u_receive"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_receive_dict, row["user_id"]), axis=1)

    with open("./json_data/u_consume_with_coupon.json", "r", encoding="utf-8") as f:
        u_consume_with_coupon_dict = json.load(f)
        train_data2["u_consume_with_coupon"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_consume_with_coupon_dict, row["user_id"]), axis=1)

    with open("./json_data/u_merchant_consume.json", "r", encoding="utf-8") as f:
        u_merchant_consume_dict = json.load(f)
        train_data2["u_merchant_consume"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_merchant_consume_dict, row["u_m_k"]), axis=1)

    with open("./json_data/u_merchant_receive.json", "r", encoding="utf-8") as f:
        u_merchant_receive_dict = json.load(f)
        train_data2["u_merchant_receive"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_merchant_receive_dict, row["u_m_k"]), axis=1)

    with open("./json_data/u_merchant_consume_with_coupon.json", "r", encoding="utf-8") as f:
        u_merchant_consume_with_coupon_dict = json.load(f)
        train_data2["u_merchant_consume_with_coupon"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_merchant_consume_with_coupon_dict, row["u_m_k"]), axis=1)

    with open("./json_data/u_receive_discount_type_count1.json", "r", encoding="utf-8") as f:
        u_receive_discount_type_count1_dict = json.load(f)
        train_data2["u_receive_discount_type_count1"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_receive_discount_type_count1_dict, row["user_id"]), axis=1)

    with open("./json_data/u_receive_discount_type_count2.json", "r", encoding="utf-8") as f:
        u_receive_discount_type_count2_dict = json.load(f)
        train_data2["u_receive_discount_type_count2"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_receive_discount_type_count2_dict, row["user_id"]), axis=1)

    with open("./json_data/u_consume_with_discount_type_count1.json", "r", encoding="utf-8") as f:
        u_consume_with_discount_type_count1_dict = json.load(f)
        train_data2["u_consume_with_discount_type_count1"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_consume_with_discount_type_count1_dict, row["user_id"]), axis=1)

    with open("./json_data/u_consume_with_discount_type_count2.json", "r", encoding="utf-8") as f:
        u_consume_with_discount_type_count2_dict = json.load(f)
        train_data2["u_consume_with_discount_type_count2"] = train_data2.apply(lambda row: __common_get_value_from_json_dict(u_consume_with_discount_type_count2_dict, row["user_id"]), axis=1)

    train_data2.to_csv(out_file, index=None)


def generate_merchant_id_feature(in_file="./feature_data/train_data3.csv",out_file="./feature_data/train_data4.csv"):
    """
    m_total：各merchant_id出现的总次数，
    m_total_ratio：各merchant_id出现的比例，
    m_consume：各merchant_id有被消费的次数，
    m_consume_ratio：各merchant_id有被消费的比例，
    m_receive：各merchant_id被领取优惠券的次数，
    m_receive_ratio：各merchant_id被领取优惠券的比例，
    m_consume_with_coupon：各merchant_id有被使用优惠券消费的次数，
    m_consume_with_coupon_ratio：各merchant_id有被使用优惠券消费的比例，
    m_consume_distance_count(1-11)：各merchant_id被消费时、11种Distance出现的总次数（稀疏矩阵表示），
    m_consume_distance_count_ratio(1-11)：各merchant_id被消费时、11种Distance出现的比例（稀疏矩阵表示），
    m_receive_discount_type_count(1-2)：各merchant_id被领取特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    m_receive_discount_type_count_ratio(1-2)：各merchant_id被领取特定折扣类型（2种）的比例（稀疏矩阵表示），
    m_consume_with_coupon_distance_type_count(1-2)各merchant_id被领取并使用特定折扣类型（2种）的各自总次数（稀疏矩阵表示），
    m_consume_with_coupon_distance_type_count_ratio(1-2)各merchant_id被领取并使用特定折扣类型（2种）的比例（稀疏矩阵表示）。
    """
    if os.path.exists(out_file):
        return
    __check_merchant_id_feature_json_file()

    train_data3 = pd.read_csv(in_file, dtype=str)

    with open("./json_data/m_total.json", "r", encoding="utf-8") as f:
        m_total_dict=json.load(f)
        m_total_dict_size=int(m_total_dict["total_size"])
        train_data3["m_total"]=train_data3.apply(lambda row:__common_get_value_from_json_dict(m_total_dict,row["merchant_id"]),axis=1)
        train_data3["m_total_ratio"] = train_data3.m_total.apply(lambda row: int(row) / m_total_dict_size)

    with open("./json_data/m_consume.json", "r", encoding="utf-8") as f:
        m_consume_dict=json.load(f)
        m_consume_dict_size=int(m_consume_dict["total_size"])
        train_data3["m_consume"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_consume_dict, row["merchant_id"]), axis=1)
        train_data3["m_consume_ratio"] = train_data3.m_consume.apply(lambda row: int(row) / m_consume_dict_size)

    with open("./json_data/m_receive.json", "r", encoding="utf-8") as f:
        m_receive_dict = json.load(f)
        m_receive_dict_size=int(m_receive_dict["total_size"])
        train_data3["m_receive"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_receive_dict, row["merchant_id"]), axis=1)
        train_data3["m_receive_ratio"] = train_data3.m_receive.apply(lambda row: int(row) / m_receive_dict_size)

    with open("./json_data/m_consume_with_coupon.json", "r", encoding="utf-8") as f:
        m_consume_with_coupon_dict = json.load(f)
        m_consume_with_coupon_dict_size=int(m_consume_with_coupon_dict["total_size"])
        train_data3["m_consume_with_coupon"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_consume_with_coupon_dict, row["merchant_id"]), axis=1)
        train_data3["m_consume_with_coupon_ratio"] = train_data3.m_consume_with_coupon.apply(lambda row: int(row) / m_consume_with_coupon_dict_size)

    with open("./json_data/m_receive_discount_type_count1.json", "r", encoding="utf-8") as f:
        m_receive_discount_type_count1_dict = json.load(f)
        m_receive_discount_type_count1_dict_size=int(m_receive_discount_type_count1_dict["total_size"])
        train_data3["m_receive_discount_type_count1"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_receive_discount_type_count1_dict, row["merchant_id"]), axis=1)
        train_data3["m_receive_discount_type_count1_ratio"] = train_data3.m_receive_discount_type_count1.apply(lambda row: int(row) / m_receive_discount_type_count1_dict_size)

    with open("./json_data/m_receive_discount_type_count2.json", "r", encoding="utf-8") as f:
        m_receive_discount_type_count2_dict = json.load(f)
        m_receive_discount_type_count2_dict_size=int(m_receive_discount_type_count2_dict["total_size"])
        train_data3["m_receive_discount_type_count2"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_receive_discount_type_count2_dict, row["merchant_id"]), axis=1)
        train_data3["m_receive_discount_type_count2_ratio"] = train_data3.m_receive_discount_type_count2.apply(lambda row: int(row) / m_receive_discount_type_count2_dict_size)

    with open("./json_data/m_consume_with_discount_type_count1.json", "r", encoding="utf-8") as f:
        m_consume_with_discount_type_count1_dict = json.load(f)
        m_consume_with_discount_type_count1_dict_size=int(m_consume_with_discount_type_count1_dict["total_size"])
        train_data3["m_consume_with_discount_type_count1"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_consume_with_discount_type_count1_dict, row["merchant_id"]), axis=1)
        train_data3["m_consume_with_discount_type_count1_ratio"] = train_data3.m_consume_with_discount_type_count1.apply(lambda row: int(row) / m_consume_with_discount_type_count1_dict_size)

    with open("./json_data/m_consume_with_discount_type_count2.json", "r", encoding="utf-8") as f:
        m_consume_with_discount_type_count2_dict = json.load(f)
        m_consume_with_discount_type_count2_dict_size=int(m_consume_with_discount_type_count2_dict["total_size"])
        train_data3["m_consume_with_discount_type_count2"] = train_data3.apply(lambda row: __common_get_value_from_json_dict(m_consume_with_discount_type_count2_dict, row["merchant_id"]), axis=1)
        train_data3["m_consume_with_discount_type_count2_ratio"] = train_data3.m_consume_with_discount_type_count2.apply(lambda row: int(row) / m_consume_with_discount_type_count2_dict_size)

    train_data3.to_csv(out_file, index=None)


def generate_coupon_id_feature(in_file="./feature_data/train_data4.csv",out_file="./feature_data/train_data5.csv"):
    """
    c_total：各coupon_id出现的总次数，
    c_total_ratio：各coupon_id出现的比例，
    c_consume：各coupon_id有被消费的次数，
    c_consume_ratio：各coupon_id有被消费的比例。
    """
    if os.path.exists(out_file):
        return
    __check_coupon_id_feature_json_file()

    train_data4 = pd.read_csv(in_file, dtype=str)

    with open("./json_data/c_total.json", "r", encoding="utf-8") as f:
        c_total_dict=json.load(f)
        c_total_dict_size=int(c_total_dict["total_size"])
        train_data4["c_total"]=train_data4.apply(lambda row:__common_get_value_from_json_dict(c_total_dict,row["coupon_id"]),axis=1)
        train_data4["c_total_ratio"] = train_data4.c_total.apply(lambda row: int(row) / c_total_dict_size)

    with open("./json_data/c_consume.json", "r", encoding="utf-8") as f:
        c_consume_dict=json.load(f)
        c_consume_dict_size=int(c_consume_dict["total_size"])
        train_data4["c_consume"] = train_data4.apply(lambda row: __common_get_value_from_json_dict(c_consume_dict, row["coupon_id"]), axis=1)
        train_data4["c_consume_ratio"] = train_data4.c_consume.apply(lambda row: int(row) / c_consume_dict_size)

    train_data4.to_csv(out_file, index=None)


def generate_distance_feature(in_file="./feature_data/train_data5.csv",out_file="./feature_data/train_data6.csv"):
    """
    d_total：各distance出现的总次数，
    d_total_ratio：各distance出现的比例，
    d_consume：各distance有消费的次数，
    d_consume_ratio：各distance有消费的比例，
    d_receive：各distance被领取优惠券的次数，
    d_receive_ratio：各distance被领取优惠券的比例，
    d_consume_with_coupon：各distance有消费并使用了优惠券的次数，
    d_consume_with_coupon_ratio：各distance有消费并使用了优惠券的比例。
    """
    if os.path.exists(out_file):
        return
    __check_distance_feature_json_file()

    train_data5 = pd.read_csv(in_file, dtype=str)

    with open("./json_data/d_total.json", "r", encoding="utf-8") as f:
        d_total_dict=json.load(f)
        d_total_dict_size=int(d_total_dict["total_size"])
        train_data5["d_total"]=train_data5.apply(lambda row:__common_get_value_from_json_dict(d_total_dict,row["distance"]),axis=1)
        train_data5["d_total_ratio"] = train_data5.d_total.apply(lambda row: int(row) / d_total_dict_size)

    with open("./json_data/d_consume.json", "r", encoding="utf-8") as f:
        d_consume_dict=json.load(f)
        d_consume_dict_size=int(d_consume_dict["total_size"])
        train_data5["d_consume"] = train_data5.apply(lambda row: __common_get_value_from_json_dict(d_consume_dict, row["distance"]), axis=1)
        train_data5["d_consume_ratio"] = train_data5.d_consume.apply(lambda row: int(row) / d_consume_dict_size)

    with open("./json_data/d_receive.json", "r", encoding="utf-8") as f:
        d_receive_dict = json.load(f)
        d_receive_dict_size=int(d_receive_dict["total_size"])
        train_data5["d_receive"] = train_data5.apply(lambda row: __common_get_value_from_json_dict(d_receive_dict, row["distance"]), axis=1)
        train_data5["d_receive_ratio"] = train_data5.d_receive.apply(lambda row: int(row) / d_receive_dict_size)

    with open("./json_data/d_consume_with_coupon.json", "r", encoding="utf-8") as f:
        d_consume_with_coupon_dict = json.load(f)
        d_consume_with_coupon_dict_size=int(d_consume_with_coupon_dict["total_size"])
        train_data5["d_consume_with_coupon"] = train_data5.apply(lambda row: __common_get_value_from_json_dict(d_consume_with_coupon_dict, row["distance"]), axis=1)
        train_data5["d_consume_with_coupon_ratio"] = train_data5.d_consume_with_coupon.apply(lambda row: int(row) / d_consume_with_coupon_dict_size)

    train_data5.to_csv(out_file, index=None)


def generate_label(in_file="./feature_data/train_data6.csv",out_file="./feature_data/train_data7.csv"):
    if os.path.exists(out_file):
        return
    train_data6 = pd.read_csv(in_file, dtype=str)

    __check_user_action_dict_json_file()
    with open("./json_data/user_action_dict.json", "r", encoding="utf-8") as f:
        user_action_dict=json.load(f)

    def _cala_day(day1,day2):
        day1,day2=str(day1),str(day2)
        return (date(int(day1[:4]), int(day1[4:6]), int(day1[6:]))-date(int(day2[:4]), int(day2[4:6]), int(day2[6:]))).days

    def _get_label(date_received,u_m_k):
        if u_m_k not in user_action_dict:
            return -1
        vs=user_action_dict[u_m_k]
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

    train_data6 = train_data6[pd.notna(train_data6.date_received)]
    train_data6["label"] = train_data6.apply(lambda row: _get_label(row['date_received'], row['u_m_k']), axis=1)

    train_data6.to_csv(out_file, index=None)



"""
生成训练数据和验证数据
"""
def convert_train_data():
    generate_discount_rate_feature()
    generate_date_received_feature()
    generate_user_id_feature()
    generate_merchant_id_feature()
    generate_coupon_id_feature()
    generate_distance_feature()
    generate_label()

"""
生成测试数据
"""
def convert_test_data():
    generate_discount_rate_feature(in_file="../data/ccf_offline_stage1_test_revised.csv",out_file="./test_data/test_data1.csv")
    generate_date_received_feature(in_file="./test_data/test_data1.csv",out_file="./test_data/test_data2.csv",temp_file="./test_data/test_data2_temp.csv")
    generate_user_id_feature(in_file="./test_data/test_data2.csv",out_file="./test_data/test_data3.csv")
    generate_merchant_id_feature(in_file="./test_data/test_data3.csv",out_file="./test_data/test_data4.csv")
    generate_coupon_id_feature(in_file="./test_data/test_data4.csv",out_file="./test_data/test_data5.csv")
    generate_distance_feature(in_file="./test_data/test_data5.csv",out_file="./test_data/test_data6.csv")


convert_train_data()
convert_test_data()


# train_data=pd.read_csv("./feature_data/train_data7.csv",dtype=str)
# zheng=train_data[train_data["label"]=="1"]
# zhong=train_data[train_data["label"]=="0"]
# fu=train_data[train_data["label"]=="-1"]
#
# out=pd.concat([zheng.sample(n=10000),zhong.sample(n=1000),fu.sample(n=1000)],axis=0)
# out.to_csv("./feature_data/train_data8.csv", index=None)
