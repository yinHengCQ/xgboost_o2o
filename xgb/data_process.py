import pandas as pd
import json,os
from datetime import date
import numpy as np


"""
根据测试集中的字段User_id,Merchant_id,Coupon_id,Discount_rate,Distance,Date_received建立特征数据：

一.Discount_rate：
    discount：换算成小数折扣
    is_man：是否是满减
        如果是满减券：
        max_man：满减情况下，至少要达到的金额
    discount_use_ratio：该类满减券的实际使用率

二.Date_received：
    week_day：换算成星期数
    # week_day_use：该星期数的实际优惠券使用率
    month_day：换算成当月第几天
    # month_day_use：该月各天的实际优惠券使用率
    
三.Distance：
    distance_consume_ratio：单一商户距离的消费几率

四.User_id：
    user_consume：用户消费的次数
    user_use_ratio：用户使用优惠券消费的比率
    user_receive_use_ratio：用户领取和使用优惠券的比重
    user_merchant_consume_total：用户消费单一商户的次数
    user_merchant_use_ratio：用户使用优惠券消费单一商户的比率
    user_distance_consume_ratio：用户在单一距离商户的消费几率   
    user_receive_use_gap：用户领取到使用优惠券的平均时间间隔
    user_receive_use_gap_max：用户领取到使用优惠券的最大时间间隔

五.Merchant_id：
    merchant_consume_total：各商户的被消费总数
    merchant_use_ratio：各商户被使用优惠券消费的几率
    merchant_distance_consume_ratio：各商户在单一距离下被消费的几率   
    # merchant_week_consume_ratio：各商户在单一星期下被消费的几率
    # merchant_month_consume_ratio：各商户在单一月份天数下被消费的几率
    
    merchant_coupon_type_count：各商户优惠券的种类
    
六.Coupon_id：
    # coupon_use_ratio：各优惠券被使用的几率
    



"""


def build_discount_feature(data_file="../data/ccf_offline_stage1_train.csv"):
    t1=pd.read_csv(data_file,dtype=str)
    t1=t1[pd.notna(t1["Discount_rate"])]

    def __s1(t1_origin):
        def __calc_discount_rate(discount_rate):
            if str(discount_rate).__contains__(":"):
                s = discount_rate.split(':')
                if len(s) == 1:
                    return float(s[0])
                else:
                    return str(round((1.0 - float(s[1]) / float(s[0])), 4))
            else:
                return str(round(float(discount_rate), 4))
        t1_origin["discount"] = t1_origin.apply(lambda row: __calc_discount_rate(row["Discount_rate"]), axis=1)
        return t1_origin

    def __s2(t1_origin):
        def __is_man(Discount_rate):
            if Discount_rate.__contains__(":"):
                return "1"
            else:
                return "0"
        t1_origin["is_man"] = t1_origin.apply(lambda row: __is_man(row["Discount_rate"]), axis=1)
        return t1_origin

    def __s3(t1_origin):
        def __max_man(Discount_rate):
            if Discount_rate.__contains__(":"):
                return str(Discount_rate).split(":")[0]
            else:
                return "-1"
        t1_origin["max_man"] = t1_origin.apply(lambda row: __max_man(row["Discount_rate"]), axis=1)
        return t1_origin

    def __s4(t1_origin):
        if os.path.exists("./json_data/discount_use_ratio.json"):
            with open("./json_data/discount_use_ratio.json","r",encoding="utf-8") as f:
                temp_use_dict=json.load(f)
        else:
            t1 = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            t1_copy = t1[["Discount_rate"]]
            t1_copy["count"] = 1
            t1_receive_out = t1_copy.groupby("Discount_rate").agg("sum").reset_index()
            temp_receive_dict = dict()
            for i in t1_receive_out.values:
                temp_receive_dict[i[0]] = int(i[1])
            t1_use = t1[pd.notna(t1["Date"])]
            t1_use["count"] = 1
            t1_man_use_out = t1_use.groupby("Discount_rate").agg("sum").reset_index()
            temp_use_dict = dict()
            for i in t1_man_use_out.values:
                temp_use_dict[i[0]] = str(round(int(i[1]) / temp_receive_dict[i[0]], 4))
            with open("./json_data/discount_use_ratio.json","w",encoding="utf-8") as f:
                json.dump(temp_use_dict, f, ensure_ascii=False)

        def __get_use_ratio(Discount_rate):
            if Discount_rate in temp_use_dict:
                return temp_use_dict[Discount_rate]
            return "-1"

        t1_origin["discount_use_ratio"] = t1_origin.apply(lambda row: __get_use_ratio(row["Discount_rate"]), axis=1)
        return t1_origin

    t1=__s1(t1)
    t1=__s2(t1)
    t1=__s3(t1)
    t1=__s4(t1)
    if data_file=="../data/ccf_offline_stage1_train.csv":
        t1.to_csv("./feature_data/t1.csv", index=None)
    else:
        t1.to_csv("./test_data/t1.csv", index=None)


def build_date_received_feature(data_file="./feature_data/t1.csv"):
    t2 = pd.read_csv(data_file, dtype=str)

    def __s1(t2_origin):
        def _calc_weekday(date_str):
            return str(int(date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])).weekday()) + 1)
        t2_origin["week_day"] = t2_origin.apply(lambda row: _calc_weekday(row["Date_received"]), axis=1)
        return t2_origin

    def __s2(t2_origin):
        if os.path.exists("./json_data/week_day_use.json"):
            with open("./json_data/week_day_use.json","r",encoding="utf-8") as f:
                t2_use_dict=json.load(f)
        else:
            t2 = pd.read_csv("./feature_data/t1.csv", dtype=str)
            t2_copy = t2[["week_day"]]
            t2_copy["count"] = 1
            t2_receive_out = t2_copy.groupby("week_day").agg("sum").reset_index()
            t2_receive_dict = dict()
            for i in t2_receive_out.values:
                t2_receive_dict[i[0]] = i[1]
            t2_use = t2[pd.notna(t2["Date"])]
            t2_use = t2_use[["week_day"]]
            t2_use["count"] = 1
            t2_use_out = t2_use.groupby("week_day").agg("sum").reset_index()
            t2_use_dict = dict()
            for i in t2_use_out.values:
                t2_use_dict[i[0]] = str(round(int(i[1]) / t2_receive_dict[i[0]], 4))
            with open("./json_data/week_day_use.json","w",encoding="utf-8") as f:
                json.dump(t2_use_dict, f, ensure_ascii=False)

        def __get_use_ratio(week_day):
            return t2_use_dict[week_day]
        t2_origin["week_day_use"] = t2_origin.apply(lambda row: __get_use_ratio(row["week_day"]), axis=1)
        return t2_origin

    def __s3(t2_origin):
        def _calc_month_day(date_str):
            return str(int(date_str[6:8]))
        t2_origin["month_day"] = t2_origin.apply(lambda row: _calc_month_day(row["Date_received"]), axis=1)
        return t2_origin

    def __s4(t2_origin):
        if os.path.exists("./json_data/month_day_use.json"):
            with open("./json_data/month_day_use.json","r",encoding="utf-8") as f:
                t2_use_dict=json.load(f)
        else:
            t2 = pd.read_csv("./feature_data/t1.csv", dtype=str)
            t2_copy = t2[["month_day"]]
            t2_copy["count"] = 1
            t2_receive_out = t2_copy.groupby("month_day").agg("sum").reset_index()
            t2_receive_dict = dict()
            for i in t2_receive_out.values:
                t2_receive_dict[i[0]] = i[1]
            t2_use = t2[pd.notna(t2["Date"])]
            t2_use = t2_use[["month_day"]]
            t2_use["count"] = 1
            t2_use_out = t2_use.groupby("month_day").agg("sum").reset_index()
            t2_use_dict = dict()
            for i in t2_use_out.values:
                t2_use_dict[i[0]] = str(round(int(i[1]) / t2_receive_dict[i[0]], 4))
            with open("./json_data/month_day_use.json","w",encoding="utf-8") as f:
                json.dump(t2_use_dict, f, ensure_ascii=False)

        def __get_use_ratio(month_day):
            return t2_use_dict[month_day]
        t2_origin["month_day_use"] = t2_origin.apply(lambda row: __get_use_ratio(row["month_day"]), axis=1)
        return t2_origin

    t2=__s1(t2)
    # __s2()
    t2=__s3(t2)
    # __s4()
    if data_file=="./feature_data/t1.csv":
        t2.to_csv("./feature_data/t2.csv", index=None)
    else:
        t2.to_csv("./test_data/t2.csv", index=None)


def build_distance_feature(data_file="./feature_data/t2.csv"):
    if os.path.exists("./json_data/distance_consume_ratio.json"):
        with open("./json_data/distance_consume_ratio.json", "r", encoding="utf-8") as f:
            distance_consume_ratio_dict = json.load(f)
    else:
        origin_data=pd.read_csv("../data/ccf_offline_stage1_train.csv",dtype=str)
        consume_data=origin_data[pd.notna(origin_data["Date"])]
        consume_total_size=consume_data.size
        consume_data=consume_data[["Distance"]]
        consume_data["Distance"]=consume_data["Distance"].replace(np.nan,"-1")
        consume_data["count"]=1
        consume_data_out=consume_data.groupby("Distance").agg("sum").reset_index()
        distance_consume_ratio_dict=dict()
        for i in consume_data_out.values:
            distance_consume_ratio_dict[i[0]]=str(round((int(i[1])/consume_total_size),4))
        with open("./json_data/distance_consume_ratio.json", "w", encoding="utf-8") as f:
            json.dump(distance_consume_ratio_dict,f,ensure_ascii=False)

    def __get_distance_consume_ratio(distance):
        if distance in distance_consume_ratio_dict:
            return distance_consume_ratio_dict[distance]
        return "0"

    t2=pd.read_csv(data_file,dtype=str)
    t2["Distance"]=t2["Distance"].replace(np.nan, "-1")
    t2["distance_consume_ratio"] = t2.apply(lambda row: __get_distance_consume_ratio(row["Distance"]), axis=1)

    if data_file=="./feature_data/t2.csv":
        t2.to_csv("./feature_data/t3.csv", index=None)
    else:
        t2.to_csv("./test_data/t3.csv", index=None)


def build_user_feature(data_file="./feature_data/t3.csv"):
    t3=pd.read_csv(data_file,dtype=str)

    def __s12(t3_origin):
        if os.path.exists("./json_data/user_consume.json") and os.path.exists("./json_data/user_use_ratio.json"):
            with open("./json_data/user_consume.json", "r", encoding="utf-8") as f:
                user_consume_dict = json.load(f)
            with open("./json_data/user_use_ratio.json", "r", encoding="utf-8") as f:
                user_use_ratio_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            consume_data = origin_data[pd.notna(origin_data["Date"])]
            user_data = consume_data[pd.notna(consume_data["Coupon_id"])]
            consume_data = consume_data[["User_id"]]
            consume_data["count"] = 1
            consume_data_out = consume_data.groupby("User_id").agg("sum").reset_index()
            user_consume_dict = dict()
            for i in consume_data_out.values:
                user_consume_dict[i[0]] = int(i[1])
            with open("./json_data/user_consume.json", "w", encoding="utf-8") as f:
                json.dump(user_consume_dict, f, ensure_ascii=False)
            user_data = user_data[["User_id"]]
            user_data["count"] = 1
            user_data_out = user_data.groupby("User_id").agg("sum").reset_index()
            user_use_ratio_dict = dict()
            for i in user_data_out.values:
                user_use_ratio_dict[i[0]] = str(round((int(i[1]) / user_consume_dict[i[0]]), 4))
            with open("./json_data/user_use_ratio.json", "w", encoding="utf-8") as f:
                json.dump(user_use_ratio_dict, f, ensure_ascii=False)

        def __get_user_consume(user_id):
            if user_id in user_consume_dict:
                return user_consume_dict[user_id]
            return "0"

        def __get_user_use_ratio(user_id):
            if user_id in user_use_ratio_dict:
                return user_use_ratio_dict[user_id]
            return "0"

        t3_origin["user_consume"] = t3_origin.apply(lambda row: __get_user_consume(row["User_id"]), axis=1)
        t3_origin["user_use_ratio"] = t3_origin.apply(lambda row: __get_user_use_ratio(row["User_id"]), axis=1)
        return t3_origin

    def __s3(t3_origin):
        if os.path.exists("./json_data/user_receive_use_ratio.json"):
            with open("./json_data/user_receive_use_ratio.json", "r", encoding="utf-8") as f:
                user_receive_use_ratio_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            receive_data = origin_data[pd.notna(origin_data["Date_received"])]
            receive_use_data = origin_data[pd.notna(origin_data["Date_received"]) & pd.notna(origin_data["Date"])]
            receive_data = receive_data[["User_id"]]
            receive_data["count"] = 1
            receive_use_data = receive_use_data[["User_id"]]
            receive_use_data["count"] = 1
            receive_data_out = receive_data.groupby("User_id").agg("sum").reset_index()
            receive_use_data_out = receive_use_data.groupby("User_id").agg("sum").reset_index()
            receive_data_dict = dict()
            for i in receive_data_out.values:
                receive_data_dict[i[0]] = int(i[1])
            user_receive_use_ratio_dict = dict()
            for i in receive_use_data_out.values:
                user_receive_use_ratio_dict[i[0]] = str(round((receive_data_dict[i[0]] / i[1]), 4))
            with open("./json_data/user_receive_use_ratio.json", "w", encoding="utf-8") as f:
                json.dump(user_receive_use_ratio_dict, f, ensure_ascii=False)

        def __get_user_receive_use_ratio(user_id):
            if user_id in user_receive_use_ratio_dict:
                return user_receive_use_ratio_dict[user_id]
            return "0"

        t3_origin["user_receive_use_ratio"] = t3_origin.apply(lambda row: __get_user_receive_use_ratio(row["User_id"]), axis=1)
        return t3_origin

    def __s45(t3_origin):
        if os.path.exists("./json_data/user_merchant_consume_total.json") and os.path.exists("./json_data/user_merchant_use_ratio.json"):
            with open("./json_data/user_merchant_consume_total.json", "r", encoding="utf-8") as f:
                user_merchant_consume_total_dict = json.load(f)
            with open("./json_data/user_merchant_use_ratio.json", "r", encoding="utf-8") as f:
                user_merchant_use_ratio_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            consume_data = origin_data[pd.notna(origin_data["Date"])]
            use_data=origin_data[pd.notna(origin_data["Date"])&pd.notna(origin_data["Coupon_id"])]
            use_data=use_data[["User_id", "Merchant_id"]]
            consume_data = consume_data[["User_id", "Merchant_id"]]
            consume_data["count"] = 1
            use_data["count"] = 1
            consume_data_out = consume_data.groupby(["User_id", "Merchant_id"]).agg("sum").reset_index()
            use_data_out=use_data.groupby(["User_id", "Merchant_id"]).agg("sum").reset_index()
            user_merchant_consume_total_dict = dict()
            user_merchant_use_ratio_dict=dict()
            for i in consume_data_out.values:
                user_merchant_consume_total_dict["{}_{}".format(i[0], i[1])] = i[2]
            with open("./json_data/user_merchant_consume_total.json", "w", encoding="utf-8") as f:
                json.dump(user_merchant_consume_total_dict, f, ensure_ascii=False)
            for i in use_data_out.values:
                k="{}_{}".format(i[0], i[1])
                user_merchant_use_ratio_dict[k]=str(round((i[2]/user_merchant_consume_total_dict[k]),4))
            with open("./json_data/user_merchant_use_ratio.json", "w", encoding="utf-8") as f:
                json.dump(user_merchant_use_ratio_dict, f, ensure_ascii=False)

        def __get_user_merchant_consume_total(user_id, merchant_id):
            k="{}_{}".format(user_id, merchant_id)
            if k in user_merchant_consume_total_dict:
                return str(user_merchant_consume_total_dict[k])
            return "0"

        def __get_user_merchant_use_ratio(user_id, merchant_id):
            k = "{}_{}".format(user_id, merchant_id)
            if k in user_merchant_consume_total_dict:
                return str(round((int(user_merchant_consume_total_dict["{}_{}".format(user_id, merchant_id)])/int(user_merchant_consume_total_dict[k])),4))
            return "0"

        t3_origin["user_merchant_consume_total"] = t3_origin.apply(lambda row: __get_user_merchant_consume_total(row["User_id"], row["Merchant_id"]), axis=1)
        t3_origin["user_merchant_use_ratio"] = t3_origin.apply(lambda row: __get_user_merchant_use_ratio(row["User_id"], row["Merchant_id"]), axis=1)
        return t3_origin

    def __s6(t3_origin):
        if os.path.exists("./json_data/user_distance_consume_ratio.json"):
            with open("./json_data/user_distance_consume_ratio.json", "r", encoding="utf-8") as f:
                user_distance_consume_ratio_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            consume_data = origin_data[pd.notna(origin_data["Date"])]
            consume_data = consume_data[["User_id", "Distance"]]
            consume_data = consume_data.replace(np.nan, "-1")
            consume_data["count"] = 1
            consume_data_out = consume_data.groupby(["User_id", "Distance"]).agg("sum").reset_index()
            user_distance_consume_ratio_dict = dict()
            for i in consume_data_out.values:
                user_distance_consume_ratio_dict["{}_{}".format(i[0], i[1])] = str(i[2])
            with open("./json_data/user_distance_consume_ratio.json", "w", encoding="utf-8") as f:
                json.dump(user_distance_consume_ratio_dict, f, ensure_ascii=False)
        with open("./json_data/user_consume.json", "r", encoding="utf-8") as f:
            user_consume_dict = json.load(f)

        def __get_user_distance_consume_ratio(user_id, distance):
            k = "{}_{}".format(user_id, distance)
            if k in user_distance_consume_ratio_dict:
                return str(round((int(user_distance_consume_ratio_dict[k])/int(user_consume_dict[user_id])),4))
            return "0"

        t3_origin["user_distance_consume_ratio"] = t3_origin.apply(lambda row: __get_user_distance_consume_ratio(row["User_id"], row["Distance"]), axis=1)
        return t3_origin

    def __s78(t3_origin):
        if os.path.exists("./json_data/user_receive_use_gap.json") and os.path.exists("./json_data/user_receive_use_gap_max.json"):
            with open("./json_data/user_receive_use_gap.json", "r", encoding="utf-8") as f:
                user_receive_use_gap_dict = json.load(f)
            with open("./json_data/user_receive_use_gap_max.json", "r", encoding="utf-8") as f:
                user_receive_use_gap_max_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            consume_data = origin_data[pd.notna(origin_data["Date"])&pd.notna(origin_data["Date_received"])]
            def __calc_day_gap(day1, day2):
                return (date(int(day1[:4]), int(day1[4:6]), int(day1[6:])) - date(int(day2[:4]), int(day2[4:6]),int(day2[6:]))).days
            consume_data["day_gap"] = consume_data.apply(lambda row: __calc_day_gap(row["Date"], row["Date_received"]), axis=1)
            consume_data_new=consume_data[["User_id","day_gap"]]
            consume_data_new_out = consume_data_new.groupby(["User_id"]).agg(["max", "mean"]).reset_index()
            user_receive_use_gap_dict=dict()
            user_receive_use_gap_max_dict=dict()
            for i in consume_data_new_out.values:
                user_receive_use_gap_dict[i[0]]=str(round(i[2],4))
                user_receive_use_gap_max_dict[i[0]] = str(i[1])
            with open("./json_data/user_receive_use_gap.json", "w", encoding="utf-8") as f:
                json.dump(user_receive_use_gap_dict,f,ensure_ascii=False)
            with open("./json_data/user_receive_use_gap_max.json", "w", encoding="utf-8") as f:
                json.dump(user_receive_use_gap_max_dict, f, ensure_ascii=False)

        def __get_user_receive_use_gap(user_id):
            if user_id in user_receive_use_gap_dict:
                return user_receive_use_gap_dict[user_id]
            return "0"

        def __get_user_receive_use_gap_max(user_id):
            if user_id in user_receive_use_gap_max_dict:
                return user_receive_use_gap_max_dict[user_id]
            return "0"

        t3_origin["user_receive_use_gap"] = t3_origin.apply(lambda row: __get_user_receive_use_gap(row["User_id"]), axis=1)
        t3_origin["user_receive_use_gap_max"] = t3_origin.apply(lambda row: __get_user_receive_use_gap_max(row["User_id"]), axis=1)
        return t3_origin

    t3=__s12(t3)
    t3=__s3(t3)
    t3=__s45(t3)
    t3=__s6(t3)
    t3=__s78(t3)
    if data_file=="./feature_data/t3.csv":
        t3.to_csv("./feature_data/t4.csv", index=None)
    else:
        t3.to_csv("./test_data/t4.csv", index=None)


def build_merchant_feature(data_file="./feature_data/t4.csv"):
    t4 = pd.read_csv(data_file, dtype=str)

    def __s12(t4_origin):
        if os.path.exists("./json_data/merchant_consume_total.json") and os.path.exists("./json_data/merchant_use_ratio.json"):
            with open("./json_data/merchant_consume_total.json", "r", encoding="utf-8") as f:
                merchant_consume_total_dict = json.load(f)
            with open("./json_data/merchant_use_ratio.json", "r", encoding="utf-8") as f:
                merchant_use_ratio_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            consume_data = origin_data[pd.notna(origin_data["Date"])]
            consume_data = consume_data[["Merchant_id"]]
            consume_data["count"] = 1
            consume_data_out = consume_data.groupby("Merchant_id").agg("sum").reset_index()
            merchant_consume_total_dict = dict()
            for i in consume_data_out.values:
                merchant_consume_total_dict[i[0]] = i[1]
            with open("./json_data/merchant_consume_total.json", "w", encoding="utf-8") as f:
                json.dump(merchant_consume_total_dict, f, ensure_ascii=False)
            use_data=origin_data[pd.notna(origin_data["Date"])&pd.notna(origin_data["Coupon_id"])]
            use_data=use_data[["Merchant_id"]]
            use_data["count"] = 1
            use_data_out = use_data.groupby("Merchant_id").agg("sum").reset_index()
            merchant_use_ratio_dict=dict()
            for i in use_data_out.values:
                merchant_use_ratio_dict[i[0]]=str(round((i[1]/merchant_consume_total_dict[i[0]]),4))
            with open("./json_data/merchant_use_ratio.json", "w", encoding="utf-8") as f:
                json.dump(merchant_use_ratio_dict, f, ensure_ascii=False)

        def __get_merchant_consume_total(merchant_id):
            if merchant_id in merchant_consume_total_dict:
                return str(merchant_consume_total_dict[merchant_id])
            return "0"
        def __get_merchant_use_ratio(merchant_id):
            if merchant_id in merchant_use_ratio_dict:
                return str(merchant_use_ratio_dict[merchant_id])
            return "0"

        t4_origin["merchant_consume_total"] = t4_origin.apply(lambda row: __get_merchant_consume_total(row["Merchant_id"]), axis=1)
        t4_origin["merchant_use_ratio"] = t4_origin.apply(lambda row: __get_merchant_use_ratio(row["Merchant_id"]), axis=1)
        return t4_origin

    def __s3(t4_origin):
        with open("./json_data/merchant_consume_total.json", "r", encoding="utf-8") as f:
            merchant_consume_total_dict = json.load(f)
        if os.path.exists("./json_data/merchant_distance_consume_ratio.json"):
            with open("./json_data/merchant_distance_consume_ratio.json", "r", encoding="utf-8") as f:
                merchant_distance_consume_ratio_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            consume_data = origin_data[pd.notna(origin_data["Date"])]
            consume_data = consume_data[["Merchant_id","Distance"]]
            consume_data["count"] = 1
            consume_data_out = consume_data.groupby(["Merchant_id","Distance"]).agg("sum").reset_index()
            merchant_distance_consume_ratio_dict = dict()
            for i in consume_data_out.values:
                merchant_distance_consume_ratio_dict["{}_{}".format(i[0],i[1])]=str(round((i[2]/int(merchant_consume_total_dict[i[0]])),4))
            with open("./json_data/merchant_distance_consume_ratio.json", "w", encoding="utf-8") as f:
                json.dump(merchant_distance_consume_ratio_dict, f, ensure_ascii=False)

        def __get_merchant_distance_consume_ratio(merchant_id, distance):
            k = "{}_{}".format(merchant_id, distance)
            if k in merchant_distance_consume_ratio_dict:
                return merchant_distance_consume_ratio_dict[k]
            return "0"

        t4_origin["merchant_distance_consume_ratio"] = t4_origin.apply(lambda row: __get_merchant_distance_consume_ratio(row["Merchant_id"], row["Distance"]), axis=1)
        return t4_origin

    def __s4(t4_origin):
        if os.path.exists("./json_data/merchant_week_consume_ratio.json"):
            with open("./json_data/merchant_week_consume_ratio.json", "r", encoding="utf-8") as f:
                merchant_week_consume_ratio_dict = json.load(f)
        else:
            with open("./json_data/merchant_consume_total.json", "r", encoding="utf-8") as f:
                merchant_consume_total_dict = json.load(f)
            t2=pd.read_csv("./feature_data/t2.csv",dtype=str)
            t2=t2[pd.notna(t2["Date"])]
            t2=t2[["Merchant_id","week_day"]]
            t2["count"]=1
            t2_out=t2.groupby(["Merchant_id","week_day"]).agg("sum").reset_index()
            merchant_week_consume_ratio_dict=dict()
            for i in t2_out.values:
                merchant_week_consume_ratio_dict["{}_{}".format(i[0],i[1])]=str(round((i[2]/int(merchant_consume_total_dict[i[0]])),4))
            with open("./json_data/merchant_week_consume_ratio.json", "w", encoding="utf-8") as f:
                json.dump(merchant_week_consume_ratio_dict, f, ensure_ascii=False)

        def __get_merchant_week_consume_ratio(merchant_id, week_day):
            k = "{}_{}".format(merchant_id, week_day)
            if k in merchant_week_consume_ratio_dict:
                return merchant_week_consume_ratio_dict[k]
            return "0"

        t4_origin["merchant_week_consume_ratio"] = t4_origin.apply(lambda row: __get_merchant_week_consume_ratio(row["Merchant_id"], row["week_day"]), axis=1)
        return t4_origin

    def __s5(t4_origin):
        if os.path.exists("./json_data/merchant_month_consume_ratio.json"):
            with open("./json_data/merchant_month_consume_ratio.json", "r", encoding="utf-8") as f:
                merchant_month_consume_ratio_dict = json.load(f)
        else:
            with open("./json_data/merchant_consume_total.json", "r", encoding="utf-8") as f:
                merchant_consume_total_dict = json.load(f)
            t2=pd.read_csv("./feature_data/t2.csv",dtype=str)
            t2=t2[pd.notna(t2["Date"])]
            t2=t2[["Merchant_id","month_day"]]
            t2["count"]=1
            t2_out=t2.groupby(["Merchant_id","month_day"]).agg("sum").reset_index()
            merchant_month_consume_ratio_dict=dict()
            for i in t2_out.values:
                merchant_month_consume_ratio_dict["{}_{}".format(i[0],i[1])]=str(round((i[2]/int(merchant_consume_total_dict[i[0]])),4))
            with open("./json_data/merchant_month_consume_ratio.json", "w", encoding="utf-8") as f:
                json.dump(merchant_month_consume_ratio_dict, f, ensure_ascii=False)

        def __get_merchant_month_consume_ratio(merchant_id, month_day):
            k = "{}_{}".format(merchant_id, month_day)
            if k in merchant_month_consume_ratio_dict:
                return merchant_month_consume_ratio_dict[k]
            return "0"

        t4_origin["merchant_month_consume_ratio"] = t4_origin.apply(lambda row: __get_merchant_month_consume_ratio(row["Merchant_id"], row["month_day"]), axis=1)
        return t4_origin

    def __s6(t4_origin):
        if os.path.exists("./json_data/merchant_coupon_type_count.json"):
            with open("./json_data/merchant_coupon_type_count.json", "r", encoding="utf-8") as f:
                merchant_coupon_type_count_dict = json.load(f)
        else:
            origin_data = pd.read_csv("../data/ccf_offline_stage1_train.csv", dtype=str)
            data = origin_data[pd.notna(origin_data["Coupon_id"])]
            data = data[["Merchant_id", "Coupon_id"]]
            data.drop_duplicates(inplace=True)
            data["count"] = 1
            data_out = data.groupby(["Merchant_id"]).agg("sum").reset_index()
            merchant_coupon_type_count_dict=dict()
            for i in data_out.values:
                merchant_coupon_type_count_dict[i[0]]=str(i[1])
            with open("./json_data/merchant_coupon_type_count.json", "w", encoding="utf-8") as f:
                json.dump(merchant_coupon_type_count_dict, f, ensure_ascii=False)

        def __get_merchant_coupon_type_count(merchant_id):
            if merchant_id in merchant_coupon_type_count_dict:
                return merchant_coupon_type_count_dict[merchant_id]
            return "0"

        t4_origin["merchant_coupon_type_count"] = t4_origin.apply(lambda row: __get_merchant_coupon_type_count(row["Merchant_id"]), axis=1)
        return t4_origin


    t4=__s12(t4)
    t4=__s3(t4)
    # t4 = __s4(t4)
    # t4 = __s5(t4)
    t4 = __s6(t4)
    if data_file=="./feature_data/t4.csv":
        t4.to_csv("./feature_data/t5.csv", index=None)
    else:
        t4.to_csv("./test_data/t5.csv", index=None)




def generate_label(data_file="./feature_data/t5.csv"):
    t6 = pd.read_csv(data_file, dtype=str)

    def __check_user_action_dict_json_file():
        if not os.path.exists("./json_data/user_action_dict.json"):
            off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv', dtype=str)
            use_data = off_train[pd.notna(off_train.Date)]

            user_action_dict = dict()
            for i in use_data.values:
                k = "{}_{}".format(i[0], i[1])
                if k not in user_action_dict:
                    user_action_dict[k] = []
                user_action_dict[k].append("{}_{}_{}_{}_{}".format(i[2], i[3], i[4], i[5], i[6]))

            with open("./json_data/user_action_dict.json", "w", encoding="utf-8") as f:
                json.dump(user_action_dict, f, ensure_ascii=False)

    __check_user_action_dict_json_file()
    with open("./json_data/user_action_dict.json", "r", encoding="utf-8") as f:
        user_action_dict=json.load(f)

    def _cala_day(day1,day2):
        day1,day2=str(day1),str(day2)
        return (date(int(day1[:4]), int(day1[4:6]), int(day1[6:]))-date(int(day2[:4]), int(day2[4:6]), int(day2[6:]))).days

    def _get_label(date_received,user_id,merchant_id):
        u_m_k="{}_{}".format(user_id,merchant_id)
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

    train_data6 = t6[pd.notna(t6.Date_received)]
    train_data6["label"] = train_data6.apply(lambda row: _get_label(row['Date_received'], row['User_id'] ,row['Merchant_id']), axis=1)
    train_data6.to_csv("./feature_data/t6.csv", index=None)



# build_discount_feature()
# build_date_received_feature()
# build_distance_feature()
# build_user_feature()
build_merchant_feature()
generate_label()



# build_discount_feature(data_file="../data/ccf_offline_stage1_test_revised.csv")
# build_date_received_feature(data_file="./test_data/t1.csv")
# build_distance_feature(data_file="./test_data/t2.csv")
# build_user_feature(data_file="./test_data/t3.csv")
build_merchant_feature(data_file="./test_data/t4.csv")
