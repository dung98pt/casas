import numpy as np 
import pandas as pd 
from datetime import time

"""Mấy tiện ích linh tinh chỗ này"""
def entity_list(in_df, conditions=[]):
    "Trả về list các entity_id, nhận vào df, condition là list string, nếu 1 string có trong entity_id thì thỏa mãn"
    result = []
    b = list(set(in_df.entity_id))
    if conditions:
        for i in b:
            for j in conditions:
                if j in i:
                    result.append(i)
                    break
    else:
        result = b
    return result

"""Kiểm tra"""
def check_valid(b_df, category="switch", entity_id=None):
    """
        Kiểm tra xem entity đó có bình thường ko, nếu ko đc sử dụng thường xuyên, hoặc chỉ là test thì ko có ý nghĩa ==> loại bỏ
    """
    if (b_df.last_updated.iloc[-1].date() - b_df.last_updated.iloc[0].date()).days < 7:
        return False
    if category=="switch": # {'on', 'off', 'unavailable', '0'}
        if len(b_df[b_df.state.isin(['on', 'off'])]) / len(b_df) < 0.35:
            return False
    elif category=="light": # {'unavailable', 'off'}
        if len(b_df[b_df.state.isin(['unavailable', 'off'])]) / len(b_df) > 0.65:
            return False
    elif category=="climate": # {'heat_cool', 'unknown', 'dry', 'fan_only', 'auto', 'cool', 'unavailable', 'off', 'heat'} 
        if len(b_df[b_df.state.isin(['unavailable', 'unknown'])]) / len(b_df) > 0.65:
            return False
    elif category=="media_player": # {'idle', nan, 'off', 'on', 'paused', 'playing', 'unavailable', 'unknown'}
        if len(b_df[b_df.state.isin(['unavailable', 'unknown', 'off', np.nan])]) / len(b_df) > 0.65:
            return False
    elif category=="automation": # {nan, 'off', 'on', 'unavailable'}
        if len(b_df[b_df.state.isin(['unavailable', np.nan])]) / len(b_df) > 0.65:
            return False
    elif category=="sensor":
        if "lux" in entity_id:
            if len(list(set(b_df.state)))<4 or len(b_df[b_df.state.isin(['unavailable', 'unknown'])]) / len(b_df) > 0.65:
                return False
        else:
            if len(list(set(b_df.state)))<8 or len(b_df[b_df.state.isin(['unavailable', 'unknown'])]) / len(b_df) > 0.65:
                return False
    return True

"""resample: Chỉ dành cho sensor nhiệt độ, độ ẩm"""
def transform_value(t, category):
    if t in ['unknown', "", np.nan, 'unavailable']:
        return 0
    if category=="t" and "-" in t:
        return 0
    if category=="h" and len(t)>2:
        return 0
    return float(t)

def change_state_un(n_df):
    result = []
    list_entity = list(set(n_df.entity_id))
    for entity in list_entity:
        en_df = n_df[n_df.entity_id == entity]
        new_state = []
        for state in en_df["state"]:
            if state in ['unavailable', np.nan, "unknown"]:
                new_state.append(["un"])
            else:
                new_state.append(state)
        en_df["state"] = new_state
        result.append(en_df)
    return result


range_humid = np.array([38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 
                        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])

range_temp = np.array([23.6, 22.9, 25.4, 25.3, 25.6, 24.2, 27.5, 21.4, 21.9, 27.7, 25.7,
                        24.6, 22.3, 24.7, 24. , 22.1, 20.2, 25.5, 23.1, 19.6, 20.9, 18.6,
                        20. , 21.2, 21.3, 22.2, 21.6, 21. , 27.2, 22.5, 24.1, 25.2, 26.9,
                        22. , 26.6, 22.7, 26. , 21.5, 27.3, 25.8, 20.4, 26.8, 27.6, 26.2,
                        23. , 25.9, 26.3, 27. , 26.7, 19. , 22.6, 25.1, 23.4, 20.5, 26.4,
                        27.1, 26.1, 24.9, 27.4, 25. , 23.2, 26.5, 23.9, 23.8, 21.7, 23.3,
                        24.4, 24.3, 18.7, 22.8, 24.5, 24.8, 23.5, 22.4, 23.7])

def resample_sensors(s_df, list_sensor, minute_sample=30, r_t=0.1, deploy_mode=False):
    # list_sensor = domain_dict["sensor"]
    if deploy_mode: 
        resample_value_t = lambda x: range_temp[np.argmin(np.abs(range_temp-x))]
        resample_value_h = lambda x: range_humid[np.argmin(np.abs(range_humid-x))]
    else:
        resample_value_t = lambda x: "{:.1f}".format((int(x*(1.0/r_t)+0.5))*r_t)
        resample_value_h = lambda x: int(x+0.5)
    result = []
    for entity in list_sensor:
        en_df = s_df[s_df.entity_id == entity]
        if "temperature" in entity:
            c = "t"
        elif "humidity" in entity: 
            c = "h"
        else:
            return result
        en_df["state"] = [transform_value(t, c) for t in en_df["state"]]
        en_df.state = en_df.state.replace(0.0, en_df.state.mean())
        frame = pd.DataFrame()
        frame['state'] = en_df.resample('{}T'.format(minute_sample))['state'].mean()
        frame = frame.dropna()   
        frame["date"] = [i.date() for i in frame.index]
        frame["time"] = [i.time() for i in frame.index]
        frame["entity_id"] = [entity]*len(frame)
        if c=='t':
            frame["state"] = [resample_value_t(i) for i in frame.state]
        if c=='h':
            frame["state"] = [resample_value_h(i) for i in frame.state]
        frame = frame[['date', 'time', 'entity_id', 'state']]
        result.append(frame)
    return result
"""Segment"""
def segment_time(df, mask=None):
    dates = list((set(df.date)))
    dates = sorted(dates)
    the_day_of_weeks = [i.weekday() for i in dates]
    t1 = time(8, 30, 00)
    t2 = time(11, 50, 00)
    t3 = time(13, 5, 00)
    t4 = time(18, 30, 00)
    t = [df[df.date==i] for i in dates] # số lượng mẫu theo từng ngày
    plus = [] # phần cộng dồn
    buff = 0
    a = [] # vector bắt đầu

    for i in t[:-1]:
        plus.append(buff + len(i))
        buff += len(i)

    for i in range(len(t)):
        b = []
        tt = t[i]
        b.append(0)
        b.append(np.sum([tt.time < t1]))
        b.append(np.sum([tt.time < t2]))
        b.append(np.sum([tt.time < t3]))
        b.append(np.sum([tt.time < t4]))
        a.append(b)
    a = np.array(a)
    a = a[:, 1:] # bỏ khoảng đầu tiên của mỗi ngày vì sẽ tính gộp từ tối hôm trước
    for i in range(len(plus)):
        a[i+1] += plus[i]

    # Thêm ngày cuối tuần là t7,cn vô
    if mask:
        assert len(mask)==len(dates)
        for i in [i for i, week_day in enumerate(the_day_of_weeks) if week_day==5 or week_day==6 or mask[i]==1]:
            if i==0:
                a[i] = np.array([0]*4)
            elif i==(len(the_day_of_weeks)-1):
                a[i] =  np.array([a[i][-1]]*4)
            else:
                a[i] =  np.array([-1]*4)
    else:
        for i in [i for i, week_day in enumerate(the_day_of_weeks) if week_day==5 or week_day==6]:
            if i==0:
                a[i] = np.array([0]*4)
            elif i==(len(the_day_of_weeks)-1):
                a[i] =  np.array([a[i][-1]]*4)
            else:
                a[i] =  np.array([-1]*4)
    b = a.copy()
    a = a.reshape(-1)
    a = [i for i in a if i != -1]
    return a, b

def labeling_activity(df, a):
    log = [""] *len(df)
    activity = [""] *len(df)
    for i in range(len(a)-1):
        if a[i+1] - a[i] > 4:
            log[a[i]] = "begin"
            log[a[i+1]-1] = "end"
            if i%4==0:
                ac = "Lam_viec"
            elif i%4==1:
                ac = "Nghi_trua"
            elif i%4==2:
                ac = "Lam_viec"
            elif i%4==3:
                ac = "Khong_lam_viec"
            activity[a[i]] = ac
            activity[a[i+1]-1] = ac
    df["log"] = log
    df["activity"] = activity
    return df

def labeling_activity_v2(df, a):
    activity = [""]*len(df)
    for i in range(len(a)-1):
        if a[i+1] - a[i] > 4:
            if i%4==0:
                ac = "Lam_viec"
            elif i%4==1:
                ac = "Nghi_trua"
            elif i%4==2:
                ac = "Lam_viec"
            elif i%4==3:
                ac = "Khong_lam_viec"
            activity[a[i]:a[i+1]] = [ac]*(a[i+1]-a[i])
    df["activity"] = activity
    return df
