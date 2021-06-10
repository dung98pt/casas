import json

def str_to_dict(a:str):
    res = json.loads(a)
    return res

def check_valid_range_day(b_df, attribute):
    if (b_df[attribute].iloc[-1].date() - b_df[attribute].iloc[0].date()).days < 7:
        return False
    return True