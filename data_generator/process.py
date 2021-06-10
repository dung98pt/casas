import sqlite3
import pandas as pd
from datetime import timedelta, date, datetime
from utils import str_to_dict, check_valid_range_day
from preprocess import *
import numpy as np 

def get_state_df(folder):
    df = pd.read_csv("{}/data.csv".format(folder), sep="\t")
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    # df['last_updated'] = df['last_updated'] + timedelta(hours=7)
    df = df.sort_values(by=['last_updated'])
    return df

def get_event_df(folder):
    events_df = pd.read_csv("{}/event.csv".format(folder), sep="\t")
    events_df["time_fired"] = pd.to_datetime(events_df["time_fired"])
    # events_df["time_fired"] = events_df["time_fired"] + timedelta(hours=7)
    events_df = events_df.sort_values(by=['time_fired'])
    return events_df

def get_auto_df(events_df):
    auto_df = events_df[events_df.event_type=="automation_triggered"]
    auto_df['entity_id'] = [str_to_dict(i)['entity_id'] for i in auto_df.event_data]
    valid_automations = [i for i in entity_list(auto_df) if check_valid_range_day(auto_df[auto_df.entity_id==i], "time_fired")]
    auto_df = auto_df[auto_df.entity_id.isin(valid_automations)]
    auto_df["date"] = [i.date() for i in auto_df.time_fired]
    auto_df["time"] = [i.time() for i in auto_df.time_fired]
    auto_df["state"] = ["triggered"] * len(auto_df)
    auto_df["last_updated"] = auto_df["time_fired"]
    auto_df = auto_df[["last_updated", "date", "time", "entity_id", "state"]]
    auto_df = auto_df[auto_df.entity_id.isin(entities)]
    auto_df = auto_df.set_index("last_updated")
    return auto_df

def get_service_df(events_df):
    service_df = events_df[events_df.event_type=="call_service"]
    entity_ids = []
    services = []
    for i in range(len(service_df)):
        entity_id, service = get_entity_and_service(str_to_dict(service_df.event_data.iloc[i]))
        entity_ids.append(entity_id) 
        services.append(service)
    service_df["entity_id"] = entity_ids
    service_df["state"] = services
    service_df["date"] = [i.date() for i in service_df.time_fired]
    service_df["time"] = [i.time() for i in service_df.time_fired]
    service_df["last_updated"] = service_df["time_fired"]
    service_df = service_df[["last_updated", "date", "time", "entity_id", "state"]]
    service_df = service_df.dropna()
    service_df = service_df[service_df.entity_id.isin(entities)]
    service_df = service_df.set_index("last_updated")
    return service_df
    
def get_entity_and_service(event_data:dict):
    try:
        event_data['service_data']['entity_id']
        if type(event_data['service_data']['entity_id'])==str:
            return event_data['service_data']['entity_id'], event_data["service"]
        if type(event_data['service_data']['entity_id'])==list:
            return event_data['service_data']['entity_id'][0], event_data["service"]
    except:
        return np.nan, np.nan

entities = ['automation.bao_khi_nhiet_do_tren_25_do',
 'automation.den_gio_thuc_day',
 'automation.nhac_atv_an_com',
#  'automation.push_den_van_phong',
 'automation.set_theme_at_startup',
 'automation.tat_den_nghi_trua',
 'automation.thuc_day_buoi_chieu',
 'automation.tu_dong_tat_dieu_hoa_khi_nhiet_do_duoi_20_do',
 'automation.zigbee_join_disabled',
 'automation.zigbee_join_enabled',
 'climate.daikin',
 'climate.daikin_2',
 'climate.dh_1',
 'climate.rm_test_on',
 'climate.test_ha_moi',
 'sensor.0x842e14fffedb848b_humidity',
 'sensor.0x842e14fffedb848b_temperature',
 'switch.0x00124b001b78ce56',
 'switch.0x00124b00226aa8a4',
 'switch.0x00124b00226aa8d5',
 'switch.0x588e81fffed97dea',
 'switch.0x842e14fffe0e1df2',
 'switch.0xbc33acfffe931149_1',
 'switch.0xbc33acfffe931149_2',
 'switch.0xbc33acfffe931149_3',
 'switch.0xbc33acfffec44e08_1',
 'switch.0xbc33acfffefe2b1e_1',
 'switch.0xec1bbdfffe871287_l1',
 'switch.0xec1bbdfffe871287_l2',
 'switch.0xec1bbdfffe871287_l3',
 'switch.0xec1bbdfffe871287_l4',
 'switch.bo_do_dien_1',
 'switch.dusun2mqtt_30ae7b6408b1_main_join',
 'switch.switchbot',
 'switch.zigbee2mqtt_main_join']
