import pandas as pd 

def view_state_of_domain(df, domain=None):
    if domain:
        state = list(set(df[df.domain==domain].state))
        entities = list(set(df[df.domain==domain].entity_id))
        print("entities: {} - total: {}\nstate: {}".format(entities, len(entities), state))
    else:
        state = list(set(df.state))
        entities = list(set(df.entity_id))
        print("entities: {} - total: {}\nstate: {}".format(entities, len(entities), state))