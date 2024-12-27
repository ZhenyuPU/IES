from IES.Utils.station_metadata import station_metadata
import numpy as np
import random
import matplotlib.pyplot as plt
from attr_dict import AttrDict
"""
    States: S_tes
    Actions: a_tes, a_qc
    Power: P_tesc, P_tesd, P_AC
    normalization: unified
    Violation: heating balance
"""
class TESS_env:
    def __init__(self):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.heating_storage
        self.time_step = 0

    def reset(self):
        self.time_step = 0
        self.soc = self.params.soc_init
        self.traces = AttrDict(
            soc     =   [],
            g_tesc  =   [],
            g_tesd  =   [],
        )
        self.traces.soc.append(self.params.soc_init)
    
    def step(self, actions):
        a_tes = actions['heating_storage'] if actions['heating_storage'] is not None else 0

        self.g_tesc = min(max(a_tes * self.params.P_ch_max, 0), self.params.capacity * (self.params.soc_max - self.soc) / self.params.eta_ch)
        self.g_tesd = - max(min(a_tes * self.params.P_dis_max, 0), self.params.capacity * (self.params.soc_min - self.soc) * self.params.eta_dis)

        # 更新储能状态
        next_soc = self.soc  + (self.g_tesc * self.params.eta_ch - self.g_tesd / self.params.eta_dis) / self.params.capacity
        self.soc = next_soc

        self.time_step += 1
        self.traces.g_tesc.append(self.g_tesc) 
        self.traces.g_tesd.append(self.g_tesd)
        self.traces.soc.append(next_soc)
        # decisions.g_tesc = self.g_tesc
        # decisions.g_tesd = self.g_tesd
        # decisions.soc = self.soc

    def reward(self):
        return 0    


class absorption_chiller:
    def __init__(self):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.absorption_chiller
        self.time_step = 0
    
    def reset(self):
        self.time_step = 0
        self.traces = AttrDict(
            g_ac = [],
            q_ac = [],
        )
    
    def step(self, actions):
        a_ac = actions['absorption_chiller'] if actions['absorption_chiller'] is not None else 0

        self.g_ac = max(a_ac * self.params.g_ac_max, 0)
        self.q_ac = self.g_ac * self.params.eta_ac

        self.traces.g_ac.append( self.g_ac ) 
        self.traces.q_ac.append( self.q_ac )
        # decisions.g_ac = self.g_ac
        # decisions.q_ac = self.q_ac

        self.time_step += 1

    def reward(self):
        return 0 