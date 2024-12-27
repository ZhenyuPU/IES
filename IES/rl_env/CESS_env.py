from IES.Utils.station_metadata import station_metadata
import numpy as np
import random
import matplotlib.pyplot as plt
from attr_dict import AttrDict
"""
    States: S_css
    Actions: a_css
    Power: P_cssc, P_cssd
    normalization: unified
    Violation: cooling balance
"""
class CESS_env:
    def __init__(self):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.cooling_storage
        self.time_step = 0

    def reset(self):
        self.time_step = 0
        self.soc = self.params.soc_init
        self.traces = AttrDict(
            soc     =   [],
            q_cssc  =   [],
            q_cssd  =   [],
        )
        self.traces.soc.append(self.soc)

    def step(self, actions):
        a_css = actions['cooling_storage'] if actions['cooling_storage'] is not None else 0

        self.q_cssc = max(a_css * self.params.P_ch_max, 0)
        # min(max(a_css * self.params.P_ch_max, 0), self.params.capacity * (self.params.soc_max - self.soc) / self.params.eta_ch)
        self.q_cssd = - min(a_css * self.params.P_dis_max, 0)
        # - max(min(a_css * self.params.P_dis_max, 0), self.params.capacity * (self.params.soc_min - self.soc) * self.params.eta_dis)
        # print(a_css * self.params.P_dis_max, self.q_cssd)

        # 更新储能状态
        next_soc = self.soc  + (self.q_cssc * self.params.eta_ch - self.q_cssd / self.params.eta_dis) / self.params.capacity
        self.soc = next_soc

        self.time_step += 1
        self.traces.soc.append(next_soc)
        self.traces.q_cssc.append(self.q_cssc)
        self.traces.q_cssd.append(self.q_cssd)
        # decisions.q_cssc = self.q_cssc
        # decisions.q_cssd = self.q_cssd
        # decisions.soc    = self.soc



    def reward(self):
        return 0 
