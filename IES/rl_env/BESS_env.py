from IES.Utils.station_metadata import station_metadata
import numpy as np
import random
import matplotlib.pyplot as plt
from attr_dict import AttrDict
"""
    States: S_bss
    Actions: a_bss
    Power: P_bssc, P_bssd
    normalization: unified
    Violation: 0
"""
class BESS_env:
    def __init__(self):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.electrical_storage
        self.time_step = 0
        self.capacity = self.params.capacity

    def reset(self):
        self.time_step = 0
        self.soc = self.params.soc_init
        self.traces = AttrDict(
            soc     =   [],
            P_bssc  =   [],
            P_bssd  =   [],
        )
        self.traces.soc.append(self.soc)

    def step(self, actions):
        a_bss = actions['electrical_storage'] if actions['electrical_storage'] is not None else 0

        self.P_bssc = min(max(a_bss * self.params.P_dis_max, 0), self.params.capacity * (self.params.soc_max - self.soc) / self.params.eta_ch)
        self.P_bssd = - max(min(a_bss * self.params.P_dis_max, 0), self.params.capacity * (self.params.soc_min - self.soc) * self.params.eta_dis)

        # 更新储能状态
        next_soc = self.soc  + (self.P_bssc * self.params.eta_ch - self.P_bssd / self.params.eta_dis) / self.params.capacity
        self.soc = next_soc

        self.time_step += 1

        self.traces.soc.append(next_soc)
        self.traces.P_bssc.append(self.P_bssc)
        self.traces.P_bssd.append(self.P_bssd)

        # decisions.P_bssc = self.P_bssc
        # decisions.P_bssd = self.P_bssd
        # decisions.soc    = self.soc

    def reward(self):
        return 0     
