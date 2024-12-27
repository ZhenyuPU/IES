"""
    BESS
"""
import numpy as np
import gurobipy as gp
from  gurobipy import GRB
from IES.Utils.station_metadata import station_metadata
from attr_dict import AttrDict


class BESS:
    def __init__(self, components, station_id):
        self.components = components
        self.station_id = station_id
        self.station_metadata = station_metadata()
        self.capacity = self.station_metadata.electrical_storage.capacity


    def variables_def(self, model, T):
        self.P_bssc  = model.addVars(T,  lb = -GRB.INFINITY,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_bssc",  vtype=GRB.CONTINUOUS )           # 电池充电功率[单位：kW]
        self.P_bssd  = model.addVars(T,  lb = -GRB.INFINITY,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_bssd",  vtype=GRB.CONTINUOUS )           # 电池放电功率[单位：kW]
        self.soc     = model.addVars(T + 1, lb = -GRB.INFINITY,  ub = GRB.INFINITY,  name = f"{self.station_id}_electrical_storage_soc",    vtype=GRB.CONTINUOUS  )
        
    def model(self, T, model):
        if not self.components['electrical_storage']['active']:
            model.addConstrs(self.P_bssc[t] == 0 for t in range(T))
            model.addConstrs(self.P_bssd[t] == 0 for t in range(T))
        else:
            self.model_defining(T, model)

    
    def model_defining(self, T, model):
        model.addConstr(self.soc[0] ==  self.traces.soc[self.time_step])  

        for t in range(T):
            model.addConstr( self.P_bssc[t] >= 0)
            model.addConstr( self.P_bssc[t] <= self.station_metadata.electrical_storage.P_ch_max)
            model.addConstr( self.P_bssd[t] >= 0)
            model.addConstr( self.P_bssd[t] <= self.station_metadata.electrical_storage.P_dis_max)
            model.addConstr( self.soc[t + 1] 
                            == self.soc[t] 
                            + ( self.P_bssc[t] * self.station_metadata.electrical_storage.eta_ch - self.P_bssd[t] / self.station_metadata.electrical_storage.eta_dis ) / self.station_metadata.electrical_storage.capacity )
            model.addConstr( self.soc[t] <= self.station_metadata.electrical_storage.soc_max )
            model.addConstr( self.soc[t] >= self.station_metadata.electrical_storage.soc_min )


    def cost(self, T):
        return 0
    

    def reset(self):
        self.traces = AttrDict(
            soc     =   [],
            P_bssc  =   [],
            P_bssd  =   [],
        )
        self.time_step = 0
        self.traces.soc.append(self.station_metadata.electrical_storage.soc_init)

    def step(self, actions):
        P_bssc, P_bssd = actions[f'{self.station_id}_P_bssc'], actions[f'{self.station_id}_P_bssd']
        next_soc = self.traces.soc[self.time_step] + (P_bssc * self.station_metadata.electrical_storage.eta_ch  - P_bssd / self.station_metadata.electrical_storage.eta_dis) / self.capacity
        self.time_step += 1

        self.traces.soc.append(next_soc)
        self.traces.P_bssc.append(P_bssc)
        self.traces.P_bssd.append(P_bssd)

    @property
    def get_reward(self):
        return 0


