"""
    CSS
"""
import numpy as np
import gurobipy as gp
from  gurobipy import GRB
from IES.Utils.station_metadata import station_metadata
from attr_dict import AttrDict



class CESS:
    def __init__(self, components, station_id):
        self.components = components
        self.station_id = station_id
        self.station_metadata = station_metadata()

    def variables_def(self, model, T):
        self.q_cssc  = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_q_cssc",  vtype=GRB.CONTINUOUS )           # 储冷水罐输入功率
        self.q_cssd  = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_q_cssd",  vtype=GRB.CONTINUOUS )           # 储冷水罐输出功率
        self.soc  = model.addVars(T + 1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_cooling_storage_soc",  vtype=GRB.CONTINUOUS )        # 储冷水罐状态

        self.x_q_cssc = model.addVars(T, lb = 0, ub = 1, name = f"{self.station_id}_x_q_cssc", vtype = GRB.BINARY)
        self.x_q_cssd = model.addVars(T, lb = 0, ub = 1, name = f"{self.station_id}_x_q_cssd", vtype = GRB.BINARY)

    def model(self, T, model):
        if not self.components['cooling_storage']['active']:
            model.addConstrs(self.q_cssc[t] == 0 for t in range(T))
            model.addConstrs(self.q_cssd[t] == 0 for t in range(T))
        else:
            self.model_defining(T, model)
    
    def model_defining(self, T, model):
        model.addConstr(self.soc[0] ==  self.traces.soc[self.time_step])
        
        for t in range(T):
            model.addConstr(self.x_q_cssc[t] + self.x_q_cssd[t] == 1)
            model.addConstr( self.q_cssc[t] >= 0)
            model.addConstr( self.q_cssc[t] <= self.x_q_cssc[t] * self.station_metadata.cooling_storage.P_ch_max)
            model.addConstr( self.q_cssd[t] >= 0)
            model.addConstr( self.q_cssd[t] <= self.x_q_cssd[t] * self.station_metadata.cooling_storage.P_dis_max)
            model.addConstr( self.soc[t + 1] 
                            == self.soc[t] 
                            + ( self.q_cssc[t] * self.station_metadata.cooling_storage.eta_ch - self.q_cssd[t] / self.station_metadata.cooling_storage.eta_dis ) / self.station_metadata.cooling_storage.capacity )
            model.addConstr( self.soc[t] <= self.station_metadata.cooling_storage.soc_max )
            model.addConstr( self.soc[t] >= self.station_metadata.cooling_storage.soc_min )



    def cost(self, T):
        return 0
    

    def reset(self):
        self.traces = AttrDict(
            soc     =   [],
            q_cssc  =   [],
            q_cssd  =   [],
        )
        self.time_step = 0
        self.traces.soc.append(self.station_metadata.cooling_storage.soc_init)

    def step(self, actions):
        q_cssc, q_cssd = actions[f'{self.station_id}_q_cssc'], actions[f'{self.station_id}_q_cssd']
        next_soc = self.traces.soc[self.time_step] + ( q_cssc * self.station_metadata.cooling_storage.eta_ch  - q_cssd / self.station_metadata.cooling_storage.eta_dis ) / self.station_metadata.cooling_storage.capacity
        self.time_step += 1

        self.traces.soc.append(next_soc)
        self.traces.q_cssc.append(q_cssc)
        self.traces.q_cssd.append(q_cssd)


    @property
    def get_reward(self):
        return 0