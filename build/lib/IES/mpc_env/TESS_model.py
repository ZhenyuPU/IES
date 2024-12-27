"""
    TESS
"""
import numpy as np
import gurobipy as gp
from  gurobipy import GRB
from IES.Utils.station_metadata import station_metadata
from attr_dict import AttrDict

class absorption_chilling:
    def __init__(self, components, station_id):
        self.components = components
        self.station_id = station_id
        self.station_metadata = station_metadata()
    
    def variables_def(self, model, T):
        self.g_AC    =  model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_g_AC",  vtype=GRB.CONTINUOUS   )           # 吸收式制冷机输入功率
        self.q_AC    =  model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_q_AC",  vtype=GRB.CONTINUOUS   )        
    
    def model(self, T, model):
        if not self.components['absorption_chiller']['active']:
            model.addConstrs( self.g_AC[t] == 0  for t in range(T))
            model.addConstrs( self.q_AC[t] == 0  for t in range(T))
        else:
            model.addConstrs( self.g_AC[t]   <= self.station_metadata.absorption_chiller.g_ac_max for t in range(T))
            model.addConstrs( self.q_AC[t]   == self.g_AC[t] * self.station_metadata.absorption_chiller.eta_ac for t in range(T))


    def reset(self):
        self.traces = AttrDict(
            g_AC = [],
            q_AC = [],
        )
        self.time_step = 0

    def step(self, actions = None):
        g_AC = actions[f'{self.station_id}_g_AC']
        q_AC = actions[f'{self.station_id}_q_AC']
        self.traces.g_AC.append( g_AC ) 
        self.traces.q_AC.append( q_AC )
        self.time_step += 1
    
    @property
    def get_reward(self):
        return 0
    
    def cost(self, T):
        return 0



class TESS:
    def __init__(self, components, station_id):
        self.components = components
        self.station_id = station_id
        self.station_metadata = station_metadata()
    
    def variables_def(self, model, T):
        self.g_tesc  =  model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_g_tesc",  vtype=GRB.CONTINUOUS )           # 储热水罐输入功率
        self.g_tesd  =  model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_g_tesd",  vtype=GRB.CONTINUOUS )           # 储热水罐输出功率
        self.soc     =  model.addVars(T + 1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_heating_storage_soc",  vtype=GRB.CONTINUOUS )        # 储热水罐状态

        self.x_g_tesc = model.addVars(T, lb = 0, ub = 1, name = f"{self.station_id}_x_g_tesc", vtype = GRB.BINARY)
        self.x_g_tesd = model.addVars(T, lb = 0, ub = 1, name = f"{self.station_id}_x_g_tesd", vtype = GRB.BINARY)


    def model(self, T, model):
        if not self.components['heating_storage']['active']:
            model.addConstrs(self.g_tesc[t] == 0 for t in range(T))
            model.addConstrs(self.g_tesd[t] == 0 for t in range(T))
        else:
            self.model_defining(T, model)


    def model_defining(self, T, model):
        model.addConstr(self.soc[0] ==  self.traces.soc[self.time_step])

        for t in range(T):
            model.addConstr(self.x_g_tesc[t] + self.x_g_tesd[t] == 1)
            model.addConstr( self.g_tesc[t] >= 0)
            model.addConstr( self.g_tesc[t] <= self.x_g_tesc[t] * self.station_metadata.heating_storage.P_ch_max)
            model.addConstr( self.g_tesd[t] >= 0)
            model.addConstr( self.g_tesd[t] <= self.x_g_tesd[t] * self.station_metadata.heating_storage.P_dis_max )
            model.addConstr( self.soc[t + 1] 
                            == self.soc[t] 
                            + ( self.g_tesc[t] * self.station_metadata.heating_storage.eta_ch - self.g_tesd[t] / self.station_metadata.heating_storage.eta_dis ) / self.station_metadata.heating_storage.capacity )
            model.addConstr( self.soc[t] <= self.station_metadata.heating_storage.soc_max)
            model.addConstr( self.soc[t] >= self.station_metadata.heating_storage.soc_min)
    

    def cost(self, T):
        return 0
    

    def reset(self):
        self.traces = AttrDict(
            soc     =   [],
            g_tesc  =   [],
            g_tesd  =   [],
        )
        self.time_step = 0
        self.traces.soc.append(self.station_metadata.heating_storage.soc_init)

    def step(self, actions=None):
        g_tesc   = actions[f'{self.station_id}_g_tesc'] # input of absorption chiller 
        g_tesd   = actions[f'{self.station_id}_g_tesd'] # generated heat 
        self.traces.g_tesc.append(g_tesc) 
        self.traces.g_tesd.append(g_tesd)

        next_soc = self.traces.soc[self.time_step] + (g_tesc * self.station_metadata.heating_storage.eta_ch - g_tesd / self.station_metadata.heating_storage.eta_dis ) / self.station_metadata.heating_storage.capacity
        self.traces.soc.append(next_soc)
        self.time_step += 1

    @property
    def get_reward(self):
        return 0


                