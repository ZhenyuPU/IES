"""
    HESS
"""


import numpy as np
import gurobipy as gp
from  gurobipy import GRB, quicksum
from IES.Utils.station_metadata import station_metadata
from attr_dict import AttrDict


class hydrogen_market:
    def __init__(self, components, station_id):
        self.components = components
        self.station_id = station_id
        self.station_metadata = station_metadata()

    def variables_def(self, model, T):
        self.hydr_purc    = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_hydr_purc", vtype=GRB.CONTINUOUS   )

    def model(self, T, model):
        if not self.components['hydrogen_market']['active']:
            model.addConstrs(self.hydr_purc[t] == 0 for t in range(T))    
        else:
            model.addConstrs(self.hydr_purc[t] <= self.station_metadata.hydrogen_market.p_hydro_max for t in range(T))


    def cost(self, T):
        return quicksum( self.station_metadata.hydrogen_market.hydrogen_pricing * self.hydr_purc[t] for t in range(T))

    def reset(self):
        self.time_step = 0
        self.traces = AttrDict(
            hydr_purc = [],
        )
    
    def step(self, actions = None):
        hydr_purc = actions[f'{self.station_id}_hydr_purc']
        self.traces.hydr_purc.append(hydr_purc)
        self.time_step += 1

    @property
    def get_reward(self):
        return self.station_metadata.hydrogen_market.hydrogen_pricing * self.traces.hydr_purc[-1]



class HESS:
    def __init__(self, components, station_id, energy_simulation, hydrogen_market):
        self.components = components
        self.station_id = station_id
        self.station_metadata = station_metadata()
        self.hydrogen_market = hydrogen_market

        self.T_amb = energy_simulation.indoor_dry_bulb_temperature

        self.capacity = self.station_metadata.hydrogen_storage.capacity
        self.params = self.station_metadata.hydrogen_storage
        self.P_el_high = self.params.P_el_high
        self.P_el_low = self.params.P_el_low
        self.P_nom_el = self.params.P_nom_el

        self.P_fc_high = self.params.P_fc_high
        self.P_fc_high = self.params.P_fc_low
        self.i_nom_el = self.params.i_nom_el
        self.i_bp_fc = self.params.i_bp_fc

        self.M = 1e5
        self.eps = 1e-4

    def variables_def(self, model, T):
        ############################################################################################## Eletrolyzer ####################################################################################################
        
        self.P_el     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el",  vtype=GRB.CONTINUOUS   )           # 电解池输入功率[单位：kW]
        self.x_P_el = model.addVars(T, lb=0, ub=1,  name=f"{self.station_id}_x_P_el",  vtype=GRB.BINARY   )         

        self.P_el_1_L1 = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el_1_L1",  vtype=GRB.CONTINUOUS   ) 
        self.x_P_el_1_L1 = model.addVars(T, lb=0, ub=1,  name=f"{self.station_id}_x_P_el_1_L1",  vtype=GRB.BINARY   )  
        self.P_el_1_L2 = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el_1_L2",  vtype=GRB.CONTINUOUS   ) 
        self.x_P_el_1_L2 = model.addVars(T, lb=0, ub=1,  name=f"{self.station_id}_x_P_el_1_L2",  vtype=GRB.BINARY   )  

        self.P_el_L1 = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el_L1",  vtype=GRB.CONTINUOUS   ) 
        self.P_el_L2 = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el_L2",  vtype=GRB.CONTINUOUS   ) 
        self.P_el_L3 = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el_L3",  vtype=GRB.CONTINUOUS   ) 

        self.x_P_el_L1 = model.addVars(T, lb=0, ub=1,  name=f"{self.station_id}_x_P_el_L1",  vtype=GRB.BINARY   )  
        self.x_P_el_L2 = model.addVars(T, lb=0, ub=1,  name=f"{self.station_id}_x_P_el_L2",  vtype=GRB.BINARY   )  
        self.x_P_el_L3 = model.addVars(T, lb=0, ub=1,  name=f"{self.station_id}_x_P_el_L3",  vtype=GRB.BINARY   )  

        self.P_el_1 = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_el_1",  vtype=GRB.CONTINUOUS   ) 


        self.i_el     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_i_el",  vtype=GRB.CONTINUOUS   )
   
        self.T_el     = model.addVars(T+1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_eT_el",  vtype=GRB.CONTINUOUS   )

        self.C        = model.addVars(T+1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_C",  vtype=GRB.CONTINUOUS   )

        ######################################################################################### Gas cleaner and tank ################################################################################################

        self.v_el     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_v_el",  vtype=GRB.CONTINUOUS   )           # 存储氢气体积速率[单位：m3/s]


        self.v_fc     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_v_fc",  vtype=GRB.CONTINUOUS   )            # 氢气输入燃料电池体积速率[单位：m3/s]
        self.v_cdg    = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_v_cdg",  vtype=GRB.CONTINUOUS   )
        self.p_tank        = model.addVars(T+1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_p_tank",  vtype=GRB.CONTINUOUS   )
        self.T_tank   = model.addVars(T+1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_T_tank",  vtype=GRB.CONTINUOUS   )
        self.soc = model.addVars(T+1,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_hydrogen_storage_soc",  vtype=GRB.CONTINUOUS   )

        
        ############################################################################################## Fuel Cell ####################################################################################################
        
        self.P_fc     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_fc",  vtype=GRB.CONTINUOUS   )            # 燃料电池输出功率[单位：kW]

        self.x_P_fc = model.addVars(T, lb=0, ub=1, name=f"{self.station_id}_x_P_fc", vtype=GRB.BINARY)

        self.i_fc     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_i_fc",  vtype=GRB.CONTINUOUS   )

        self.g_fc     = model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_g_fc",  vtype=GRB.CONTINUOUS   )
       

    def model(self, T, model):
        if not self.components['hydrogen_storage']['active']:
            model.addConstrs(self.P_el[t] == 0 for t in range(T))
            model.addConstrs(self.g_fc[t] == 0 for t in range(T))
            model.addConstrs(self.P_fc[t] == 0 for t in range(T))
        else:
            self.model_defining(T, model)

    

    def model_defining(self, T, model):
        ###########################################################################################   Eletrolyzer   ###############################################################################################
        
        ### volume rate of hydrogen
        model.addConstrs(self.v_el[t] <= self.params.z1 * self.T_el[t] + self.params.z0 + self.params.z_low * (self.P_el[t] - self.params.P_nom_el) for t in range(T))
        model.addConstrs(self.v_el[t] <= self.params.z1 * self.T_el[t] + self.params.z0 + self.params.z_high * (self.P_el[t] - self.params.P_nom_el) for t in range(T))
        model.addConstrs(self.v_el[t] >= 0 for t in range(T))
        model.addConstrs(self.v_el[t] <= self.M * self.x_P_el[t] for t in range(T)) # TODO x_P_el

        ### electrolyzer
        model.addConstr(self.T_el[0] == self.traces.T_el[self.time_step])
        model.addConstrs(self.T_el[t+1] == self.params.j1 * self.T_el[t] + self.params.j2 * self.P_el[t] + self.params.j0 for t in range(T))
        model.addConstrs(self.T_el[t] <= self.params.T_el_high for t in range(T+1))

        ### electrolyzer stack current
        model.addConstrs(self.P_el_1_L1[t] >= self.x_P_el_1_L1[t] * (self.params.P_el_low - self.params.P_nom_el_1) for t in range(T))
        model.addConstrs(self.P_el_1_L1[t] <= self.x_P_el_1_L1[t] * (self.params.P_nom_el_1 - self.params.P_nom_el_1) for t in range(T))

        model.addConstrs(self.P_el_1_L2[t] >= self.x_P_el_1_L2[t] * (self.params.P_nom_el_1 - self.params.P_nom_el_1) for t in range(T))
        model.addConstrs(self.P_el_1_L2[t] <= self.x_P_el_1_L2[t] * (self.params.P_el_high - self.params.P_nom_el_1) for t in range(T))

        model.addConstrs(self.i_el[t] == self.params.h1 * self.T_el[t] + self.params.h0 + self.params.h_low * self.P_el_1_L1[t] + self.params.h_high * self.P_el_1_L2[t] for t in range(T))
        model.addConstrs(self.x_P_el[t] == self.x_P_el_1_L1[t] + self.x_P_el_1_L2[t] for t in range(T))

        ### electrolyzer power
        model.addConstrs(self.P_el[t] >= self.x_P_el[t] * self.params.P_el_low for t in range(T))
        model.addConstrs(self.P_el[t] <= self.x_P_el[t] * self.params.P_el_high for t in range(T))

        ### stack 1 power
        model.addConstrs(self.P_el[t] == self.P_el_L1[t] + self.P_el_L2[t] + self.P_el_L3[t] for t in range(T))
        model.addConstrs(self.x_P_el[t] == self.x_P_el_L1[t] + self.x_P_el_L2[t] + self.x_P_el_L3[t]  for t in range(T))

        model.addConstrs(self.P_el_L1[t] >= self.params.P_el_low * self.x_P_el_L1[t] for t in range(T))
        model.addConstrs(self.P_el_L1[t] <= self.params.P_nom_el /4 * self.x_P_el_L1[t] for t in range(T))

        model.addConstrs(self.P_el_L2[t] >= self.params.P_nom_el / 4 * self.x_P_el_L2[t] for t in range(T))
        model.addConstrs(self.P_el_L2[t] <= self.params.P_nom_el * self.x_P_el_L2[t] for t in range(T))

        model.addConstrs(self.P_el_L3[t] >= self.params.P_nom_el * self.x_P_el_L3[t] for t in range(T))
        model.addConstrs(self.P_el_L3[t] <= self.params.P_el_high * self.x_P_el_L3[t] for t in range(T))

        ### stack 1
        model.addConstrs(self.P_el_1[t] == self.P_el_L1[t] + self.params.P_nom_el / 4 * self.x_P_el_L2[t] + self.P_el_L3[t] / 4 for t in range(T))

        model.addConstrs(self.P_el_1[t] == self.P_el_1_L1[t] + self.P_el_1_L2[t] + self.params.P_nom_el_1 for t in range(T))


        # overload
        model.addConstrs(self.C[t+1] >= self.C[t] + (self.i_el[t] - self.params.i_nom_el) for t in range(T))
        model.addConstrs(self.C[t+1] >= 0 for t in range(T))
        model.addConstrs(self.C[t] <= self.params.C_high for t in range(T+1))
        model.addConstr(self.C[0] == self.traces.C[self.time_step])

        ####################################################################################### Gas cleaner and tank ##############################################################################################
        
        model.addConstrs(self.v_cdg[t] == self.params.alpha * self.v_el[t] for t in range(T))

        model.addConstrs(self.soc[t+1] == self.soc[t] + (self.params.rho * (self.v_cdg[t] - self.v_fc[t]) + self.hydrogen_market.hydr_purc[t]) / self.capacity for t in range(T))
        model.addConstr(self.soc[0] == self.traces.soc[self.time_step])
        model.addConstrs(self.soc[t] >= self.params.soc_min for t in range(T+1))
        model.addConstrs(self.soc[t] <= self.params.soc_max for t in range(T+1))

        ### hydrogen tank pressure
        model.addConstrs(self.p_tank[t] == (self.params.b0 + self.params.b1 * self.T_tank[t]) * (self.soc[t] * self.capacity / self.params.V_tank) for t in range(T))
        model.addConstrs(self.p_tank[t] >= self.params.p_tank_low for t in range(T))
        model.addConstrs(self.p_tank[t] <= self.params.p_tank_high for t in range(T))

        ### tank temperature
        model.addConstrs(self.T_tank[t+1] == self.params.g0 * self.T_tank[t] + self.params.g1 * self.T_amb[t] for t in range(T))
        model.addConstr(self.T_tank[0] == self.traces.T_tank[self.time_step])

        ############################################################################################ Fuel Cell ####################################################################################################
        model.addConstrs(self.v_fc[t] == self.params.c * self.i_fc[t] for t in range(T))
        
        model.addConstrs(self.i_fc[t] >= self.params.s1 * self.P_fc[t] for t in range(T))
        model.addConstrs(self.i_fc[t] >= self.params.s2 * (self.P_fc[t] - self.params.u_bp_fc) + self.params.i_bp_fc for t in range(T))
        model.addConstrs(self.i_fc[t] >= 0 for t in range(T))
        model.addConstrs(self.i_fc[t] <= self.M * self.x_P_fc[t] for t in range(T))

        model.addConstrs(self.P_fc[t] >= self.x_P_fc[t] * self.params.P_fc_low for t in range(T))
        model.addConstrs(self.P_fc[t] <= self.x_P_fc[t] * self.params.P_fc_high for t in range(T))

        if self.components['heating_storage']['active']:
            model.addConstrs(self.g_fc[t] == (1 - self.params.eta_fc) * self.P_fc[t] / self.params.eta_fc * self.params.eta_fc_rec for t in range(T))
        else:
            model.addConstrs(self.g_fc[t] == 0 for t in range(T))






    # def model_defining(self, T, model):
    #     #######################################################################################  初始化储氢罐设备状态  ############################################################################################
    #     model.addConstr(self.soc[0]  == self.traces.soc[self.time_step])     # 储氢罐储氢质量
    #     model.addConstr(self.T_el[0]   == self.traces.T_el[self.time_step])     # 电解槽温度
    #     model.addConstr(self.T_tank[0] == self.traces.T_tank[self.time_step])    # 储氢罐初始温度
    #     model.addConstr(self.C[0]      == self.traces.C[self.time_step])         # load counter
    
    #     ### 设备运行物理约束

    #     ###########################################################################################   Eletrolyzer   ###############################################################################################
            
    #     ### range of P_el
    #     model.addConstrs(self.P_el[t] <= self.P_el_high for t in range(T))
    #     ### volume of hydrogen in the electrolyzer
    #     low = [0, self.P_el_low, self.P_nom_el, self.P_el_high]
    #     high = [self.P_el_low, self.P_nom_el, self.P_el_high, self.M]

    #     model.addConstrs(self.A_v_el_0[t] == self.params.z1 * self.T_el[t] + self.params.z0 - self.params.z_low * self.P_nom_el for t in range(T))
    #     model.addConstrs(self.A_v_el_1[t] == self.params.z1 * self.T_el[t] + self.params.z0 - self.params.z_high * self.P_nom_el for t in range(T))
    #     A = [
    #         [0, 0],
    #         [self.params.z_low, [self.A_v_el_0[t] for t in range(T)]],
    #         [self.params.z_high, [self.A_v_el_1[t] for t in range(T)]],
    #         [0, 0]
    #     ]
    #     self.get_piecewise_fun(low, high, A, self.P_el, self.v_el, T, name='v_el', model=model)
    
    #     ### Power of Stack 1 in the electrolyzer

    #     low = [0, self.params.P_el_low, self.params.P_nom_el / 4, self.params.P_nom_el, self.params.P_el_high]
    #     high = [self.params.P_el_low, self.params.P_nom_el / 4, self.params.P_nom_el, self.params.P_el_high, self.M]
    #     A = [
    #         [0, 0], 
    #         [1, 0], 
    #         [0, self.params.P_nom_el_1 / 4], 
    #         [1./4, 0], 
    #         [0, 0]]
    #     self.get_piecewise_fun(low, high, A, self.P_el, self.P_el_1, T, name='P_el_1', model=model)
            
    #     ### Current of the electrolyzer
    #     low = [0, self.params.P_nom_el_1, self.params.P_nom_el, self.params.P_el_high]
    #     high = [self.params.P_nom_el_1, self.params.P_nom_el, self.params.P_el_high, self.M]

    #     model.addConstrs(self.A_i_el_0[t] == self.params.h1 * self.T_el[t] + self.params.h0 - self.params.h_low * self.params.P_nom_el_1  for t in range(T))
    #     model.addConstrs(self.A_i_el_1[t] == self.params.h1 * self.T_el[t] + self.params.h0 - self.params.h_high * self.params.P_nom_el_1  for t in range(T))
    #     A = [
    #         [0, 0], 
    #         [self.params.h_low, [self.A_i_el_0[t] for t in range(T)]], 
    #         [self.params.h_high, [self.A_i_el_1[t] for t in range(T)]], 
    #         [0, 0]]
        
    #     self.get_piecewise_fun(low, high, A, self.P_el_1, self.i_el, T, name='i_el', model=model)


    #     ####################################################################################### Gas cleaner and tank ##############################################################################################
        
    #     for t in range(T):
    #         ### volume flow velocity of the electrolyzer
    #         model.addConstr(self.v_cdg[t] == self.params.alpha * self.v_el[t])

    #         ### presure of the tank
    #         model.addConstr(self.p_tank[t]     == (self.params.b0 + self.params.b1 * self.T_tank[t]) * self.soc[t] * self.params.capacity / self.params.V_tank ) 
    #         model.addConstr(self.p_tank[t] >= self.params.p_tank_low)
    #         model.addConstr(self.p_tank[t] <= self.params.p_tank_high)

    #     ############################################################################################ Fuel Cell ####################################################################################################
            
    #     ### P_fc
    #     # P_fc的范围
    #     model.addConstrs(self.P_fc[t] <= self.params.P_fc_high for t in range(T))

    #     low = [0, self.params.P_fc_low, self.params.u_bp_fc, self.params.P_fc_high]
    #     high = [self.params.P_fc_low, self.params.u_bp_fc, self.params.P_fc_high, self.M]

    #     A = [
    #         [0, 0], 
    #         [self.params.s1, 0], 
    #         [self.params.s2, self.params.i_bp_fc - self.params.s2 * self.params.u_bp_fc], 
    #         [0, 0]]
    #     self.get_piecewise_fun(low, high, A, self.P_fc, self.i_fc, T, name='i_fc', model=model)
        
    #     ### volume flow velocity of fuel cell
    #     model.addConstrs(self.v_fc[t] == self.params.c * self.i_fc[t] for t in range(T))

    #     ### 产热
    #     if self.components['fuel_cell_chp']['active']:
    #         model.addConstrs(self.g_fc[t] == self.params.eta_fc_rec * (1 - self.params.eta_fc) * self.P_fc[t] / self.params.eta_fc for t in range(T))
    #     else:
    #         model.addConstrs(self.g_fc[t] == 0 for t in range(T))

    #     #######################################################################################  状态转移，系统动态特性  ############################################################################################

    #     model.addConstrs(self.T_el[t+1] == self.params.j1 * self.T_el[t] + self.params.j2 * self.P_el[t] + self.params.j0 for t in range(T))
    #     model.addConstrs(self.T_el[t] <= self.params.T_el_high for t in range(T + 1))

    #     model.addConstrs(self.T_tank[t+1] == self.params.g0 * self.T_tank[t] + self.params.g1 * self.T_amb[self.time_step + t] for t in range(T))

    #     model.addConstrs(self.soc[t+1] == self.soc[t] + (self.params.rho * (self.v_cdg[t] - self.v_fc[t]) + self.hydrogen_market.hydr_purc[t]) / self.params.capacity for t in range(T))
    #     model.addConstrs( self.soc[t] >= self.params.soc_min for t in range(1, T + 1))
    #     model.addConstrs( self.soc[t] <= self.params.soc_max for t in range(1, T + 1))

    #     model.addConstrs(self.C[t+1] >= self.C[t] +  (self.i_el[t] - self.params.i_nom_el) for t in range(T))
        
    #     model.addConstrs(self.C[t] >= 0 for t in range(T + 1))
    #     model.addConstrs(self.C[t] <= self.params.C_high for t in range(T + 1))


    
    def get_piecewise_fun(self, low, high, A, U, Y, T, name, model):
        """
        定义分段线性函数的约束和目标函数。

        Args:
            low (list): 每段的下界向量。
            high (list): 每段的上界向量。
            A (list of list): 分段函数的系数矩阵 [[a_0, a_1], ...]，表示 y = a_1 * x + a_0。
            U (Var): 决策变量，用于存储分段函数的结果。
            Y (Var): 分段输出变量。
            name (str): 变量命名前缀。
            model (gurobipy.Model): Gurobi 模型实例。
        """
        L = len(low)  # 分段数

        # 定义 Gurobi 变量
        X = model.addVars(L, T, vtype=GRB.BINARY, name=f"{name}_X")  # 0-1 矩阵
        Z = model.addVars(L, T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"{name}_Z")  # 连续变量

        # 约束 1: 每个时间步长只能激活一个分段
        model.addConstrs((quicksum(X[i, j] for i in range(L)) == 1 for j in range(T)), name=f"{name}_single_segment")

        # 约束 2: 确保 Z[i, j] 在对应分段范围内
        for j in range(T):
            model.addConstr(Z[0, j] <= X[0, j] * high[0], name=f"{name}_upper_bound_0")
            model.addConstr(Z[0, j] >= X[0, j] * low[0], name=f"{name}_lower_bound_0")
            for i in range(L-1):
                model.addConstr(Z[i+1, j] <= X[i+1, j] * high[i+1] + self.eps, name=f"{name}_upper_bound_{i+1}")
                model.addConstr(Z[i+1, j] >= X[i+1, j] * low[i+1] + self.eps, name=f"{name}_lower_bound_{i+1}")

        # 约束 3: 确保 U[j] 是所有分段变量的之和
        model.addConstrs((U[j] == quicksum(Z[i, j] for i in range(L)) for j in range(T)), name=f"{name}_U_sum")

        # 约束 4: 分段函数求和的逻辑关系，得到输出量Y[j]是所有分段i结果的加权和
        model.addConstrs((
            Y[j] == quicksum(
                (A[i][0] * Z[i, j] + (A[i][1][j] if isinstance(A[i][1], list) else A[i][1])) * X[i, j]
                for i in range(L)
            ) for j in range(T)
        ), name=f"{name}_piecewise_fun")


    def reset(self):
        # todo
        self.traces = AttrDict(
            soc     =  [],
            T_el    =  [],
            C       =  [],
            T_tank  =  [],
            p_tank  =  [],
            P_el    =  [],
            v_el    =  [],
            i_el    =  [],
            v_cdg   =  [],
            P_fc    =  [],
            i_fc    =  [],
            v_fc    =  [], 
            g_fc    =  [],
            P_el_1  =  []
        )
        self.time_step = 0
        self.traces.soc.append(self.params.soc_init)       # randomly initialize the hydrogen tank soc 
        self.traces.T_el.append( self.params.T_el_init )
        self.traces.C.append(self.params.C_init)
        self.traces.T_tank.append( self.params.T_tank_init )

        p_tank_init = (self.params.b0 + self.params.b1 * self.traces.T_tank[self.time_step]) * ( self.traces.soc[self.time_step] * self.params.capacity / self.params.V_tank )  # compute initial tank pressure
        self.traces.p_tank.append(p_tank_init)


    def step(self, actions):
        P_el   =  actions[f'{self.station_id}_P_el']  # DC power consumption of electrolyzer 
        P_fc    =  actions[f'{self.station_id}_P_fc']   # electricity generated by fuel cell 
        i_fc    =  actions[f'{self.station_id}_i_fc']   # electricity generated by fuel cell 
        g_fc    =  actions[f'{self.station_id}_g_fc']   # heat generated by fuel cell
        v_el   =  actions[f'{self.station_id}_v_el']  # volume rate of hydrogen generated by electrolyzer
        i_el   =  actions[f'{self.station_id}_i_el']  # stack current of electrolyzer 
        v_fc    =  actions[f'{self.station_id}_v_fc']   # volume rate of hydrogen consumed by fuel cell 
        P_el_1 =  actions[f'{self.station_id}_P_el_1']  # DC power distributed to the first stack of electrolyzer  
        v_cdg   =  actions[f'{self.station_id}_v_cdg']
        hydr_purc =  actions[f'{self.station_id}_hydr_purc']


        self.traces.P_el.append(P_el)
        self.traces.v_el.append(v_el)
        self.traces.i_el.append(i_el)
        self.traces.v_cdg.append(v_cdg)
        self.traces.P_fc.append(P_fc)
        self.traces.i_fc.append(i_fc)
        self.traces.v_fc.append(v_fc)
        self.traces.g_fc.append(g_fc)
        self.traces.P_el_1.append(P_el_1)

        soc_next = self.traces.soc[self.time_step] + ( self.params.rho * (v_cdg - v_fc) + hydr_purc )/self.params.capacity

        T_el_next = self.params.j1 * self.traces.T_el[self.time_step] + self.params.j2 * self.traces.P_el[self.time_step] + self.params.j0

        C_next    = max(0, self.traces.C[self.time_step] + (i_el - self.params.i_nom_el) )
        T_tank_next = self.params.g0 * self.traces.T_tank[self.time_step] + self.params.g1 * self.T_amb[self.time_step]
        p_tank_next  = ( self.params.b0 + self.params.b1 * T_tank_next) * (soc_next * self.capacity/self.params.V_tank) 

        self.traces.soc.append(soc_next)
        self.traces.T_el.append(T_el_next)
        self.traces.C.append(C_next)
        self.traces.T_tank.append(T_tank_next)
        self.traces.p_tank.append(p_tank_next)

        self.time_step += 1


    @property
    def get_reward(self):
        return 0
    
    def cost(self, T):
        return 0