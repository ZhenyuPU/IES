from IES.mpc_env.HESS_model import HESS, hydrogen_market
from IES.mpc_env.BESS_model import BESS
from IES.mpc_env.TESS_model import TESS, absorption_chilling
from IES.mpc_env.CESS_model import CESS
from IES.Utils.station_metadata import station_metadata
from IES.dataset import energy_simulation, Pricing, Weather, solar_thermal_collector

import numpy as np
import gurobipy as gp
from  gurobipy import GRB, quicksum
import os
import json
import time
import pkg_resources

class AttrDict(dict):
    """A dictionary that allows attribute-style access."""
    
    def __getattr__(self, key):
        # If the key is not found, define it with a default value (None or any default)
        if key not in self:
            self[key] = None  # You can change None to any default value you like
        return self[key]
    
    def __setattr__(self, key, value):
        # Avoid recursion by checking if we are setting an attribute that already exists in the dict
        if key == '__dict__':
            super().__setattr__(key, value)  # Handle special case for the internal dictionary
        else:
            self[key] = value
    
    def __delattr__(self, key):
        # Avoid recursion when deleting an attribute
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'{key}' not found in AttrDict")

class power_market:
    def __init__(self, components, station_id, pricing):
        self.station_metadata = station_metadata()
        self.components = components
        self.station_id = station_id
        self.pricing = pricing
        self.electricity_price = self.pricing.electricity_price
        if hasattr(self.pricing, 'electricity_price_pred'):
            self.electricity_price_pred = self.pricing.electricity_price_pred
        else:
            self.electricity_price_pred = self.pricing.electricity_price

    def reset(self):
        self.traces = AttrDict(
            P_g_buy = [],
            P_g_sell = [],
        )
        self.time_step = 0

    def step(self, actions):
        P_g_buy = actions[f'{self.station_id}_P_g_buy']
        P_g_sell = actions[f'{self.station_id}_P_g_sell']
        self.traces.P_g_buy.append(P_g_buy)
        self.traces.P_g_sell.append(P_g_sell)
        self.time_step += 1

    def variables_def(self, model, T):
        self.P_g_buy    =   model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_g_buy",  vtype=GRB.CONTINUOUS )   # 与电网买卖电量[单位：kW]
        self.P_g_sell   =   model.addVars(T,  lb = 0,  ub = GRB.INFINITY,  name=f"{self.station_id}_P_g_sell",  vtype=GRB.CONTINUOUS )  # 与电网买卖电量[单位：kW]

    def model(self, T, model):
        if self.components['power_market']['active']:
            model.addConstrs( self.P_g_buy[t]  <=  self.station_metadata.power_market.p_grid_max for t in range(T))
            model.addConstrs( self.P_g_sell[t] <=  self.station_metadata.power_market.p_grid_max for t in range(T))
        else:
            model.addConstrs( self.P_g_buy[t]  == 0 for t in range(T))
            model.addConstrs( self.P_g_sell[t] == 0 for t in range(T))

    def cost(self, T):
        return quicksum(self.electricity_price_pred[self.time_step + t] * self.P_g_buy[t] - self.station_metadata.power_market.selling_price * self.P_g_sell[t] for t in range(T))
    
    @property
    def get_reward(self):
        return self.pricing.electricity_price[self.time_step-1] * self.traces.P_g_buy[-1] - self.station_metadata.power_market.selling_price * self.traces.P_g_sell[-1]



class EnergyHubEnv:
    def __init__(self, 
                 components = None,
                 DATASET_NAME = None,
                 station_id = None,
                 data_path = None,
                 simulation_start_time_step=None,
                 simulation_end_time_step=None, pred_data=None) -> None:
        """_summary_

        Args:
            penalty_factor (int, optional): demand penalty factor. Defaults to 10.
            program (str, optional): choose the dataset (train or test) to be aligned with the RL results. Defaults to 'test'.
            components (_type_, optional): system components. Defaults to None.
        """
        self.station_metadata  =   station_metadata()
        self.station_id = station_id

        self.components        = self.get_component(components)

        self.simulation_start_time_step = simulation_start_time_step
        self.simulation_end_time_step   = simulation_end_time_step

        # 模块定义
        self.energy_simulation = energy_simulation(components=self.components, simulation_start_time_step=self.simulation_start_time_step, simulation_end_time_step=self.simulation_end_time_step, DATASET_NAME=DATASET_NAME, station_id=self.station_id, data_path=data_path, pred_data=pred_data)
        self.pricing = Pricing(components=self.components, simulation_start_time_step=self.simulation_start_time_step, simulation_end_time_step=self.simulation_end_time_step, DATASET_NAME=DATASET_NAME, data_path=data_path, pred_data=pred_data)
        self.weather = Weather(components=self.components, simulation_start_time_step=self.simulation_start_time_step, simulation_end_time_step=self.simulation_end_time_step, DATASET_NAME=DATASET_NAME, data_path=data_path, pred_data=pred_data)

        self.solar_thermal_collector = solar_thermal_collector(self.weather, self.components)

        self.power_market  = power_market(self.components, self.station_id, self.pricing)
        self.hydrogen_market = hydrogen_market(self.components, self.station_id)

        self.electrical_storage = BESS(self.components, self.station_id)
        self.hydrogen_storage = HESS(self.components, self.station_id, self.energy_simulation, self.hydrogen_market)
        self.heating_storage = TESS(self.components, self.station_id)
        self.absorption_chiller = absorption_chilling(self.components, self.station_id)
        self.cooling_storage = CESS(self.components, self.station_id)

        self.get_forecast_data()




    def get_forecast_data(self):
        self.non_shiftable_load = self.energy_simulation.non_shiftable_load
        if hasattr(self.energy_simulation, 'non_shiftable_load_pred'):
            self.non_shiftable_load_pred = self.energy_simulation.non_shiftable_load_pred
        else:
            self.non_shiftable_load_pred = self.energy_simulation.non_shiftable_load
        
        self.heating_demand = self.energy_simulation.heating_demand
        if hasattr(self.energy_simulation, 'heating_demand_pred'):
            self.heating_demand_pred = self.energy_simulation.heating_demand_pred
        else:
            self.heating_demand_pred = self.energy_simulation.heating_demand

        self.cooling_demand = self.energy_simulation.cooling_demand
        if hasattr(self.energy_simulation, 'cooling_demand_pred'):
            self.cooling_demand_pred = self.energy_simulation.cooling_demand_pred
        else:
            self.cooling_demand_pred = self.energy_simulation.cooling_demand
        
        self.solar_generation = self.energy_simulation.solar_generation
        if hasattr(self.energy_simulation, 'solar_generation_pred'):
            self.solar_generation_pred = self.energy_simulation.solar_generation_pred
        else:
            self.solar_generation_pred = self.energy_simulation.solar_generation
        
        self.P_solar_heat = self.solar_thermal_collector.P_solar_heat
        if hasattr(self.solar_thermal_collector, 'P_solar_heat_pred'):
            self.P_solar_heat_pred = self.solar_thermal_collector.P_solar_heat_pred
        else:
            self.P_solar_heat_pred = self.solar_thermal_collector.P_solar_heat




    def variables_def(self, model, T):
        
        self.power_market.variables_def(model, T)
        self.hydrogen_market.variables_def(model, T)
        self.electrical_storage.variables_def(model, T) 
        self.hydrogen_storage.variables_def(model, T)
        self.heating_storage.variables_def(model, T)
        self.absorption_chiller.variables_def(model, T)
        self.cooling_storage.variables_def(model, T)

        return model
    
    def model(self, T, model):
        """prediction optimization: N = time_start - time_end + 1, which refers to the external data (current + prediction)

        Args:
            model (gurobi.model): gurobi model
            params (dict): store starting and ending indices
            method (str, optional): choose whether to predict. Defaults to 'PRED'.
            selected_day (_type_, optional): select the day to execute the optimization. Defaults to None.
            forecast_data (_type_, optional): forecasting data. Defaults to None(of no use).
            states (dict, optional): keep the last states as the inital states of the next step. Defaults to {}.

        Returns:
            tuple: model(gurobi.model), obj(objective), decisions(of no use)
        """

        # 变量定义
        model = self.variables_def(model, T)

        self.power_market.model(T, model)
        self.electrical_storage.model(T, model)
        self.hydrogen_storage.model(T, model)
        self.hydrogen_market.model(T, model)
        self.heating_storage.model(T, model)
        self.absorption_chiller.model(T, model)
        self.cooling_storage.model(T, model)

        # 电力平衡  
        model.addConstrs( (self.power_market.P_g_sell[t] + self.hydrogen_storage.P_el[t] + self.electrical_storage.P_bssc[t] + self.non_shiftable_load_pred[self.time_step + t]) 
                            <= (self.power_market.P_g_buy[t] +  self.solar_generation_pred[self.time_step + t] + self.hydrogen_storage.P_fc[t] + self.electrical_storage.P_bssd[t]) for t in range(T)) 
       
        
        # 热平衡
        model.addConstrs( (self.heating_storage.g_tesc[t] + self.heating_demand_pred[self.time_step + t] + self.absorption_chiller.g_AC[t]) 
                        <=(self.hydrogen_storage.g_fc[t] + self.heating_storage.g_tesd[t] + self.P_solar_heat_pred[self.time_step + t]) for t in range(T)) 
       

        # 冷平衡
        model.addConstrs( (self.cooling_storage.q_cssc[t] + self.cooling_demand_pred[self.time_step + t]) 
                        <= (self.absorption_chiller.q_AC[t]  + self.cooling_storage.q_cssd[t]) for t in range(T))

        ### Objective function
        obj = self.hydrogen_storage.cost(T) + self.electrical_storage.cost(T) + self.heating_storage.cost(T) + self.cooling_storage.cost(T) + self.power_market.cost(T) + self.hydrogen_market.cost(T) + self.absorption_chiller.cost(T)
    
        return model, obj
    

    def reset(self):
        self.time_step = 0
        self.power_market.reset()
        self.hydrogen_market.reset()
        self.energy_simulation.reset()
        self.weather.reset()
        self.pricing.reset()
        self.electrical_storage.reset()
        self.hydrogen_storage.reset()
        self.heating_storage.reset()
        self.cooling_storage.reset()
        self.absorption_chiller.reset()


    def step(self, actions = None):
        self.energy_simulation.step(actions)
        self.weather.step(actions)
        self.pricing.step(actions)
        self.power_market.step(actions)
        self.hydrogen_market.step(actions)
        self.electrical_storage.step(actions)
        self.hydrogen_storage.step(actions)
        self.heating_storage.step(actions)
        self.absorption_chiller.step(actions)
        self.cooling_storage.step(actions)
        reward = self.get_reward
        
        self.time_step += 1

        done = True if self.time_step == self.simulation_end_time_step - self.simulation_start_time_step else False 
        self.update_state()

        return self.states, reward, done, {}
    

    @property
    def get_reward(self):
        return self.power_market.get_reward + self.hydrogen_market.get_reward + self.electrical_storage.get_reward + self.hydrogen_storage.get_reward + self.cooling_storage.get_reward + self.heating_storage.get_reward + self.absorption_chiller.get_reward

    def update_state(self):
        self.states = AttrDict()
        for key, value in self.components.items():
            if value['active']:
                if any(sub in key for sub in ['month', 'day_type', 'hour', 'load', 'demand', 'solar_generation', 'indoor_dry_bulb_temperature']):
                    self.states[key] = getattr(self.energy_simulation, key)[self.time_step]
            elif 'storage' in key:
                    self.state_kwargs[key] = getattr(self, key).traces.soc[self.time_step]
            elif 'power_market' in key:
                self.states[key] = self.pricing.electricity_price[self.time_step]
            else:
                self.states[key] = getattr(self.weather, key)[self.time_step]


    @property
    def schema(self):
        # 获取包内的 CSV 文件路径
        file_path = pkg_resources.resource_filename(
            'IES', 'Utils/schema.json'
        )
        with open(file_path, 'r') as file:
            return json.load(file)
        
    @property
    def _components(self):
        return self.schema['components']
        
    
    def get_component(self, component):
        if component is None:
            return self.schema['components']
        
        component_all = self.schema['components']
        for key, value in component_all.items():
            if key not in component:
                component_all[key]['active'] = False
        
        return component_all

