import json
import numpy as np
from importlib.resources import files
from IES.Utils.station_metadata import station_metadata
import pandas as pd
from pathlib import Path

# TODO 在这里获取预测的值
# TODO RL能够结合LSTM （后续再研究）
# TODO solar generation
class energy_simulation:
    def __init__(self, 
                 components, 
                 simulation_start_time_step, simulation_end_time_step, 
                 data_path=None, 
                 DATASET_NAME=None, 
                 station_id=None, 
                 pred_data: dict = None
            ):
        """
            pred_data: {'': , '': ,...}
        """

        self.components = components
        self.station_metadata = station_metadata()
        self.simulation_start_time_step = simulation_start_time_step
        self.simulation_end_time_step =  simulation_end_time_step
        self.data_path = data_path
        self.pred_data = pred_data
        self.DATASET_NAME = DATASET_NAME
        self.station_id = station_id

        self.non_shiftable_load_cap = self.station_metadata.energy_simulation.non_shiftable_load_cap  # 200 kW 
        self.heating_demand_cap     = self.station_metadata.energy_simulation.heating_demand_cap      # 100 kW
        self.dhw_demand_cap         = self.station_metadata.energy_simulation.dhw_demand_cap
        self.cooling_demand_cap     = self.station_metadata.energy_simulation.cooling_demand_cap      # 150 kW 
        self.solar_generation_cap   = self.station_metadata.energy_simulation.solar_generation_cap    # 300 kW

        self.get_data()

        self.observation_max, self.observation_min = self.get_observation_min_max()

    def reset(self):
        self.time_step = 0
    
    def step(self, action=None):
        self.time_step += 1
        # action.update({key: getattr(self, key)[self.time_step] for key in self.observations})

    def get_data(self):
        if self.pred_data:
            df_pred = pd.DataFrame(self.pred_data)
            for col in df_pred.columns.to_list():
                setattr(self, col + '_pred', df_pred[col][self.simulation_start_time_step : self.simulation_end_time_step + 24].to_numpy())
                if col in ['non_shiftable_load_pred', 'heating_demand_pred', 'dhw_demand_pred', 'cooling_demand_pred', 'solar_generation_pred']:
                    data = getattr(self, col + '_pred')
                    scaled_data = data / max(max(data), 1e-3) * getattr(self, col + '_cap')
                    setattr(self, col + '_pred', scaled_data)
    
        if not self.data_path:
            csv_path = files("IES.data").joinpath(f"{self.DATASET_NAME}/Building_{int(self.station_id[9:])}.csv")
            df = pd.read_csv(csv_path)
            df.fillna(0.1, inplace=True)

        else:
            df = pd.read_csv(Path(self.data_path))
        
        for col in df.columns.to_list():
            setattr(self, col, df[col][self.simulation_start_time_step : self.simulation_end_time_step + 24].to_numpy())
            if col in ['non_shiftable_load', 'heating_demand', 'dhw_demand', 'cooling_demand', 'solar_generation']:
                 data = getattr(self, col)
                 scaled_data = data / max(max(data), 1e-3) * getattr(self, col + '_cap')
                 setattr(self, col, scaled_data)

        self.heating_demand = self.heating_demand + self.dhw_demand  # 结合两种需求，预测数据已经包含

        if not self.components['non_shiftable_load']['active']:
            self.non_shiftable_load = np.zeros_like(self.non_shiftable_load)
            self.non_shiftable_load_pred = np.zeros_like(self.non_shiftable_load)
        if not self.components['heating_demand']['active']:
            self.heating_demand = np.zeros_like(self.non_shiftable_load)
            self.heating_demand_pred = np.zeros_like(self.non_shiftable_load)
        if not self.components['cooling_demand']['active']:
            self.cooling_demand = np.zeros_like(self.non_shiftable_load)
            self.cooling_demand_pred = np.zeros_like(self.non_shiftable_load)
        if not self.components['solar_generation']['active']:
            self.solar_generation = np.zeros_like(self.non_shiftable_load)
            self.solar_generation_pred = np.zeros_like(self.non_shiftable_load)

        
        self.observations = df.columns.to_list() 

    def get_observation_min_max(self):
        min_value = 0
        max_value = 0
        for key, value in self.components.items():
            if value['active']:
                max_value = max(max(getattr(self, key)), max_value)
                min_value = min(min(getattr(self, key)), min_value)
        return max_value + 1e-3, min_value


    

class Pricing:
    def __init__(self, 
                 components, 
                 simulation_start_time_step, simulation_end_time_step, 
                 data_path=None, 
                 DATASET_NAME=None, 
                 pred_data=None
            ):
        self.components = components
        self.simulation_start_time_step = simulation_start_time_step
        self.simulation_end_time_step =  simulation_end_time_step
        self.data_path = data_path
        self.pred_data = pred_data
        self.DATASET_NAME = DATASET_NAME
        self.get_data()
    
    def get_data(self):
        if self.pred_data:
            if hasattr(self.pred_data, 'electricity_price') or 'electricity_price' in self.pred_data:
                self.electricity_price_pred = self.pred_data['electricity_price']
            else:
                first_key = next(iter(self.pred_data))  # 获取字典中的第一个键
                shape = np.shape(self.pred_data[first_key])  # 获取其值的维度
                self.electricity_price_pred = np.zeros(shape)

        if not self.data_path:
            csv_path = files("IES.data").joinpath(f"{self.DATASET_NAME}/pricing.csv")

            df = pd.read_csv(csv_path)
            df.fillna(0.1, inplace=True)

        else:
            df = pd.read_csv(Path(self.data_path))
        self.electricity_price = df['electricity_price'][self.simulation_start_time_step : self.simulation_end_time_step + 24].to_numpy()

        if not self.components['power_market']['active']:
            self.electricity_price = np.zeros_like(self.electricity_price)
    
    def reset(self):
        self.time_step = 0
    
    def step(self, action=None):
        self.time_step += 1
        # action.update({'electricity_price': self.electricity_price})

        

class Weather:
    def __init__(self, 
                 components, 
                 simulation_start_time_step, simulation_end_time_step, 
                 data_path=None, 
                 DATASET_NAME=None, 
                 station_id=None, 
                 pred_data=None
            ):
        """
            pred_data: {'': , '': ,...}
        """

        self.components = components
        self.station_metadata = station_metadata()
        self.simulation_start_time_step = simulation_start_time_step
        self.simulation_end_time_step =  simulation_end_time_step
        self.data_path = data_path
        self.pred_data = pred_data
        self.DATASET_NAME = DATASET_NAME
        self.get_data()
        self.observation_max, self.observation_min = self.get_observation_min_max()

    def reset(self):
        self.time_step = 0
    
    def step(self, action=None):
        self.time_step += 1
        # action.update({key: getattr(self, key)[self.time_step] for key in self.observations})

    def get_data(self):

        if self.pred_data:
            if hasattr(self.pred_data, 'direct_solar_irradiance') or 'direct_solar_irradiance' in self.pred_data:
                self.direct_solar_irradiance_pred = self.pred_data['direct_solar_irradiance']
            else:
                first_key = next(iter(self.pred_data))  # 获取字典中的第一个键
                shape = np.shape(self.pred_data[first_key])  # 获取其值的维度
                self.direct_solar_irradiance_pred = np.zeros(shape)

        if not self.data_path:
            csv_path = files("IES.data").joinpath(f"{self.DATASET_NAME}/weather.csv")
            df = pd.read_csv(csv_path)
            df.fillna(0.1, inplace=True)

        else:
            df = pd.read_csv(Path(self.data_path))
        
        for col in df.columns.to_list():
            setattr(self, col, df[col][self.simulation_start_time_step : self.simulation_end_time_step + 24].to_numpy())

        self.observations = df.columns.to_list() 
    
    def get_observation_min_max(self):
        min_value = 0
        max_value = 0
        for key, value in self.components.items():
            if value['active']:
                max_value = max(max(getattr(self, key)), max_value)
                min_value = min(min(getattr(self, key)), min_value)
        return max_value + 1e-3, min_value


class solar_thermal_collector:
    def __init__(self, weather, component):
        self.weather = weather
        self.components = component
        self.station_metadata = station_metadata()
        self.area = self.station_metadata.solar_thermal_collector.solar_heat_collector_area
        self.eta = self.station_metadata.solar_thermal_collector.eta_solar_heat
        self.get_data()

    def get_data(self):
        if hasattr(self.weather, 'direct_solar_irradiance_pred'):
            self.P_solar_heat_pred = self.weather.direct_solar_irradiance_pred * self.area * self.eta / max(self.weather.direct_solar_irradiance_pred)
        
        if  not self.components['direct_solar_irradiance']['active']:
            self.P_solar_heat = np.zeros_like(self.weather.direct_solar_irradiance)
        else:
            self.P_solar_heat = self.weather.direct_solar_irradiance * self.area * self.eta / max(self.weather.direct_solar_irradiance)
    
    def reset(self):
        self.time_step = 0


    def step(self, actions):
        self.time_step += 1
        # actions.update({'P_solar_heat': self.P_solar_heat})
    