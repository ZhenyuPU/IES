from IES.Utils.station_metadata import station_metadata
from IES.dataset import energy_simulation, Pricing, Weather, solar_thermal_collector
from IES.rl_env.HESS_env import HESS_env, hydrogen_market
from IES.rl_env.BESS_env import BESS_env
from IES.rl_env.TESS_env import TESS_env, absorption_chiller
from IES.rl_env.CESS_env import CESS_env


import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import json
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




class EnergyHubEnvLearn(gym.Env):
    def __init__(self, observation = None,
                 action_name = None,
                 DATASET_NAME = None,
                 station_id = None,
                 data_path = None,
                 simulation_start_time_step=None,
                 simulation_end_time_step=None, pred_data=None) -> None:
        gym.Env.__init__(self) 

        self.observation = self.get_observation(observation)
        self.action_name = self.get_action_name(action_name)

        self.station_metadata  =   station_metadata()
        self.station_id = station_id
        self.simulation_start_time_step = simulation_start_time_step
        self.simulation_end_time_step = simulation_end_time_step

        self.energy_simulation = energy_simulation(components=self.observation, simulation_start_time_step=self.simulation_start_time_step, simulation_end_time_step=self.simulation_end_time_step, DATASET_NAME=DATASET_NAME, station_id=self.station_id, data_path=data_path, pred_data=pred_data)
        self.pricing = Pricing(components=self.observation, simulation_start_time_step=self.simulation_start_time_step, simulation_end_time_step=self.simulation_end_time_step, DATASET_NAME=DATASET_NAME, data_path=data_path, pred_data=pred_data)
        self.weather = Weather(components=self.observation, simulation_start_time_step=self.simulation_start_time_step, simulation_end_time_step=self.simulation_end_time_step, DATASET_NAME=DATASET_NAME, data_path=data_path, pred_data=pred_data)
        self.solar_thermal_collector = solar_thermal_collector(self.weather, self.observation)

        self.power_market = power_market(self.pricing)
        self.hydrogen_market = hydrogen_market()

        self.electrical_storage = BESS_env()
        self.hydrogen_storage = HESS_env(self.energy_simulation, self.hydrogen_market)
        self.heating_storage = TESS_env()
        self.absorption_chiller = absorption_chiller()
        self.cooling_storage = CESS_env()

        # TODO self.state_space, self.action_space

    def reset(self):
        self.time_step = 0
        self.power_market.reset()
        self.hydrogen_market.reset()
        self.energy_simulation.reset()
        self.weather.reset()
        self.pricing.reset()

        self.power_market.reset()
        self.hydrogen_market.reset()
        self.electrical_storage.reset()
        self.hydrogen_storage.reset()
        self.heating_storage.reset()
        self.absorption_chiller.reset()
        self.cooling_storage.reset()

        self.update_states()

        return self.norm(self.state_kwargs)
    
    def update_states(self):
        self.state_kwargs = AttrDict()
        for key, value in self.observation.items():
            if value['active']:
                if any(sub in key for sub in ['month', 'day_type', 'hour', 'load', 'demand', 'solar_generation', 'indoor_dry_bulb_temperature']):
                    self.state_kwargs[key] = getattr(self.energy_simulation, key)[self.time_step]
                elif 'storage' in key:
                    self.state_kwargs[key] = getattr(self, key).soc
                elif 'power_market' in key:
                    self.state_kwargs[key] = self.pricing.electricity_price[self.time_step]
                else:
                    self.state_kwargs[key] = getattr(self.weather, key)[self.time_step]


    
    def norm(self, state_kwargs):
        state_norm = AttrDict()
        # sin-cos归一化
        for key, value in state_kwargs.items():
            if any(sub in key for sub in ['month', 'hour', 'day_type']):
                max_value = max(getattr(self.energy_simulation, key))
                state_norm[f'{key}_sin'] = self._sin_norm(value, max_value)
                state_norm[f'{key}_cos'] = self._cos_norm(value, max_value)
            elif any(sub in key for sub in ['load', 'demand', 'solar_generation', 'indoor_dry_bulb_temperature']):
                value_max = max(getattr(self.energy_simulation, key))
                value_min = min(getattr(self.energy_simulation, key))
                state_norm[f'{key}'] = self.min_max(value, value_min, value_max)
            elif 'power_market' in key:
                value_max = max(self.pricing.electricity_price)
                value_min = min(self.pricing.electricity_price)
                state_norm[f'{key}'] = self.min_max(value, value_min, value_max)
            elif 'storage' in key:
                state_norm[f'{key}'] = value
            else:
                value_max = max(getattr(self.weather, key))
                value_min = min(getattr(self.weather, key))
                state_norm[f'{key}'] = self.min_max(value, value_min, value_max)
        return np.array(list(state_norm.values()))
        
    

    def _sin_norm(self, data_K, data):
        return np.sin(2 * np.pi * data_K / np.max(data))

    def _cos_norm(self, data_K, data):
        return np.cos(2 * np.pi * data_K / np.max(data))
    
    def min_max(self, data, data_min, data_max):
        return (data - data_min) / (data_max - data_min + 1e-3)

        
    def step(self, actions):
        self.action_mapping(actions)
        self.energy_simulation.step(actions)
        self.weather.step(actions)
        self.pricing.step(actions)
        
        self.power_market.step(self.action_kwargs)
        self.hydrogen_market.step(self.action_kwargs)
        self.electrical_storage.step(self.action_kwargs)
        self.hydrogen_storage.step(self.action_kwargs)
        self.heating_storage.step(self.action_kwargs)
        self.absorption_chiller.step(self.action_kwargs)
        self.cooling_storage.step(self.action_kwargs)
        
        reward, violation = self.reward()
        self.time_step += 1
        self.update_states()

        done = True if self.time_step == self.simulation_end_time_step - self.simulation_start_time_step else False 

        return self.norm(self.state_kwargs), reward, done, violation
    
    def reward(self):
        cost = self.power_market.reward() + self.hydrogen_market.reward() + self.electrical_storage.reward() + self.hydrogen_storage.reward() + self.heating_storage.reward() + self.cooling_storage.reward()

        balance_elec = np.max(self.hydrogen_storage.P_el + self.electrical_storage.P_bssc + self.energy_simulation.non_shiftable_load[self.time_step] - self.power_market.P_g - self.energy_simulation.solar_generation[self.time_step] - self.hydrogen_storage.P_fc - self.electrical_storage.P_bssd, 0)

        balance_heat = np.max(self.heating_storage.g_tesc + self.energy_simulation.heating_demand[self.time_step] + self.absorption_chiller.g_ac - (self.hydrogen_storage.g_fc + self.heating_storage.g_tesd + self.solar_thermal_collector.P_solar_heat[self.time_step]), 0)

        balance_cool = np.max((self.cooling_storage.q_cssc + self.energy_simulation.cooling_demand[self.time_step]) - (self.absorption_chiller.q_ac  + self.cooling_storage.q_cssd), 0)

        violation = balance_elec + balance_heat + balance_cool
        reward = - cost - violation
        return cost, violation

        
    @property
    def schema(self):
        file_path = pkg_resources.resource_filename(
            'IES', 'Utils/schema.json'
        )
        with open(file_path, 'r') as file:
            return json.load(file)

    def get_observation(self, observation):
        if observation is None:
            return self.schema['observations']
        
        observation_all = self.schema['observations']
        for key, value in observation_all.items():
            if key not in observation:
                observation_all[key]['active'] = False
        return observation_all
    

    def action_mapping(self, actions):
        self.action_kwargs = {}
        i = 0
        for idx, (key, value) in enumerate(self.action_name.items()):
            if value['active']:
                self.action_kwargs[key] = actions[i]
                i += 1
            else:
                self.action_kwargs[key] = None

    def get_action_name(self, action_name):
        if action_name is None:
            return self.schema['actions']
        action_all = self.schema['actions']
        for key, value in action_all.items():
            if key not in action_name:
                action_all[key]['active'] = False
        return action_all
    


class power_market:
    def __init__(self, pricing):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.power_market
        self.pricing = pricing
        self.time_step = 0
    
    def reset(self):
        self.time_step = 0
        self.traces = AttrDict(
            P_g_buy = [],
            P_g_sell = [],
        )
    
    def step(self, actions):
        a_g = actions['power_market'] if actions['power_market'] else 0
        self.P_g = max(a_g * self.params.p_grid_max, 0)
        self.traces.P_g_buy.append(max(self.P_g, 0))
        self.traces.P_g_sell.append(- min(self.P_g, 0))
        self.time_step += 1

    def reward(self):
        return (self.pricing.electricity_price[self.time_step-1] - self.params.selling_price)/2 * np.abs(self.P_g) + (self.pricing.electricity_price[self.time_step-1] + self.params.selling_price)/2 * self.P_g
    




    
# class HIES_Env(HESS_env, BESS_env, TESS_env, CESS_env, gym.Env):
#     def __init__(self, 
#                  observations = None, 
#                  action_name = None, 
#                  program='train', 
#                  K=1, 
#                  data_path = None,
#                  start_time_step = 0, 
#                  end_time_step = 24,
#                  penalty_cost = None,
#                  penalty_violation = None):
        
#         HESS_env.__init__(self, program=program, data_path=data_path)
#         BESS_env.__init__(self, program=program, data_path=data_path)
#         TESS_env.__init__(self, program=program, data_path=data_path)
#         CESS_env.__init__(self, program=program, data_path=data_path)
#         gym.Env.__init__(self)  # 初始化 gym.Env

#         self.dynamic_observations = ["electricity_price", "pv_power", "electrical_demand", "heating_demand", "cooling_demand", "T_a_indoor", "outdoor_drybulb_temperature", "outdoor_relative_humidity"]

#         self.env_options = HIES_Options(program, data_path)
#         self.MAX_DAY = self.env_options.MAX_DAY # 数据集最大天数
#         self.day = 0        # 选择episode dayz
#         self.time_step = 0  # 每一个episode从0开始
#         self.K = K          # 考虑历史K步信息
#         self.DELTA = 1      # time interval

#         self.program  = program     # 获取训练模式

#         self.penalty_cost = penalty_cost
#         self.penalty_vio  = penalty_violation

#         self.start_time_step = start_time_step  # 数据开始时间步
#         self.end_time_step = end_time_step      # 数据结束时间步
#         self.start_day = self.start_time_step // 24         # 开始天
#         self.end_day   = self.end_time_step // 24           # 结束天
#         self.T = 23 #self.end_time_step - self.start_time_step         # episode time step

#         self.observations = observations if observations is not None else self.env_options.OBSERVATIONS     # 获取选择的状态
#         self.action_name = action_name
#         self.action_dict = self.__action()
    
#         self.state_dim = self._state_dim()

#     @property
#     def action_dim(self):
#         return len(self.action_name)


#     def get_info(self):
#         return self.day, self.time_step

#     @property
#     def schema(self):
#         file_path = pkg_resources.resource_filename(
#             'IES', 'Utils/HIES_config/schema.json'
#         )
#         with open(file_path, 'r') as file:
#             return json.load(file)

#     def observation_all(self):
#         # 将所有状态初始化为 True
#         observation_all = self.schema['observations']
        
#         # 遍历 self.observation_all 中的每个状态
#         for key, value in observation_all.items():
#             # 检查该状态是否在 self.observation 中被定义
#             if key not in self.observations:
#                 # 如果该状态没有被定义，设置其 active 为 False
#                 observation_all[key]['active'] = False
#             if key in self.observations and key in self.dynamic_observations:
#                 observation_all[key]['dim'] = self.K

#         return observation_all
    
#     def action_all(self):
#         # 将所有状态初始化为 True
#         action_all = self.schema['actions']
        
#         # 遍历 self.observation_all 中的每个状态
#         for key, value in action_all.items():
#             # 检查该状态是否在 self.observation 中被定义
#             if key not in self.action_name:
#                 # 如果该状态没有被定义，设置其 active 为 False
#                 action_all[key]['active'] = False
#         return action_all
    
#     def __action(self):
#         return self.action_all()

#     def __observation(self):
#         """
#         调用 observation_all 来更新并返回 self.observation_all 中的状态。
#         假设 observation_all 返回一个字典类型。
#         """
#         return self.observation_all()

#     def _state_dim(self):
#         """
#         计算所有活动状态的维度总和。
#         """
#         dim = 0
#         self._observation = self.__observation()  # 获取 observation 的字典
#         for k, v in self._observation.items():  # 遍历字典中的键值对
#             if self._observation[k]['active']:  # 检查是否为激活状态
#                 dim += v['dim']  # 累加维度
#         return dim

#     # 获取数据的历史K步信息
#     def get_K_step(self, data, day, time_step, K):
#         t_start = (day * 24 + time_step - K + 1) % 24
#         day_start = (day * 24 + time_step - K + 1) // 24
#         if day_start == day:
#             array = data[day][t_start:time_step+1]
#         else:
#             start_row = data[day_start][t_start:]   # 开始的那一天
#             end_row = data[day][:time_step+1]       # 结束的那一天
#             mid_row = data[day_start+1:day]         # 如果时间尺度大于24h否则是[]
#             if mid_row.size > 0:
#                 array = np.hstack([start_row] + list(mid_row) + [end_row])
#             else:
#                 array = np.hstack([start_row, end_row])
#             # 从历史数据开始那一天的一号位开始（有0号，但预测数据是从上一天的1点开始到第二天的零点）
#         return array
    
#     def reset(self, day=1):
#         """
#             重新选择某一天, 每个episode结束之后 ?
#         """
#         if self.program == 'train':
#             self.day = random.randint(1, 360)
#         # self.day = self.start_day
#         else:
#             self.day = day
#         self.time_step = 0

#         # 各组件环境初始化
#         if self._observation['HESS_states']['active']:
#             self.HESS_states = HESS_env.reset(self, day=self.day, time_step=self.time_step)
#             self.HESS_state_dim = len(self.HESS_states)
#         else:
#             self.HESS_states = None

#         if self._observation['BESS_states']['active']:
#             self.BESS_states = BESS_env.reset(self, day=self.day, time_step=self.time_step)
#             self.BESS_state_dim = len(self.BESS_states)
#         else:
#             self.BESS_states = None

#         if self._observation['TESS_states']['active']:
#             self.TESS_states = TESS_env.reset(self, day=self.day, time_step=self.time_step)
#             self.TESS_state_dim = len(self.TESS_states)
#         else:
#             self.TESS_states = None

#         if self._observation['CESS_states']['active']:
#             self.CESS_states = CESS_env.reset(self, day=self.day, time_step=self.time_step)
#             self.CESS_state_dim = len(self.CESS_states)
#         else:
#             self.CESS_states = None

#         self.state = self._build_states()
#         return  self._norm_old()
    
#     def _sin_norm(self, data_K, data):
#         return np.sin(2 * np.pi * data_K / np.max(data))

#     def _cos_norm(self, data_K, data):
#         return np.cos(2 * np.pi * data_K / np.max(data))
    
#     def _norm(self):
#         if self.observations == self.env_options.OBSERVATIONS:
            
#             lambda_b_sin      = self._sin_norm(self.state_dict['lambda_b'], self.env_options.lambda_b)
#             lambda_b_cos      = self._cos_norm(self.state_dict['lambda_b'], self.env_options.lambda_b)

#             P_solar_gen_sin   = self._sin_norm(self.state_dict['P_solar_gen'], self.env_options.P_solar_gen)
#             P_solar_gen_cos   = self._cos_norm(self.state_dict['P_solar_gen'], self.env_options.P_solar_gen)

#             Q_ED_sin          = self._sin_norm(self.state_dict['Q_ED'], self.env_options.Q_ED)
#             Q_ED_cos          = self._cos_norm(self.state_dict['Q_ED'], self.env_options.Q_ED)

#             Q_HD_sin          = self._sin_norm(self.state_dict['Q_HD'], self.env_options.Q_HD)
#             Q_HD_cos          = self._cos_norm(self.state_dict['Q_HD'], self.env_options.Q_HD)

#             Q_CD_sin          = self._sin_norm(self.state_dict['Q_CD'], self.env_options.Q_CD)
#             Q_CD_cos          = self._cos_norm(self.state_dict['Q_CD'], self.env_options.Q_CD)


#             T_a_sin           = self._sin_norm(self.state_dict['T_a'], self.env_options.T_a)
#             T_a_cos           = self._cos_norm(self.state_dict['T_a'], self.env_options.T_a)


#             T_dry_od_sin      = self._sin_norm(self.state_dict['T_dry_od'], self.env_options.T_dry_od)
#             T_dry_od_cos      = self._cos_norm(self.state_dict['T_dry_od'], self.env_options.T_dry_od)

#             humid_rela_od_sin = self._sin_norm(self.state_dict['humid_rela_od'], self.env_options.humid_rela_od)
#             humid_rela_od_cos = self._cos_norm(self.state_dict['humid_rela_od'], self.env_options.humid_rela_od)

#             day_sin           = self._sin_norm(self.state_dict['day'], 7)
#             day_cos           = self._cos_norm(self.state_dict['day'], 7)

#             hour_sin          = self._sin_norm(self.state_dict['hour'], 24)
#             hour_cos          = self._cos_norm(self.state_dict['hour'], 24)

#             BESS_states_norm  = BESS_env.norm_(self, self.BESS_states)
#             TESS_state_norm   = TESS_env.norm_(self, self.TESS_states)
#             CESS_states_norm  = CESS_env.norm_(self, self.CESS_states)
#             HESS_states_norm  = HESS_env.norm_(self, self.HESS_states)

#             state_norm = np.concatenate([lambda_b_sin, lambda_b_cos, P_solar_gen_sin, P_solar_gen_cos, Q_ED_sin, Q_ED_cos, Q_HD_sin, Q_HD_cos, Q_CD_sin, Q_CD_cos, T_a_sin, T_a_cos, T_dry_od_sin, T_dry_od_cos, humid_rela_od_sin, humid_rela_od_cos, BESS_states_norm, TESS_state_norm, CESS_states_norm, HESS_states_norm, [day_sin], [day_cos], [hour_sin], [hour_cos]])

#         return state_norm
    
#     def _norm_old(self):
#         """
#         Normalize the state variables based on their maximum values defined in the environment options.

#         Returns:
#             np.ndarray: Normalized state vector containing active components.
#         """
#         # Helper function to normalize a state component if observation is active
#         def normalize_state(key, state_value, max_value):
#             return state_value / max_value if self._observation[key]['active'] else None

#         # Normalize individual components
#         lambda_b = normalize_state('electricity_price', self.state_dict['lambda_b'], np.max(self.env_options.lambda_b[self.day-1:self.day+1]))
#         P_solar_gen = normalize_state('pv_power', self.state_dict['P_solar_gen'], self.env_options.norm_max)
#         Q_ED = normalize_state('electrical_demand', self.state_dict['Q_ED'], self.env_options.norm_max)
#         Q_HD = normalize_state('heating_demand', self.state_dict['Q_HD'], self.env_options.norm_max)
#         Q_CD = normalize_state('cooling_demand', self.state_dict['Q_CD'], self.env_options.norm_max)
#         T_a = normalize_state('T_a_indoor', self.state_dict['T_a'], self.env_options.norm_max)
#         T_dry_od = normalize_state('outdoor_drybulb_temperature', self.state_dict['T_dry_od'], np.max(self.env_options.norm_dynamics))
#         humid_rela_od = normalize_state('outdoor_relative_humidity', self.state_dict['humid_rela_od'], np.max(self.env_options.norm_dynamics))

#         # Normalize subsystem states
#         BESS_states_norm = BESS_env.norm_(self, self.BESS_states) if self._observation['BESS_states']['active'] else None
#         TESS_state_norm = TESS_env.norm_(self, self.TESS_states) if self._observation['TESS_states']['active'] else None
#         CESS_states_norm = CESS_env.norm_(self, self.CESS_states) if self._observation['CESS_states']['active'] else None
#         HESS_states_norm = HESS_env.norm_(self, self.HESS_states) if self._observation['HESS_states']['active'] else None

#         # Normalize time-related components
#         day = normalize_state('day', self.state_dict['day'], 7)
#         hour = normalize_state('hour', self.state_dict['hour'], 24)

#         # Combine all normalized components, filtering out None values
#         state_norm = np.concatenate([x for x in [
#             lambda_b, P_solar_gen, Q_ED, Q_HD, Q_CD, T_a, T_dry_od, humid_rela_od,
#             BESS_states_norm, TESS_state_norm, CESS_states_norm, HESS_states_norm,
#             day, hour
#         ] if x is not None])

#         return state_norm

    
#     def _build_states(self, initial=False):
#         """
#         Construct the state vector for the environment based on active observations.

#         Args:
#             initial (bool): Whether to reset subsystem states to their initial values.

#         Returns:
#             np.ndarray: Concatenated state vector containing the relevant state variables.
#         """
#         # Helper function to fetch data only if observation is active
#         def get_state_component(key, env_var):
#             return self.get_K_step(env_var, self.day, self._time_step, self.K) if self._observation[key]['active'] else None

#         # Retrieve active components for the state vector
#         lambda_b = get_state_component('electricity_price', self.env_options.lambda_b)  # Electricity buy price
#         P_solar_gen = get_state_component('pv_power', self.env_options.P_solar_gen)      # Solar power generation
#         Q_ED = get_state_component('electrical_demand', self.env_options.Q_ED)           # Electricity demand
#         Q_HD = get_state_component('heating_demand', self.env_options.Q_HD)              # Heating demand
#         Q_CD = get_state_component('cooling_demand', self.env_options.Q_CD)              # Cooling demand
#         T_a = get_state_component('T_a_indoor', self.env_options.T_a)                    # Indoor temperature
#         T_dry_od = get_state_component('outdoor_drybulb_temperature', self.env_options.T_dry_od)  # Outdoor dry-bulb temperature
#         humid_rela_od = get_state_component('outdoor_relative_humidity', self.env_options.humid_rela_od)  # Outdoor relative humidity

#         # Time-related components
#         day = np.array([self.time_step // 24 % 7]) if self._observation['day']['active'] else None  # Day of the week (0-6)
#         hour = np.array([self._time_step]) if self._observation['hour']['active'] else None         # Hour of the day

#         # Reset subsystem states if `initial` is True
#         if initial:
#             self.BESS_states = BESS_env.reset(self, day=self.day, time_step=self.time_step) if self._observation['BESS_states']['active'] else None
#             self.TESS_states = TESS_env.reset(self, day=self.day, time_step=self.time_step) if self._observation['TESS_states']['active'] else None
#             self.CESS_states = CESS_env.reset(self, day=self.day, time_step=self.time_step) if self._observation['CESS_states']['active'] else None
#             self.HESS_states = HESS_env.reset(self, day=self.day, time_step=self.time_step) if self._observation['HESS_states']['active'] else None

#         # Subsystem states
#         subsystem_states = [self.BESS_states, self.TESS_states, self.CESS_states, self.HESS_states]

#         # Combine all state components, filtering out None values
#         state_components = [lambda_b, P_solar_gen, Q_ED, Q_HD, Q_CD, T_a, T_dry_od, humid_rela_od, *subsystem_states, day, hour]
#         state = np.concatenate([component for component in state_components if component is not None])

#         # Save state dictionary for potential debugging or reference
#         self.state_dict = {
#             'lambda_b': lambda_b,
#             'P_solar_gen': P_solar_gen,
#             'Q_ED': Q_ED,
#             'Q_HD': Q_HD,
#             'Q_CD': Q_CD,
#             'T_a': T_a,
#             'T_dry_od': T_dry_od,
#             'humid_rela_od': humid_rela_od,
#             'BESS_states': self.BESS_states,
#             'TESS_states': self.TESS_states,
#             'CESS_states': self.CESS_states,
#             'HESS_states': self.HESS_states,
#             'day': day,
#             'hour': hour
#         }

#         return state


#     @property
#     def _time_step(self):
#         return self.time_step % 24
    

#     @property
#     def active_bess(self):
#         return 'BESS_states' in self.observations
    
#     @property
#     def active_tess(self):
#         return 'TESS_states' in self.observations
    
#     @property
#     def active_cess(self):
#         return 'CESS_states' in self.observations
    
#     @property
#     def active_hess(self):
#         return 'HESS_states' in self.observations
    

#     def get_decisions(self, actions):
#         """
#         Update decision variables based on the provided actions and active subsystems.

#         Args:
#             actions (tuple): Actions determined by the agent.
#         """
#         self.action_map(actions)        # 将action映射到对应的名字

#         # Handle Hydrogen Energy Storage System (HESS)
#         HESS_env.get_decisions(self, self.action_kwargs, self.HESS_states, self.decisions)
#         P_el = self.decisions['P_el']
#         P_fc = self.decisions['P_fc']

#         # Handle Battery Energy Storage System (BESS)
#         BESS_env.get_decisions(self, self.action_kwargs, self.BESS_states, self.decisions)
#         P_bssc = self.decisions['P_bssc']
#         P_bssd = self.decisions['P_bssd']

#         # Handle Thermal Energy Storage System (TESS)
#         TESS_env.get_decisions(self, self.action_kwargs, self.TESS_states, self.decisions)

#         # Handle Cooling Energy Storage System (CESS)
#         CESS_env.get_decisions(self, self.action_kwargs, self.CESS_states, self.decisions)

#         # Calculate power from the grid
#         if self.action_kwargs['power_grid']:
#             a_g = self.action_kwargs['power_grid']
#             P_g = max(a_g * self.env_options.P_g_max, 0)
#         else:
#             P_g = (
#                 self.electricity_demand +
#                 P_bssc + P_bssd + P_el - P_fc -
#                 self.pv_power
#             )

#         self.decisions['P_g'] = P_g


#     def action_map(self, actions):
#         self.action_kwargs = {}
#         i = 0
#         for idx, (key, value) in enumerate(self.action_dict.items()):
#             if value['active']:
#                 self.action_kwargs[key] = actions[i]
#                 i += 1
#             else:
#                 self.action_kwargs[key] = None



#     def step(self, actions):
#         # 获取决策变量，安全限制和决策相关量的计算
#         self.decisions = {}
#         self.get_decisions(actions)
#         # 获取reward
#         reward, cost, violation = self.get_reward()

#         HESS_env.step(self, self.HESS_states, self.decisions)
#         BESS_env.step(self, self.BESS_states, self.decisions)
#         TESS_env.step(self, self.TESS_states, self.decisions)
#         CESS_env.step(self, self.CESS_states, self.decisions)

#         if self.time_step < self.T:
#             self.time_step = self.time_step + 1  # 更新当前时刻
#             # self.day       = self.time_step // 24 + self.start_day
#             HESS_env.time_step = self.time_step
#             BESS_env.time_step = self.time_step
#             TESS_env.time_step = self.time_step
#             CESS_env.time_step = self.time_step
#             done = False
#             next_state = self._build_states()
#         else:
#             self.day        = self.day + 1
#             self.time_step  = 0
#             next_state      = self._build_states(initial=True)
#             done            = True

#         self.state = next_state
#         return self._norm_old(), reward, done, [cost, violation]
    
#     @property
#     def electricity_price(self):
#         return self.env_options.lambda_b[self.day][self._time_step]
    
#     @property
#     def electricity_demand(self):
#         return self.env_options.Q_ED[self.day][self._time_step]
    
#     @property
#     def pv_power(self):
#         return self.env_options.P_solar_gen[self.day][self._time_step]

#     @property
#     def _penalty_vio(self):
#         if self.penalty_vio is None:
#             return self.env_options.penalty
#         else:
#             return self.penalty_vio
    
#     @property
#     def _penalty_cost(self):
#         if self.penalty_cost is None:
#             soc_hss = self.HESS_states[0] / self.env_options.S_hss_max
#             penalty_ele = 1.0 
#             return penalty_ele
#         else:
#             return self.penalty_ele

#     def get_reward(self):
#         # 获取不同元器件的违反约束
#         violation = HESS_env.get_reward(self, self.HESS_states, self.decisions, self.active_hess) + BESS_env.get_reward(self) + TESS_env.get_reward(self, self.decisions, self.active_tess) + CESS_env.get_reward(self, self.decisions, self.active_cess) 

#         P_g      =  self.decisions['P_g'] 
#         m_buy    =  self.decisions['m_buy']
#         P_bssc   =  self.decisions['P_bssc']
#         P_bssd   =  self.decisions['P_bssd']
#         P_el     =  self.decisions['P_el']
#         P_fc     =  self.decisions['P_fc']

#         violation_P_g = max(self.electricity_demand + P_bssc + P_el - P_bssd - P_fc - self.pv_power - P_g, 0) if self.action_kwargs['power_grid'] else 0

#         # 总的违反约束
#         violation += violation_P_g
#         # 买电成本
#         Cost_ele = ( self.electricity_price - self.env_options.lambda_s )/2 * np.abs(P_g) + ( self.electricity_price + self.env_options.lambda_s )/2 * P_g
#         # 买氢成本
#         Cost_hss  = self.env_options.lambda_B * m_buy
#         cost = Cost_ele + Cost_hss
#         # reward
#         reward = - (Cost_ele * self._penalty_cost + Cost_hss + violation * self._penalty_vio)

#         return reward, cost, violation       

