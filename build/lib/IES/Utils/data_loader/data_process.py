import numpy as np
import pandas as pd
import os

def build_data_process(source_path, save_path, program):
    solar_radiation = pd.DataFrame()
    outdoor_drybulb_temperature = pd.DataFrame()
    outdoor_relative_humidity = pd.DataFrame()
    electricity_demand = pd.DataFrame()
    heating_demand = pd.DataFrame()
    cooling_demand = pd.DataFrame()
    T_a = pd.DataFrame()
    for filename in os.listdir(source_path):
        if filename.startswith('Building') and filename.endswith('.csv'):
            filepath = os.path.join(source_path, filename)
            df = pd.read_csv(filepath)
            electricity_demand[filename] = df['Equipment Electric Power [kWh]']
            heating_demand[filename] = df['DHW Heating [kWh]']
            cooling_demand[filename] = df['Cooling Load [kWh]']
            T_a[filename] = df['Indoor Temperature [C]']
        elif filename == 'weather.csv':
            filepath = os.path.join(source_path, filename)
            df = pd.read_csv(filepath)
            solar_radiation = df['Direct Solar Radiation [W/m2]']
            outdoor_drybulb_temperature = df['Outdoor Drybulb Temperature [C]']
            outdoor_relative_humidity = df['Outdoor Relative Humidity [%]']
    electricity_demand = pd.DataFrame(electricity_demand.sum(axis=1))[:365*24]
    heating_demand = pd.DataFrame(heating_demand.sum(axis=1))[:365*24]
    cooling_demand = pd.DataFrame(cooling_demand.sum(axis=1))[:365*24]
    T_a = pd.DataFrame(T_a.mean(axis=1))[:365*24] 
    solar_radiation = solar_radiation[:365*24] 
    solar_radiation = pd.DataFrame(solar_radiation.to_numpy().reshape(int(len(solar_radiation)/24), 24))

    outdoor_drybulb_temperature = outdoor_drybulb_temperature[:365*24]
    outdoor_drybulb_temperature = pd.DataFrame(outdoor_drybulb_temperature.to_numpy().reshape(int(len(outdoor_drybulb_temperature)/24), 24))

    outdoor_relative_humidity = outdoor_relative_humidity[:365*24]
    outdoor_relative_humidity = pd.DataFrame(outdoor_relative_humidity.to_numpy().reshape(int(len(outdoor_relative_humidity)/24), 24))

    electricity_demand = pd.DataFrame(electricity_demand.to_numpy().reshape(int(len(electricity_demand)/24), 24))

    heating_demand = pd.DataFrame(heating_demand.to_numpy().reshape(int(len(heating_demand)/24), 24))

    cooling_demand = pd.DataFrame(cooling_demand.to_numpy().reshape(int(len(cooling_demand)/24), 24))

    T_a = pd.DataFrame(T_a.to_numpy().reshape(int(len(T_a)/24), 24))

    solar_radiation.to_csv(save_path+f'/solar_radiation_{program}.csv')
    outdoor_drybulb_temperature.to_csv(save_path+f'/outdoor_drybulb_temperature_{program}.csv')
    outdoor_relative_humidity.to_csv(save_path+f'/outdoor_relative_humidity_{program}.csv')
    electricity_demand.to_csv(save_path+f'/electricity_demand_{program}.csv')
    heating_demand.to_csv(save_path+f'/heating_demand_{program}.csv')
    cooling_demand.to_csv(save_path+f'/cooling_demand_{program}.csv')
    T_a.to_csv(save_path+f'/T_a_{program}.csv')





def price_process(source_path, save_path, program):
    df = pd.read_csv(source_path)
    electricity_price = df['system_energy_price_da'] / 100

    electricity_price = pd.DataFrame(electricity_price[:-1].to_numpy().reshape(int(len(electricity_price)/24), 24))
    electricity_price.to_csv(save_path+f'/electricity_price_{program}.csv')



"""
    price solar e_demand h_demand c_demand T_a

    8760(365 * 24), 6(node num)

"""


def graph_data_contruct(source_path, save_path, program):
    price       =   pd.read_csv(source_path+f'/electricity_price_{program}.csv', index_col=0).to_numpy().reshape(1, -1)

    solar_radiation =   pd.read_csv(source_path+f'/solar_radiation_{program}.csv', index_col=0).to_numpy().reshape(1, -1)
    outdoor_drybulb_temperature =   pd.read_csv(source_path+f'/outdoor_drybulb_temperature_{program}.csv', index_col=0).to_numpy().reshape(1, -1)
    outdoor_relative_humidity =   pd.read_csv(source_path+f'/outdoor_relative_humidity_{program}.csv', index_col=0).to_numpy().reshape(1, -1)

    e_demand    =   pd.read_csv(source_path+f'/electricity_demand_{program}.csv', index_col=0).to_numpy().reshape(1, -1)
    h_demand    =   pd.read_csv(source_path+f'/heating_demand_{program}.csv', index_col=0).to_numpy().reshape(1, -1)
    c_demand    =   pd.read_csv(source_path+f'/cooling_demand_{program}.csv', index_col=0).to_numpy().reshape(1, -1)
    T_a       =   pd.read_csv(source_path+f'/T_a_{program}.csv', index_col=0).to_numpy().reshape(1, -1)

    if program == 'train':
        dataset     =   np.concatenate((price, solar_radiation, outdoor_drybulb_temperature, outdoor_relative_humidity, e_demand, h_demand, c_demand, T_a), axis = 0).T
    else:
        # 裁剪数组到前 7632 行
        solar_radiation = solar_radiation[:, :7632]
        outdoor_drybulb_temperature = outdoor_drybulb_temperature[:, :7632]
        outdoor_relative_humidity = outdoor_relative_humidity[:, :7632]
        e_demand = e_demand[:, :7632]
        h_demand = h_demand[:, :7632]
        c_demand = c_demand[:, :7632]
        T_a = T_a[:, :7632]

        dataset     =   np.concatenate((price, solar_radiation, outdoor_drybulb_temperature, outdoor_relative_humidity, e_demand, h_demand, c_demand, T_a), axis = 0).T

    np.savetxt(save_path + f'/HIES_data_{program}.csv', dataset, delimiter=',')


def train_test_split():
    build_data_process(source_path='data/citylearn_challenge_2021', save_path='datasets/data', program='train')
    price_process(source_path='data/electricity_price/da_hrl_lmps.csv', save_path='datasets/data', program='train')
    graph_data_contruct(source_path='data/data', save_path='datasets', program='train')

    build_data_process(source_path='data/citylearn_challenge_2020_climate_zone_1', save_path='datasets/data', program='test')
    price_process(source_path='data/electricity_price/da_hrl_lmps_2024.csv', save_path='datasets/data', program='test')
    graph_data_contruct(source_path='data/data', save_path='datasets', program='test')

if __name__ == '__main__':
    # citylearn_challenge_2021 
    # citylearn_challenge_2022
    train_test_split()



        


