# file for storing the device metadata
import pandas as pd


# electrical storage 
class station_metadata:
    def __init__(self):
        self.T = 24
        self.electrical_storage = electrical_storage()
        self.hydrogen_storage = hydrogen_storage()
        self.heating_storage = heating_storage()
        self.cooling_storage = cooling_storage()
        self.absorption_chiller = absorption_chiller()
        self.energy_simulation = energy_simulation()
        self.solar_generator = solar_generator()
        self.solar_thermal_collector = solar_thermal_collector()
        self.power_market = power_market()
        self.hydrogen_market = hydrogen_market()



class electrical_storage:
      eta_ch    = 0.95
      eta_dis   = 0.95
      capacity  = 50    
      P_ch_max  = 20
      P_dis_max = 20
      soc_init  = 0.1
      soc_min   = 0
      soc_max   = 1.0


class hydrogen_storage:
      ############################################################################## Electrolyzer   ##############################################################################
      # 电解槽设备参数（EL）
      P_el_low  = 15                              # 电解槽最低功率                          
      P_el_high = 200                            # 电解槽最大功率

      T_el_high = 70                             # 电解槽最大温度
      T_el_init = 20                             # 电解槽最低温度

      P_nom_el   = 100                             # 电解槽名义功率
      P_nom_el_1 = 25                            # 电解槽一个stack的名义功率

      # 电解槽输送氢气速率参数
      z1      = 1.618 * 10**(-5) * 3600                # m3/(℃h)      
      z0      = 1.490 * 10**(-2) * 3600                # m3/(j)  
      z_low   = 1.530 * 10**(-4) * 3600             # m3/(h kW)  
      z_high  = 1.195 * 10**(-4) * 3600            # m3/(h kW)  

      # 电解槽温度T_el动态变化
      j1 = 0.919                                  # -
      j2 = 7.572 * 10**(-3)                       # ℃/(kW)
      j0 = 3.958                                  # ℃

      # 电解槽电流i_el计算参数
      h1          = 0.6731                                  # A/℃                 
      h0          = 235.25                                  # A
      h_low       = 10.987                               # A/(kW)
      h_high      = 8.9992                              # A/(kW)
      i_nom_el    = 300                             # A

      # 电解槽电流过载公式参数
      C_high = 75                                 # Ah
      C_init = 0
      ############################################################################## 储氢罐和gas cleaner参数 ##############################################################################
      # 储氢罐的参数
      alpha   = 0.697           # CDG效率
      # 储氢罐压强变化参数
      b0      = 11.5 * 1e5 / 1e5       # m2/s2
      b1      = 4.16 * 1e3 / 1e5       # m2/(℃ s2)
      V_tank  = 10             # 储氢罐体积（m3)
      rho     = 8.99 * 1e-2   # 氢气密度（kg/m3)

      capacity     = 50
      p_tank_low   = 0              # 储氢罐压强最小值（bar)
      p_tank_high = 45             # 储氢罐压强最大值（bar)

      m_buy_max  = 50      # maximum hydorgen to buy or sell in the market [kg]
      soc_max    = 1.0
      soc_min    = 0.0
      soc_init   = 0.0     # initial soc of hydrogen tank

      # 储氢罐温度变化参数
      g0 = 0.94               # -
      g1 = 5.91 * 1e-2        # -
      T_tank_init = 20        # 初始温度

      ############################################################################## Fuel cell   ##############################################################################
      P_fc_low  = 30              # fc最小功率
      P_fc_high = 100             # fc最大功率

      c  = 0.21                # FC电流与氢气吸收速率系数 Nm3/C ???
      # FC电流计算相关参数
      s1 = 2.56               # 1/kV
      s2 = 3.31               # 1/kV
      u_bp_fc = 47.97         # 名义功率
      i_bp_fc = 122.8032        # 名义电流

      # FC热回收系数
      eta_fc = 0.3                               # 燃料电池氢气发电效率     
      eta_fc_rec = 0.8                           # 燃料电池热回收系数
      LHV_H2 = 33.2                              # 氢气低热值low heat value (单位：kWh/kg) 

class heating_storage:
      capacity  = 100
      P_ch_max  = 50
      P_dis_max = 50
      eta_ch    = 0.9
      eta_dis   = 0.9
      soc_init  = 0.1 
      soc_min   = 0
      soc_max   = 1.0

class cooling_storage:
      capacity  = 200
      P_ch_max  = 40
      P_dis_max = 40
      eta_ch    = 0.9
      eta_dis   = 0.9
      soc_init  = 0.1 
      soc_min   = 0
      soc_max   = 1.0

class absorption_chiller:
      g_ac_max = 200  # [kW]
      eta_ac  = 0.94

class energy_simulation:
      non_shiftable_load_cap = 200
      heating_demand_cap  = 100
      cooling_demand_cap = 150
      dhw_demand_cap = 150
      solar_generation_cap = 300

class solar_generator:
      solar_generation_cap = 1000


class solar_thermal_collector:
      solar_heat_collector_area = 400  # solar thermal collector area [m2]
      eta_solar_heat = 0.762


class power_market:
      p_grid_max = 500
      selling_price = 0

class hydrogen_market:
      p_hydro_max = 0  # [kg]
      hydrogen_pricing = 1.0
