"""
    CESS 设备参数配置文件
"""

import pandas as pd

class CESSConfig:
    capacity  = 200
    P_ch_max  = 40
    P_dis_max = 40
    eta_ch    = 0.9
    eta_dis   = 0.9
    soc_init  = 0.1 
    soc_min   = 0
    soc_max   = 1.0

    # absorption chiller
    g_ac_max = 200  # [kW]
    eta_ac  = 0.94