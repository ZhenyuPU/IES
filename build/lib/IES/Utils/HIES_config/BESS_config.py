"""
    BESS 设备参数配置文件
"""

import pandas as pd

class BESSConfig:
    eta_ch    = 0.95
    eta_dis   = 0.95
    capacity  = 50    
    P_ch_max  = 20
    P_dis_max = 20
    soc_init  = 0.1
    soc_min   = 0
    soc_max   = 1.0