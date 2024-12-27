"""
    TESS 设备参数配置文件
"""

import pandas as pd

class TESSConfig:
    capacity  = 100
    P_ch_max  = 50
    P_dis_max = 50
    eta_ch    = 0.9
    eta_dis   = 0.9
    soc_init  = 0.1 
    soc_min   = 0
    soc_max   = 1.0