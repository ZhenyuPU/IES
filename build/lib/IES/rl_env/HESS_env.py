from IES.Utils.station_metadata import station_metadata
import numpy as np
import random
import matplotlib.pyplot as plt
from attr_dict import AttrDict
"""
    States: S_hss, T_el, C, T_tank
    Actions: a_el, a_fc
    Power: P_el, P_fc
    normalization: unified
    Violation: 
"""
class HESS_env:
    def __init__(self, energy_simulation, hydrogen_market):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.hydrogen_storage
        self.hydrogen_market = hydrogen_market
        self.T_amb = energy_simulation.indoor_dry_bulb_temperature
        self.time_step = 0

        self.capacity = self.params.capacity

    def reset(self):
        self.time_step = 0
        self.soc = self.params.soc_init
        self.T_el   = self.params.T_el_init
        self.C      = self.params.C_init
        self.T_tank = self.params.T_tank_init

        p_tank_init = (self.params.b0 + self.params.b1 * self.T_tank) * (self.soc * self.params.capacity / self.params.V_tank)

        self.p_tank = p_tank_init

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
        self.traces.soc.append(self.params.soc_init)       # randomly initialize the hydrogen tank soc 
        self.traces.T_el.append( self.params.T_el_init )
        self.traces.C.append(self.params.C_init)
        self.traces.T_tank.append( self.params.T_tank_init )
    
    def step(self, actions):
        if actions['electrolyzer']:
            a_el = actions['electrolyzer']
            P_el = np.clip(self.params.P_el_low + (self.params.P_el_high - self.params.P_el_low) * a_el, 0, self.params.P_el_high) if a_el >= 0 else 0
        else:
            P_el = 0
        
        if actions['fuel_cell'] is not None:
            a_fc = actions['fuel_cell']
            P_fc = np.clip(self.params.P_fc_low + (self.params.P_fc_high - self.params.P_fc_low) * a_fc, 0, self.params.P_fc_high) if a_fc >= 0 else 0

        else:
            P_fc = 0

        hydr_purc = self.hydrogen_market.hydr_purc

        #################################### Electrolyzer #######################################

        # volume rate of hydrogen in the eletrolyzer
        if self.params.P_el_low <= P_el <= self.params.P_nom_el:
            v_el = self.params.z1 * self.T_el + self.params.z0 + self.params.z_low * (P_el - self.params.P_nom_el)
        elif self.params.P_nom_el < P_el <= self.params.P_el_high:
            v_el = self.params.z1 * self.T_el + self.params.z0 + self.params.z_high * (P_el - self.params.P_nom_el)     
        else:
            v_el = 0

        # stack 1 power
        if self.params.P_el_low <= P_el <= self.params.P_nom_el/4:
            P_el_1 = P_el
        elif self.params.P_nom_el / 4 < P_el <= self.params.P_nom_el:
            P_el_1 = self.params.P_nom_el / 4
        elif self.params.P_nom_el < P_el <= self.params.P_el_high:
            P_el_1 = P_el / 4
        else:
            P_el_1 = 0

        # Current
        if self.params.P_el_low <= P_el_1 <= self.params.P_nom_el_1:
            i_el = self.params.h1 * self.T_el + self.params.h0 + self.params.h_low * (P_el_1 - self.params.P_nom_el_1)
        elif self.params.P_nom_el_1 < P_el_1 <= self.params.P_el_high:
            i_el = self.params.h1 * self.T_el + self.params.h0 + self.params.h_high * (P_el_1 - self.params.P_nom_el_1)
        else:
            i_el = 0

        # overload dynamics
        C_next    = max(0, self.C + (i_el - self.params.i_nom_el) )

        # temperature of electrolyzer
        T_el_next = self.params.j1 * self.T_el + self.params.j2 * P_el + self.params.j0


        #################################### Fuel Cell #######################################

        # Current of FC
        if self.params.P_fc_low <= P_fc <= self.params.u_bp_fc:
            i_fc = self.params.s1 * P_fc
        elif self.params.u_bp_fc < P_fc <= self.params.P_fc_high:
            i_fc = self.params.s2 * (P_fc - self.params.u_bp_fc) + self.params.i_bp_fc
        else:
            i_fc = 0
        
        # volume rate of fuel cell
        v_fc = self.params.c * i_fc
        # 燃料电池产热量
        g_fc = (1 - self.params.eta_fc) * P_fc / self.params.eta_fc * self.params.eta_fc_rec if actions['heating_storage'] else 0 



         #################################### Gas cleaner and Tank ##############################

        # gas cleaner
        v_cdg = self.params.alpha * v_el

        # tank
        # p_tank = (self.params.b0 + self.params.b1 * self.T_tank) * (self.soc * self.params.capacity / self.params.V_tank)

        soc_next = self.soc + ( self.params.rho * (v_cdg - v_fc) + hydr_purc )/self.params.capacity

        T_tank_next = self.params.g0 * self.T_tank + self.params.g1 * self.T_amb[self.time_step]

        p_tank_next  = ( self.params.b0 + self.params.b1 * T_tank_next) * (soc_next * self.capacity / self.params.V_tank)  


        self.P_el   = P_el
        self.v_el   = v_el
        self.i_el   = i_el
        self.v_cdg  = v_cdg
        self.P_fc   = P_fc
        self.i_fc   = i_fc
        self.v_fc   = v_fc
        self.g_fc   = g_fc
        self.P_el_1 = P_el_1

        self.soc    = soc_next
        self.T_el   = T_el_next
        self.C      = C_next
        self.T_tank = T_tank_next
        self.p_tank = p_tank_next


        self.traces.P_el.append(P_el)
        self.traces.v_el.append(v_el)
        self.traces.i_el.append(i_el)
        self.traces.v_cdg.append(v_cdg)
        self.traces.P_fc.append(P_fc)
        self.traces.i_fc.append(i_fc)
        self.traces.v_fc.append(v_fc)
        self.traces.g_fc.append(g_fc)
        self.traces.P_el_1.append(P_el_1)

        self.traces.soc.append(soc_next)
        self.traces.T_el.append(T_el_next)
        self.traces.C.append(C_next)
        self.traces.T_tank.append(T_tank_next)
        self.traces.p_tank.append(p_tank_next)


        # decisions.P_el   = P_el
        # decisions.v_el   = v_el
        # decisions.i_el   = i_el
        # decisions.v_cdg  = v_cdg
        # decisions.P_fc   = P_fc
        # decisions.i_fc   = i_fc
        # decisions.v_fc   = v_fc
        # decisions.g_fc   = g_fc
        # decisions.P_el_1 = P_el_1

        # decisions.soc    = soc_next
        # decisions.T_el   = T_el_next
        # decisions.C      = C_next
        # decisions.T_tank = T_tank_next
        # decisions.p_tank = p_tank_next


        self.time_step += 1

    def reward(self):
        return 0
    


class hydrogen_market:
    def __init__(self):
        self.station_metadata = station_metadata()
        self.params = self.station_metadata.hydrogen_market
        self.time_step = 0
    
    def reset(self):
        self.time_step = 0
        self.traces = AttrDict(
            hydr_purc = [],
        )
    
    def step(self, actions):
        a_hydr_purc = actions['hydrogen_purchase'] if actions['hydrogen_purchase'] is not None else 0
        self.hydr_purc = max(a_hydr_purc * self.params.p_hydro_max, 0)
        # decisions.hydr_purc = self.hydr_purc
        self.traces.hydr_purc.append(self.hydr_purc)
        self.time_step += 1

    def reward(self):
        return self.hydr_purc * self.params.hydrogen_pricing