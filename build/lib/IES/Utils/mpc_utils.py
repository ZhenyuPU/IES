import numpy as np
from IES.agent.mpc import MPC_Controller



def run_MPC(self, env):
    mpc_controller = MPC_Controller()
    observations = env.reset()

    while not env.terminated:
        action = mpc_controller.run()
        observations, reward, done, _ = env.step(action)
        
    