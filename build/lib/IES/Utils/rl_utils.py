from tqdm import tqdm
import streamlit as st
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt
import pandas as pd
from Utils.param_config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self): 
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 
    
    @property
    def size(self):
         return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def add_exploration_noise(env, action, num_episodes, current_episode):
    k = ( np.log(3) - np.log(0.02) )/(0.2 * num_episodes)
    sigma = np.maximum( 3 * np.exp(-k * current_episode), 0.02) 
    action = action + sigma * np.random.randn(env.action_dim)          # 给动作添加噪声，增加探索
    return action


def train_off_policy_agent(env, agent, replay_buffer,
                           file_name, save_name = None):
    seed = Config.seed
    num_episodes = Config.num_episodes
    minimal_size = Config.minimal_size
    iter_num = Config.iter_num

    set_seed(seed)
    return_list    = []
    trace_list     = []            
    violation_list = []
    cost_list      = []
    for i in range(iter_num):
        with tqdm(total=int(num_episodes/iter_num), desc='Iteration %d' % i) as pbar:     
            for i_episode in range( int(num_episodes/iter_num) ):
                current_episode = num_episodes/iter_num * i + i_episode
                episode_return    = 0               # 计算每个episode的汇报
                episode_violation = 0               # 计算每个episode的约束违反惩罚
                episode_cost      = 0               # 计算每个episode累积费用
                ep_step = 0

                trace = {}                          # store the epoch trajectories
                trace['day']       = []                 
                trace['state' ]    = []
                trace['action']    = []
                trace['decision']  = []
                trace['reward']    = []
                trace['violation'] = []
                trace['cost']      = []
                
                state = env.reset()  
                trace['day'].append(env.day) 
                done = False 
                while not done:
                    ep_step += 1
                    action = agent.take_action(state)
                    action = add_exploration_noise(env, action, num_episodes, current_episode)
                    next_state, reward, done, violation, cost = env.step(action)     # next_state已归一化
                    
                    trace['state'].append(env.state)            # 存储当前状态 (未归一化状态)
                    trace['action'].append(action)              # 存储当前行动
                    trace['decision'].append(env.decisions)          # 存储当前决策
                    trace['reward'].append(reward)              # 存储当前收益
                    trace['violation'].append(violation)
                    trace['cost'].append(cost)
                    
                    replay_buffer.add(state, action, reward / 1e2, next_state, done) 
                    state                = next_state
                    episode_return      += reward
                    episode_violation   += violation
                    episode_cost        += cost
                    # 训练模式启动更新
                    if replay_buffer.size > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample()
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                
                return_list.append(episode_return)
                trace_list.append(trace)
                violation_list.append(episode_violation )
                cost_list.append(episode_cost )
                
                if (i_episode + 1) % iter_num == 0:
                    pbar.set_postfix({
                        'Iteration': i,
                        'Episode': f'{i_episode + 1}/{int(num_episodes/iter_num)}',
                        'Return': f'{np.mean(return_list[-10:]):.3f}',
                        'Cost': f'{np.mean(cost_list[-10:]):.3f}',
                        'Violation': f'{np.mean(violation_list[-10:]):.3f}'
                    })
                pbar.update(1)
    if save_name == None:
        save_name = file_name

    agent.save_results(file_name)   # 保存网络参数
    # 保存训练数据
    np.savez(f'result/rl_model/train_data_{file_name}.npz', 
                return_list = return_list,
                trace_list = trace_list,
                violation_list = violation_list,
                cost_list = cost_list)
    print('Finished training and Models Saved !')
    return return_list, trace_list, violation_list, cost_list



def test_off_policy_agent(env, agent, file_name, start_day):
    seed = Config.seed
    set_seed(seed)
    return_list    = []
    trace_list     = []             # 存储(states, actions, rewards, violations, costs)
    violation_list = []
    cost_list      = []
    agent.load_results(file_name) 
    num_episodes = Config.test_episodes
    iter_num = Config.test_iter_num

    for i in range(iter_num):
        with tqdm(total=int(num_episodes/iter_num), desc='Iteration %d' % i) as pbar:     
            for i_episode in range( int(num_episodes/iter_num) ):
                current_episode = num_episodes/iter_num * i + i_episode
                episode_return    = 0               # 计算每个episode的汇报
                episode_violation = 0               # 计算每个episode的约束违反惩罚
                episode_cost      = 0               # 计算每个episode累积费用
                ep_step = 0

                trace = {}                          # store the epoch trajectories
                trace['day']       = []                 
                trace['state' ]    = []
                trace['action']    = []
                trace['decision']  = []
                trace['reward']    = []
                trace['violation'] = []
                trace['cost']      = []     
                
                state = env.reset(start_day) 
                trace['day'].append(env.day) 
                start_day += 1
                done = False 
                while not done:
                    ep_step += 1
                    action = agent.take_action(state)  
                    next_state, reward, done, violation, cost = env.step(action)     # next_state已归一化
                    
                    trace['state'].append(env.state)            # 存储当前状态 (未归一化状态)
                    trace['action'].append(action)              # 存储当前行动
                    trace['decision'].append(env.decisions)          # 存储当前决策
                    trace['reward'].append(reward)              # 存储当前收益
                    trace['violation'].append(violation)
                    trace['cost'].append(cost)

                    state                = next_state
                    episode_return      += reward
                    episode_violation   += violation
                    episode_cost        += cost
                
                return_list.append(episode_return)
                trace_list.append(trace)
                violation_list.append(episode_violation )
                cost_list.append(episode_cost )
                
                pbar.set_postfix({
                    'Iteration': i,
                    'Episode': f'{i_episode + 1}/{int(num_episodes/iter_num)}',
                    'Return': f'{np.mean(return_list):.3f}',
                    'Cost': f'{np.mean(cost_list):.3f}',
                    'Violation': f'{np.mean(violation_list):.3f}'
                })
                pbar.update(1)
    if save_name == None:
        save_name = file_name

    np.savez(f'result/rl_test/test_data_{save_name}.npz', 
                reward = return_list,
                cost = cost_list,
                violation = violation_list,
                trace = trace_list)
    print('Finished testing and Traces Saved !' )
    return return_list, trace_list, violation_list, cost_list




class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		device,
		max_size=1e6,
		batch_size=256,
		max_action=1,
		normalize_actions=True,
		prioritized=True
	):
	
		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.prioritized = prioritized
		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.max_priority = 1

		self.normalize_actions = max_action if normalize_actions else 1
        
	
	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action/self.normalize_actions
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		
		if self.prioritized:
			self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
        

	def sample(self):
		if self.prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device)*csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
		else:
			self.ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)


	def update_priority(self, priority):
		self.priority[self.ind] = priority.reshape(-1).detach()
		self.max_priority = max(float(priority.max()), self.max_priority)


	def reset_max_priority(self):
		self.max_priority = float(self.priority[:self.size].max())

  
