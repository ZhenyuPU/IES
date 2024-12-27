"""
    Training parameter configuration
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



import numpy as np
import torch
import torch.nn.functional as F

class Config:
    ###################################### ENV PARAMETERS (if continuous days)##############################################
    DAY_COUNT = 21
    DAY_START = 80
    TIME_STEP_PER_DAY = 24
    START_TIME_STEP = DAY_START * TIME_STEP_PER_DAY
    END_TIME_STEP = (DAY_START + DAY_COUNT) * TIME_STEP_PER_DAY

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ###################################### COMMON TRAINING PARAMETERS ##############################################
    seed = 42
    num_episodes = 3000
    iter_num     = 10

    batch_size   = 1024
    buffer_size  = 100000
    minimal_size = 10000
    gamma     = 0.95
    actor_lr  = 0.0004
    critic_lr = 0.005
    tau       = 0.001  # Soft update parameter

    hidden_dim = 128
    
    action_bound = 1
    scaled_action_indices = np.array([0, 1, 2], dtype=np.int32)
    scaled_action_indices_tanh = np.array([3, 4, 5, 6], dtype=np.int32)

    ###################################### COMMON TEST PARAMETERS ##############################################
    START_DAY  = 100
    test_episodes = 30   # 测试天数
    test_iter_num = 10    # 设置迭代次数（任意值）

    ###################################### DDPG ##############################################
    sigma = 0.02   # 噪声
    ###################################### PPO ##############################################
    lmbda = 0.9
    epochs = 5
    repeat_times = 2 ** 3
    eps = 0.2
    if_per_or_gae = False

    ###################################### SAC ##############################################
    alpha_lr = 5e-4

    ###################################### TD7 ##############################################
    exploration_noise = 0.1
    target_update_rate = 250

    target_policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    alpha = 0.4
    min_priority = 1
    lmbda_td7 = 0.1

    # Checkpointing
    max_eps_when_checkpointing = 20
    steps_before_checkpointing = 75e4
    reset_weight = 0.9

    # Encoder Model
    zs_dim = 256
    enc_hdim = 256
    enc_activ = F.elu
    encoder_lr = 3e-4

    # Critic Model
    critic_hdim = 256
    critic_activ = F.elu
    critic_lr = 3e-4

    # Actor Model
    actor_hdim = 256
    actor_activ = F.relu
    actor_lr = 3e-4

    ###################################### PRED ##############################################
    units = 6
    pred = 'LSTM'
    time_step = 24
    horizon = 24
    dropout_rate = 0.5
    leaky_rate = 0.2
    hidden_size = 64
    output_size = 256

