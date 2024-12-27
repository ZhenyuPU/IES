import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import rcParams

from IES.Utils.station_metadata import HIES_Options
from IES.rl_env.StationLearnEnv import EnergyHubEnvLearn

from Utils import rl_utils
from Utils.prediction_utils import Args



def plot_all(*data_arrays, labels, title='Reward', start_idx=200, lim=[-10000, -1000]):
    # 设置全局字体为 Times New Roman
    rcParams['font.family'] = 'Times New Roman'
    # 确保输入数据和标签长度一致
    if not data_arrays:
        raise ValueError("Please input at least one array!")
    
    # 如果没有提供标签，自动生成默认标签
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(data_arrays))]
    elif len(labels) != len(data_arrays):
        raise ValueError("数据数组数量与标签数量不一致！")
    
    data_arrays = [data[start_idx:] for data in data_arrays]
    episodes_list = list(range(len(data_arrays[0])))

    colors = plt.cm.tab10.colors  # 获取颜色循环表

    # 平滑数据
    smoothed_data_arrays = [rl_utils.moving_average(data, 99) for data in data_arrays]

    # 创建主图
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    for i, (data, smooth_data, label) in enumerate(zip(data_arrays, smoothed_data_arrays, labels)):
        color = colors[i % len(colors)]  # 循环取颜色
        ax.fill_between(episodes_list, data, data, color=color, alpha=0.2)
        ax.plot(episodes_list[:len(smooth_data)], smooth_data, color=color, label=label)

    ax.set_xlabel('Episode', fontsize=24)
    ax.set_ylabel(f'{title}', fontsize=24)
    ax.legend(
        loc='lower right',
        fontsize='large',
        prop={'size': 12, 'weight': 'bold', 'family': 'serif'},
        title_fontsize=14
    )
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 嵌套子图
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="center right")

    for i, (data, smooth_data, label) in enumerate(zip(data_arrays, smoothed_data_arrays, labels)):
        color = colors[i % len(colors)]  # 循环取颜色
        ax_inset.plot(episodes_list[:len(data)], data, color=color, alpha=0.3)
        ax_inset.plot(episodes_list[:len(smooth_data)], smooth_data, color=color, label=label)

    ax_inset.set_xlim(5000, 6000)
    ax_inset.set_ylim(lim[0], lim[1])
    # ax_inset.set_title("Zoomed Inset", fontsize=10)
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    # 显示图像
    plt.tight_layout()
    plt.savefig(f'result/comparison/train_curve_{title}')
    plt.show()




def _box_plot(*args, name='reward', labels=None, capping_factor=1.5):
    """
    绘制多个算法的箱线图，并对数据进行截断处理。

    Parameters:
        *args: 任意多个数组，每个数组表示一种算法的结果。
        name (str): 保存文件时使用的名称，默认为'reward'。
        labels (list): 算法的标签，如果没有提供，默认使用'Algorithm1', 'Algorithm2', ...
        capping_factor (float): 截断因子，默认为1.5，表示IQR外1.5倍的位置。
    
    """
    rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 如果没有提供labels，则自动生成标签
    if labels is None:
        labels = [f'Algorithm{i+1}' for i in range(len(args))]
    
    # 确保输入的数组数量和标签数量一致
    assert len(args) == len(labels), "The number of labels must match the number of input arrays."
    
    # 将输入的数组转换为字典，并构造DataFrame
    data = {label: arg for label, arg in zip(labels, args)}
    
    # 对每个数组进行截断处理
    for label, arg in data.items():
        Q1 = np.percentile(arg, 25)
        Q3 = np.percentile(arg, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - capping_factor * IQR
        upper_bound = Q3 + capping_factor * IQR
        # 截断数据
        data[label] = np.clip(arg, lower_bound, upper_bound)
    
    # 将截断后的数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 绘制箱线图
    plt.grid(visible=True, linestyle='--', linewidth=1.0, alpha=0.7)
    sns.boxplot(data=df)
    plt.ylabel(f'{name}', fontsize=16)
    # plt.xlabel('Algorithm')  # 可选：如果需要显示算法名
    plt.savefig(f'result/rl_test/test_boxplot_{name}.png')
    plt.show()



# def get_decisions(DAYS, trace_list, T=24):
#     env = HIES_Env()
#     env_options = HIES_Options(program='test')
#     S_bss  = np.zeros( (DAYS, T) )
#     S_hss  = np.zeros( (DAYS, T) )
#     S_tes  = np.zeros( (DAYS, T) )
#     S_css  = np.zeros( (DAYS, T) )
#     P_bssc = np.zeros( (DAYS, T) )
#     P_bssd = np.zeros( (DAYS, T) )

#     p_t   =  np.zeros( (DAYS, T) ) 
#     T_ely =  np.zeros( (DAYS, T) )
#     T_tank =  np.zeros( (DAYS, T) )
#     C =  np.zeros( (DAYS, T) )

#     P_EL  =  np.zeros( (DAYS, T) )
#     P_FC  =  np.zeros( (DAYS, T) )
#     P_g   =  np.zeros( (DAYS, T) )
#     g_AC  =  np.zeros( (DAYS, T) )
#     g_FC  =  np.zeros( (DAYS, T) )

#     g_tesc = np.zeros( (DAYS, T) )
#     g_tesd = np.zeros( (DAYS, T) )

#     q_cssc = np.zeros( (DAYS, T) )
#     q_cssd = np.zeros( (DAYS, T) )
#     m_buy    = np.zeros( (DAYS, T) )
#     P_EL   = np.zeros( (DAYS, T) )
#     P_FC   = np.zeros( (DAYS, T) )
#     P_CO   = np.zeros( (DAYS, T) )
#     v_el   = np.zeros( (DAYS, T) )
#     i_el   = np.zeros( (DAYS, T) )
#     v_cdg   = np.zeros( (DAYS, T) )
#     i_fc   = np.zeros( (DAYS, T) )
#     v_fc   = np.zeros( (DAYS, T) )

#     reward = np.zeros( (DAYS, T) )
#     violation = np.zeros( (DAYS, T), dtype = int)
#     for day in range(DAYS):
#         trace = trace_list[day]
#         for t in range(T):
#             P_bssc[day][t] =  trace['decision'][t]['P_bssc']
#             P_bssd[day][t] =  trace['decision'][t]['P_bssd']
#             g_tesc[day][t] =  trace['decision'][t]['g_tesc']
#             g_tesd[day][t] =  trace['decision'][t]['g_tesd']
#             P_g[day][t]    =  trace['decision'][t]['P_g']
#             # P_CO[day][t]   =  trace['decision'][t]['P_CO']
#             g_AC[day][t]   =  trace['decision'][t]['g_AC']
#             g_FC[day][t]   =  trace['decision'][t]['g_FC']
#             p_t[day][t]    =  trace['decision'][t]['p_t']    # p_t 
            
#             q_cssc[day][t] =  trace['decision'][t]['q_cssc']
#             q_cssd[day][t] =  trace['decision'][t]['q_cssd']
#             m_buy[day][t]    =  trace['decision'][t]['m_buy']
#             P_EL[day][t]   =  trace['decision'][t]['P_el']
#             P_FC[day][t]   =  trace['decision'][t]['P_fc']

#             v_el[day][t]   =  trace['decision'][t]['v_el']
#             i_el[day][t]   =  trace['decision'][t]['i_el']
#             v_cdg[day][t]   =  trace['decision'][t]['v_cdg']
#             i_fc[day][t]   =  trace['decision'][t]['i_fc']
#             v_fc[day][t]   =  trace['decision'][t]['v_fc']

#             idx = -env.BESS_state_dim - env.TESS_state_dim - env.CESS_state_dim- env.HESS_state_dim - 2
#             S_bss[day][t] =  trace['state'][t][idx:idx + env.BESS_state_dim] 
#             idx += env.BESS_state_dim
#             S_tes[day][t] =  trace['state'][t][idx:idx + env.TESS_state_dim]
#             idx += env.TESS_state_dim
#             S_css[day][t] =  trace['state'][t][idx:idx + env.CESS_state_dim]
#             idx += env.CESS_state_dim

#             S_hss[day][t]   =  trace['state'][t][idx:idx + env.HESS_state_dim][0]
#             T_ely[day][t]   =  trace['state'][t][idx:idx + env.HESS_state_dim][1]
#             C[day][t]   =  trace['state'][t][idx:idx + env.HESS_state_dim][2]
#             T_tank[day][t]   =  trace['state'][t][idx:idx + env.HESS_state_dim][3]

#             reward[day][t] = trace['reward'][t]
#             violation[day][t] = trace['violation'][t]

#     q_AC = g_AC * env_options.eta_AC
#     P_g_buy = np.maximum(P_g, 0)
#     P_g_sell = np.minimum(P_g, 0)
#     decisions = {
#         "P_g_buy": P_g_buy,
#         "P_g_sell": P_g_sell,
#         "P_bssc": P_bssc,
#         "P_bssd": P_bssd,
#         "m_buy": m_buy,
#         "P_EL": P_EL,
#         "P_FC": P_FC,
#         "g_FC": g_FC,
#         "g_tesc": g_tesc,
#         "g_tesd": g_tesd,
#         "g_AC": g_AC,
#         "q_AC": q_AC,
#         "q_cssc": q_cssc,
#         "q_cssd": q_cssd,
#         "S_bss": S_bss,
#         "S_hss": S_hss,
#         "S_tes": S_tes,
#         "S_css": S_css,
#         "p_t": p_t,
#         "T_ely": T_ely,
#         "T_tank": T_tank,
#         "C": C,

#         "v_el": v_el,
#         "i_el": i_el,
#         "v_cdg": v_cdg,
#         "v_fc": v_fc,
#         "i_fc": i_fc,
#     }

#     return decisions




def get_decisions_MPC(DAYS, trace_list, T=24):
    env_options = HIES_Options(program='test')
    S_bss  = np.zeros( (DAYS, T+1) )
    S_hss  = np.zeros( (DAYS, T+1) )
    S_tes  = np.zeros( (DAYS, T+1) )
    S_css  = np.zeros( (DAYS, T+1) )
    P_bssc = np.zeros( (DAYS, T) )
    P_bssd = np.zeros( (DAYS, T) )
    m_buy  = np.zeros( (DAYS, T) )

    p_t   =  np.zeros( (DAYS, T+1) ) 
    T_ely =  np.zeros( (DAYS, T+1) )
    T_tank =  np.zeros( (DAYS, T+1) )
    C =  np.zeros( (DAYS, T+1) )

    P_EL  =  np.zeros( (DAYS, T) )
    P_FC  =  np.zeros( (DAYS, T) )
    P_g_buy   =  np.zeros( (DAYS, T) )
    P_g_sell  =  np.zeros( (DAYS, T) )
    g_AC  =  np.zeros( (DAYS, T) )
    g_FC  =  np.zeros( (DAYS, T) )

    g_tesc = np.zeros( (DAYS, T) )
    g_tesd = np.zeros( (DAYS, T) )

    q_cssc = np.zeros( (DAYS, T) )
    q_cssd = np.zeros( (DAYS, T) )
    P_EL   = np.zeros( (DAYS, T) )
    P_FC   = np.zeros( (DAYS, T) )
    P_CO   = np.zeros( (DAYS, T) )
    v_el   = np.zeros( (DAYS, T) )
    i_el   = np.zeros( (DAYS, T) )
    v_cdg   = np.zeros( (DAYS, T) )
    i_fc   = np.zeros( (DAYS, T) )
    v_fc   = np.zeros( (DAYS, T) )

    reward = np.zeros( (DAYS, T) )
    violation = np.zeros( (DAYS, T), dtype = int)

    for day in range(DAYS):
        trace = trace_list[day]
        P_bssc[day] =  trace['P_bssc']
        P_bssd[day] =  trace['P_bssd']
        g_tesc[day] =  trace['g_tesc']
        g_tesd[day] =  trace['g_tesd']
        P_g_buy[day]    =  trace['P_g_buy']
        P_g_sell[day]   = trace['P_g_sell']
        # P_CO[day]   =  trace['P_CO']
        g_AC[day]   =  trace['g_AC']
        g_FC[day]   =  trace['g_fc']
        p_t[day]    =  trace['p']    # p_t 

        v_el[day]   =  trace['v_el']
        i_el[day]   =  trace['i_el']
        v_cdg[day]   =  trace['v_cdg']
        i_fc[day]   =  trace['i_fc']
        v_fc[day]   =  trace['v_fc']
        
        q_cssc[day] =  trace['q_cssc']
        q_cssd[day] =  trace['q_cssd']
        # m_B[day]    =  trace['m_B']
        P_EL[day]   =  trace['P_el']
        P_FC[day]   =  trace['P_fc']
        m_buy[day]  =  trace['m_buy']

        S_bss[day] =  trace['S_bss']
        S_tes[day] =  trace['S_tes']
        S_css[day] =  trace['S_css']

        S_hss[day]   =  trace['m_hss']
        T_ely[day]   =  trace['T_el']
        C[day]   =  trace['C']
        T_tank[day]   =  trace['T_tank']

    q_AC = g_AC * env_options.eta_AC
    decisions = {
        "P_g_buy": P_g_buy,
        "P_g_sell": P_g_sell,
        "P_bssc": P_bssc,
        "P_bssd": P_bssd,
        "m_buy": m_buy,
        "P_EL": P_EL,
        "P_FC": P_FC,
        "g_FC": g_FC,
        "g_tesc": g_tesc,
        "g_tesd": g_tesd,
        "g_AC": g_AC,
        "q_AC": q_AC,
        "q_cssc": q_cssc,
        "q_cssd": q_cssd,
        "S_bss": S_bss,
        "S_hss": S_hss,
        "S_tes": S_tes,
        "S_css": S_css,
        "p_t": p_t,
        "T_ely": T_ely,
        "T_tank": T_tank,
        "C": C,

        "v_el": v_el,
        "i_el": i_el,
        "v_cdg": v_cdg,
        "v_fc": v_fc,
        "i_fc": i_fc,
    }

    return decisions


# def test_RL(model, save_file, selected_day=70):
#     # 设置随机种子
#     seed = 42
#     rl_utils.set_seed(seed)

#     ### 创建训练环境
#     env_options    = HIES_Options(program='test')
#     penalty_factor = training_config['penalty_factor']


#     ### 定义训练参数
#     hidden_dim    =  training_config['hidden_dim']
#     tau           =  training_config['tau']        
#     buffer_size   =  training_config['buffer_size']
#     minimal_size  =  training_config['minimal_size']
#     batch_size    =  training_config['batch_size']
#     sigma         =  training_config['sigma']
#     actor_lr      = training_config['actor_lr']
#     critic_lr     = training_config['critic_lr']
#     gamma         = training_config['gamma']
#     file_name     = 'DDPG_model'
#     action_bound  = training_config['action_bound']
#     device        =  training_config['device']
#     replay_buffer =  rl_utils.ReplayBuffer( buffer_size )
#     # PPO特有参数定义
#     lmbda         = training_config['lmbda']
#     epochs        = training_config['epochs']
#     eps           = training_config['eps']
#     # SAC特有参数定义
#     alpha_lr       = training_config['alpha_lr']

#     scaled_action_indices = training_config['scaled_action_indices']


#     ### 定义TGCN预测参数
#     units        = training_config['units']
#     stack_cnt    = training_config['stack_cnt']
#     time_step    = training_config['time_step']
#     horizon      = training_config['horizon']
#     dropout_rate = training_config['dropout_rate']
#     leaky_rate   = training_config['leaky_rate']
#     hidden_size  = training_config['hidden_size']
#     output_size = training_config['output_size']
#     scaled_action_indices_tanh = training_config['scaled_action_indices_tanh']


#     state_dim      = time_step * 6 + 8
#     action_dim     = 7
#     env            = HIES_Env(program='test', K=time_step, state_dim=state_dim, action_dim=action_dim)                 # 创建一个HEMS系统（训练环境）

#     target_entropy = -env.action_dim
#     ### 设置回溯步长K
#     K = env.K
#     pred           = training_config['pred']
#     # 测试参数
#     selected_day = selected_day # start_day
#     num_episodes  = 30   # end_day - start_day
#     iter_num      = 10

#     if model == 'DDPG':

#         agent = DDPG(state_dim, hidden_dim, hidden_size, action_dim, 
#                     action_bound, scaled_action_indices, 
#                     sigma, actor_lr, critic_lr, tau, gamma, 
#                     units, stack_cnt, env.K, horizon, dropout_rate, leaky_rate, 
#                     device, pred=pred)  ### 创建训练agent


#         return_list_DDPG, trace_list_DDPG, violation_list_PPO, cost_list_DDPG = rl_utils.train_off_policy_agent(
#             env, agent, num_episodes, 
#             replay_buffer, minimal_size, batch_size, 
#             program = 'test', 
#             seed=seed, 
#             file_name = f'{model}_Model', 
#             save_name = save_file,
#             iter_num=iter_num,
#             day=selected_day
#         )
#     elif model =='SAC':
#         alpha_lr = 3e-4
#         target_entropy = -env.action_dim

#         agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
#                             actor_lr, critic_lr, alpha_lr, target_entropy, tau,
#                             gamma, scaled_action_indices_tanh , device, hidden_size, units, K, horizon, pred, output_size)

#         return_list_SAC, trace_list_SAC, violation_list_SAC, cost_list_SAC = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, 
#             minimal_size, batch_size, 
#             program = 'test', seed=seed, file_name = f'{model}_model', 
#             save_name=save_file,
#             iter_num=iter_num, 
#             day=selected_day)
        

# def test_opt(model, save_file, selected_day=70):
#     start_day = selected_day
#     end_day = start_day + 30
#     day = 0
#     selected_day = start_day + day

#     env_options = HIES_Options(program='test')
#     args = Args()
#     Solver = MILP_Solver.Solver(pred_method='LSTM')
#     if model == 'OPT':
#         reward_opt, violation_opt, cost_opt, decision_opt = Solver.run_OPT_model(start_day=start_day, end_day=end_day, save_file=save_file)