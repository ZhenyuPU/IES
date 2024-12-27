# Integrated Energy System Environment

## 我的代码框架

### 功能介绍

功能: 实现综合能源系统环境

综合能源系统配件:

- PV
- solar thermal collector
- electrical storage
- Hydrogen electrolyzer
- hydrogen tank
- fuel cell CHP
- heating storage 
- absorption chiller
- cooling storage
- non_shiftable_load
- heating demand (heating demand + dwt demand)
- cooling demand

### 框架

IES/

├── agent/ (rl, mpc control)

│   ├── ddpg.py

│   ├── sac.py

│   ├── ppo.py

│   ├── mpc.py


├── mpc_env/

│   ├── StationOptEnv.py


├── rl_env/

│   ├── StationLearnEnv.py


├── Utils/

│   ├── station_metadata.py

│   ├── schema.json (存储组件信息)

│   ├── rl_utils.py

│   ├── test_utils.py


├── dataset.py (处理数据)

├── data/ (数据保存)


example/

├── main_MPC.ipynb (MPC测试)

├── main_RL.ipynb  (RL测试)


可选功能:
- 只包含电力的能源系统
- 包含电, 氢 系统
- - 包含热、冷系统
- 包含电, 氢, 热系统
- 包含电, 氢, 热, 冷系统

数据集:
- 自动选择数据集 (默认操作)
- 手动选择, 给出合适的数据集(结构需要与package中的数据集匹配)

rl定义:
- 自动选择状态空间和动作空间, 其中动作空间受到状态空间影响, 根据定义的状态空间定义动作
- 状态空间: 详见shema.json
- 动作空间: 详见shema.json
- 归一化: sin-cos and min-max

mpc定义:
- 组件选择：详见shema.json







