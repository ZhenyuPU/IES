import gurobipy as gp
from  gurobipy import GRB
from collections import defaultdict
import numpy as np


class Opt_Solver:
    def __init__(self, env, horizon):
        self.env = env
        self.horizon = horizon


    def run(self):
        model = gp.Model('opt_model')
        model.setParam("OutputFlag", 0)
        model, obj = self.env.model(self.horizon, model)
        model.setObjective(obj, sense = GRB.MINIMIZE)
        model.optimize() 
        if  model.status == gp.GRB.OPTIMAL:
            decision_result = self.get_result(model)
        elif model.status == gp.GRB.INFEASIBLE:
            model.computeIIS()
            model.write("infeasible.ilp")  # 写入一个文件以分析不可行原因
            print("Model is infeasible, see infeasible.ilp for details")
            # 打开并读取 infeasible.ilp 文件
            with open("infeasible.ilp", "r") as file:
                content = file.readlines()
            # 打印文件内容
            for line in content:
                print(line.strip())
            print("model is infeasible!\n")
            raise RuntimeError("Terminating program due to infeasible model.")
        elif model.status == gp.GRB.UNBOUNDED:
            print("model is unbounded!\n")
            raise RuntimeError("Terminating program due to unbounded model.")
        else:
            print(f"Optimization ended with status: {model.status}\n" )
            raise RuntimeError("Terminating program due to unexpected optimization status.")  
        
        return decision_result


    def get_result(self, model):
        parsed_result = defaultdict(dict)
        
        # 遍历模型变量
        for var in model.getVars():
            var_name, *indices = var.VarName.replace("]", "").split("[")
            indices = tuple(map(int, indices[0].split(","))) if indices else ()
            if indices:
                parsed_result[var_name][indices] = var.X
            else:
                parsed_result[var_name] = var.X
        
        # 将 defaultdict(dict) 转换为所需的格式
        formatted_result = {
            var_name: np.array([value for _, value in sorted(values.items())])
            if isinstance(values, dict) else values
            for var_name, values in parsed_result.items()
        }
        
        return formatted_result




