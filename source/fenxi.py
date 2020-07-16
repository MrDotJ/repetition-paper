import gurobipy as gurobi
import numpy as np
from copy import deepcopy as copy
import matplotlib.pyplot as plt


# success !
# need test for multi-area-multi-time
# TODO: why the ub and lb doesn't apply to the voltage_square ????
# TODO: p2 always be zero,  don't know why?

# first i change the generator cost
# second i change the load_conference

def print_info(info):
    print(info.this_voltage_square)
    print(info.that_voltage_square)
    print(info.power_flow)
    print(info.react_flow)
    print(info.this_node_pressure)
    print(info.that_node_pressure)
    print(info.gas_flow_in)
    print(info.gas_flow_out)


class connection_line_info:
    def __init__(self):
        self.this_voltage_square = 0.0
        self.that_voltage_square = 0.0
        self.power_flow = 0.0  # out -> to the connected node
        self.react_flow = 0.0  # out -> to the connected node
        self.this_node_pressure = 0.0
        self.that_node_pressure = 0.0
        self.gas_flow_in = 0.0
        self.gas_flow_out = 0.0


# 基本参数
plt.figure(figsize=(20, 10.5))
T = 1
g_link = 1
g_connection = [
    [1],
    [0]
]
player_num = 2

# 算法调节参数
g_tao = 100
PUNISH = 1
OUTER_LOOP = [1500, 3000, 30, 30] + [30] * 50
PCCP_COUNT = 1
G_K = 1.2

# 价格信息参数
power_price = [[0, 5], [5, 0]]  # 单位购电成本
well_output_price = 1  # 单位产气成本
LOAD_COEFF = 100  # 负荷效应系数
GAS_LOAD_COEFF = 100  # 负荷效应系数
GAS_PRICE = 2  # 天然气购买费用
g_gas_price_aux = [1, -1, 1]
GRID_PRICE  = 8
# 辅助数据结构
# TODO:       one_line_0_with_1          one_line_0_with_2
# [   [ [info_0 info_1 ... info_T], [info_0 info_1 ... info_T], ]                              <===== area_0
#            one_line_1_with_0          one_line_1_with_2           one_line_1_with_3
#     [ [info_0 info_1 ... info_T], [info_0 info_1 ... info_T], [info_0 info_1 ... info_T],]   <===== area_1
#            one_line_2_with_0          one_line_2_with_1
#     [ [info_0 info_1 ... info_T], [info_0 info_1 ... info_T],]  ]                            <===== area_2
# g_info[ 0 ]    [ 0 ]      [ 0 ]
#      area_0   0_with_i   time_0    ====> connection_line_info
g_info = [
    [
        [
            connection_line_info()
            for ____time in range(T)
        ] for ____line in range(len(g_connection[____area]))
    ] for ____area in range(len(g_connection))
]
# TODO: g_lam format:
#  [ [line1 [<> <> <> <> <> <> <> <>] | [<> <> <> <> <> <> <> <>] |...T...],
#   [line2 [<> <> <> <> <> <> <> <>] | [<> <> <> <> <> <> <> <>] |...T...],
#   [line3 [<> <> <> <> <> <> <> <>] | [<> <> <> <> <> <> <> <>] |...T...],
#   ...
#   [linen [<> <> <> <>] | [<> <> <> <>] |...T...]]
#  g_lam   [0]      [0]   = [x,      x,       x,            x  ,           x,            x,             x,       x]
#        0-line   0-time  this_v  that_v  power_f_in  react_f_in   this_pressure  that_pressure   flow_IN    flow_OUT
g_lam = [
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        for _____time in range(T)
    ] for ______line in range(g_link)
]

g_grid_power_in = [[0] * T for region___ in range(player_num)]
g_lam_power_grid = [0] * T
g_grid_react_in = [[0] * T for region____ in range(player_num)]
g_lam_react_grid = [0] * T
g_lam_power_grid_negative = [0] * T
g_lam_react_grid_negative = [0] * T


# g_lam_index format
# [ area1[line, line, line ...],
#   area2[line, line, line ...],
#   ...]
# g_lam_index [0]
#    all index for 0-area
g_lam_index = [
    [0],
    [0]
]

abcd = []

lam_h1 = []
lam_h2 = []
lam_h3 = []
lam_h4 = []
lam_h5 = []
lam_h6 = []
lam_h7 = []
lam_h8 = []
lam_h9 = []
lam_h10 = []
lam_h11 = []
lam_h12 = []
lam_h13 = []
lam_h14 = []
lam_h15 = []
lam_h16 = []
gxdiv100 = []
gxdiv101 = []
gxdiv102 = []
gxdiv103 = []
gxdiv104 = []
gxdiv105 = []
gxdiv106 = []
gxdiv107 = []
gxdiv108 = []
oldlam = []
oldlam1 = []
oldlam2 = []
oldlam3 = []
oldlam4 = []
oldlam5 = []
oldlam6 = []
oldlam7 = []
oldlam8 = []


class PowerNet:
    def __init__(self, system_info, node_info, line_info, gas_node_info, gas_line_info):
        self.index = system_info['index']
        self.T = system_info['T']

        # ------------- generator of non-gas ----------------------
        self.gen_num = node_info['gen_num']  # add virtual node at last as the connected node
        self.gen_index = node_info['gen_index']
        self.gen_power_min = node_info['gen_power_min']
        self.gen_power_max = node_info['gen_power_max']
        self.gen_react_min = node_info['gen_react_min']
        self.gen_react_max = node_info['gen_react_max']
        self.gen_cost_a = node_info['gen_cost_a']
        self.gen_cost_b = node_info['gen_cost_b']
        self.gen_cost_c = node_info['gen_cost_c']
        self.grid_index = [4]
        # ---------------- power bus node ---------------------------
        self.bus_num = node_info['bus_num']  # add virtual node at last
        self.bus_voltage_min = node_info['bus_voltage_min']
        self.bus_voltage_max = node_info['bus_voltage_max']
        # ----------------- power load -------------------------------
        self.load_num = node_info['load_num']
        self.load_index = node_info['load_index']
        self.load_power_min = node_info['load_power_min']
        self.load_power_max = node_info['load_power_max']
        self.load_react_min = node_info['load_react_min']
        self.load_react_max = node_info['load_react_max']
        # --------------- power connection info -----------------------
        self.bus_num_outside = node_info['bus_num_outside']
        self.connection_area = system_info['connection_area']
        self.connection_index = node_info['connection_index']
        # ------------------- power line info -------------------------
        self.line_num = line_info['line_num']
        self.line_current_capacity = line_info['line_current_capacity']
        self.line_start_point = line_info['line_start_point']
        self.line_end_point = line_info['line_end_point']
        self.line_resistance = line_info['line_resistance']
        self.line_reactance = line_info['line_reactance']
        # ------------------ gas node info -------------------------
        self.gas_node_num = gas_node_info['gas_node_num']
        self.node_pressure_min = gas_node_info['node_pressure_min']
        self.node_pressure_max = gas_node_info['node_pressure_max']
        # ------------------ gas well info -------------------------
        self.well_num = gas_node_info['gas_well_num']
        self.well_index = gas_node_info['well_index']  # [0,0,4,5]
        self.well_output_min = gas_node_info['well_output_min']
        self.well_output_max = gas_node_info['well_output_max']
        # ------------------ gas load info -------------------------
        self.gas_load_index = gas_node_info['load_index']
        self.gas_load_min = gas_node_info['gas_load_min']
        self.gas_load_max = gas_node_info['gas_load_max']
        self.gas_load_num = gas_node_info['gas_load_num']
        # ----------------- gas generator --------------------------
        self.gen_gas_num = gas_node_info['gen_gas_num']
        self.gen_gas_index = gas_node_info['gen_gas_index']
        self.gen_gas_index_power = gas_node_info['gen_gas_index_power']
        self.gen_gas_min = gas_node_info['gen_gas_min']
        self.gen_gas_max = gas_node_info['gen_gas_max']
        self.gen_gas_efficiency = gas_node_info['gen_gas_efficiency']
        # ----------------- gas line info -------------------------
        self.weymouth = gas_line_info['weymouth']  # for easy, it should contain all line(include active line)
        self.gas_line_num = gas_line_info['gas_line_num']
        self.gas_line_start_point = gas_line_info['gas_line_start_point']  # gas flow out
        self.gas_line_end_point = gas_line_info['gas_line_end_point']  # gas flow in
        self.gas_line_pack_coefficient = gas_line_info['gas_line_pack_coefficient']
        self.gas_line_pack_initial = gas_line_info['gas_line_pack_initial']
        self.gas_line_active = gas_line_info['gas_line_active']
        self.gas_flow_in_max = gas_line_info['gas_flow_in_max']
        self.gas_flow_out_max = gas_line_info['gas_flow_out_max']
        # ------------------- gas compressor info ------------------
        self.compressor_num = gas_line_info['compressor_num']
        self.compressor_start_point = gas_line_info['compressor_start_point']
        self.compressor_end_point = gas_line_info['compressor_end_point']
        self.compressor_coefficient = gas_line_info['compressor_coefficient']
        self.compressor_max_flow = gas_line_info['compressor_max_flow']
        self.compressor_energy_consumption = gas_line_info['compressor_energy_consumption']
        # ----------------------------------------gas information end
        # ------------------model------------------------------------
        self.model = gurobi.Model()
        self.basic_objective = None
        self.addition_objective = None
        self.objective = None
        self.constrain_update = []
        self.objs = []
        self.lams = []
        self.dual = []
        self.dual_addition = 0
        self.norm_addition = 0
        self.global_dual_addition_power = 0
        self.global_dual_addition_react = 0
        self.global_norm_addition_power = 0
        self.global_norm_addition_react = 0
        # -------------------- power system var -------------------
        self.power_gen = None
        self.react_gen = None
        self.voltage_square = None
        self.line_current_square = None
        self.line_power_flow = None
        self.line_react_flow = None
        self.power_load = None
        self.react_load = None
        self.power_from_grid = None
        self.react_from_grid = None
        # -------------------- gas system var ----------------------
        self.node_pressure = None
        self.well_output = None
        self.gas_load = None
        self.gen_gas_power = None
        self.gas_flow_in = None
        self.gas_flow_out = None
        self.linepack = None
        self.compressor_out = None
        self.compressor_in = None
        self.gas_source = None
        self.pccp = None
        # ------------------------ old info -------------------------
        self.info = [
            [
                connection_line_info()
                for _ in range(self.T)
            ] for __ in range(len(self.connection_area))
        ]
        # TODO: self.info  [0]                   [0]
        #          self.index_with_i    at     time_0
        self.old_value = [
            [
                connection_line_info()
                for _ in range(self.T)
            ] for __ in range(len(self.connection_area))
        ]
        # TODO: self.old_value   [0]                 [0]
        #                 self.index_with_i   at   time_0
        self.gas_flow_in_old = [
            [
                0.2 for _ in range(self.T)
            ] for __ in range(self.gas_line_num)
        ]
        self.gas_flow_out_old = [
            [
                0.2 for _ in range(self.T)
            ] for __ in range(self.gas_line_num)
        ]
        self.node_pressure_old = [
            [
                0.2 for _ in range(self.T)
            ] for __ in range(self.gas_node_num)
        ]
        self.old_power_from_grid = [0] * T
        self.old_react_from_grid = [0] * T

    # ---------- power system ---------------------------------------
    def power_gen_connected_with(self, node):
        result = np.where(np.array(self.gen_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.power_gen[i, time])
            result_list.append(per)
        return np.array(result_list)

    def react_gen_connected_with(self, node):
        result = np.where(np.array(self.gen_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.react_gen[i, time])
            result_list.append(per)
        return np.array(result_list)

    def load_power_connected_with(self, node):
        result = np.where(np.array(self.load_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.power_load[i, time])
            result_list.append(per)
        return np.array(result_list)

    def load_react_connected_with(self, node):
        result = np.where(np.array(self.load_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.react_load[i, time])
            result_list.append(per)
        return np.array(result_list)

    def grid_power_in_connected_with(self, node):
        result = np.where(np.array(self.grid_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.power_from_grid[time])
            result_list.append(per)
        return np.array(result_list)

    def grid_react_in_connected_with(self, node):
        result = np.where(np.array(self.grid_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.react_from_grid[time])
            result_list.append(per)
        return np.array(result_list)

    # power flow in/out of the node
    def power_flow_in_connected_with(self, node):
        result = np.where(np.array(self.line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_power_flow[i, time])
            result_list.append(per)
        return np.array(result_list)

    def power_flow_out_connected_with(self, node):
        result = np.where(np.array(self.line_start_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_power_flow[i, time])
            result_list.append(per)
        return np.array(result_list)

    def raect_flow_in_connected_with(self, node):
        result = np.where(np.array(self.line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_react_flow[i, time])
            result_list.append(per)
        return np.array(result_list)

    def react_flow_out_connected_with(self, node):
        result = np.where(np.array(self.line_start_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_react_flow[i, time])
            result_list.append(per)
        return np.array(result_list)

    def current_in_connected_with(self, node):
        result = np.where(np.array(self.line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_current_square[i, time])
            result_list.append(per)
        return np.array(result_list)

    def resistance_in_connected_with(self, node):
        result = np.where(np.array(self.line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_resistance[i])
            result_list.append(per)
        return np.array(result_list)

    def reactance_in_connected_with(self, node):
        result = np.where(np.array(self.line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per = []
            for time in range(self.T):
                per.append(self.line_reactance[i])
            result_list.append(per)
        return np.array(result_list)

    # ---------- gas system -----------------------------------------
    def well_connected_with(self, node):
        result = np.where(np.array(self.well_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_well = []
            for time in range(self.T):
                per_well.append(self.well_output[i, time])
            result_list.append(per_well)
        return np.array(result_list)

    def load_connected_with(self, node):
        result = np.where(np.array(self.gas_load_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_load = []
            for time in range(self.T):
                per_load.append(self.gas_load[i, time])
            result_list.append(per_load)
        return np.array(result_list)

    def p2g_connected_with(self, node):
        return np.array([[0] * self.T])

    def gas_flow_out_connected_with(self, node):  # 从这个节点流出 连接 start
        result = np.where(np.array(self.gas_line_start_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_out = []
            for time in range(self.T):
                per_out.append(self.gas_flow_in[i, time])
            result_list.append(per_out)
        return np.array(result_list)

    def gas_flow_in_connected_with(self, node):  # 流入这个节点
        result = np.where(np.array(self.gas_line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_in = []
            for time in range(self.T):
                per_in.append(self.gas_flow_out[i, time])
            result_list.append(per_in)
        return np.array(result_list)

    def gen_connected_with(self, node):  # list of expression
        result = np.where(np.array(self.gen_gas_index) == node)  # this node is gas node
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_gen = []
            for time in range(self.T):
                per_gen.append(self.gen_gas_power[i, time] / self.gen_gas_efficiency[i])  # change to gas
            result_list.append(per_gen)
        return np.array(result_list)

    def gas_to_power_connected_with(self, node):
        result = np.where(np.array(self.gen_gas_index_power) == node)  # this node is power node
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_gen = []
            for time in range(self.T):
                per_gen.append(self.gen_gas_power[i, time])  # just power
            result_list.append(per_gen)
        return np.array(result_list)

    # ----------- auxiliary key function ----------------------------
    def get_dual(self, this_info, that_info, start_point):
        # this or that of two areas is same!
        # 这里 我们 认为 that_info 始终遵循 一个 全局的 方向 即 0 的 this that 与 1 的 this that 始终 是 相同 的
        # 但是 this_info 却把 内部的节点 作为 this， 外部 作为 that
        if start_point != 0:
            diff1 = this_info.this_voltage_square - that_info.this_voltage_square
            diff2 = -1 * this_info.this_voltage_square + that_info.this_voltage_square
            diff3 = this_info.that_voltage_square - that_info.that_voltage_square
            diff4 = -1 * this_info.that_voltage_square + that_info.that_voltage_square
        else:
            diff1 = this_info.that_voltage_square - that_info.this_voltage_square
            diff2 = -1 * this_info.that_voltage_square + that_info.this_voltage_square
            diff3 = this_info.this_voltage_square - that_info.that_voltage_square
            diff4 = -1 * this_info.this_voltage_square + that_info.that_voltage_square

        if start_point != 0:  # this is start point
            diff5 = this_info.power_flow - that_info.power_flow
            diff6 = -1 * this_info.power_flow + that_info.power_flow
            diff7 = this_info.react_flow - that_info.react_flow
            diff8 = -1 * this_info.react_flow + that_info.react_flow
        else:
            diff5 = -1 * this_info.power_flow - that_info.power_flow
            diff6 = 1 * this_info.power_flow + that_info.power_flow
            diff7 = -1 * this_info.react_flow - that_info.react_flow
            diff8 = 1 * this_info.react_flow + that_info.react_flow

        #
        if start_point != 0:
            diff9 = this_info.this_node_pressure - that_info.this_node_pressure
            diff10 = -1 * this_info.this_node_pressure + that_info.this_node_pressure
            diff11 = this_info.that_node_pressure - that_info.that_node_pressure
            diff12 = -1 * this_info.that_node_pressure + that_info.that_node_pressure
        else:
            diff9 = this_info.that_node_pressure - that_info.this_node_pressure
            diff10 = -1 * this_info.that_node_pressure + that_info.this_node_pressure
            diff11 = this_info.this_node_pressure - that_info.that_node_pressure
            diff12 = -1 * this_info.this_node_pressure + that_info.that_node_pressure

        diff13 = this_info.gas_flow_in - that_info.gas_flow_in
        diff14 = -1 * this_info.gas_flow_in + that_info.gas_flow_in
        diff15 = this_info.gas_flow_out - that_info.gas_flow_out
        diff16 = -1 * this_info.gas_flow_out + that_info.gas_flow_out

        return [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8,
                1 * diff9, 1 * diff10, 1 * diff11, 1 * diff12,
                1 * diff13, 1 * diff14, 1 * diff15, 1 * diff16]

    def get_sub(self, this_info, this_info_old, start_point):
        # this_info_old 应该遵守全局 的 顺序
        diff = 0
        if start_point != 0:  # this is start point
            diff = diff + \
                   (this_info.this_voltage_square - this_info_old.this_voltage_square) * \
                   (this_info.this_voltage_square - this_info_old.this_voltage_square) + \
                   (this_info.that_voltage_square - this_info_old.that_voltage_square) * \
                   (this_info.that_voltage_square - this_info_old.that_voltage_square) + \
                   (this_info.power_flow - this_info_old.power_flow) * \
                   (this_info.power_flow - this_info_old.power_flow) + \
                   (this_info.react_flow - this_info_old.react_flow) * \
                   (this_info.react_flow - this_info_old.react_flow) + \
                   (this_info.this_node_pressure - this_info_old.this_node_pressure) * \
                   (this_info.this_node_pressure - this_info_old.this_node_pressure) + \
                   (this_info.that_node_pressure - this_info_old.that_node_pressure) * \
                   (this_info.that_node_pressure - this_info_old.that_node_pressure) + \
                   (this_info.gas_flow_in - this_info_old.gas_flow_in) * \
                   (this_info.gas_flow_in - this_info_old.gas_flow_in) + \
                   (this_info.gas_flow_out - this_info_old.gas_flow_out) * \
                   (this_info.gas_flow_out - this_info_old.gas_flow_out)
        else:
            diff = diff + \
                   (this_info.that_voltage_square - this_info_old.this_voltage_square) * \
                   (this_info.that_voltage_square - this_info_old.this_voltage_square) + \
                   (this_info.this_voltage_square - this_info_old.that_voltage_square) * \
                   (this_info.this_voltage_square - this_info_old.that_voltage_square) + \
                   (-1 * this_info.power_flow - this_info_old.power_flow) * \
                   (-1 * this_info.power_flow - this_info_old.power_flow) + \
                   (-1 * this_info.react_flow - this_info_old.react_flow) * \
                   (-1 * this_info.react_flow - this_info_old.react_flow) + \
                   (this_info.that_node_pressure - this_info_old.this_node_pressure) * \
                   (this_info.that_node_pressure - this_info_old.this_node_pressure) + \
                   (this_info.this_node_pressure - this_info_old.that_node_pressure) * \
                   (this_info.this_node_pressure - this_info_old.that_node_pressure) + \
                   (this_info.gas_flow_in - this_info_old.gas_flow_in) * \
                   (this_info.gas_flow_in - this_info_old.gas_flow_in) + \
                   (this_info.gas_flow_out - this_info_old.gas_flow_out) * \
                   (this_info.gas_flow_out - this_info_old.gas_flow_out)
        return diff

    def get_lam(self, index, start_point):
        lam = g_lam[index]
        lam_T = []
        for i in range(self.T):
            lam_copy = lam[i].copy()
            lam_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if start_point != 0:  # this is start_point
                lam_t = lam_copy
            else:
                lam_t[0] = lam_copy[1]
                lam_t[1] = lam_copy[0]
                lam_t[2] = lam_copy[3]
                lam_t[3] = lam_copy[2]
                lam_t[4] = lam_copy[5]
                lam_t[5] = lam_copy[4]
                lam_t[6] = lam_copy[7]
                lam_t[7] = lam_copy[6]
                lam_t[8] = lam_copy[9]
                lam_t[9] = lam_copy[8]
                lam_t[10] = lam_copy[11]
                lam_t[11] = lam_copy[10]
                lam_t[12] = lam_copy[13]
                lam_t[13] = lam_copy[12]
                lam_t[14] = lam_copy[15]
                lam_t[15] = lam_copy[14]
            lam_T.extend(lam_t)
        return lam_T

    def build_model(self):
        # add var
        self.power_gen = self.model.addVars(
            self.gen_num, self.T,
            lb=[[self.gen_power_min[i]] * self.T for i in range(self.gen_num)],
            ub=[[self.gen_power_max[i]] * self.T for i in range(self.gen_num)],
            name='power_gene')
        self.react_gen = self.model.addVars(
            self.gen_num, self.T,
            lb=[[self.gen_react_min[i]] * self.T for i in range(self.gen_num)],
            ub=[[self.gen_react_max[i]] * self.T for i in range(self.gen_num)],
            name='reactive_gene')
        self.power_load = self.model.addVars(
            self.load_num, self.T,
            lb=self.load_power_min,  # [[self.load_power_min[i]] * self.T for i in range(self.load_num)],
            ub=self.load_power_max,  # [[self.load_power_max[i]] * self.T for i in range(self.load_num)],
            name='power_load')
        self.react_load = self.model.addVars(
            self.load_num, self.T,
            lb=self.load_react_min,  # [[self.load_react_min[i]] * self.T for i in range(self.load_num)],
            ub=self.load_react_max,  # [[self.load_react_max[i]] * self.T for i in range(self.load_num)],
            name='react_load')
        self.voltage_square = self.model.addVars(
            self.bus_num, self.T,
            lb=[[self.bus_voltage_min[i] * self.bus_voltage_min[i]] * self.T
                for i in range(self.bus_num)],
            ub=[[self.bus_voltage_max[i] * self.bus_voltage_max[i]] * self.T
                for i in range(self.bus_num)],
            name='bus_voltage_square')
        self.line_current_square = self.model.addVars(
            self.line_num, self.T,
            ub=[[self.line_current_capacity[i] * self.line_current_capacity[i]] * self.T
                for i in range(self.line_num)],
            name='line_current_square')
        self.line_power_flow = self.model.addVars(
            self.line_num, self.T,
            lb=-10, ub=10,  # TODO: key error, core error
            name='line_power_flow')
        self.line_react_flow = self.model.addVars(
            self.line_num, self.T,
            lb=-10, ub=10,
            name='line_react_flow')
        self.well_output = self.model.addVars(self.well_num, self.T,
                                              lb=[[self.well_output_min[i]] * self.T for i in range(self.well_num)],
                                              ub=[[self.well_output_max[i]] * self.T for i in range(self.well_num)],
                                              name='gas_well_outputs')
        self.node_pressure = self.model.addVars(self.gas_node_num, self.T,
                                                lb=[[self.node_pressure_min[i]] * self.T for i in
                                                    range(self.gas_node_num)],
                                                ub=[[self.node_pressure_max[i]] * self.T for i in
                                                    range(self.gas_node_num)],
                                                name='node_pressure')
        self.gas_flow_in = self.model.addVars(self.gas_line_num, self.T,
                                              ub=[[self.gas_flow_in_max[i]] * self.T for i in range(self.gas_line_num)],
                                              lb=[[0 * self.gas_flow_in_max[i]] * self.T for i in
                                                  range(self.gas_line_num)],
                                              name='gas_flow_in')
        self.gas_flow_out = self.model.addVars(self.gas_line_num, self.T,
                                               ub=[[self.gas_flow_out_max[i]] * self.T for i in
                                                   range(self.gas_line_num)],
                                               lb=[[0 * self.gas_flow_out_max[i]] * self.T for i in
                                                   range(self.gas_line_num)],
                                               name='gas_flow_out')
        self.gas_load = self.model.addVars(self.gas_load_num, self.T,
                                           lb=self.gas_load_min, ub=self.gas_load_max,
                                           name='gas_load')
        self.gen_gas_power = self.model.addVars(self.gen_gas_num, self.T,
                                                lb=[[self.gen_gas_min[i]] * self.T for i in range(self.gen_gas_num)],
                                                ub=[[self.gen_gas_max[i]] * self.T for i in range(self.gen_gas_num)],
                                                name='gen_gas_power')
        self.linepack = self.model.addVars(self.gas_line_num, self.T, name='gas_linepack')
        self.pccp = self.model.addVars(self.gas_line_num, self.T, lb=0, name='pccp')
        self.power_from_grid = self.model.addVars(self.T, lb=-2, ub=2, name='power_from_grid')
        self.react_from_grid = self.model.addVars(self.T, lb=-2, ub=2, name='react_from_grid')
        self.model.update()
        # ----------- construct the info structure --------------------------------
        for i in range(len(self.connection_area)):
            line_T = []
            line_start = self.line_num - len(self.connection_area)  # 5-2 = 3   3 4
            bus_start = self.bus_num - len(self.connection_area)
            gas_node_start = self.gas_node_num - len(self.connection_area)
            gas_line_start = self.gas_line_num - len(self.connection_area)
            this_index = self.connection_index[i]
            this_index_gas = self.connection_index[i]  # TODO: we assume gas and power have the same connection index
            for time in range(self.T):
                line_t = connection_line_info()
                line_t.power_flow = self.line_power_flow[i + line_start, time]
                line_t.react_flow = self.line_react_flow[i + line_start, time]
                line_t.this_voltage_square = self.voltage_square[this_index, time]
                line_t.that_voltage_square = self.voltage_square[i + bus_start, time]
                line_t.this_node_pressure = self.node_pressure[this_index_gas, time]
                line_t.that_node_pressure = self.node_pressure[i + gas_node_start, time]
                line_t.gas_flow_in = self.gas_flow_in[i + gas_line_start, time]
                line_t.gas_flow_out = self.gas_flow_out[i + gas_line_start, time]
                line_T.append(line_t)
            self.info[i] = line_T

        # ----------- node power balance -----------------
        for node in range(self.bus_num):
            Power = self.power_gen_connected_with(node)
            React = self.react_gen_connected_with(node)
            Power_Load = self.load_power_connected_with(node)
            React_Load = self.load_react_connected_with(node)
            Power_In = self.power_flow_in_connected_with(node)
            Power_Out = self.power_flow_out_connected_with(node)
            React_In = self.raect_flow_in_connected_with(node)
            React_Out = self.react_flow_out_connected_with(node)
            Current_In = self.current_in_connected_with(node)
            resistance = self.resistance_in_connected_with(node)
            reactance = self.reactance_in_connected_with(node)
            G2P = self.gas_to_power_connected_with(node)
            Power_Grid_In = self.grid_power_in_connected_with(node)
            React_Grid_In = self.grid_react_in_connected_with(node)
            for time in range(self.T):
                self.model.addConstr(
                    lhs=sum(Power[:, time]) + sum(G2P[:, time]) + sum(Power_Grid_In[:, time]) +
                        sum(Power_In[:, time] - resistance[:, time] * Current_In[:, time]),
                    rhs=sum(Power_Load[:, time]) + sum(Power_Out[:, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='power_balance')
                self.model.addConstr(
                    lhs=sum(React[:, time]) + sum(React_Grid_In[:, time]) +
                        sum(React_In[:, time] - reactance[:, time] * Current_In[:, time]),
                    rhs=sum(React_Load[:, time]) + sum(React_Out[:, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='react_balance')

        # ----------- line voltage drop ------------------
        for i in range(self.line_num):
            start_point = self.line_start_point[i]
            end_point = self.line_end_point[i]
            resistance = self.line_resistance[i]
            reactance = self.line_reactance[i]
            impedance_square = reactance * reactance + resistance * resistance
            for time in range(self.T):
                self.model.addConstr(
                    lhs=self.voltage_square[end_point, time] -
                        self.voltage_square[start_point, time],
                    rhs=impedance_square * self.line_current_square[i, time] -
                        2 * (resistance * self.line_power_flow[i, time] +
                             reactance * self.line_react_flow[i, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='voltage_drop')
                self.model.addConstr(
                    lhs=self.line_power_flow[i, time] * self.line_power_flow[i, time] +
                        self.line_react_flow[i, time] * self.line_react_flow[i, time],
                    rhs=self.line_current_square[i, time] * self.voltage_square[start_point, time],
                    sense=gurobi.GRB.LESS_EQUAL,
                    # sense=gurobi.GRB.EQUAL,
                    name='flow_relax')

        # ----------- gas node balance  ------------------
        for node in range(self.gas_node_num):
            # for all passive and active   # use numpy !!!! return [[]] format
            Well = self.well_connected_with(node)  # 节点node对应的well变量
            Load = self.load_connected_with(node)
            Gen = self.gen_connected_with(node)  # this change Power to Gas   # considered efficiency !!!!!!!
            P2G = self.p2g_connected_with(node)  # this is just gas
            Line_Out = self.gas_flow_out_connected_with(node)
            Line_In = self.gas_flow_in_connected_with(node)
            for time in range(self.T):
                self.model.addConstr(
                    lhs=sum(Well[:, time]) + sum(P2G[:, time]) + sum(Line_In[:, time]),  # source
                    rhs=sum(Gen[:, time]) + sum(Load[:, time]) + sum(Line_Out[:, time]),  # load
                    sense=gurobi.GRB.EQUAL,
                    name='gas_nodal_balance_node')

        # ----------- line pack passive ------------------
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                start_point = self.gas_line_start_point[line]
                end_point = self.gas_line_end_point[line]
                linepack_coefficient = self.gas_line_pack_coefficient[line]
                for time in range(self.T):
                    self.model.addConstr(
                        lhs=self.linepack[line, time],
                        rhs=linepack_coefficient *
                            (self.node_pressure[start_point, time] + self.node_pressure[end_point, time]),
                        sense=gurobi.GRB.EQUAL,
                        name='linePack')

        # ----------- passive Pack-T ---------------------
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                for time in range(self.T):
                    if time == 0:
                        self.model.addConstr(
                            lhs=self.linepack[line, 0] - self.linepack[line, self.T - 1],
                            rhs=self.gas_flow_in[line, 0] - self.gas_flow_out[line, 0],
                            sense=gurobi.GRB.EQUAL,
                            name='linepack_with_time_' + str(time) + '_line' + str(line))
                    else:
                        self.model.addConstr(
                            lhs=self.linepack[line, time] - self.linepack[line, time - 1],
                            rhs=self.gas_flow_in[line, time] - self.gas_flow_out[line, time],
                            sense=gurobi.GRB.EQUAL,
                            name='linepack_with_time_' + str(time) + '_line' + str(line))

        # ----------- Pack Less Init ---------------------
        linepack_sum = 0  # ? passive or active
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                linepack_sum = linepack_sum + self.linepack[line, self.T - 1]
        self.model.addConstr(linepack_sum <= self.gas_line_pack_initial)

        # ---------active gas-consume active pressure-increase---------------------------
        for line in range(self.gas_line_num):
            if line in self.gas_line_active:
                thisIndex = self.gas_line_active.index(line)
                compressor_coeff = self.compressor_coefficient[thisIndex]
                start_point = self.gas_line_start_point[line]
                end_point = self.gas_line_end_point[line]
                max_flow = self.compressor_max_flow[thisIndex]
                energy_consumption = 1 - self.compressor_energy_consumption[thisIndex]
                for time in range(self.T):
                    # self.model.addConstr(self.gas_flow_in[line, time] <= max_flow)
                    self.model.addConstr(self.node_pressure[end_point, time] <=
                                         compressor_coeff * self.node_pressure[start_point, time])
                    # add flow quantities for gas compressors
                    self.model.addConstr(self.gas_flow_out[line, time] ==
                                         energy_consumption * self.gas_flow_in[line, time])

        # ------------- weymouth passive ------------------------
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                start_point = self.gas_line_start_point[line]
                end_point = self.gas_line_end_point[line]
                weymouth = self.weymouth[line]
                for time in range(self.T):
                    self.model.addConstr(
                        lhs=((self.gas_flow_in[line, time] + self.gas_flow_out[line, time]) / 2) *
                            ((self.gas_flow_in[line, time] + self.gas_flow_out[line, time]) / 2),
                        rhs=weymouth * (self.node_pressure[start_point, time] *
                                        self.node_pressure[start_point, time] -
                                        self.node_pressure[end_point, time] *
                                        self.node_pressure[end_point, time]),
                        sense=gurobi.GRB.LESS_EQUAL,
                        name='weymouth')
                    self.model.addConstr(self.node_pressure[start_point, time] >= self.node_pressure[end_point, time])
                    # self.model.addConstr(self.gas_flow_in[line, time] == self.gas_flow_out[line, time])
                    self.constrain_update.append(
                        self.model.addConstr(
                            lhs=weymouth * self.node_pressure[start_point, time] *
                                self.node_pressure[start_point, time] - (
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) *
                                        (self.gas_flow_in[line, time] + self.gas_flow_out[line, time]) / 2 -
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) *
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) / 4 -
                                        weymouth * self.node_pressure_old[end_point][time] *
                                        self.node_pressure_old[end_point][time] +
                                        2 * weymouth * self.node_pressure[end_point, time] *
                                        self.node_pressure_old[end_point][time]
                                ),
                            rhs=self.pccp[line, time],
                            sense=gurobi.GRB.LESS_EQUAL,
                            name='pccp_less'
                        )
                    )
        # ------------- gas system end --------------------------
        abcdefg = 0
        # ------------- construct object ------------------------
        first_line = self.line_num - len(self.connection_area)
        self.objs = []
        # 购电成本
        for t in range(T):
            self.objs.append((self.power_from_grid[t] + self.react_from_grid[t]) * GRID_PRICE)
        # 发电成本
        self.gen_power_cost = []
        for gen in range(self.gen_num - len(self.connection_area)):
            per = 0
            for time in range(self.T):
                per = per + \
                      self.power_gen[gen, time] * self.gen_cost_a[gen] + \
                      self.power_gen[gen, time] * self.power_gen[gen, time] * self.gen_cost_b[gen] + \
                      self.gen_cost_c[gen]
            self.gen_power_cost.append(per)
            self.objs.append(per)
        self.gen_power_cost = sum(self.gen_power_cost)

        # 电负荷效益函数
        self.load_power_cost = []
        for load in range(self.load_num - len(self.connection_area)):
            per = 0
            for time in range(self.T):
                load_ref = (self.load_power_max[load][time] + self.load_power_min[load][time]) / 2
                per = per + \
                      LOAD_COEFF * (self.power_load[load, time] - load_ref) * \
                      (self.power_load[load, time] - load_ref)
            self.objs.append(per)
            self.load_power_cost.append(per)
        self.load_power_cost = sum(self.load_power_cost)

        # 气负荷效益函数，我们要刨去最后一个负荷
        self.load_gas_cost = []
        if self.index == 1:
            for load in range(self.gas_load_num - len(self.connection_area)):
                per = 0
                for time in range(self.T):
                    load_ref = (self.gas_load_min[load][time] + self.gas_load_max[load][time]) / 2
                    per = per + \
                          GAS_LOAD_COEFF * (self.gas_load[load, time] - load_ref) * \
                          (self.gas_load[load, time] - load_ref)
                self.objs.append(per)
                self.load_gas_cost.append(per)
        if self.index == 0:
            for load in range(self.gas_load_num):
                per = 0
                for time in range(self.T):
                    load_ref = (self.gas_load_min[load][time] + self.gas_load_max[load][time]) / 2
                    per = per + \
                          GAS_LOAD_COEFF * (self.gas_load[load, time] - load_ref) * \
                          (self.gas_load[load, time] - load_ref)
                self.objs.append(per)
                self.load_gas_cost.append(per)
        self.load_gas_cost = sum(self.load_gas_cost)

        # 购电费用
        self.power_buy_cost = []
        for line in range(len(self.connection_area)):
            connect_to = self.connection_area[line]
            per_area = 0
            for time in range(self.T):
                per_area = per_area + (self.line_power_flow[first_line + line, time] +
                                       self.line_react_flow[first_line + line, time]) * \
                           power_price[self.index][connect_to] * 1
            self.objs.append(per_area)
            self.power_buy_cost.append(per_area)
        self.power_buy_cost = sum(self.power_buy_cost)

        # 购气费用
        self.gas_buy_cost = []
        for conn in range(len(self.connection_area)):
            gas_line_num = self.gas_line_num - len(self.connection_area)
            for time in range(T):
                self.objs.append(GAS_PRICE *  # gas_buy_price  购气成本
                                 (self.gas_flow_out[gas_line_num + conn, time] + self.gas_flow_in[
                                     gas_line_num + conn, time])
                                 * g_gas_price_aux[self.index])
                self.gas_buy_cost.append(GAS_PRICE *  # gas_buy_price  购气成本
                                         (self.gas_flow_out[gas_line_num + conn, time] + self.gas_flow_in[
                                             gas_line_num + conn, time])
                                         * g_gas_price_aux[self.index])
        self.gas_buy_cost = sum(self.gas_buy_cost)

        # 天然气井的生产费用
        self.well_produce_cost = []
        if self.index == 0:
            for well in range(self.well_num - len(self.connection_area)):
                for time in range(T):
                    self.objs.append(self.well_output[well, time] * well_output_price)
                    self.well_produce_cost.append(self.well_output[well, time] * well_output_price)
        if self.index == 1:
            for well in range(self.well_num):
                for time in range(T):
                    self.objs.append(self.well_output[well, time] * well_output_price)
                    self.well_produce_cost.append(self.well_output[well, time] * well_output_price)
        self.well_produce_cost = sum(self.well_produce_cost)

        # 燃气轮机的生产费用
        for gas_gen in range(self.gen_gas_num):
            for time in range(T):
                self.objs.append(self.gen_gas_power[gas_gen, time] * 0)  # 燃气轮机相当于负荷，只有偏差成本
        objective = sum(self.objs)
        self.basic_objective = objective

    def update_model(self, tao):
        global g_info
        self.lams = []
        # obtain the lam of this player
        for i, index in enumerate(g_lam_index[self.index]):
            connect_to = self.connection_area[i]
            is_start_point = 0
            if connect_to > self.index:
                is_start_point = 1
            self.lams.extend(self.get_lam(index, is_start_point))

        # construct the dual object
        self.dual = []  # [ ---time---,  ---time--- ]
        for i in range(len(self.connection_area)):
            for time in range(self.T):
                connect_to = self.connection_area[i]
                line_index = g_connection[connect_to].index(self.index)
                that_info = g_info[connect_to][line_index][time]
                this_info = self.info[i][time]
                is_start_point = 0
                if connect_to > self.index:
                    is_start_point = 1
                self.dual.extend(self.get_dual(this_info, that_info, is_start_point))
        self.dual_addition = sum([PUNISH * a * b for a, b in zip(self.dual, self.lams)])

        global_lambda_power = copy(g_lam_power_grid)
        global_grid_power_in = copy(g_grid_power_in)
        global_grid_power_in[self.index] = [0] * T  # 3 * T
        grid_power_in_var = np.array([self.power_from_grid[t] - 1 for t in range(T)])   # here we add the limit
        grid_power_per_time = (np.array(global_grid_power_in).sum(axis=0) + grid_power_in_var).tolist()  # 1 * T
        self.global_dual_addition_power = sum([PUNISH * a * b for a, b in zip(grid_power_per_time, global_lambda_power)])
        global_lambda_react = copy(g_lam_react_grid)
        global_grid_react_in = copy(g_grid_react_in)
        global_grid_react_in[self.index] = [0] * T  # 3 * T
        grid_react_in_var = np.array([self.react_from_grid[t] - 1 for t in range(T)])   # here we add the limit
        grid_react_per_time = (np.array(global_grid_react_in).sum(axis=0) + grid_react_in_var).tolist()  # 1 * T
        self.global_dual_addition_react = sum([PUNISH * a * b for a, b in zip(grid_react_per_time, global_lambda_react)])

        global_lambda_power_negative = copy(g_lam_power_grid_negative)
        global_grid_power_in_negative = copy(g_grid_power_in)
        global_grid_power_in_negative[self.index] = [0] * T  # 3 * T
        grid_power_in_var_negative = np.array([-1 * self.power_from_grid[t] - 1 for t in range(T)])   # here we add the limit
        grid_power_per_time_negative = (-1 * np.array(global_grid_power_in_negative).sum(axis=0) + grid_power_in_var_negative).tolist()  # 1 * T
        self.global_dual_addition_power_negative = sum([PUNISH * a * b for a, b in zip(grid_power_per_time_negative, global_lambda_power_negative)])
        global_lambda_react_negative = copy(g_lam_react_grid_negative)
        global_grid_react_in_negative = copy(g_grid_react_in)
        global_grid_react_in_negative[self.index] = [0] * T  # 3 * T
        grid_react_in_var_negative = np.array([-1 * self.react_from_grid[t] - 1 for t in range(T)])   # here we add the limit
        grid_react_per_time_negative = (-1 * np.array(global_grid_react_in_negative).sum(axis=0) + grid_react_in_var_negative).tolist()  # 1 * T
        self.global_dual_addition_react_negative = sum([PUNISH * a * b for a, b in zip(grid_react_per_time_negative, global_lambda_react_negative)])

        self.dual_addition = self.dual_addition + self.global_dual_addition_power + self.global_dual_addition_react + \
                             self.global_dual_addition_power_negative + self.global_dual_addition_react_negative

        # construct the norm object
        self.norm_addition = 0
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]
            is_start_point = 0
            if connect_to > self.index:
                is_start_point = 1
            for time in range(self.T):
                self.norm_addition = self.norm_addition + \
                                     self.get_sub(self.info[i][time], self.old_value[i][time], is_start_point)

        self.global_norm_addition_power = \
            sum([(self.power_from_grid[t] - self.old_power_from_grid[t]) *
                 (self.power_from_grid[t] - self.old_power_from_grid[t]) for t in range(T)])
        self.global_norm_addition_react = \
            sum([(self.react_from_grid[t] - self.old_react_from_grid[t]) *
                 (self.react_from_grid[t] - self.old_react_from_grid[t]) for t in range(T)])
        self.norm_addition = self.norm_addition + self.global_norm_addition_power + self.global_norm_addition_react

        self.addition_objective = 1 * self.dual_addition + tao / 2 * self.norm_addition
        self.objective = self.basic_objective + self.addition_objective

    def optimize(self):
        self.model.Params.OutputFlag = 0
        self.model.setObjective(self.objective + gurobi.quicksum(self.pccp) * G_K)
        self.model.optimize()

        for line in range(self.gas_line_num):
            for time in range(self.T):
                self.gas_flow_in_old[line][time] = self.gas_flow_in[line, time].getAttr('X')

        for line in range(self.gas_line_num):
            for time in range(self.T):
                self.gas_flow_out_old[line][time] = self.gas_flow_out[line, time].getAttr('X')

        for node in range(self.gas_node_num):
            for time in range(self.T):
                self.node_pressure_old[node][time] = self.node_pressure[node, time].getAttr('X')

        for i in range(len(self.connection_area)):
            for time in range(self.T):
                this_index = self.connection_index[i]
                connect_to = self.connection_area[i]
                this_index_gas = self.connection_index[i]  # we assume gas and power have the same index
                is_start_point = 0
                if connect_to > self.index:
                    is_start_point = 1
                line = connection_line_info()
                line_start = self.line_num - len(self.connection_area)
                bus_start = self.bus_num - len(self.connection_area)
                gas_node_start = self.gas_node_num - len(self.connection_area)
                gas_line_start = self.gas_line_num - len(self.connection_area)
                # ---------- update power flow --------------
                if is_start_point != 0:  # this is start point
                    line.power_flow = self.line_power_flow[i + line_start, time].getAttr('X')
                    line.react_flow = self.line_react_flow[i + line_start, time].getAttr('X')
                else:
                    line.power_flow = self.line_power_flow[i + line_start, time].getAttr('X') * (-1)
                    line.react_flow = self.line_react_flow[i + line_start, time].getAttr('X') * (-1)
                # -------- update voltage ------------
                if is_start_point != 0:  # this is start point
                    line.this_voltage_square = self.voltage_square[this_index, time].getAttr('X')
                    line.that_voltage_square = self.voltage_square[i + bus_start, time].getAttr('X')
                else:
                    line.this_voltage_square = self.voltage_square[i + bus_start, time].getAttr('X')
                    line.that_voltage_square = self.voltage_square[this_index, time].getAttr('X')
                # ------- update pressure -----------
                if is_start_point != 0:  # this is start point
                    line.this_node_pressure = self.node_pressure[this_index_gas, time].getAttr('X')
                    line.that_node_pressure = self.node_pressure[i + gas_node_start, time].getAttr('X')
                else:
                    line.this_node_pressure = self.node_pressure[i + gas_node_start, time].getAttr('X')
                    line.that_node_pressure = self.node_pressure[this_index_gas, time].getAttr('X')
                # -------- update gas flow -----------
                line.gas_flow_in = self.gas_flow_in[i + gas_line_start, time].getAttr('X')
                line.gas_flow_out = self.gas_flow_out[i + gas_line_start, time].getAttr('X')
                # u p d a t e   g _ i n f o
                g_info[self.index][i][time] = line

        g_grid_power_in[self.index] = [self.power_from_grid[t].getAttr('X') for t in range(T)]
        g_grid_react_in[self.index] = [self.react_from_grid[t].getAttr('X') for t in range(T)]

    def set_old_value(self, old):  # old_value should be consisted with the g_info
        for area in range(len(self.connection_area)):
            for time in range(self.T):
                self.old_value[area][time].this_voltage_square = old[area][time].this_voltage_square
                self.old_value[area][time].that_voltage_square = old[area][time].that_voltage_square
                self.old_value[area][time].power_flow = old[area][time].power_flow
                self.old_value[area][time].react_flow = old[area][time].react_flow
                self.old_value[area][time].this_node_pressure = old[area][time].this_node_pressure
                self.old_value[area][time].that_node_pressure = old[area][time].that_node_pressure
                self.old_value[area][time].gas_flow_in = old[area][time].gas_flow_in
                self.old_value[area][time].gas_flow_out = old[area][time].gas_flow_out
        for time in range(T):
            self.old_power_from_grid[time] = g_grid_power_in[self.index][time]
            self.old_react_from_grid[time] = g_grid_react_in[self.index][time]

    def cal_gap(self):
        result = []
        for line in range(self.gas_line_num):
            for time in range(self.T):
                self.gas_flow_in_old[line][time] = self.gas_flow_in[line, time].getAttr('X')
        for line in range(self.gas_line_num):
            for time in range(self.T):
                self.gas_flow_out_old[line][time] = self.gas_flow_out[line, time].getAttr('X')
        for node in range(self.gas_node_num):
            for time in range(self.T):
                self.node_pressure_old[node][time] = self.node_pressure[node, time].getAttr('X')
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                start_point = self.gas_line_start_point[line]
                end_point = self.gas_line_end_point[line]
                weymouth = self.weymouth[line]
                for time in range(self.T):
                    lhs = ((self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) / 2) * \
                          ((self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) / 2)
                    rhs = weymouth * (self.node_pressure_old[start_point][time] *
                                      self.node_pressure_old[start_point][time] -
                                      self.node_pressure_old[end_point][time] *
                                      self.node_pressure_old[end_point][time])
                    result.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
        return max(result)

    def update_outer_model(self):
        self.model.remove(self.constrain_update)
        self.constrain_update = []
        # weymouth for passive line
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                start_point = self.gas_line_start_point[line]
                end_point = self.gas_line_end_point[line]
                weymouth = self.weymouth[line]
                for time in range(self.T):
                    self.constrain_update.append(
                        self.model.addConstr(
                            lhs=weymouth * self.node_pressure[start_point, time] *
                                self.node_pressure[start_point, time] - (
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) *
                                        (self.gas_flow_in[line, time] + self.gas_flow_out[line, time]) / 2 -
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) *
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) / 4 -
                                        weymouth * self.node_pressure_old[end_point][time] *
                                        self.node_pressure_old[end_point][time] +
                                        2 * weymouth * self.node_pressure[end_point, time] *
                                        self.node_pressure_old[end_point][time]
                                ),
                            rhs=self.pccp[line, time],
                            sense=gurobi.GRB.LESS_EQUAL,
                            name='weymouth_relax'
                        ))
        pccp_value = []
        for i in range(self.gas_line_num):
            pccp_value.append(self.pccp[i, 0].getAttr('X'))
        if abs(max(pccp_value)) < 0.005:
            print('a_a_a_a_a_a_a_a_a_a_a_a_a_a_a_a_a_amazing_______________________')


class PlayerN1:
    def __init__(self):
        self.gx = []
        self.old_value = [0] * g_link * T * 16
        self.dual_express = 0
        self.norm_express = 0
        self.objective = 0
        self.old_grid_power_value = [0] * T
        self.old_grid_react_value = [0] * T
        self.old_grid_power_value_negative = [0] * T
        self.old_grid_react_value_negative = [0] * T

    def sub(self, lhs, rhs):
        gx = []
        gx.append(lhs.this_voltage_square - rhs.this_voltage_square)
        gx.append(-1 * lhs.this_voltage_square + rhs.this_voltage_square)
        gx.append(lhs.that_voltage_square - rhs.that_voltage_square)
        gx.append(-1 * lhs.that_voltage_square + rhs.that_voltage_square)
        gx.append(lhs.power_flow - rhs.power_flow)
        gx.append(-1 * lhs.power_flow + rhs.power_flow)
        gx.append(lhs.react_flow - rhs.react_flow)
        gx.append(-1 * lhs.react_flow + rhs.react_flow)
        gx.append(1 * (lhs.this_node_pressure - rhs.this_node_pressure))
        gx.append(1 * (-1 * lhs.this_node_pressure + rhs.this_node_pressure))
        gx.append(1 * (lhs.that_node_pressure - rhs.that_node_pressure))
        gx.append(1 * (-1 * lhs.that_node_pressure + rhs.that_node_pressure))
        gx.append(1 * (lhs.gas_flow_in - rhs.gas_flow_in))
        gx.append(1 * (-1 * lhs.gas_flow_in + rhs.gas_flow_in))
        gx.append(1 * (lhs.gas_flow_out - rhs.gas_flow_out))
        gx.append(1 * (-1 * lhs.gas_flow_out + rhs.gas_flow_out))
        return gx

    def optimize(self, tao):
        global g_lam_power_grid
        global g_lam_react_grid
        global g_lam_power_grid_negative
        global g_lam_react_grid_negative

        model = gurobi.Model()

        self.dual_express = 0
        self.norm_express = 0
        self.objective = 0
        self.gx = []

        for i in range(len(g_connection)):
            for connect_to in g_connection[i]:
                if i < connect_to:
                    for time in range(T):
                        lhs = g_info[i][g_connection[i].index(connect_to)][time]
                        rhs = g_info[connect_to][g_connection[connect_to].index(i)][time]
                        self.gx.extend(self.sub(lhs, rhs))

        duals = model.addVars(len(self.gx))
        model.update()

        self.dual_express = gurobi.quicksum(
            1 * duals[i] * self.gx[i] for i in range(len(self.gx))
        )
        gx_grid_power = (np.array(g_grid_power_in).sum(axis=0) + np.array([-1] * T)).tolist()   # here we add limit
        gx_grid_react = (np.array(g_grid_react_in).sum(axis=0) + np.array([-1] * T)).tolist()
        duals_grid_power = model.addVars(T)
        duals_grid_react = model.addVars(T)

        gx_grid_power_negative = (-1 * np.array(g_grid_power_in).sum(axis=0) + np.array([-1] * T)).tolist()   # here we add limit
        gx_grid_react_negative = (-1 * np.array(g_grid_react_in).sum(axis=0) + np.array([-1] * T)).tolist()
        duals_grid_power_negative = model.addVars(T)
        duals_grid_react_negative = model.addVars(T)


        dual_grid_power_express = gurobi.quicksum(
            1 * duals_grid_power[i] * gx_grid_power[i] for i in range(T)
        )
        dual_grid_react_express = gurobi.quicksum(
            1 * duals_grid_react[i] * gx_grid_react[i] for i in range(T)
        )

        dual_grid_power_express_negative = gurobi.quicksum(
            1 * duals_grid_power_negative[i] * gx_grid_power_negative[i] for i in range(T)
        )
        dual_grid_react_express_negative = gurobi.quicksum(
            1 * duals_grid_react_negative[i] * gx_grid_react_negative[i] for i in range(T)
        )
        self.dual_express = self.dual_express + dual_grid_power_express + dual_grid_react_express + \
                            dual_grid_power_express_negative + dual_grid_react_express_negative


        self.norm_express = gurobi.quicksum(
            (duals[i] - self.old_value[i]) * (duals[i] - self.old_value[i])
            for i in range(len(self.gx)))
        norm_grid_power_express = gurobi.quicksum(
            (duals_grid_power[i] - self.old_grid_power_value[i]) * (duals_grid_power[i] - self.old_grid_power_value[i])
            for i in range(T)
        )
        norm_grid_react_express = gurobi.quicksum(
            (duals_grid_react[i] - self.old_grid_react_value[i]) * (duals_grid_react[i] - self.old_grid_react_value[i])
            for i in range(T)
        )
        norm_grid_power_express_negative = gurobi.quicksum(
            (duals_grid_power_negative[i] - self.old_grid_power_value_negative[i]) *
            (duals_grid_power_negative[i] - self.old_grid_power_value_negative[i])
            for i in range(T)
        )
        norm_grid_react_express_negative = gurobi.quicksum(
            (duals_grid_react_negative[i] - self.old_grid_react_value_negative[i]) *
            (duals_grid_react_negative[i] - self.old_grid_react_value_negative[i])
            for i in range(T)
        )
        self.norm_express = self.norm_express + norm_grid_power_express + norm_grid_react_express + \
                            norm_grid_power_express_negative + norm_grid_react_express_negative



        gxdiv100.append(-1 * self.gx[0] / 100)
        gxdiv101.append(-1 * self.gx[1] / 100)
        gxdiv102.append(-1 * self.gx[2] / 100)
        gxdiv103.append(-1 * self.gx[3] / 100)
        gxdiv104.append(-1 * self.gx[4] / 100)
        gxdiv105.append(-1 * self.gx[5] / 100)
        gxdiv106.append(-1 * self.gx[6] / 100)
        gxdiv107.append(-1 * self.gx[7] / 100)
        gxdiv108.append(-1 * self.gx[8] / 100)

        oldlam.append(self.old_value[0])
        oldlam1.append(self.old_value[1])
        oldlam2.append(self.old_value[2])
        oldlam3.append(self.old_value[3])
        oldlam4.append(self.old_value[4])
        oldlam5.append(self.old_value[5])
        oldlam6.append(self.old_value[6])
        oldlam7.append(self.old_value[7])
        global g_tao

        self.objective = -1 * self.dual_express + (tao / 2) * self.norm_express
        model.setObjective(self.objective)
        model.Params.OutputFlag = 0
        model.optimize()
        dual_value = []
        pos = 0
        for line in range(g_link):
            lam_T = []
            for time in range(T):
                lam_t = []
                for _m_m_ in range(16):
                    lam_t.append(duals[pos].getAttr('X'))
                    # lam_t.append((self.old_value[pos] - self.gx[pos] / g_tao) if (self.old_value[pos] - self.gx[pos] / g_tao) > 0 else 0)
                    pos = pos + 1
                lam_T.append(lam_t)
            dual_value.append(lam_T)

        g_lam_power_grid = [duals_grid_power[t].getAttr('X') for t in range(T)]
        g_lam_react_grid = [duals_grid_react[t].getAttr('X') for t in range(T)]
        g_lam_power_grid_negative = [duals_grid_power_negative[t].getAttr('X') for t in range(T)]
        g_lam_react_grid_negative = [duals_grid_react_negative[t].getAttr('X') for t in range(T)]
        return copy(dual_value)

    def set_old_value(self, old_value):
        self.old_value = copy(old_value)
        self.old_grid_power_value = copy(g_lam_power_grid)
        self.old_grid_react_value = copy(g_lam_react_grid)
        self.old_grid_power_value_negative = copy(g_lam_power_grid_negative)
        self.old_grid_react_value_negative = copy(g_lam_react_grid_negative)


def getPowerNet():
    # -------------- p l a y e r 0 ----------------
    a_1 = 3
    player_index = 0
    system_info_0 = {
        'index': player_index,
        'T': T,
        'connection_area': g_connection[player_index]
    }
    node_info_0 = {
        'gen_num': 3 + 1,  # the outside node as node-12 connected with index-3 with line-12
        'gen_index': [0, 0, 5, 12],
        'gen_power_min': [0, 0, 0, 0],  # 0
        'gen_power_max': [0.2, 0.2, 0.2, 10],  # 1.2
        'gen_react_min': [0, 0, 0, 0],  # 0
        'gen_react_max': [0.3, 0.3, 0.3, 10],  # 1.2
        'gen_cost_a': [1, 1.3, 0.9, 0],
        'gen_cost_b': [0.1, 0.1, 0.1, 0],
        'gen_cost_c': [0.1, 0.1, 0.1, 0],
        'bus_num': 12 + 1,
        'bus_voltage_min': [0.95 * 1] * (12 + 1),
        'bus_voltage_max': [1.05 * 1] * (12 + 1),
        'load_num': 8 + 1,
        'load_index': [2, 3, 4, 7, 8, 9, 10, 11, 12],
        'load_power_min': np.array(
            [[0.10, 0.11, 0.12, 0.10, 0.09, 0.12, 0.10, 0.12, 0.12, 0.12],
             [0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.25, 0.22, 0.20],
             [0.20, 0.19, 0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.22, 0.20],
             [0.10, 0.12, 0.10, 0.10, 0.10, 0.13, 0.12, 0.10, 0.09, 0.09],
             [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
             [0.10, 0.11, 0.12, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12],
             [0.20, 0.20, 0.18, 0.20, 0.22, 0.22, 0.22, 0.20, 0.20, 0.20],
             [0.10, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])[:, 0:T].tolist(),  # 1.3
        'load_power_max': np.array(
            [[0.15, 0.16, 0.17, 0.18, 0.17, 0.15, 0.14, 0.14, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.26, 0.27, 0.25, 0.26, 0.24, 0.23, 0.25],
             [0.25, 0.23, 0.24, 0.26, 0.27, 0.28, 0.29, 0.27, 0.26, 0.25],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.35, 0.40, 0.42, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.28, 0.29, 0.24, 0.23, 0.20, 0.20, 0.20],
             [0.15, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])[:, 0:T].tolist(),  # 1.7
        'load_react_min': np.array(
            [[0.10, 0.11, 0.12, 0.10, 0.09, 0.12, 0.10, 0.12, 0.12, 0.12],
             [0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.25, 0.22, 0.20],
             [0.20, 0.19, 0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.22, 0.20],
             [0.10, 0.12, 0.10, 0.10, 0.10, 0.13, 0.12, 0.10, 0.09, 0.09],
             [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
             [0.10, 0.11, 0.12, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12],
             [0.20, 0.20, 0.18, 0.20, 0.22, 0.22, 0.22, 0.20, 0.20, 0.20],
             [0.10, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])[:, 0:T].tolist(),
        'load_react_max': np.array(
            [[0.15, 0.16, 0.17, 0.18, 0.17, 0.15, 0.14, 0.14, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.26, 0.27, 0.25, 0.26, 0.24, 0.23, 0.25],
             [0.25, 0.23, 0.24, 0.26, 0.27, 0.28, 0.29, 0.27, 0.26, 0.25],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.35, 0.40, 0.42, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.28, 0.29, 0.24, 0.23, 0.20, 0.20, 0.20],
             [0.15, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])[:, 0:T].tolist(),  # 1.7
        'bus_num_outside': 1,
        'connection_index': [1],  # the outer area power/gas connect with this index
    }
    line_info_0 = {
        'line_num': 11 + 1,
        'line_current_capacity': [10] * (11 + 1),
        'line_start_point': [0, 1, 0, 3, 0, 5, 8, 5, 6, 5, 6, 12],
        'line_end_point': [1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 1],
        'line_resistance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist(),
        'line_reactance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist()
    }
    gas_node_info_0 = {
        'gas_node_num': 3 + a_1 + 1,
        'node_pressure_min': [0] * (3 + a_1 + 1),
        'node_pressure_max': [10] * (3 + a_1 + 1),
        'gas_well_num': 1 + 1,
        'well_index': [1, 3 + a_1],
        'well_output_min': [0, 0],
        'well_output_max': [0.2, 2],
        'gas_load_num': 2 + 1,  #
        'load_index': [0, 2, 5],
        'gas_load_min': (1 * (np.array([[0.1, 0.11, 0.12, 0.11, 0.12, 0.11, 0.12, 0.11, 0.10, 0.10],
                                        [0.1, 0.09, 0.08, 0.09, 0.10, 0.11, 0.11, 0.11, 0.09, 0.08],

                                        (np.array(
                                            [0.1, 0.09, 0.09, 0.09, 0.10, 0.10, 0.10, 0.09, 0.08, 0.07]) * 1).tolist(),
                                        # (np.array([0.1, 0.10, 0.10, 0.10, 0.11, 0.11, 0.11, 0.13, 0.12, 0.10]) * 0.01).tolist()
                                        ])[:, 0:T])).tolist(),
        'gas_load_max': (1 * (np.array([[0.2, 0.21, 0.22, 0.21, 0.23, 0.24, 0.21, 0.26, 0.23, 0.24],
                                        [0.2, 0.22, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
                                        (np.array(
                                            [0.2, 0.19, 0.18, 0.20, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]) * 1).tolist(),
                                        # (np.array([0.2, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]) * 0.01).tolist()
                                        ])[:, 0:T])).tolist(),
        'gen_gas_num': 1,
        'gen_gas_index': [2],  # the gas generator index in the gas system
        'gen_gas_index_power': [3],  # the gas generator index in the power system
        'gen_gas_min': [0],  # this is power
        'gen_gas_max': [0.3],  # this is power
        'gen_gas_efficiency': [2],  # 0.05 gas => 0.5 power
    }
    gas_line_info_0 = {
        'weymouth': [0.4] * (2 + a_1 + 1),
        'gas_line_num': 2 + a_1 + 1,
        # 'gas_line_start_point': [1, 1, 1, 4, 4,  6],  # gas flow out
        # 'gas_line_end_point': [0, 2, 4, 3, 5, 1],  # gas flow in
        'gas_line_start_point': [1, 1, 1, 3, 4, 6],  # gas flow out
        'gas_line_end_point': [0, 2, 3, 4, 5, 1],  # gas flow in
        'gas_line_pack_coefficient': [0.5] * (2 + a_1 + 1),
        'gas_line_pack_initial': 20,
        'gas_flow_in_max': [1] * (2 + a_1 + 1),  # unused
        'gas_flow_out_max': [1] * (2 + a_1 + 1),  # unused
        'gas_line_active': [],
        'compressor_num': 0,
        'compressor_start_point': [],
        'compressor_end_point': [],
        'compressor_coefficient': [],
        'compressor_max_flow': [],
        'compressor_energy_consumption': [],
    }
    player_index = player_index + 1
    # -------------- p l a y e r 1 ----------------
    a_2 = 2
    system_info_1 = {
        'index': player_index,
        'T': T,
        'connection_area': g_connection[player_index]
    }
    node_info_1 = {
        'gen_num': 3 + 1,  # the outside node as node-12 connected with index-3 with line-12
        'gen_index': [0, 0, 5, 12],
        'gen_power_min': [0.3, 0.3, 0.3, 0],  # 0 - 3
        'gen_power_max': [1, 0.8, 1.2, 10],
        'gen_react_min': [0.3, 0.3, 0.3, 0],  # 0 - 3
        'gen_react_max': [1, 0.8, 1.2, 10],
        'gen_cost_a': [1, 1.3, 0.9, 0],
        'gen_cost_b': [0.1, 0.1, 0.1, 0],
        'gen_cost_c': [0.1, 0.1, 0.1, 0],
        'bus_num': 12 + 1,
        'bus_voltage_min': [0.95 * 1] * (12 + 1),
        'bus_voltage_max': [1.05 * 1] * (12 + 1),
        'load_num': 8 + 1,
        'load_index': [2, 3, 4, 7, 8, 9, 10, 11, 12],
        'load_power_min': (np.array(
            [[0.10, 0.10, 0.10, 0.10, 0.09, 0.12, 0.10, 0.12, 0.12, 0.12],
             [0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.25, 0.22, 0.20],
             [0.20, 0.19, 0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.22, 0.20],
             [0.10, 0.12, 0.10, 0.10, 0.10, 0.13, 0.12, 0.10, 0.09, 0.09],
             [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
             [0.10, 0.11, 0.12, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12],
             [0.20, 0.20, 0.18, 0.20, 0.22, 0.22, 0.22, 0.20, 0.20, 0.20],
             [0.10, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])[:, 0:T] * 1.3).tolist(),  # 1.3
        'load_power_max': (np.array(
            [[0.15, 0.16, 0.17, 0.18, 0.17, 0.15, 0.14, 0.14, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.26, 0.27, 0.25, 0.26, 0.24, 0.23, 0.25],
             [0.25, 0.23, 0.24, 0.26, 0.27, 0.28, 0.29, 0.27, 0.26, 0.25],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.35, 0.40, 0.42, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.28, 0.29, 0.24, 0.23, 0.20, 0.20, 0.20],
             [0.15, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])[:, 0:T] * 1.3).tolist(),  # 1.7
        'load_react_min': (np.array(
            [[0.10, 0.11, 0.12, 0.10, 0.09, 0.12, 0.10, 0.12, 0.12, 0.12],
             [0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.25, 0.22, 0.20],
             [0.20, 0.19, 0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.22, 0.20],
             [0.10, 0.12, 0.10, 0.10, 0.10, 0.13, 0.12, 0.10, 0.09, 0.09],
             [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
             [0.10, 0.11, 0.12, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12],
             [0.20, 0.20, 0.18, 0.20, 0.22, 0.22, 0.22, 0.20, 0.20, 0.20],
             [0.10, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])[:, 0:T] * 1.3).tolist(),
        'load_react_max': (np.array(
            [[0.15, 0.16, 0.17, 0.18, 0.17, 0.15, 0.14, 0.14, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.26, 0.27, 0.25, 0.26, 0.24, 0.23, 0.25],
             [0.25, 0.23, 0.24, 0.26, 0.27, 0.28, 0.29, 0.27, 0.26, 0.25],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.35, 0.40, 0.42, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [0.25, 0.26, 0.27, 0.28, 0.29, 0.24, 0.23, 0.20, 0.20, 0.20],
             [0.15, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
             [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])[:, 0:T] * 1.3).tolist(),
        'bus_num_outside': 1,
        'connection_index': [1],  # the outer area power/gas connect with this index
    }
    line_info_1 = {
        'line_num': 11 + 1,
        'line_current_capacity': [10] * (11 + 1),
        'line_start_point': [0, 1, 0, 3, 0, 5, 8, 5, 6, 5, 6, 12],
        'line_end_point': [1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 1],
        'line_resistance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist(),
        'line_reactance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist()
    }
    gas_node_info_1 = {
        'gas_node_num': 3 + a_2 + 1,
        'node_pressure_min': [0] * (3 + a_2 + 1),
        'node_pressure_max': [10] * (3 + a_2 + 1),
        'gas_well_num': 1,
        'well_index': [0],
        'well_output_min': [0],
        'well_output_max': [2],
        'gas_load_num': 1 + 2 + 1,
        'load_index': [2, 3, 4, 5],
        'gas_load_min': (np.array([[0.1, 0.11, 0.12, 0.11, 0.12, 0.11, 0.12, 0.11, 0.10, 0.10],
                                   [0.1, 0.09, 0.08, 0.09, 0.10, 0.11, 0.11, 0.11, 0.09, 0.08],
                                   [0.1, 0.09, 0.08, 0.09, 0.10, 0.11, 0.11, 0.11, 0.09, 0.08],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   ])[:, 0:T]).tolist(),
        'gas_load_max': (np.array([[0.2, 0.21, 0.22, 0.21, 0.23, 0.24, 0.21, 0.26, 0.23, 0.24],
                                   [0.2, 0.22, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
                                   [0.2, 0.22, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                   ])[:, 0:T]).tolist(),
        'gen_gas_num': 1,
        'gen_gas_index': [2],  # the gas generator index in the gas system
        'gen_gas_index_power': [2],  # the gas generator index in the power system
        'gen_gas_min': [0],  # this is power
        'gen_gas_max': [0.2],
        'gen_gas_efficiency': [2],
    }
    gas_line_info_1 = {
        'weymouth': [0.4] * (2 + a_2 + 1),
        'gas_line_num': 2 + a_2 + 1,
        'gas_line_start_point': [0, 1, 1, 3, 1],  # gas flow out
        'gas_line_end_point': [1, 2, 3, 4, 5],  # gas flow in
        'gas_line_pack_coefficient': [0.5] * (2 + a_2 + 1),
        'gas_line_pack_initial': 20,
        'gas_flow_in_max': [1] * (2 + a_2 + 1),  # unused
        'gas_flow_out_max': [1] * (2 + a_2 + 1),  # unused
        'gas_line_active': [],
        'compressor_num': 0,
        'compressor_start_point': [],
        'compressor_end_point': [],
        'compressor_coefficient': [],
        'compressor_max_flow': [],
        'compressor_energy_consumption': [],
    }
    player_index = player_index + 1

    p1 = PowerNet(system_info_0, node_info_0, line_info_0, gas_node_info_0, gas_line_info_0)
    p2 = PowerNet(system_info_1, node_info_1, line_info_1, gas_node_info_1, gas_line_info_1)
    return [p1, p2, PlayerN1()]


def update_old_value():
    for i, player in enumerate(g_players):
        player.set_old_value(copy(g_info[i]))
    temp_lam = []
    for line in range(g_link):
        for time in range(T):
            for index in range(16):
                temp_lam.append(g_lam[line][time][index])
    g_playerN1.set_old_value(copy(temp_lam))


def sub_info(a, b):
    return (a.this_voltage_square - b.this_voltage_square) * (a.this_voltage_square - b.this_voltage_square) + \
           (a.that_voltage_square - b.that_voltage_square) * (a.that_voltage_square - b.that_voltage_square) + \
           (a.power_flow - b.power_flow) * (a.power_flow - b.power_flow) + \
           (a.react_flow - b.react_flow) * (a.react_flow - b.react_flow) + \
           (a.this_node_pressure - b.this_node_pressure) * (a.this_node_pressure - b.this_node_pressure) + \
           (a.that_node_pressure - b.that_node_pressure) * (a.that_node_pressure - b.that_node_pressure) + \
           (a.gas_flow_in - b.gas_flow_in) * (a.gas_flow_in - b.gas_flow_in) + \
           (a.gas_flow_out - b.gas_flow_out) * (a.gas_flow_out - b.gas_flow_out)


def sub_norm(a, b):
    sum = 0
    for i in range(len(g_connection)):
        for j in range(len(g_connection[i])):
            for k in range(T):
                sum += sub_info(a[i][j][k], b[i][j][k])
    return sum


def calculate_NE():
    global g_lam
    count_best_response = 0
    old_info = 0
    while count_best_response < 30:
        old_info = copy(g_info)
        for i, player in enumerate(g_players):
            # get the data for the player i
            player.update_model(g_tao)  # 填充x_i 以及lam_i
            player.optimize()
        # update the lam_dual variable
        g_lam = copy(g_playerN1.optimize(g_tao))
        # update the response
        if sub_norm(old_info, copy(g_info)) < 0.000001:
            print(count_best_response + 1)
            break
        count_best_response = count_best_response + 1


def calculate_GNE(iijj):
    outer_loop_count = 0
    global result_plt
    global result_plt1
    global result_plt2
    global result_plt3
    global result_plt4
    global result_plt5
    global result_plt6
    global result_plt7
    global result_plt8
    global result_plt9
    global result_plt10
    global result_plt11
    global result_plt12
    global result_plt13
    global result_plt14
    global result_plt15
    result_plt = []
    result_plt1 = []
    result_plt2 = []
    result_plt3 = []
    result_plt4 = []
    result_plt5 = []
    result_plt6 = []
    result_plt7 = []
    result_plt8 = []
    result_plt9 = []
    result_plt10 = []
    result_plt11 = []
    result_plt12 = []
    result_plt13 = []
    result_plt14 = []
    result_plt15 = []
    lam_h1 = []
    lam_h2 = []
    lam_h3 = []
    lam_h4 = []
    lam_h5 = []
    lam_h6 = []
    lam_h7 = []
    lam_h8 = []
    lam_h9 = []
    lam_h10 = []
    lam_h11 = []
    lam_h12 = []
    lam_h13 = []
    lam_h14 = []
    lam_h15 = []
    lam_h16 = []
    global gap1
    global gap2
    global PUNISH
    gap1 = []
    gap2 = []
    while outer_loop_count < OUTER_LOOP[iijj]:
        print(outer_loop_count)
        # give xn, lam_n, calculate the equilibrium
        calculate_NE()
        # 现在我们得到了一个新的NE，我们应该把这个NE设为参照值
        update_old_value()
        outer_loop_count = outer_loop_count + 1
        if outer_loop_count == 250:
            breakthis = 1;
        result_plt.append(g_info[0][0][0].this_voltage_square - g_info[1][0][0].this_voltage_square)
        result_plt1.append(g_info[0][0][0].that_voltage_square - g_info[1][0][0].that_voltage_square)
        result_plt2.append(g_info[0][0][0].power_flow - g_info[1][0][0].power_flow)
        result_plt3.append(g_info[0][0][0].react_flow - g_info[1][0][0].react_flow)
        result_plt4.append(1 * (g_info[0][0][0].this_node_pressure - g_info[1][0][0].this_node_pressure))
        result_plt5.append(1 * (g_info[0][0][0].that_node_pressure - g_info[1][0][0].that_node_pressure))
        result_plt6.append(1 * (g_info[0][0][0].gas_flow_in - g_info[1][0][0].gas_flow_in))
        result_plt7.append(1 * (g_info[0][0][0].gas_flow_out - g_info[1][0][0].gas_flow_out))
        result_plt8.append(g_info[1][0][0].this_voltage_square)
        result_plt9.append(g_info[1][0][0].that_voltage_square)
        result_plt10.append(g_info[1][0][0].power_flow)
        result_plt11.append(g_info[1][0][0].react_flow)
        result_plt12.append(g_info[1][0][0].this_node_pressure)
        result_plt13.append(g_info[1][0][0].that_node_pressure)
        result_plt14.append(g_info[1][0][0].gas_flow_in)
        result_plt15.append(g_info[1][0][0].gas_flow_out)
        lam_h1.append(g_lam[0][0][0])
        lam_h2.append(g_lam[0][0][1])
        lam_h3.append(g_lam[0][0][2])
        lam_h4.append(g_lam[0][0][3])
        lam_h5.append(g_lam[0][0][4])
        lam_h6.append(g_lam[0][0][5])
        lam_h7.append(g_lam[0][0][6])
        lam_h8.append(g_lam[0][0][7])
        lam_h9.append(g_lam[0][0][8])
        lam_h10.append(g_lam[0][0][9])
        lam_h11.append(g_lam[0][0][10])
        lam_h12.append(g_lam[0][0][11])
        lam_h13.append(g_lam[0][0][12])
        lam_h14.append(g_lam[0][0][13])
        lam_h15.append(g_lam[0][0][14])
        lam_h16.append(g_lam[0][0][15])
        if p1.cal_gap() > 1000:
            gap1.append(-10)
        else:
            gap1.append(p1.cal_gap())
        if p2.cal_gap() > 1000:
            gap2.append(-10)
        else:
            gap2.append(p2.cal_gap())

    # plot value and difference
    plt.plot(result_plt, label='diff0')
    plt.plot(result_plt1, label='diff1')
    plt.plot(result_plt2, label='diff2')
    plt.plot(result_plt3, label='diff3')
    plt.plot(result_plt4, label='diff4')
    plt.plot(result_plt5, label='diff5')
    plt.plot(result_plt6, label='diff6')
    plt.plot(result_plt7, label='diff7')
    plt.plot(result_plt8, '*', label='this_voltage_square')
    plt.plot(result_plt9, '*', label='that_voltage_square')
    plt.plot(result_plt10, '*', label='power_flow')
    plt.plot(result_plt11, '*', label='react_flow')
    plt.plot(result_plt12, '*', label='this_node_pressure')
    plt.plot(result_plt13, '*', label='that_node_pressure')
    plt.plot(result_plt14, '*', label='gas_flow_in')
    plt.plot(result_plt15, '*', label='gas_flow_out')
    plt.xlabel('iterator count')
    plt.ylabel('value')
    plt.legend(loc='best')
    plt.savefig('decision_variable' + str(iijj) + '.png')
    plt.cla()

    # plot the shadow price
    plt.plot(lam_h1, '.', label='1')
    plt.plot(lam_h2, '.', label='2')
    plt.plot(lam_h3, '.', label='3')
    plt.plot(lam_h4, '.', label='4')
    plt.plot(lam_h5, '*', label='5')
    plt.plot(lam_h6, '*', label='6')
    plt.plot(lam_h7, '*', label='7')
    plt.plot(lam_h8, '*', label='8')
    plt.legend(loc='best')
    plt.savefig('shadow_price_power' + str(iijj) + '.png')
    plt.cla()

    plt.plot(lam_h9, '.', label='9')
    plt.plot(lam_h10, '.', label='10')
    plt.plot(lam_h11, '.', label='11')
    plt.plot(lam_h12, '.', label='12')
    plt.plot(lam_h13, '*', label='13')
    plt.plot(lam_h14, '*', label='14')
    plt.plot(lam_h15, '*', label='15')
    plt.plot(lam_h16, '*', label='16')
    plt.legend(loc='best')
    plt.savefig('shadow_price_gas' + str(iijj) + '.png')
    plt.cla()


def calculate_pccp():
    global abcd
    abcd = []
    global G_K
    pccp_loop = 0
    while pccp_loop < PCCP_COUNT:
        pccp_loop = pccp_loop + 1
        G_K = G_K * 2
        calculate_GNE(pccp_loop)
        for player in g_players:
            player.update_outer_model()
            print('player gap : ' + str(player.cal_gap()))
        abcd.append([p1.cal_gap(), p2.cal_gap()])
    #    calculate_GNE(15)
    plt.plot(abcd)
    plt.show()


def ddd():
    print_info(g_info[0][0][0])
    print("-------------------")
    print_info(g_info[1][0][0])


if __name__ == '__main__':
    result_plt = []
    result_plt1 = []
    result_plt2 = []
    result_plt3 = []
    result_plt4 = []
    result_plt5 = []
    result_plt6 = []
    result_plt7 = []
    result_plt8 = []
    result_plt9 = []
    result_plt10 = []
    result_plt11 = []
    result_plt12 = []
    result_plt13 = []
    result_plt14 = []
    result_plt15 = []
    gap1 = []
    gap2 = []
    all_players = getPowerNet()
    g_players = all_players[:player_num]
    g_playerN1 = all_players[player_num]
    [p1, p2] = g_players
    pn = g_playerN1
    for player_g in g_players:
        player_g.build_model()
    calculate_pccp()
    pycharm_debug = 2


def getCostValue(player):
    print('self.gen_power_cost: ' + str(player.gen_power_cost.getValue()))
    print('self.load_power_cost: ' + str(player.load_power_cost.getValue()))
    print('self.load_gas_cost: ' + str(player.load_gas_cost.getValue()))
    print('self.power_buy_cost: ' + str(player.power_buy_cost.getValue()))
    print('self.gas_buy_cost: ' + str(player.gas_buy_cost.getValue()))
    print('self.well_produce_cost: ' + str(player.well_produce_cost.getValue()))
