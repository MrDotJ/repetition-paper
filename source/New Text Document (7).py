# scratch 4 5 6 7
# add more constrains
# change all the direction of gas IN/OUT to the actual direction
# TODO: deal with it
import gurobipy as gurobi
#import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import math







def make_lam_index():
    result = []
    for p in range(player_num):
        node_p = node_t[p].copy()  # [[0 1],[0 2],[0 3]]
        per_p = []
        for line_index in range(len(node_p)):  # [0 1] [0 2] [0 3]
            line = node_p[line_index]
            start = line[0]
            end = line[1]
            if start < end:
                per_p = \
                    per_p + list(itertools.chain(
                        *[[start * 2 * T + 2 * t, start * 2 * T + 2 * t + 1, end * 2 * T + 2 * t,
                           end * 2 * T + 2 * t + 1]
                          for t in range(T)]))
            else:
                per_p = \
                    per_p + list(itertools.chain(*[
                        [start * 2 * T + 2 * t + 1, start * 2 * T + 2 * t, end * 2 * T + 2 * t + 1, end * 2 * T + 2 * t]
                        for t in range(T)]))
        result.append(per_p)
    return result







# test addMVar
#                       [0]
#                        o  (0)(2)(4)
#                         \
#                          \
#                           \  (1)(3)(5)
#                 [2]o-------o[1]
#                    (7)   (6)
#                    (9)   (8)
#                    (11)  (10)
#
price = [[0, 6, 6],
         [6, 0, 6],
         [6, 6, 0]]
g_link = 2
injection = [[], [], []]
#          angle pressure In Out   TODO: IN-OUT BUG SOURCE
node_t = [[[0, 1], [2, 3], [4, 5]],
          [[1, 0], [3, 2], [5, 4], [6, 7], [8, 9], [10, 11]],
          [[7, 6], [9, 8], [11, 10]]]
g_connection = [[1],
                [0, 2],
                [1]]
connection_index = [[1],
                    [1, 1],
                    [1]]
player_num = len(price)
all_node = g_link * 2 * 3
# about topology UP
T = 4
g_tao = 100
PUNISH = 300
OUTER_LOOP = 300
INNER_FIX = 0.001
# about algorithm UP

g_lam = [0] * (2 * all_node * T)
g_lam_index = make_lam_index()
g_angles = [
    [[0] * T
     for i in range(2 * len(node_t[p]))]
    for p in range(player_num)
]
g_gas_price_aux  = [1, -1, 1]
#                  buy sell bny




















#      a n g l e         p r e s s u r e    i n    o u t
#    0         1        2        3          4          5      6     7        8        9          10     11        12       13        14      15      16          17
# [ [000..] [000...]   [000..] [000...]   [000..] [000...]
#   [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]
#   [000..] [000...]  [000..] [000...]  [000..] [000...]
#   [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]  [000..] [000...]
#   [000..] [000...]  [000..] [000...]  [000..] [000...]  ]
# ind = g_connection[connect_to].index(self.index)   0 1 2 ==> 2 3   8 9    14 15   ind * 6 + 2/3
#                                                                                   ind * 6 + 4/5
class Region:
    def __init__(self, index, X_raw, node_info, connection_info, gas_line_info, gas_node_info):
        self.index = index
        self.T = T
        # give basic topology and information
        self.node_count = 0
        self.X_raw = X_raw
        # connection_information
        self.connection_index = connection_info['connection_index']
        self.connection_x = connection_info['connection_x']
        self.connection_area = connection_info['connection_area']
        self.connection_exchange_max = connection_info['connection_exchange_max']
        # node_information
        self.node_info = node_info
        self.X_B1 = []
        # all about gas
        # gas node information
        self.gas_node_num = gas_node_info['gas_node_num']
        self.node_pressure_min = gas_node_info['node_pressure_min']
        self.node_pressure_max = gas_node_info['node_pressure_max']
        self.well_num = gas_node_info['gas_well_num']
        self.well_index = gas_node_info['well_index']  # [0,0,4,5]
        self.well_output_min = gas_node_info['well_output_min']
        self.well_output_max = gas_node_info['well_output_max']
        self.gas_load_index = gas_node_info['load_index']
        self.gas_load_min = gas_node_info['gas_load_min']
        self.gas_load_max = gas_node_info['gas_load_max']
        self.gas_load_num = gas_node_info['gas_load_num']
        self.gen_gas_num = gas_node_info['gen_gas_num']
        self.gen_gas_index = gas_node_info['gen_gas_index']
        self.gen_gas_index_power = gas_node_info['gen_gas_index_power']
        self.gen_gas_min = gas_node_info['gen_gas_min']
        self.gen_gas_max = gas_node_info['gen_gas_max']
        self.gen_gas_efficiency = gas_node_info['gen_gas_efficiency']
        # gas line information
        self.weymouth = gas_line_info['weymouth']  # for easy, it should contain all line(include active line)
        self.gas_line_num = gas_line_info['gas_line_num']
        self.gas_line_start_point = gas_line_info['gas_line_start_point']  # gas flow out
        self.gas_line_end_point = gas_line_info['gas_line_end_point']  # gas flow in
        self.gas_line_pack_coefficient = gas_line_info['gas_line_pack_coefficient']
        self.gas_line_pack_initial = gas_line_info['gas_line_pack_initial']
        self.gas_line_active = gas_line_info['gas_line_active']
        self.gas_flow_in_max = gas_line_info['gas_flow_in_max']
        self.gas_flow_out_max = gas_line_info['gas_flow_out_max']
        self.compressor_num = gas_line_info['compressor_num']
        self.compressor_start_point = gas_line_info['compressor_start_point']
        self.compressor_end_point = gas_line_info['compressor_end_point']
        self.compressor_coefficient = gas_line_info['compressor_coefficient']
        self.compressor_max_flow = gas_line_info['compressor_max_flow']
        self.compressor_energy_consumption = gas_line_info['compressor_energy_consumption']
        # gas information end

        # model
        self.model = gurobi.Model()
        self.load_vars = None
        self.gene_vars = None
        self.power_injections = None
        self.angles_inside = []
        self.angles_outside = []
        self.power_injections_outside = []
        self.object_basic = None
        self.object_addition = None
        self.object = None
        self.free_var = None

        # all about gas model
        self.node_pressure = None
        self.well_output = None
        self.gas_load = None
        self.gen_gas_power = None
        # line
        self.gas_flow_in = None
        self.gas_flow_out = None
        self.linepack = None
        self.compressor_out = None
        self.compressor_in = None
        self.gas_source = None
        # gas end
        # old value
        self.old_value = [[0] * T for i in range(len(self.connection_area) * 2 * 3)]
        self.gas_flow_in_old = [[0.01 for jqy in range(self.T)] for jjqqyy in range(self.gas_line_num)]
        self.gas_flow_out_old = [[0.01 for jqy in range(self.T)] for jjqqyyy in range(self.gas_line_num)]
        self.node_pressure_old = [[0.01 for jqy in range(self.T)] for jjqqyy in range(self.gas_node_num)]
        self.pccp = None
        self.constrain_update = []

    def build_B(self):
        # build_B
        self.node_count = len(self.X_raw)
        self.X_B1 = []
        for i in range(self.node_count):
            self.X_B1.append([0] * self.node_count)
        for row in range(self.node_count):
            for column in range(self.node_count):
                if row == column:
                    X_i = self.X_raw[row]
                    B_ii = 0
                    for value in X_i:
                        if value != 0:
                            B_ii = B_ii + 1 / value
                    self.X_B1[row][column] = B_ii
                else:
                    if self.X_raw[row][column] != 0:
                        self.X_B1[row][column] = -1 / self.X_raw[row][column]
        X_B1_expand = np.array(self.X_B1, dtype=np.float64) * 0
        X_B1_expand[:-1, :-1] = np.array(self.X_B1)[:-1, :-1]
        self.X_B1 = X_B1_expand.tolist()

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

    def gen_connected_with(self, node):  # list of expression
        result = np.where(np.array(self.gen_gas_index) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_gen = []
            for time in range(self.T):
                per_gen.append(self.gen_gas_power[i, time] / self.gen_gas_efficiency[i])
            result_list.append(per_gen)
        return np.array(result_list)

    def p2g_connected_with(self, node):
        return np.array([[0] * self.T])

    def gas_flow_out_connected_with(self, node):
        result = np.where(np.array(self.gas_line_end_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_out = []
            for time in range(self.T):
                per_out.append(self.gas_flow_out[i, time])
            result_list.append(per_out)
        return np.array(result_list)

    def gas_flow_in_connected_with(self, node):
        result = np.where(np.array(self.gas_line_start_point) == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result[0]:
            per_in = []
            for time in range(self.T):
                per_in.append(self.gas_flow_in[i, time])
            result_list.append(per_in)
        return np.array(result_list)

    def get_gas_to_power(self, node, time):
        result = 0
        if node in self.gen_gas_index_power:
            index = self.gen_gas_index_power.index(node)
            result = self.gen_gas_power[index, time]
        else:
            result = 0
        return result

    def build_model(self):
        global price
        index = self.index
        # build B_1 matrix
        self.build_B()
        objects = []
        constrains = []
        # build for gas system
        self.well_output = \
            self.model.addVars(self.well_num, self.T,
                               lb=[[self.well_output_min[i]] * self.T for i in range(self.well_num)],
                               ub=[[self.well_output_max[i]] * self.T for i in range(self.well_num)],
                               name='gas_well_outputs')
        self.node_pressure = \
            self.model.addVars(self.gas_node_num, self.T,
                               lb=[[self.node_pressure_min[i]] * self.T for i in range(self.gas_node_num)],
                               ub=[[self.node_pressure_max[i]] * self.T for i in range(self.gas_node_num)],
                               name='node_pressure')
        self.gas_flow_in = \
            self.model.addVars(self.gas_line_num, self.T,
                               ub=[[self.gas_flow_in_max[i]] * self.T for i in range(self.gas_line_num)],
                               lb=[[-1 * self.gas_flow_in_max[i]] * self.T for i in range(self.gas_line_num)],
                               name='gas_flow_in')
        self.gas_flow_out = \
            self.model.addVars(self.gas_line_num, self.T,
                               ub=[[self.gas_flow_out_max[i]] * self.T for i in range(self.gas_line_num)],
                               lb=[[-1 * self.gas_flow_out_max[i]] * self.T for i in range(self.gas_line_num)],
                               name='gas_flow_out')
        self.gas_load = \
            self.model.addVars(self.gas_load_num, self.T,
                               lb=self.gas_load_min,
                               ub=self.gas_load_max,
                               name='gas_load')
        self.gen_gas_power = \
            self.model.addVars(self.gen_gas_num, self.T,
                               lb=[[self.gen_gas_min[i]] * self.T for i in range(self.gen_gas_num)],
                               ub=[[self.gen_gas_max[i]] * self.T for i in range(self.gen_gas_num)], )
        self.linepack = self.model.addVars(self.gas_line_num, self.T, name='gas_linepack')
        self.pccp = self.model.addVars(self.gas_line_num, self.T,
                                       lb=0, name='pccp')

        self.model.update()
        # add constrain
        # gas nodal balance for all passive and active
        for node in range(self.gas_node_num):
            # use numpy !!!! return [[]] format
            Well = self.well_connected_with(node)  # 节点node对应的well变量
            Load = self.load_connected_with(node)
            # considered efficiency !!!!!!!
            Gen = self.gen_connected_with(node)  # this change Power to Gas
            P2G = self.p2g_connected_with(node)  # this is just gas
            Line_Out = self.gas_flow_out_connected_with(node)
            Line_In = self.gas_flow_in_connected_with(node)
            # Line_Out_diff = self.gas_flow_out_diff_connected_with(node)
            for time in range(self.T):
                self.model.addConstr(
                    lhs=sum(Well[:, time]) + sum(P2G[:, time]) + sum(Line_Out[:, time]),  # source
                    rhs=sum(Gen[:, time]) + sum(Load[:, time]) + sum(Line_In[:, time]),  # load
                    sense=gurobi.GRB.EQUAL,
                    name='gas_nodal_balance_node')

        # line pack passive line     passive 线路的存量公式  (14)
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
                        sense=gurobi.GRB.EQUAL)
        # passive 线路的存量时间公式 (15) for passive line
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                for time in range(self.T):
                    if time == 0:
                        self.model.addConstr(
                            lhs=self.linepack[line, 0] - self.linepack[line, self.T - 1],
                            rhs=self.gas_flow_in[line, 0] - self.gas_flow_out[line, 0],
                            sense=gurobi.GRB.EQUAL)
                    else:
                        self.model.addConstr(
                            lhs=self.linepack[line, time] - self.linepack[line, time - 1],
                            rhs=self.gas_flow_in[line, time] - self.gas_flow_out[line, time],
                            sense=gurobi.GRB.EQUAL)
        # T-1's 存量 小于 初值
        linepack_sum = 0  # ? passive or active
        for line in range(self.gas_line_num):
            if line not in self.gas_line_active:
                linepack_sum = linepack_sum + self.linepack[line, self.T - 1]
        self.model.addConstr(linepack_sum <= self.gas_line_pack_initial)

        # active pipeline pressure-increase & gas-consume
        # active 线路的 气压升压 公式 以及 能耗公式 (16)(18)
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

        # weymouth for passive line
        # passive 线路的well 方程 (14)
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
                        sense=gurobi.GRB.LESS_EQUAL)

                    self.constrain_update.append(
                        self.model.addConstr(
                            lhs=weymouth * self.node_pressure[start_point, time] *
                                self.node_pressure[start_point, time] - (
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) *
                                        (self.gas_flow_in[line, time] + self.gas_flow_out[line, time]) / 2 -
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) *
                                        (self.gas_flow_in_old[line][time] + self.gas_flow_out_old[line][time]) / 4 -
                                        weymouth * self.node_pressure[end_point, time] *
                                        self.node_pressure[end_point, time] +
                                        2 * weymouth * self.node_pressure[end_point, time] *
                                        self.node_pressure_old[end_point][time]
                                ),
                            rhs=self.pccp[line, time],
                            sense=gurobi.GRB.LESS_EQUAL
                        )
                    )
        # gas system end

        # set node_pressure_start    node_pressure_end         gas_flow_in         gas_flow_out
        # construct power model
        # add variables
        # 每一个节点既有负荷又有generator
        # 每个节点添加负荷变量，发电变量，内部节点的注入功率变量，以及联系
        # for node in range(self.node_count):
        self.load_vars = self.model.addVars(
            self.node_count, T,
            lb=[[self.node_info[node]['min_load']] * T for node in range(self.node_count)],
            ub=[[self.node_info[node]['max_load']] * T for node in range(self.node_count)],
            obj=0,
            name='Region_' + str(self.index) + '_loads')
        self.gene_vars = self.model.addVars(
            self.node_count, T,
            lb=[[self.node_info[node]['min_power']] * T for node in range(self.node_count)],
            ub=[[self.node_info[node]['max_power']] * T for node in range(self.node_count)],
            obj=0,
            name='Region_' + str(self.index) + '_powers')
        self.power_injections = self.model.addVars(
            self.node_count, T,
            lb=-10000, ub=10000,
            name='Region_' + str(self.index) + '_power_injection')

        # 功率注入的等式
        for node in range(self.node_count):
            for time in range(T):
                self.model.addConstr(
                    -1 * self.load_vars[node, time] + self.gene_vars[node, time] ==
                    self.power_injections[node, time],
                    name='power_injection' + str(node) + '_time_' + str(time))

        # 添加内部/外部相角变量以及外部注入功率变量，**相角的范围是多少？**
        self.angles_inside = self.model.addVars(
            self.node_count, T, lb=-3.14, ub=3.14, name='angle_inside')
        self.angles_outside = self.model.addVars(
            len(self.connection_area), T, lb=-3.14, ub=3.14, name='angle_outside')
        # 外部功率注入
        self.power_injections_outside = \
            self.model.addVars(len(self.connection_area), T,
                               lb=[[-1 * m_max] * T for m_max in self.connection_exchange_max],
                               ub=[[1 * m_max] * T for m_max in self.connection_exchange_max],
                               name='Region_' + str(self.index) + '_power_injection_outside')
        # 外部功率注入直流潮流
        for conn in range(len(self.connection_area)):
            for time in range(T):
                self.model.addConstr(self.power_injections_outside[conn, time] ==
                                     (self.angles_outside[conn, time] / self.connection_x[conn] -
                                      self.angles_inside[self.connection_index[conn], time] / self.connection_x[
                                          conn]))  # the B is B^ not B^^
        # x should satisfy the max-power-flow

        # 内部线路直流潮流的计算公式 except the reference node
        ang_ins = lambda _time: [self.angles_inside[i, _time] for i in range(self.node_count)]
        external_injection_pos = 0
        for row in range(self.node_count - 1):
            for time in range(T):
                gas_to_power = self.get_gas_to_power(row, time)
                if row in self.connection_index:  # if the index has connection with outside area
                    self.model.addConstr(np.array(self.X_B1[row]).reshape((1, -1)).dot(
                        np.array(ang_ins(time)).reshape((-1, 1)))[0][0] ==  # TODO # here is wrong
                                         self.power_injections[row, time] +
                                         self.power_injections_outside[external_injection_pos, time] +
                                         gas_to_power)
                    if time == T - 1:
                        external_injection_pos = external_injection_pos + 1
                else:  # else this index does not have external injection
                    self.model.addConstr(
                        np.array(self.X_B1[row]).reshape((1, -1)).dot(
                            np.array(ang_ins(time)).reshape((-1, 1)))[0][0] ==
                        self.power_injections[row, time] + gas_to_power)
        # 添加发电机的爬坡约束
        node_count_local_var = self.node_count
        get_gen = lambda: np.array([[self.gene_vars[_index_, _time_] for _time_ in range(T)]
                                    for _index_ in range(node_count_local_var)])
        gen_var = get_gen()
        gen_ramp_up_expr = gen_var[:, 1: T] - gen_var[:, 0: T - 1]
        gen_ramp_down_expr = gen_var[:, 0: T - 1] - gen_var[:, 1: T]
        gen_ramp_up = [self.node_info[gen]['gen_ramp_up'] for gen in range(self.node_count)]
        gen_ramp_down = [self.node_info[gen]['gen_ramp_down'] for gen in range(self.node_count)]
        for gen in range(self.node_count):
            for time in range(T - 1):
                self.model.addConstr(
                    gen_ramp_up_expr[gen, time] <= gen_ramp_up[gen],
                    name='ramp_up_gen' + str(gen) + '_time_' + str(time))
                self.model.addConstr(
                    gen_ramp_down_expr[gen, time] <= gen_ramp_down[gen],
                    name='ramp_down_gen' + str(gen) + '_time_' + str(time))
        # 添加 目标函数 注意正负
        for node in range(self.node_count):
            for time in range(T):
                objects.append(
                    self.node_info[node]['load_coeff'] *
                    (self.load_vars[node, time] - self.node_info[node]['load_ref']) *
                    (self.load_vars[node, time] - self.node_info[node]['load_ref']))  # 舒适成本
                objects.append(
                    self.node_info[node]['power_coeff_a'] * (self.gene_vars[node, time] * self.gene_vars[node, time]) +
                    (self.node_info[node]['power_coeff_b'] + self.gene_vars[node, time]) +
                    self.node_info[node]['power_coeff_c'])  # 发电成本
        for conn in range(len(self.connection_area)):
            for time in range(T):
                objects.append(price[self.index][self.connection_area[conn]] *
                               self.power_injections_outside[conn, time])  # 购电成本
        line_num = self.gas_line_num - len(self.connection_area)
        for conn in range(len(self.connection_area)):
            for time in range(T):
                objects.append(3 *  # gas_buy_price  购气成本
                               self.gas_flow_in[line_num + conn, time] * g_gas_price_aux[self.index])

        # add constrain - power balance
        power_balance = self.model.addConstr(
            gurobi.quicksum(self.load_vars) ==
            gurobi.quicksum(self.gene_vars) + gurobi.quicksum(self.power_injections_outside) +
            gurobi.quicksum(self.gen_gas_power),
            name='Region_' + str(index) + '_power_balance'
        )
        constrains.append(power_balance)
        # add object
        self.object_basic = sum(objects)

    def update_model(self, tao):
        global g_angles
        global g_connection
        global g_lam  # TODO: SECOND WRONG

        lams = []
        lam_index = g_lam_index[self.index]
        for index in lam_index:
            lams.append(g_lam[index])  # first, all time for "one line", then all time for another line

        duals = []
        if self.index == 0:
            self.free_var = [0] * T
        else:
            self.free_var = self.model.addVars(T, lb=-3.14, ub=3.14, name='free_var')
            # self.free_var = self.model.addVar(lb=0, ub=0, name='free_var')

        # for each power node
        for i in range(len(self.connection_area)):
            # ========================= a n g l e ====================================
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            ind = g_connection[connect_to].index(self.index)
            for time in range(T):
                thisvar_inside = self.angles_inside[self.connection_index[i], time]
                thisvar_outside = g_angles[connect_to][ind * 6 + 1][time]
                thatvar_inside = self.angles_outside[i, time]
                thatvar_outside = g_angles[connect_to][ind * 6][time]
                # ***********************************************
                duals.append(thisvar_inside + self.free_var[time] - thisvar_outside)
                duals.append(-1 * thisvar_inside - self.free_var[time] + thisvar_outside)
                duals.append(thatvar_inside + self.free_var[time] - thatvar_outside)
                duals.append(-1 * thatvar_inside - self.free_var[time] + thatvar_outside)
                # this is for one line, for all time t: [(0, 1, 2, 3)t=0,()t=1, ... ] + [another line]
            # for each line gas node pressure
            # ========================= p r e s s u r e ===================================
            # 外部属性的node pressure 以及 gas_flow_in 都是最后面的节点
            # for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            ind = g_connection[connect_to].index(self.index)
            num_node_gas = self.gas_node_num - len(self.connection_area)
            for time in range(T):
                thisvar_inside = self.node_pressure[self.connection_index[i], time]
                thisvar_outside = g_angles[connect_to][ind * 6 + 2 + 1][time]
                thatvar_inside = self.node_pressure[num_node_gas + i, time]
                thatvar_outside = g_angles[connect_to][ind * 6 + 2][time]
                # ***********************************************
                duals.append(thisvar_inside - thisvar_outside)
                duals.append(-1 * thisvar_inside + thisvar_outside)
                duals.append(thatvar_inside - thatvar_outside)
                duals.append(-1 * thatvar_inside + thatvar_outside)
                # this is for one line, for all time t: [(0, 1, 2, 3)t=0,()t=1, ... ] + [another line]
            # ======================== g a s - f l o w - i n /  o u t ===============================
            # for each line gas flow in/ gas flow out
            # for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            ind = g_connection[connect_to].index(self.index)
            num_line_gas = self.gas_line_num - len(self.connection_area)
            for time in range(T):
                thisvar_inside = self.gas_flow_in[num_line_gas + i, time]
                thisvar_outside = g_angles[connect_to][ind * 6 + 4][time]
                thatvar_inside = self.gas_flow_out[num_line_gas + i, time]
                thatvar_outside = g_angles[connect_to][ind * 6 + 4 + 1][time]
                # ***********************************************
                if self.index < connect_to:
                    duals.append(thisvar_inside - thisvar_outside)
                    duals.append(-1 * thisvar_inside + thisvar_outside)
                    duals.append(1 * thatvar_inside - thatvar_outside)
                    duals.append(-1 * thatvar_inside + thatvar_outside)
                else:
                    duals.append(1 * thatvar_inside - thatvar_outside)
                    duals.append(-1 * thatvar_inside + thatvar_outside)
                    duals.append(thisvar_inside - thisvar_outside)
                    duals.append(-1 * thisvar_inside + thisvar_outside)

                # this is for one line, for all time t: [(0, 1, 2, 3)t=0,()t=1, ... ] + [another line]
        dual_addition = sum([a * b * PUNISH for a, b in zip(duals, lams)])

        # construct norm object
        norm_addition = 0
        for i in range(len(self.connection_area)):
            # ========================= a n g l e ====================================
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            ind = g_connection[connect_to].index(self.index)
            for time in range(T):
                thisvar_inside = self.angles_inside[self.connection_index[i], time]  # 与 connect_to 区域相连的线路的内部的相角
                thatvar_inside = self.angles_outside[i, time]
                thisvar_outside = g_angles[connect_to][ind * 6 + 1][time]
                thatvar_outside = g_angles[connect_to][ind * 6][time]
                norm_addition = norm_addition + \
                                (thisvar_inside + self.free_var[time] - self.old_value[6 * i][time]) * \
                                (thisvar_inside + self.free_var[time] - self.old_value[6 * i][time]) + \
                                (thatvar_inside + self.free_var[time] - self.old_value[6 * i + 1][time]) * \
                                (thatvar_inside + self.free_var[time] - self.old_value[6 * i + 1][time])
                if self.index != 0:
                    dual_addition = dual_addition + \
                                    10 * ((thisvar_inside + self.free_var[time]) - thisvar_outside) * \
                                    ((thisvar_inside + self.free_var[time]) - thisvar_outside) + \
                                    10 * ((thatvar_inside + self.free_var[time]) - thatvar_outside) * \
                                    ((thatvar_inside + self.free_var[time]) - thatvar_outside)
                # select the best reference point
            # ========================= p r e s s u r e ===================================
            # for i in range(len(self.connection_area)):
            num_node_gas = self.gas_node_num - len(self.connection_area)
            for time in range(T):
                thisvar_inside = self.node_pressure[self.connection_index[i], time]
                thatvar_inside = self.node_pressure[num_node_gas + i, time]
                norm_addition = norm_addition + \
                                (thisvar_inside - self.old_value[6 * i + 2][time]) * \
                                (thisvar_inside - self.old_value[6 * i + 2][time]) + \
                                (thatvar_inside - self.old_value[6 * i + 3][time]) * \
                                (thatvar_inside - self.old_value[6 * i + 3][time])
                # select the best reference point
            # ======================== g a s - f l o w - i n /  o u t ===============================
            # for i in range(len(self.connection_area)):
            num_line_gas = self.gas_line_num - len(self.connection_area)
            for time in range(T):
                thisvar_inside = self.gas_flow_in[num_line_gas + i, time]
                thatvar_inside = self.gas_flow_out[num_line_gas + i, time]
                norm_addition = norm_addition + \
                                (thisvar_inside - self.old_value[6 * i + 4][time]) * \
                                (thisvar_inside - self.old_value[6 * i + 4][time]) + \
                                (thatvar_inside - self.old_value[6 * i + 5][time]) * \
                                (thatvar_inside - self.old_value[6 * i + 5][time])
                # select the best reference point
        # add difference
        self.object_addition = dual_addition + \
                               tao / 2 * norm_addition
        self.object = self.object_basic + self.object_addition

    def optimize_model(self):  # calculate the response
        k = 1.4
        j = 0
        print(str(self.index) + 'start')
        while j < 20:
            j = j + 1
            k = k * 1.4
            if k > 50:
                k = 150
            self.model.setObjective(self.object + gurobi.quicksum(self.pccp) * k)
            self.model.Params.OutputFlag = 0
            self.model.optimize()
            #print(j)
            for line in range(self.gas_line_num):
                for time in range(self.T):
                    self.gas_flow_in_old[line][time] = self.gas_flow_in[line, time].getAttr('X')

            for line in range(self.gas_line_num):
                for time in range(self.T):
                    self.gas_flow_out_old[line][time] = self.gas_flow_out[line, time].getAttr('X')

            for node in range(self.gas_node_num):
                for time in range(self.T):
                    self.node_pressure_old[node][time] = self.node_pressure[node, time].getAttr('X')

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
                                            weymouth * self.node_pressure[end_point, time] *
                                            self.node_pressure[end_point, time] +
                                            2 * weymouth * self.node_pressure[end_point, time] *
                                            self.node_pressure_old[end_point][time]
                                    ),
                                rhs=self.pccp[line, time],
                                sense=gurobi.GRB.LESS_EQUAL
                            ))

        print(str(self.index) + 'start')
        injection[self.index] = copy.deepcopy([[self.power_injections_outside[i, time].getAttr('X')
                                                for time in range(T)]
                                               for i in range(len(self.connection_area))])
        exchange_angle = []

        print(str(self.index) + 'end')

        for i in range(len(self.connection_area)):
            # ========================= a n g l e ====================================
            exchange_angle_this = []
            exchange_angle_that = []
            for time in range(T):
                thisvar_inside = self.angles_inside[self.connection_index[i], time]
                thatvar_inside = self.angles_outside[i, time]
                if self.index == 0:
                    freevar = 0
                else:
                    freevar = self.free_var[time].getAttr('X')
                exchange_angle_this.append(thisvar_inside.getAttr('X') + freevar)
                exchange_angle_that.append(thatvar_inside.getAttr('X') + freevar)
            exchange_angle.append(exchange_angle_this)
            exchange_angle.append(exchange_angle_that)
            # ========================= p r e s s u r e ===================================
            exchange_press_this = []
            exchange_press_that = []
            num_node_gas = self.gas_node_num - len(self.connection_area)
            for time in range(T):
                thisvar_inside = self.node_pressure[self.connection_index[i], time]
                thatvar_inside = self.node_pressure[num_node_gas + i, time]
                exchange_press_this.append(thisvar_inside.getAttr('X'))
                exchange_press_that.append(thatvar_inside.getAttr('X'))
            exchange_angle.append(exchange_press_this)
            exchange_angle.append(exchange_press_that)
            # ======================== g a s - f l o w - i n /  o u t ===============================
            exchange_flow_in_this = []
            exchange_flow_out_that = []
            num_line_gas = self.gas_line_num - len(self.connection_area)
            for time in range(T):
                thisvar_inside = self.gas_flow_in[num_line_gas + i, time]
                thatvar_inside = self.gas_flow_out[num_line_gas + i, time]
                exchange_flow_in_this.append(thisvar_inside.getAttr('X'))
                exchange_flow_out_that.append(thatvar_inside.getAttr('X'))
            exchange_angle.append(exchange_flow_in_this)
            exchange_angle.append(exchange_flow_out_that)
        return exchange_angle

    def set_old_value(self, old):  # 01 02 03 04 [[ x * T],[ x * T],...]
        for i in range(len(self.old_value)):
            for time in range(T):
                self.old_value[i][time] = old[i][time]
class playerNp1:
    def __init__(self):
        self.old_value = [0] * (g_link * 2 * 3 * 2 * T)

    def optimize(self, tao):  # [[01 02] [10 12] [20 21]]
        global g_angles
        model = gurobi.Model()
        gx = []
        for i in range(len(g_connection)):  # per area
            for connect_to in g_connection[i]:  # per area - area
                if i < connect_to:  # just 0-1 no 1-0
                    for time in range(T):
                        # inside - outside
                        # for region i
                        #    o ----------- o
                        #   this          that
                        thisvar_inside = g_angles[i][6 * g_connection[i].index(connect_to)][time]
                        thisvar_outside = g_angles[connect_to][6 * g_connection[connect_to].index(i) + 1][time]
                        gx.append(thisvar_inside - thisvar_outside)
                        gx.append(-1 * thisvar_inside + thisvar_outside)
                    for time in range(T):
                        thatvar_inside = g_angles[i][6 * g_connection[i].index(connect_to) + 1][time]
                        thatvar_outside = g_angles[connect_to][6 * g_connection[connect_to].index(i)][time]
                        gx.append(thatvar_inside - thatvar_outside)
                        gx.append(-1 * thatvar_inside + thatvar_outside)

                    for time in range(T):
                        # inside - outside
                        # for region i
                        #    o ----------- o
                        #   this          that
                        thisvar_inside = g_angles[i][6 * g_connection[i].index(connect_to) + 2][time]
                        thisvar_outside = g_angles[connect_to][6 * g_connection[connect_to].index(i) + 3][time]
                        gx.append(thisvar_inside - thisvar_outside)
                        gx.append(-1 * thisvar_inside + thisvar_outside)
                    for time in range(T):
                        thatvar_inside = g_angles[i][6 * g_connection[i].index(connect_to) + 3][time]
                        thatvar_outside = g_angles[connect_to][6 * g_connection[connect_to].index(i) + 2][time]
                        gx.append(thatvar_inside - thatvar_outside)
                        gx.append(-1 * thatvar_inside + thatvar_outside)

                    for time in range(T):
                        # inside - outside
                        # for region i
                        #    o ----------- o
                        #   this          that
                        thisvar_inside = g_angles[i][6 * g_connection[i].index(connect_to) + 4][time]
                        thisvar_outside = g_angles[connect_to][6 * g_connection[connect_to].index(i) + 5][time]
                        gx.append(thisvar_inside + thisvar_outside)
                        gx.append(-1 * thisvar_inside - thisvar_outside)
                    for time in range(T):
                        thatvar_inside = g_angles[i][6 * g_connection[i].index(connect_to) + 5][time]
                        thatvar_outside = g_angles[connect_to][6 * g_connection[connect_to].index(i) + 4][time]
                        gx.append(thatvar_inside + thatvar_outside)
                        gx.append(-1 * thatvar_inside - thatvar_outside)
        duals = model.addVars(len(gx))
        # duals * gx
        dual_express = gurobi.quicksum(
            duals[i] * gx[i] for i in range(len(gx)))

        # norm
        norm_express = gurobi.quicksum(
            (duals[i] - self.old_value[i]) * (duals[i] - self.old_value[i])
            for i in range(len(gx)))

        # objective
        objective = -1 * dual_express + tao / 2 * norm_express

        model.setObjective(objective)
        model.Params.OutputFlag = 0
        model.optimize()
        dual_value = []
        for i in range(len(gx)):
            dual_value.append(duals[i].getAttr('X'))
        # print(dual_value)
        return dual_value

    def set_old_value(self, old_value):
        self.old_value = old_value.copy()



























def getPlayer(player_info):
    instance = Region(
        player_info['index'],
        player_info['X_raw'],
        player_info['node_info'],
        player_info['connection_info'],
        player_info['gas_line_info'],
        player_info['gas_node_info']
    )
    return instance
def factory():
    # ================player 0==============================
    p_index = 0
    X_raw_0 = [
        [0, 0.1, 0, 0, 0],
        [0.1, 0, 0.1, 0, 0.1],
        [0, 0.1, 0, 0.1, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0.1, 0, 0, 0]
    ]
    node_info_0 = [  # 8 - 12 - 10      3 - 6
        {
            'min_load': 5,
            'max_load': 7,
            'min_power': 1,
            'max_power': 1,
            'load_coeff': 5,
            'load_ref': 6,
            'power_coeff_a': 10.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 1,
            'max_load': 2,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 4,
            'load_ref': 1.5,
            'power_coeff_a': 10.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 2,
            'max_load': 3,
            'min_power': 2,
            'max_power': 5,
            'load_coeff': 3,
            'load_ref': 2.5,
            'power_coeff_a': 10.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 2,
            'max_power': 4,
            'load_coeff': 1,
            'load_ref': 0,
            'power_coeff_a': 10.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 1,
            'max_load': 1,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 1,
            'load_ref': 1,
            'power_coeff_a': 10.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },

    ]
    connection_info_0 = {
        'connection_index': connection_index[p_index],
        'connection_x': [0.1] * len(g_connection[p_index]),
        'connection_area': g_connection[p_index],
        'connection_exchange_max': [100] * len(g_connection[p_index])
    }
    gas_node_info_0 = {
        'gas_node_num': 3 + 1,
        'node_pressure_min': [0] * (3 + 1),
        'node_pressure_max': [2] * (3 + 1),
        'gas_well_num': 0 + 1,
        'well_index': [3],
        'well_output_min': [0],
        'well_output_max': [1],
        'gas_load_num': 2,
        'load_index': [0, 2],
        'gas_load_min': [[0.1] * T, [0.1] * T],
        'gas_load_max': [[0.2] * T, [0.2] * T],
        'gen_gas_num': 1,
        'gen_gas_index': [2],
        'gen_gas_index_power': [3],
        'gen_gas_min': [1],  # this is power
        'gen_gas_max': [5],
        'gen_gas_efficiency': [10],

    }
    gas_line_info_0 = {
        'weymouth': [1] * (2 + 1),
        'gas_line_num': 2 + 1,
        'gas_line_start_point': [1, 1, 3],  # gas flow out
        'gas_line_end_point': [0, 2, 1],  # gas flow in
        'gas_line_pack_coefficient': [1] * (2 + 1),
        'gas_line_pack_initial': 2,
        'gas_flow_in_max': [5] * (2 + 1),  # unused
        'gas_flow_out_max': [5] * (2 + 1),  # unused
        'gas_line_active': [],
        'compressor_num': 0,
        'compressor_start_point': [],
        'compressor_end_point': [],
        'compressor_coefficient': [],
        'compressor_max_flow': [],
        'compressor_energy_consumption': [],
    }
    player0_info = {
        'index': 0,
        'X_raw': X_raw_0,
        'node_info': node_info_0,
        'connection_info': connection_info_0,
        'gas_node_info': gas_node_info_0,
        'gas_line_info': gas_line_info_0
    }
    p_index = p_index + 1

    # =================player 1==========================
    X_raw_1 = [
        [0, 0.1, 0, 0, 0],
        [0.1, 0, 0.1, 0, 0.1],
        [0, 0.1, 0, 0.1, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0.1, 0, 0, 0]
    ]
    node_info_1 = [  # 8 - 10 - 9      4 - 19
        {
            'min_load': 1,
            'max_load': 1,
            'min_power': 1,
            'max_power': 1,
            'load_coeff': 1,
            'load_ref': 1,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 5,
            'load_coeff': 1,
            'load_ref': 1,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 1,
            'max_load': 2,
            'min_power': 0,
            'max_power': 3,
            'load_coeff': 1,
            'load_ref': 1.5,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 4,
            'max_load': 5,
            'min_power': 1,
            'max_power': 6,
            'load_coeff': 4,
            'load_ref': 4.5,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 2,
            'max_load': 2,
            'min_power': 2,
            'max_power': 4,
            'load_coeff': 2,
            'load_ref': 2,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
    ]
    connection_info_1 = {
        'connection_index': connection_index[p_index],
        'connection_x': [0.1] * len(g_connection[p_index]),
        'connection_area': g_connection[p_index],
        'connection_exchange_max': [100] * len(g_connection[p_index])
    }
    gas_node_info_1 = {
        'gas_node_num': 3 + 2,
        'node_pressure_min': [0] * (3 + 2),
        'node_pressure_max': [2] * (3 + 2),
        'gas_well_num': 1 ,
        'well_index': [0],
        'well_output_min': [0],
        'well_output_max': [1.5],
        'gas_load_num': 1 + 2,
        'load_index': [2, 3, 4],
        'gas_load_min': [[0.1] * T , [0] * T,  [0] * T],
        'gas_load_max': [[0.2] * T , [1] * T,  [1] * T],
        'gen_gas_num': 1,
        'gen_gas_index': [2],
        'gen_gas_index_power': [3],
        'gen_gas_min': [1],  # this is power
        'gen_gas_max': [5],
        'gen_gas_efficiency': [10],
    }
    gas_line_info_1 = {
        'weymouth': [1] * (2 + 2),
        'gas_line_num': 2 + 2,
        'gas_line_start_point': [0, 1, 1, 1],  # gas flow out
        'gas_line_end_point': [1, 2, 3, 4],  # gas flow in
        'gas_line_pack_coefficient': [1] * (2 + 2),
        'gas_line_pack_initial': 2,
        'gas_flow_in_max': [5] * (2 + 2),  # unused
        'gas_flow_out_max': [5] * (2 + 2),  # unused
        'gas_line_active': [],
        'compressor_num': 0,
        'compressor_start_point': [],
        'compressor_end_point': [],
        'compressor_coefficient': [],
        'compressor_max_flow': [],
        'compressor_energy_consumption': [],
    }
    player1_info = {
        'index': 1,
        'X_raw': X_raw_1,
        'node_info': node_info_1,
        'connection_info': connection_info_1,
        'gas_node_info': gas_node_info_1,
        'gas_line_info': gas_line_info_1
    }
    p_index = p_index + 1

    # =================player 2==========================
    X_raw_2 = [
        [0, 0.1, 0, 0, 0],
        [0.1, 0, 0.1, 0, 0.1],
        [0, 0.1, 0, 0.1, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0.1, 0, 0, 0]
    ]
    node_info_2 = [  # 9 - 13 - 11      5 - 8
        {
            'min_load': 5,
            'max_load': 7,
            'min_power': 1,
            'max_power': 1,
            'load_coeff': 5,
            'load_ref': 6,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 1,
            'max_load': 2,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 4,
            'load_ref': 1.5,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 2,
            'max_load': 3,
            'min_power': 2,
            'max_power': 3,
            'load_coeff': 3,
            'load_ref': 2.5,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 2,
            'max_power': 4,
            'load_coeff': 1,
            'load_ref': 0,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },
        {
            'min_load': 1,
            'max_load': 1,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 1,
            'load_ref': 1,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
            'gen_ramp_up': 5,
            'gen_ramp_down': 5
        },

    ]
    connection_info_2 = {
        'connection_index': connection_index[p_index],
        'connection_x': [0.1] * len(g_connection[p_index]),
        'connection_area': g_connection[p_index],
        'connection_exchange_max': [100] * len(g_connection[p_index])
    }
    gas_node_info_2 = {
        'gas_node_num': 3 + 1,
        'node_pressure_min': [0] * (3 + 1),
        'node_pressure_max': [2] * (3 + 1),
        'gas_well_num': 0 + 1,
        'well_index': [3],
        'well_output_min': [0] * (0 + 1),
        'well_output_max': [1] * (0 + 1),
        'gas_load_num': 2,
        'load_index': [0, 2],
        'gas_load_min': [[0.1] * T, [0.1] * T],
        'gas_load_max': [[0.2] * T, [0.2] * T],
        'gen_gas_num': 1,
        'gen_gas_index': [2],
        'gen_gas_index_power': [3],
        'gen_gas_min': [1],  # this is power
        'gen_gas_max': [5],
        'gen_gas_efficiency': [10],
    }
    gas_line_info_2 = {
        'weymouth': [1] * (2 + 1),
        'gas_line_num': 2 + 1,
        'gas_line_start_point': [1, 1, 3],  # gas flow out
        'gas_line_end_point': [0, 2, 1],  # gas flow in
        'gas_line_pack_coefficient': [1] * (2 + 1),
        'gas_line_pack_initial': 2,
        'gas_flow_in_max': [5] * (2 + 1),  # unused
        'gas_flow_out_max': [5] * (2 + 1),  # unused
        'gas_line_active': [],
        'compressor_num': 0,
        'compressor_start_point': [],
        'compressor_end_point': [],
        'compressor_coefficient': [],
        'compressor_max_flow': [],
        'compressor_energy_consumption': [],
    }
    player2_info = {
        'index': 2,
        'X_raw': X_raw_2,
        'node_info': node_info_2,
        'connection_info': connection_info_2,
        'gas_node_info': gas_node_info_2,
        'gas_line_info': gas_line_info_2
    }
    p_index = p_index + 1

    player1 = getPlayer(player0_info)
    player2 = getPlayer(player1_info)
    player3 = getPlayer(player2_info)
    playerN1 = playerNp1()
    return [player1, player2, player3, playerN1]
def sub_norm(a, b):
    # return np.linalg.norm(np.array(a) - np.array(b))
    norm = 0.0
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(T):
                norm = norm + (a[i][j][k] - b[i][j][k]) * (a[i][j][k] - b[i][j][k])
    return math.sqrt(norm)























def calculate_NE():
    global g_lam
    count_best_response = 0
    g_angles_old = 0
    while count_best_response < 10:
        # TODO: maybe 30 is a little small
        g_angles_old = copy.deepcopy(g_angles)
        for i, player in enumerate(g_players):
            # get the data for the player i
            player.update_model(g_tao)  # 填充x_i 以及lam_i
            player_i_result = player.optimize_model()
            g_angles[i] = player_i_result.copy()
        # update the lam_dual variable
        g_lam = g_playerN1.optimize(g_tao).copy()
        # update the response
        dfd = sub_norm(g_angles_old, g_angles)
        print('=========>' + str(dfd) + '<==========')
        if dfd < INNER_FIX:
            print(count_best_response)
            break
        count_best_response = count_best_response + 1
def set_oldValue():
    for i, player in enumerate(g_players):
        player.set_old_value(g_angles[i].copy())
    g_playerN1.set_old_value(g_lam.copy())
def start():
    global g_angles
    result_plt = []
    result_plt1 = []
    result_plt2 = []
    result_plt3 = []
    # initial
    for player in g_players:
        player.build_model()
    # start the outer loop
    outer_loop_count = 0
    while outer_loop_count < OUTER_LOOP:
        print(outer_loop_count)
        # give xn, lam_n, calculate the equilibrium
        calculate_NE()
        # 现在我们得到了一个新的NE，我们应该把这个NE设为参照值
        set_oldValue()
        outer_loop_count = outer_loop_count + 1
        # result_plt.append(g_angles[0][0])
        # result_plt1.append(g_angles[1][0])
        # result_plt2.append(g_angles[0][0] - g_angles[1][0])
        result_plt.append(injection[0][0][0])
        result_plt1.append(injection[1][0][0] + injection[0][0][0])
        result_plt2.append((g_angles[1][2][0] - g_angles[0][3][0]) * 1)
        result_plt3.append(1 * (g_angles[1][3][0] - g_angles[0][2][0]))
        # result_plt2.append(g_lam[0] + g_lam[1])
        # set all value in g_ex to zero
        if outer_loop_count != OUTER_LOOP:
            g_angles = [
                [[0] * T
                 for i in range(2 * len(node_t[p]))]
                for p in range(player_num)
            ]
    plt.plot(result_plt, label='0->1')
    plt.plot(result_plt1, '-r', label='1->0')
    plt.plot(result_plt2, '-g', label='diff')
    plt.plot(result_plt3, '*b', label='diff')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('x-node-micro-grid.svg')











if __name__ == '__main__':
    all_players = factory()
    g_players = all_players[:player_num]
    g_playerN1 = all_players[player_num]
    [p1, p2, p3] = g_players
    pn = g_playerN1
    for player in g_players:
        player.build_model()