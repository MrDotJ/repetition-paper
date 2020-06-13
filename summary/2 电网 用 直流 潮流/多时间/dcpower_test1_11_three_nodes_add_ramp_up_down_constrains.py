# add more constrains
import gurobipy as gurobi
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
# test addMVar
price = [[0, 6, 6],
         [6, 0, 6],
         [6, 6, 0]]
# [[6.68847635620013, -1.5043844856390587],
#  [-7.530935327521067, -2.349528058932746],
#  [0.6367732484834399, 1.577324838479214]]

T = 1
g_tao = 100
# g_lam = [0, 1,   2, 3,   4, 5,   6, 7,   8, 9,   10, 11]
# #       (0)     (1)     (2)     (3)     (4)     (5)
g_lam = [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T
#       (0)          (1)          (2)          (3)          (4)          (5)
#                       [0]
#                        o
#                  (2)  / \  (0)
#                      /   \
#                (3)  /     \  (1)
#                 [2]o-------o[1]
#                     (5)  (4)
#               0 - 1         2 - 3
# g_lam_index = [[0, 1, 2, 3,   4, 5, 6, 7],
# #               1 - 0         4   - 5
#                [3, 2, 1, 0,   8, 9, 10, 11],
# #               3 - 2         5 - 4
#                [7, 6, 5, 4,   11, 10, 9, 8]]
g_lam_index = [list(itertools.chain( *[ [2*t, 2*t+1, 2*T+2*t, 2*T+2*t+1] for t in range(T) ] )) +
# for time t  all time t for one line   [(0)t=0,          (1)t=0              ] + [(0)t=1, (1)t=1] + ... [(0)t=T-1, (1)t=T-1]
               list(itertools.chain(*[[4*T+2*t, 4*T+2*t+1, 6*T+2*t, 6*T+2*t+1] for t in range(T)])),
#    all time t for another line      [(2)t=0,          (3)t=0              ] + ...
               list(itertools.chain(*[[2*T+2*t+1, 2*T+2*t, 2*t+1, 2*t] for t in range(T)])) +
#                                    [(1)                     (0)    ]
               list(itertools.chain(*[[8*T+2*t, 8*T+2*t+1, 10*T+2*t, 10*T+2*t+1] for t in range(T)])),
#                                     [(4)                      (5)
               list(itertools.chain(*[[6*T+2*t+1, 6*T+2*t, 4*T+2*t+1, 4*T+2*t] for t in range(T)])) +
#                                     [(3)               (2)
               list(itertools.chain(*[[10*T+2*t+1, 10*T+2*t, 8*T+2*t+1, 8*T+2*t] for t in range(T)]))]
#                                     [(5)                      (4)
# g_angles = [[0, 1, 2, 3],
#             [1, 0, 2, 3],
#             [3, 2, 2, 1]]
# g_angles = [[[0] * T, [1] * T, [2] * T, [3] * T],
#             [[1] * T, [0] * T, [2] * T, [3] * T],
#             [[3] * T, [2] * T, [2] * T, [1] * T]]
g_angles = [[[0] * T, [1] * T, [2] * T, [3] * T],
            [[1] * T, [0] * T, [2] * T, [3] * T],
            [[3] * T, [2] * T, [2] * T, [1] * T]]
g_connection = [[1, 2],
                [0, 2],
                [0, 1]]
# injection = [[0, 0],
#              [0, 0],
#              [0, 0]]
injection = [[],[],[]]
g_link = 3
player_num = 3

class Region:
    def __init__(self, index, X_raw, node_info, connection_info):
        self.index = index
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
        # old value
        self.old_value = [[0] * T for i in range(len(self.connection_area) * 2)]

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

    def build_model(self):
        global price
        index = self.index
        # build B_1 matrix
        self.build_B()
        objects = []
        constrains = []
        # construct model
        # add variables
        # 每一个节点既有负荷又有generator
        # 每个节点添加负荷变量，发电变量，内部节点的注入功率变量，以及联系
        # for node in range(self.node_count):
        self.load_vars = self.model.addVars(
                self.node_count, T,
                lb=[[self.node_info[node]['min_load']]*T for node in range(self.node_count)],
                ub=[[self.node_info[node]['max_load']]*T for node in range(self.node_count)],
                obj=0,
                name='Region_' + str(self.index) + '_loads')
        self.gene_vars = self.model.addVars(
                self.node_count, T,
                lb=[[self.node_info[node]['min_power']]*T for node in range(self.node_count)],
                ub=[[self.node_info[node]['max_power']]*T for node in range(self.node_count)],
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
                    name='power_injection' + str(node) + '_time_' + str(time) )

        # 添加内部/外部相角变量以及外部注入功率变量，**相角的范围是多少？**
        self.angles_inside = self.model.addVars(
            self.node_count, T,  lb=-3.14, ub=3.14, name='angle_inside')
        self.angles_outside = self.model.addVars(
            len(self.connection_area), T, lb=-3.14, ub=3.14, name='angle_outside')
        # 外部功率注入
        self.power_injections_outside = \
            self.model.addVars(len(self.connection_area), T,
                               lb=[[-1 * m_max]*T for m_max in self.connection_exchange_max],
                               ub=[[1 * m_max]*T for m_max in self.connection_exchange_max],
                               name='Region_' + str(self.index) + '_power_injection_outside')
        # 外部功率注入直流潮流
        for conn in range(len(self.connection_area)):
            for time in range(T):
                self.model.addConstr(self.power_injections_outside[conn, time] ==
                (self.angles_outside[conn, time] / self.connection_x[conn] -
                 self.angles_inside[self.connection_index[conn], time] / self.connection_x[conn]))  # the B is B^ not B^^
        # x should satisfy the max-power-flow

        # 内部线路直流潮流的计算公式 except the reference node
        ang_ins = lambda _time : [self.angles_inside[i, _time] for i in range(self.node_count)]
        external_injection_pos = 0
        for row in range(self.node_count - 1):
            for time in range(T):
                if row in self.connection_index:  # if the index has connection with outside area
                    self.model.addConstr(np.array(self.X_B1[row]).reshape((1, -1)).dot(
                                         np.array(ang_ins(time)).reshape((-1, 1)))[0][0] ==  #TODO # here is wrong
                                         self.power_injections[row, time] +
                                         self.power_injections_outside[external_injection_pos, time])
                    if time == T-1:
                        external_injection_pos = external_injection_pos + 1
                else:  # else this index does not have external injection
                    self.model.addConstr(
                        np.array(self.X_B1[row]).reshape((1, -1)).dot(
                        np.array(ang_ins(time)).reshape((-1, 1)))[0][0] ==
                        self.power_injections[row, time])
        # 添加发电机的爬坡约束
        node_count_local_var = self.node_count
        get_gen = lambda : np.array([[self.gene_vars[_index_, _time_] for _time_ in range(T)]
                                                for _index_ in range(node_count_local_var)])
        gen_var = get_gen()
        gen_ramp_up_expr = gen_var[:, 1 : T] - gen_var[:, 0 : T-1]
        gen_ramp_down_expr = gen_var[:, 0 : T-1] - gen_var[:, 1 : T]
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
        # 注意正负
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

        # add constrain - power balance
        power_balance = self.model.addConstr(
            gurobi.quicksum(self.load_vars) ==
            gurobi.quicksum(self.gene_vars) + gurobi.quicksum(self.power_injections_outside),
            name='Region_' + str(index) + '_power_balance'
        )
        constrains.append(power_balance)
        # add object
        self.object_basic = sum(objects)

    def update_model(self, tao):
        global g_angles
        global g_connection
        global g_lam #TODO: SECOND WRONG
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

        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            for time in range(T):
                #    o--------o
                #   this     that
                # ************************************************
                # this var
                thisvar_inside = self.angles_inside[self.connection_index[i], time]  # 与 connect_to 区域相连的线路的内部的相角
                ind = g_connection[connect_to].index(self.index)
                # area 1 for point 0 at time t:
                thisvar_outside = g_angles[connect_to][ind * 2 + 1][time]  # TODO:change the g_angle
                # 以区域二为例， g_angles[1] 选择区域2 的全部相角
                # g_connection[1].index(0) 获得区域2中对于区域1的索引，是0， 那么他与区域1联系的两个相角在0，1子维中
                # this 是第二个所以是 1. 然后往下索引时间
                # that var
                thatvar_inside = self.angles_outside[i, time]
                thatvar_outside = g_angles[connect_to][ind * 2][time]

                # ***********************************************
                duals.append(thisvar_inside + self.free_var[time] - thisvar_outside)
                duals.append(-1 * thisvar_inside - self.free_var[time] + thisvar_outside)
                duals.append(thatvar_inside + self.free_var[time] - thatvar_outside)
                duals.append(-1 * thatvar_inside - self.free_var[time] + thatvar_outside)
                # this is for one line, for all time t: [(0, 1, 2, 3)t=0,()t=1, ... ] + [another line]
        dual_addition = sum([a * b * 100 for a, b in zip(duals, lams)])

        # construct norm object
        norm_addition = 0
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            for time in range(T):
                thisvar_inside = self.angles_inside[self.connection_index[i], time]  # 与 connect_to 区域相连的线路的内部的相角
                thatvar_inside = self.angles_outside[i, time]
                ind = g_connection[connect_to].index(self.index)
                thisvar_outside = g_angles[connect_to][ind * 2 + 1][time]
                thatvar_outside = g_angles[connect_to][ind * 2][time]
                norm_addition = norm_addition + \
                                (thisvar_inside + self.free_var[time] - self.old_value[2 * i][time]) * \
                                (thisvar_inside + self.free_var[time] - self.old_value[2 * i][time]) + \
                                (thatvar_inside + self.free_var[time] - self.old_value[2 * i + 1][time]) * \
                                (thatvar_inside + self.free_var[time] - self.old_value[2 * i + 1][time])
                if self.index != 0:
                    dual_addition = dual_addition + \
                                    10 * ((thisvar_inside + self.free_var[time]) - thisvar_outside) * \
                                    ((thisvar_inside + self.free_var[time]) - thisvar_outside) + \
                                    10 * ((thatvar_inside + self.free_var[time]) - thatvar_outside) * \
                                    ((thatvar_inside + self.free_var[time]) - thatvar_outside)
                # select the best reference point
        # add difference
        self.object_addition = dual_addition + \
                               tao / 2 * norm_addition
        self.object = self.object_basic + self.object_addition
        self.model.setObjective(self.object)

    def optimize_model(self):  # calculate the response
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        injection[self.index] = copy.deepcopy([[self.power_injections_outside[i, time].getAttr('X')
                                                for time in range(T)]
                                                for i in range(len(self.connection_area))])
        exchange_angle = []


        for i in range(len(self.connection_area)):
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
        return exchange_angle

    def set_old_value(self, old):  # 01 02 03 04 [[ x * T],[ x * T],...]
        for i in range(len(self.old_value)):
            for time in range(T):
                self.old_value[i][time] = old[i][time]











class playerNp1:
    def __init__(self):
        self.old_value = [0] * (g_link * 2 * 2*T)

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
                        thisvar_inside = g_angles[i][2 * g_connection[i].index(connect_to)][time]
                        thisvar_outside = g_angles[connect_to][2 * g_connection[connect_to].index(i) + 1][time]
                        gx.append(thisvar_inside - thisvar_outside)
                        gx.append(-1 * thisvar_inside + thisvar_outside)
                    for time in range(T):
                        thatvar_inside = g_angles[i][2 * g_connection[i].index(connect_to) + 1][time]
                        thatvar_outside = g_angles[connect_to][2 * g_connection[connect_to].index(i)][time]
                        gx.append(thatvar_inside - thatvar_outside)
                        gx.append(-1 * thatvar_inside + thatvar_outside)
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
        player_info['connection_info'])
    return instance
def factory():
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
        'connection_index': [1, 2],
        'connection_x': [0.1, 0.1],
        'connection_area': [1, 2],
        'connection_exchange_max': [100, 100]
    }
    player0_info = {
        'index': 0,
        'X_raw': X_raw_0,
        'node_info': node_info_0,
        'connection_info': connection_info_0
    }

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
        'connection_index': [1, 2],
        'connection_x': [0.1, 0.1],
        'connection_area': [0, 2],
        'connection_exchange_max': [100, 100]
    }
    player1_info = {
        'index': 1,
        'X_raw': X_raw_1,
        'node_info': node_info_1,
        'connection_info': connection_info_1,
    }

    X_raw_2 = [
        [0, 0.1, 0, 0, 0],
        [0.1, 0, 0.1, 0, 0.1],
        [0, 0.1, 0, 0.1, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0.1, 0, 0, 0]
    ]
    #                             ◜¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯◝
    #                             |  0      1      2        |
    #                             |  o---------o---------o  |
    #                             |            |         |  |
    #                             |            |         |  |
    #                             |            |         |  |
    #                             |            o         o  |
    #                             |            4         3  |
    #                             |          (ref)          |
    #                             |                         |
    #                             |                    area0|
    #                             ◟________________________◞
    #
    #
    #
    #           ◜¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯◝            ◜¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯◝
    #          |   0         1         2  |            |  0         1      2     |
    #          |   o---------o---------o  |            |  o---------o---------o  |
    #          |             |         |  |            |            |         |  |
    #          |             |         |  |            |            |         |  |
    #          |             |         |  |            |            |         |  |
    #          |             o         o  |            |            o         o  |
    #          |             4         3  |            |            4         3  |
    #          |                          |            |                         |
    #          |                     area2|            |                    area1|
    #          ◟_________________________◞            ◟________________________◞
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
        'connection_index': [1, 2],
        'connection_x': [0.1, 0.1],
        'connection_area': [0, 1],
        'connection_exchange_max': [100, 100]
    }
    player2_info = {
        'index': 2,
        'X_raw': X_raw_2,
        'node_info': node_info_2,
        'connection_info': connection_info_2
    }
    player1 = getPlayer(player0_info)
    player2 = getPlayer(player1_info)
    player3 = getPlayer(player2_info)
    playerN1 = playerNp1()
    return [player1, player2, player3, playerN1]







def sub_norm(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


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
        if sub_norm(g_angles_old, g_angles) < 0.000001:
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
    while outer_loop_count < 1000:
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
        result_plt1.append(injection[1][1][0] + injection[2][1][0])
        result_plt2.append(injection[0][0][0] + injection[1][0][0])
        result_plt3.append(injection[2][0][0] + injection[0][1][0])
        # result_plt2.append(g_lam[0] + g_lam[1])
        # set all value in g_ex to zero
        if outer_loop_count != 1000:
            g_angles = (np.array(g_angles)*0).tolist()
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
    start()