import gurobipy as gurobi
import numpy as np
from DCpower.config2 import *
import copy


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
                                      self.angles_inside[self.connection_index[conn], time] /
                                      self.connection_x[conn]))
        # x should satisfy the max-power-flow

        # 内部线路直流潮流的计算公式 except the reference node
        ang_ins = lambda _time: [self.angles_inside[i, _time] for i in range(self.node_count)]
        external_injection_pos = 0
        for row in range(self.node_count - 1):
            for time in range(T):
                if row in self.connection_index:  # if the index has connection with outside area
                    self.model.addConstr(np.array(self.X_B1[row]).reshape((1, -1)).dot(
                        np.array(ang_ins(time)).reshape((-1, 1)))[0][0] ==  # TODO # here is wrong
                                         self.power_injections[row, time] +
                                         self.power_injections_outside[external_injection_pos, time])
                    if time == T - 1:
                        external_injection_pos = external_injection_pos + 1
                else:  # else this index does not have external injection
                    self.model.addConstr(
                        np.array(self.X_B1[row]).reshape((1, -1)).dot(
                            np.array(ang_ins(time)).reshape((-1, 1)))[0][0] ==
                        self.power_injections[row, time])
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
        # 注意正负
        for node in range(self.node_count):
            for time in range(T):
                objects.append(
                    self.node_info[node]['load_coeff'] *
                    (self.load_vars[node, time] - self.node_info[node]['load_ref']) *
                    (self.load_vars[node, time] - self.node_info[node]['load_ref']))  # 舒适成本
                objects.append(
                    self.node_info[node]['power_coeff_a'] * (self.gene_vars[node, time] *
                                                             self.gene_vars[node, time]) +
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
        global g_lam  # TODO: SECOND WRONG
        lams = []
        lam_index = g_lam_index[self.index]
        for index in lam_index:
            lams.append(g_lam[index])
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
                thisvar_inside = self.angles_inside[self.connection_index[i], time]
                ind = g_connection[connect_to].index(self.index)
                # area 1 for point 0 at time t:
                thisvar_outside = g_angles[connect_to][ind * 2 + 1][time]  # TODO:change the g_angle
                # 以区域二为例， g_angles[1] 选择区域2 的全部相角
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
        dual_addition = sum([a * b * 1 for a, b in zip(duals, lams)])

        # construct norm object
        norm_addition = 0
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            for time in range(T):
                thisvar_inside = self.angles_inside[self.connection_index[i], time]
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
        self.old_value = [0] * (g_link * 2 * 2 * T)

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
                        thisvar_outside = \
                            g_angles[connect_to][2 * g_connection[connect_to].index(i) + 1][time]
                        gx.append(thisvar_inside - thisvar_outside)
                        gx.append(-1 * thisvar_inside + thisvar_outside)
                    for time in range(T):
                        thatvar_inside = g_angles[i][2 * g_connection[i].index(connect_to) + 1][time]
                        thatvar_outside = \
                            g_angles[connect_to][2 * g_connection[connect_to].index(i)][time]
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
