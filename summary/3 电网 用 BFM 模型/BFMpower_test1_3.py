import gurobipy as gurobi
import numpy as np
from copy import deepcopy as copy
import matplotlib.pyplot as plt


# success !
# need test for multi-area-multi-time
# TODO: why the ub and lb doesn't apply to the voltage_square ????
# TODO: p2 always be zero,  don't know why?

def print_info(info):
    print(info.this_voltage_square)
    print(info.that_voltage_square)
    print(info.power_flow)
    print(info.react_flow)


class connection_line_info:
    def __init__(self):
        self.this_voltage_square = 0.0
        self.that_voltage_square = 0.0
        self.power_flow = 0.0               # out -> to the connected node
        self.react_flow = 0.0               # out -> to the connected node
        self.this_node_pressure = 0.0
        self.that_node_pressure = 0.0
        self.gas_flow_in = 0.0
        self.gas_flow_out = 0.0


# branch flow model ! success
T = 4
g_link = 1
g_connection = [
    [1],
    [0]
]
player_num = 2
g_tao = 100

#TODO:       one_line_0_with_1          one_line_0_with_2
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


class PowerNet:
    def __init__(self, system_info, node_info, line_info):
        self.index = system_info['index']
        self.T = system_info['T']

        # ------------- generator of non-gas ----------------------
        self.gen_num = node_info['gen_num']             # add virtual node at last as the connected node
        self.gen_index = node_info['gen_index']
        self.gen_power_min = node_info['gen_power_min']
        self.gen_power_max = node_info['gen_power_max']
        self.gen_react_min = node_info['gen_react_min']
        self.gen_react_max = node_info['gen_react_max']
        self.gen_cost_a = node_info['gen_cost_a']
        self.gen_cost_b = node_info['gen_cost_b']
        self.gen_cost_c = node_info['gen_cost_c']
        # ---------------- power bus node ---------------------------
        self.bus_num = node_info['bus_num']             # add virtual node at last
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
        self.connection_area = node_info['connection_area']
        self.connection_index = node_info['connection_index']
        # ------------------- power line info -------------------------
        self.line_num = line_info['line_num']
        self.line_current_capacity = line_info['line_current_capacity']
        self.line_start_point = line_info['line_start_point']
        self.line_end_point = line_info['line_end_point']
        self.line_resistance = line_info['line_resistance']
        self.line_reactance = line_info['line_reactance']
        # ------------------ gas info------------------

        # model
        self.model = gurobi.Model()
        # -------------------- power system var -------------------
        self.power_gen = None
        self.react_gen = None
        self.voltage_square = None
        self.line_current_square = None
        self.line_power_flow = None
        self.line_react_flow = None
        self.power_load = None
        self.react_load = None
        # ----------------------- model info ------------------------
        self.basic_objective = None
        self.addition_objective = None
        self.objective = None
        # ------------------------ old info -------------------------
        self.info = [
            [
                connection_line_info()
                for _ in range(self.T)
            ] for __ in range(len(self.connection_area))
        ]
        #TODO: self.info  [0]                   [0]
        #          self.index_with_i    at     time_0
        self.old_value = [
            [
                connection_line_info()
                for _ in range(self.T)
            ] for __ in range(len(self.connection_area))
        ]
        #TODO: self.old_value   [0]                 [0]
        #                 self.index_with_i   at   time_0

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

    def get_dual(self, this_info, that_info, start_point):
        assert 0
        # this or that of two areas is same!
        diff1 = this_info.this_voltage_square - that_info.this_voltage_square
        diff2 = -1 * this_info.this_voltage_square + that_info.this_voltage_square
        diff3 = this_info.that_voltage_square - that_info.that_voltage_square
        diff4 = -1 * this_info.that_voltage_square + that_info.that_voltage_square
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
        # this or that of two areas is same!
        diff9 = this_info.this_node_pressure - that_info.this_node_pressure
        diff10 = -1 * this_info.this_node_pressure + that_info.this_node_pressure
        diff11 = this_info.that_node_pressure - that_info.that_node_pressure
        diff12 = -1 * this_info.that_node_pressure + that_info.that_node_pressure
        if start_point != 0:  # this is start point
            diff13 = this_info.gas_flow_in - that_info.gas_flow_in
            diff14 = -1 * this_info.gas_flow_in + that_info.gas_flow_in
            diff15 = this_info.gas_flow_out - that_info.gas_flow_out
            diff16 = -1 * this_info.gas_flow_out + that_info.gas_flow_out
        else:
            diff13 = -1 * this_info.gas_flow_in - that_info.gas_flow_in
            diff14 = 1 * this_info.gas_flow_in + that_info.gas_flow_in
            diff15 = -1 * this_info.gas_flow_out - that_info.gas_flow_out
            diff16 = 1 * this_info.gas_flow_out + that_info.gas_flow_out
        return [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8,
                diff9, diff10, diff11, diff12, diff13, diff14, diff15, diff16]
    def get_sub(self, this_info, this_info_old, start_point):
        assert 0   # this must be wrong
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
                   (this_info.this_node_pressure - this_info_old.this_node_pressure) * \
                   (this_info.this_node_pressure - this_info_old.this_node_pressure) + \
                   (this_info.that_node_pressure - this_info_old.that_node_pressure) * \
                   (this_info.that_node_pressure - this_info_old.that_node_pressure) + \
                   (this_info.gas_flow_in - this_info_old.gas_flow_in) * \
                   (this_info.gas_flow_in - this_info_old.gas_flow_in) + \
                   (this_info.gas_flow_out - this_info_old.gas_flow_out) * \
                   (this_info.gas_flow_out - this_info_old.gas_flow_out)
        return diff
    def get_lam(self, index, start_point):
        assert 0       # this must be wrong
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
                lam_t[8] = lam_copy[1]
                lam_t[9] = lam_copy[0]
                lam_t[10] = lam_copy[3]
                lam_t[11] = lam_copy[2]
                lam_t[12] = lam_copy[5]
                lam_t[13] = lam_copy[4]
                lam_t[14] = lam_copy[7]
                lam_t[15] = lam_copy[6]
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
            lb=[[self.load_power_min[i]] * self.T for i in range(self.load_num)],
            ub=[[self.load_power_max[i]] * self.T for i in range(self.load_num)],
            name='power_load')
        self.react_load = self.model.addVars(
            self.load_num, self.T,
            lb=[[self.load_react_min[i]] * self.T for i in range(self.load_num)],
            ub=[[self.load_react_max[i]] * self.T for i in range(self.load_num)],
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
        for i in range(len(self.connection_area)):
            line_T = []
            line_start = self.line_num - len(self.connection_area)  # 5-2 = 3   3 4
            bus_start = self.bus_num - len(self.connection_area)
            this_index = self.connection_index[i]
            for time in range(self.T):
                line_t = connection_line_info()
                line_t.power_flow = self.line_power_flow[i + line_start, time]
                line_t.react_flow = self.line_react_flow[i + line_start, time]
                line_t.this_voltage_square = self.voltage_square[this_index, time]
                line_t.that_voltage_square = self.voltage_square[i + bus_start, time]
                line_T.append(line_t)
            # self.info.append(line_T)
            self.info[i] = line_T

        # node power balance
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
            for time in range(self.T):
                self.model.addConstr(
                    lhs=sum(Power[:, time]) +
                        sum(Power_In[:, time] - resistance[:, time] * Current_In[:, time]),
                    rhs=sum(Power_Load[:, time]) + sum(Power_Out[:, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='power_balance')
                self.model.addConstr(
                    lhs=sum(React[:, time]) +
                        sum(React_In[:, time] - reactance[:, time] * Current_In[:, time]),
                    rhs=sum(React_Load[:, time]) + sum(React_Out[:, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='react_balance')

        # voltage drop
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

        self.objs = []
        for gen in range(self.gen_num - 1):  # TODO: here is wrong, not 1, but the len(self.connection_area)
            per = 0
            for time in range(self.T):
                per = per + \
                      self.power_gen[gen, time] * self.gen_cost_a[gen] + \
                      self.power_gen[gen, time] * self.power_gen[gen, time] * self.gen_cost_b[gen] + \
                      self.gen_cost_c[gen]
            self.objs.append(per)

        for load in range(self.load_num - 1):  # except the last node #TODO: here is wrong for complex topology
            per = 0
            for time in range(self.T):
                load_ref = (self.load_power_max[load] + self.load_power_min[load]) / 2
                per = per + \
                      1 * (self.power_load[load, time] - load_ref) * \
                      (self.power_load[load, time] - load_ref)
            self.objs.append(per)
        # add cost for buying power
        first_gen = self.gen_num - len(self.connection_area)  # for example 5 - 2 = 3 , so we start with 3
        first_line = self.line_num - len(self.connection_area)
        for power_in in range(len(self.connection_area)):
            # for every area
            per_area = 0
            for time in range(self.T):
                # per_area = per_area + (self.power_gen[first_gen + power_in, time] -
                #                       self.power_load[first_gen + power_in, time]) * 50 * -1
                per_area = per_area + self.line_power_flow[first_line + power_in, time] * 50 * (1)
            self.objs.append(per_area)

        objective = sum(self.objs)
        self.basic_objective = objective

    def update_model(self, tao):
        global g_info
        self.lams = []
        for i, index in enumerate(g_lam_index[self.index]):
            connect_to = self.connection_area[i]
            is_start_point = 0
            if connect_to > self.index:
                is_start_point = 1
            self.lams.extend(self.get_lam(index, is_start_point))

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
        self.dual_addition = sum([10 * a * b for a, b in zip(self.dual, self.lams)])

        self.norm_addition = 0
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]
            is_start_point = 0
            if connect_to > self.index:
                is_start_point = 1
            for time in range(self.T):
                self.norm_addition = self.norm_addition + \
                                     self.get_sub(self.info[i][time], self.old_value[i][time], is_start_point)
        self.addition_objective = self.dual_addition + tao / 2 * self.norm_addition
        self.objective = self.basic_objective + self.addition_objective
        self.model.setObjective(self.objective)

    def optimize(self):
        #  self.model.Params.NonConvex = 2
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        for i in range(len(self.connection_area)):
            for time in range(self.T):
                this_index = self.connection_index[i]
                connect_to = self.connection_area[i]
                is_start_point = 0
                if connect_to > self.index:
                    is_start_point = 1
                line = connection_line_info()
                line_start = self.line_num - len(self.connection_area)
                bus_start = self.bus_num - len(self.connection_area)
                if is_start_point != 0:  # this is start point
                    line.power_flow = self.line_power_flow[i + line_start, time].getAttr('X')
                    line.react_flow = self.line_react_flow[i + line_start, time].getAttr('X')
                else:
                    line.power_flow = self.line_power_flow[i + line_start, time].getAttr('X') * (-1)
                    line.react_flow = self.line_react_flow[i + line_start, time].getAttr('X') * (-1)
                if is_start_point != 0:  # this is start point
                    line.this_voltage_square = self.voltage_square[this_index, time].getAttr('X')
                    line.that_voltage_square = self.voltage_square[i + bus_start, time].getAttr('X')
                else:
                    line.this_voltage_square = self.voltage_square[i + bus_start, time].getAttr('X')
                    line.that_voltage_square = self.voltage_square[this_index, time].getAttr('X')
                g_info[self.index][i][time] = line

    def set_old_value(self, old):  # old_value should be consisted with the g_info
        for area in range(len(self.connection_area)):
            for time in range(self.T):
                self.old_value[area][time].this_voltage_square = old[area][time].this_voltage_square
                self.old_value[area][time].that_voltage_square = old[area][time].that_voltage_square
                self.old_value[area][time].power_flow = old[area][time].power_flow
                self.old_value[area][time].react_flow = old[area][time].react_flow


class PlayerN1:
    def __init__(self):
        self.old_value = [0] * g_link * T * 8

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
        return gx

    def optimize(self, tao):
        model = gurobi.Model()
        self.gx = []
        for i in range(len(g_connection)):
            for connect_to in g_connection[i]:
                if i < connect_to:
                    for time in range(T):
                        lhs = g_info[i][g_connection[i].index(connect_to)][time]
                        rhs = g_info[connect_to][g_connection[connect_to].index(i)][time]
                        self.gx.extend(self.sub(lhs, rhs))
        duals = model.addVars(len(self.gx))
        self.dual_express = gurobi.quicksum(
            100 * duals[i] * self.gx[i] for i in range(len(self.gx))
        )
        self.norm_express = gurobi.quicksum(
            (duals[i] - self.old_value[i]) * (duals[i] - self.old_value[i])
            for i in range(len(self.gx)))

        self.objective = -1 * self.dual_express + tao / 2 * self.norm_express
        model.setObjective(self.objective)
        model.Params.OutputFlag = 0
        model.optimize()
        dual_value = []
        pos = 0
        for line in range(g_link):
            lam_T = []
            for time in range(T):
                lam_t = []
                for _m_m_ in range(8):
                    lam_t.append(duals[pos].getAttr('X'))
                    pos = pos + 1
                lam_T.append(lam_t)
            dual_value.append(lam_T)
        # print(dual_value)
        return copy(dual_value)

    def set_old_value(self, old_value):
        self.old_value = copy(old_value)


def getPowerNet():
    system_info_0 = {
        'index': 0,
        'T': T
    }
    node_info_0 = {
        'gen_num': 3 + 1,  # the outside node as node-12 connected with index-3 with line-12
        'gen_index': [0, 0, 5, 12],
        'gen_power_min': [0, 0, 0, 0],  # 0
        'gen_power_max': [0.4, 0.4, 0.4, 1000000],  # 1.2
        'gen_react_min': [0, 0, 0, 0],  # 0
        'gen_react_max': [0.4, 0.4, 0.4, 1000000],  # 1.2
        'gen_cost_a': [0.1, 0.13, 0.09, 0],
        'gen_cost_b': [0.01, 0.01, 0.01, 0.5],
        'gen_cost_c': [0.1, 0.1, 0.1, 0],
        'bus_num': 12 + 1,
        'bus_voltage_min': [0.9 * 1] * (12 + 1),
        'bus_voltage_max': [1.1 * 1] * (12 + 1),
        'load_num': 8 + 1,
        'load_index': [2, 3, 4, 7, 8, 9, 10, 11, 12],
        'load_power_min': [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0],  # 1.3
        'load_power_max': [0.15, 0.25, 0.25, 0.15, 0.35, 0.15, 0.25, 0.15, 100000],  # 1.7
        'load_react_min': [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0],  # 1.3
        'load_react_max': [0.15, 0.25, 0.25, 0.15, 0.35, 0.15, 0.25, 0.15, 100000],  # 1.7
        'bus_num_outside': 1,
        'connection_area': [1],
        'connection_index': [5],
    }
    line_info_0 = {
        'line_num': 11 + 1,
        'line_current_capacity': [10] * (11 + 1),
        'line_start_point': [0, 1, 0, 3, 0, 5, 8, 5, 6, 5, 6, 12],
        'line_end_point': [1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 5],
        'line_resistance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist(),
        'line_reactance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist()
    }
    system_info_1 = {
        'index': 1,
        'T': T
    }
    node_info_1 = {
        'gen_num': 3 + 1,  # the outside node as node-12 connected with index-3 with line-12
        'gen_index': [0, 0, 5, 12],
        'gen_power_min': [0.9, 0.7, 1.0, 0],  # 0 - 3
        'gen_power_max': [1, 0.8, 1.2, 1000000],
        'gen_react_min': [0.9, 0.7, 1.0, 0],  # 0 - 3
        'gen_react_max': [1, 0.8, 1.2, 1000000],
        'gen_cost_a': [0.1, 0.13, 0.09, 0],
        'gen_cost_b': [0.01, 0.01, 0.01, 0.5],
        'gen_cost_c': [0.1, 0.1, 0.1, 0],
        'bus_num': 12 + 1,
        'bus_voltage_min': [0.9 * 1] * (12 + 1),
        'bus_voltage_max': [1.1 * 1] * (12 + 1),
        'load_num': 8 + 1,
        'load_index': [2, 3, 4, 7, 8, 9, 10, 11, 12],
        'load_power_min': [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0],  # 1.3
        'load_power_max': [0.15, 0.25, 0.25, 0.15, 0.35, 0.15, 0.25, 0.15, 100000],  # 1.7
        'load_react_min': [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0],  # 1.3
        'load_react_max': [0.15, 0.25, 0.25, 0.15, 0.35, 0.15, 0.25, 0.15, 100000],  # 1.7
        'bus_num_outside': 1,
        'connection_area': [0],
        'connection_index': [5],
    }
    line_info_1 = {
        'line_num': 11 + 1,
        'line_current_capacity': [1000000] * (11 + 1),
        'line_start_point': [0, 1, 0, 3, 0, 5, 8, 5, 6, 5, 6, 12],
        'line_end_point': [1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 5],
        'line_resistance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist(),
        'line_reactance': (np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]) / 10).tolist()
    }
    p1 = PowerNet(system_info_0, node_info_0, line_info_0)
    p2 = PowerNet(system_info_1, node_info_1, line_info_1)
    return [p1, p2, PlayerN1()]


def update_old_value():
    for i, player in enumerate(g_players):
        player.set_old_value(copy(g_info[i]))
    temp_lam = []
    for line in range(g_link):
        for time in range(T):
            for index in range(8):
                temp_lam.append(g_lam[line][time][index])
    g_playerN1.set_old_value(copy(temp_lam))


def sub_info(a, b):
    return (a.this_voltage_square - b.this_voltage_square) * (a.this_voltage_square - b.this_voltage_square) + \
           (a.that_voltage_square - b.that_voltage_square) * (a.that_voltage_square - b.that_voltage_square) + \
           (a.power_flow - b.power_flow) * (a.power_flow - b.power_flow) + \
           (a.react_flow - b.react_flow) * (a.react_flow - b.react_flow)


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
        if sub_norm(old_info, copy(g_info)) < 0.001:
            print(count_best_response + 1)
            break
        count_best_response = count_best_response + 1


def start():
    global g_info
    result_plt = []
    result_plt1 = []
    result_plt2 = []
    outer_loop_count = 2
    for player in g_players:
        player.build_model()
    while outer_loop_count < 100:
        print(outer_loop_count)
        calculate_NE()
        update_old_value()
        outer_loop_count = outer_loop_count + 1
        result_plt.append(g_info[0][0][0].power_flow)
        result_plt1.append(g_info[1][0][0].power_flow)
        result_plt2.append(g_info[0][0][0].power_flow - g_info[1][0][0].power_flow)
        g_info = [[[connection_line_info() for ____time in range(T)]
                   for ____line in range(len(g_connection[____area]))]
                  for ____area in range(len(g_connection))]
    plt.plot(result_plt, label='0->1')
    plt.plot(result_plt1, '-r', label='1->0')
    plt.plot(result_plt2, '-g', label='diff')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    all_players = getPowerNet()
    g_players = all_players[:player_num]
    g_playerN1 = all_players[player_num]
    start()
