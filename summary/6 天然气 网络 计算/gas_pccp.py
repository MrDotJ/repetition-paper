import gurobipy as gurobi
import numpy as np
from matplotlib import pyplot as plt

class GasNet:
    def __init__(self, system_info, gas_line_info, gas_node_info):
        # system information
        self.index = system_info['index']
        self.T = system_info['T']
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
        # model
        self.model = gurobi.Model()
        # node
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
        #####
        self.gas_flow_in_old = [[0.01 for jqy in range(self.T)] for jjqqyy in range(self.gas_line_num)]
        self.gas_flow_out_old = [[0.01 for jqy in range(self.T)] for jjqqyyy in range(self.gas_line_num)]
        self.node_pressure_old = [[0.01 for jqy in range(self.T)] for jjqqyy in range(self.gas_node_num)]
        self.pccp = None
        self.constrain_update = []

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

    # def gas_flow_out_diff_connected_with(self, node):
    #     result = np.where(np.array(self.gas_line_energy_consum) == node)
    #     if result[0].size == 0:
    #         return np.array([[0] * self.T])
    #     result_list = []
    #     for i in result[0]:
    #         per_in = []
    #         for time in range(self.T):
    #             per_in.append(self.gas_flow_in[i, time])
    #         result_list.append(per_in)
    #     return np.array(result_list)
    #
    def build_base(self):
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
        gas_line_inactive_num = self.gas_line_num - len(self.gas_line_active)
        self.pccp = self.model.addVars(self.gas_line_num, self.T,
                                       lb = 0, name='pccp')

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

        # line pack passive line
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
        linepack_sum = 0  # ? passive or active
        for line in range(self.gas_line_num):
            linepack_sum = linepack_sum + self.linepack[line, self.T - 1]
        self.model.addConstr(linepack_sum <= self.gas_line_pack_initial)

        # active pipeline pressure-increase & gas-consume
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
                                  (self.gas_flow_in[line, time] + self.gas_flow_out[line,time]) / 2 -
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

        self.test = self.model.addVar()
        self.ppp = self.model.addConstr(self.test == 1)



    def optimize(self):
        self.result = []
        j = 0
        k = 1.2
        while j < 20:
            j = j + 1
            k = k * 1.4
            if k > 50:
                k = 50
            self.model.setObjective(gurobi.quicksum(self.pccp) * k)
            self.model.optimize()
            print(self.test)
            self.model.remove(self.ppp)
            self.ppp = self.model.addConstr(self.test == j)
            #result.append(self.pccp[0,0].getAttr('X'))
            self.result.append(
            self.node_pressure[0, 0].getAttr('X') * self.node_pressure[0, 0].getAttr('X') - \
            self.node_pressure[1, 0].getAttr('X') * self.node_pressure[1, 0].getAttr('X') - \
            (self.gas_flow_in[0, 0].getAttr('X') + self.gas_flow_out[0, 0].getAttr('X')) * \
            (self.gas_flow_in[0, 0].getAttr('X') + self.gas_flow_out[0, 0].getAttr('X')) / 4)

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
                                      (self.gas_flow_in[line, time] + self.gas_flow_out[line,time]) / 2 -
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
        plt.plot(self.result)
        plt.show()


    def get_result(self):
        return self.gas_flow_in[0].getAttr('X')


def get_weymouth_matrix():
    weymouth = [[1, 1, 0, 4000, 0.89, 0.01077, 0.00005],
                [2, 1, 4, 5000, 0.89, 0.01077, 0.00005],
                [3, 5, 4, 6000, 0.89, 0.01077, 0.00005],
                [4, 4, 2, 6000, 0.89, 0.01077, 0.00005],
                [5, 6, 3, 6000, 0.89, 0.01077, 0.00005],
                [6, 3, 1, 5000, 0.89, 0.01077, 0.00005]]
    wey = np.array(weymouth)
    wey[:, 4] = wey[:, 4] / 2
    Temperature = 281.15
    R = 5.18 * 1e2
    Z = 0.8
    pho = 0.6106
    coeff1 = 3600 * 3600 * 3.14 * 3.14 / 16
    coeff2 = Z * Temperature * pho * pho * R
    coeff3 = 3600 * 3.14 / 8
    coeff4 = R * Temperature * Z * pho
    result_wey = coeff1 * np.power(wey[:, 4], 5) / wey[:, 3] / wey[:, 5] / coeff2
    result_pack = coeff3 * wey[:, 3] * wey[:, 4] * wey[:, 4] / coeff4
    return [result_wey.tolist(), result_pack.tolist()]


def get_load_min_max():
    a = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    load_base = np.array(a)
    min_coeff = 0.8
    max_coeff = 1.2
    distr_coeff = [0.4, 0.2, 0.4]
    min_load = np.concatenate(((load_base * min_coeff * distr_coeff[0]).reshape((1, 9)),
                               (load_base * min_coeff * distr_coeff[1]).reshape((1, 9)),
                               (load_base * min_coeff * distr_coeff[2]).reshape((1, 9))), axis=0)
    max_load = np.concatenate(((load_base * max_coeff * distr_coeff[0]).reshape((1, 9)),
                               (load_base * max_coeff * distr_coeff[1]).reshape((1, 9)),
                               (load_base * max_coeff * distr_coeff[2]).reshape((1, 9))), axis=0)
    return [min_load, max_load]


def build_gas_network():
    T = 5
    [wey, pack] = get_weymouth_matrix()
    [min_load, max_load] = get_load_min_max()
    system_info = {
        'index': 0,
        'T': T
    }
    gas_node_info = {
        'gas_node_num': 7,
        'node_pressure_min': [75, 140, 100, 70, 150, 160, 100],
        'node_pressure_max': (np.array([120, 170, 150, 120, 200, 240, 140]) * 1).tolist(),
        'gas_well_num': 2,
        'well_index': [6, 5],  # [0,0,4,5]
        'well_output_min': [0, 0],
        'well_output_max': [20, 20],
        'gas_load_num': 3,
        'load_index': [0, 2, 3],
        'gas_load_min': (min_load[:, 0:T]).tolist(),
        'gas_load_max': (max_load[:, 0:T]).tolist(),
        'gen_gas_num': 2,
        'gen_gas_index': [3, 0],
        'gen_gas_min': [0, 0],
        'gen_gas_max': [5, 5],
        'gen_gas_efficiency': [0.48, 0.45],
    }

    # gas node information
    gas_line_info = {
        # 'weymouth': (np.array(wey) / 100).tolist(),  # for easy, it should contain all line(include active line)
        'weymouth': [1] * 6,
        'gas_line_num': 6,
        'gas_line_start_point': [1, 1, 5, 4, 6, 3],  # gas flow out
        'gas_line_end_point': [0, 4, 4, 2, 3, 1],  # gas flow in
        # 'gas_line_pack_coefficient': (np.array(pack) / 10).tolist(),
        'gas_line_pack_coefficient': [1] * 6,
        'gas_line_pack_initial': 100,  # up 152.22 is feasible!
        'gas_line_active': [1, 5],
        'gas_flow_in_max': [1000, 20000, 1000, 1000, 1000, 20000],  # unused
        'gas_flow_out_max': [1000, 20000, 1000, 1000, 1000, 20000],  # unused
        'compressor_num': 2,
        'compressor_start_point': [1, 3],
        'compressor_end_point': [4, 1],
        'compressor_coefficient': [2.5, 2.5],
        'compressor_max_flow': [20000, 20000],
        'compressor_energy_consumption': [0.03, 0.04],
    }

    gas_node_info_1 = {
        'gas_node_num': 3 + 2,
        'node_pressure_min': [0] * (3 + 2),
        'node_pressure_max': [1] * (3 + 2),
        'gas_well_num': 1 ,
        'well_index': [0],
        'well_output_min': [0],
        'well_output_max': [1],
        'gas_load_num': 1 + 2,
        'load_index': [2, 3, 4],
        'gas_load_min': [[0.1] * T] * 3,
        'gas_load_max': [[0.2] * T] * 3,
        'gen_gas_num': 0,
        'gen_gas_index': [],
        'gen_gas_min': [],
        'gen_gas_max': [],
        'gen_gas_efficiency': [],
    }
    gas_line_info_1 = {
        'weymouth': [1] * (2 + 2) ,
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
    # gas line information
    gas_1 = GasNet(system_info, gas_line_info_1, gas_node_info_1)
    return gas_1


def start():
    gas_1 = build_gas_network()
    gas_1.build_base()
    return gas_1


if __name__ == '__main__':
    gas = start()
    gas.optimize()
    debug = 2
