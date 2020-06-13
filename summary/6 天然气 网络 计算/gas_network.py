# this is wrong
import gurobipy as gurobi
import numpy as np


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

    def well_connected_with(self, node):
        result = np.where(self.well_index == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result:
            per_well = []
            for time in range(self.T):
                per_well.append(self.well_output[i, time])
            result_list.append(per_well)
        return np.array(result_list)

    def load_connected_with(self, node):
        result = np.where(self.gas_load_index == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result:
            per_load = []
            for time in range(self.T):
                per_load.append(self.gas_load[i, time])
            result_list.append(per_load)
        return np.array(result_list)

    def gen_connected_with(self, node):  # list of expression
        result = np.where(self.gen_gas_index == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result:
            per_gen = []
            for time in range(self.T):
                per_gen.append(self.gen_gas_power[i, time] / self.gen_gas_efficiency[i])
            result_list.append(per_gen)
        return np.array(result_list)

    def p2g_connected_with(self, node):
        return np.array([[0] * self.T])

    def gas_flow_out_connected_with(self, node):
        result = np.where(self.gas_line_start_point == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result:
            per_out = []
            for time in range(self.T):
                per_out.append(self.gas_flow_out[i, time])
            result_list.append(per_out)
        return np.array(result_list)

    def gas_flow_in_connected_with(self, node):
        result = np.where(self.gas_line_end_point == node)
        if result[0].size == 0:
            return np.array([[0] * self.T])
        result_list = []
        for i in result:
            per_in = []
            for time in range(self.T):
                per_in.append(self.gas_flow_in[i, time])
            result_list.append(per_in)
        return np.array(result_list)

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
                               # ub=[[self.gas_flow_in_max[i] * self.T] for i in range(self.gas_line_num)],
                               name='gas_flow_in')
        self.gas_flow_out = \
            self.model.addVars(self.gas_line_num, self.T,
                               # ub=[[self.gas_flow_out_max[i] * self.T] for i in range(self.gas_line_num)],
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
        # self.linepack_max = self.model.addVars(
        #     (self.gas_line_num, self.T),
        #     name='gas_linepack_max')
        # self.linepack_min = self.model.addVars(
        #     (self.gas_line_num, self.T),
        #     name='gas_linepack_min')
        # self.pressure_min = self.model.addVars(
        #     (self.node_count, self.T),
        #     name='node_pressure_min')
        # self.pressure_max = self.model.addVars(
        #     (self.node_count, self.T),
        #     name='node_pressure_max')
        # self.linepack_more = self.model.addVar(
        #     name='linepack_more')
        # self.linepack_less = self.model.addVar(
        #     name='linepack_less')
        break_line = 1

        # add constrain
        # gas nodal balance
        for node in range(self.gas_node_num):
            # use numpy !!!! return [[]] format
            Well = self.well_connected_with(node)  # 节点node对应的well变量
            Load = self.load_connected_with(node)
            # considered efficiency !!!!!!!
            Gen = self.gen_connected_with(node)  # this change Power to Gas
            P2G = self.p2g_connected_with(node)  # this is just gas
            Line_Out = self.gas_flow_out_connected_with(node)
            Line_In = self.gas_flow_in_connected_with(node)
            for time in range(self.T):
                self.model.addConstr(
                    lhs=sum(Well[:, time]) + sum(P2G[:, time]) + sum(Line_In[:, time]),  # source
                    rhs=sum(Gen[:, time]) + sum(Load[:, time]) + sum(Line_Out[:, time]),  # load
                    sense=gurobi.GRB.EQUAL,
                    name='gas_nodal_balance_node')
        # line pack
        for line in range(self.gas_line_num):
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
        linepack_sum = 0
        for line in range(self.gas_line_num):
            linepack_sum = linepack_sum + self.linepack[line, self.T - 1]
        self.model.addConstr(linepack_sum <= self.gas_line_pack_initial)

        # active pipeline
        for line in range(self.gas_line_num):
            if line in self.gas_line_active:
                thisIndex = self.gas_line_active.index(line)
                compressor_coeff = self.compressor_coefficient[thisIndex]
                start_point = self.gas_line_start_point[line]
                end_point = self.gas_line_end_point[thisIndex]
                max_flow = self.compressor_max_flow[thisIndex]
                energy_consumption = 1 - self.compressor_energy_consumption[thisIndex]
                for time in range(self.T):
                    self.model.addConstr(self.gas_flow_in[line, time] <= max_flow)
                    self.model.addConstr(self.node_pressure[end_point, time] <=
                                         compressor_coeff * self.node_pressure[start_point, time])
                    # add flow quantities for gas compressors
                    self.model.addConstr(self.gas_flow_out[line, time] ==
                                         energy_consumption * self.gas_flow_in[line, time])

        # weymouth
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
                        sense=gurobi.GRB.EQUAL)

        self.model.setObjective(0)

        self.model.Params.NonConvex = 2
        self.model.optimize()

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
    min_load = [
        [5520 * 28.3 * 0.8 * 0.4, 4920 * 28.3 * 0.8 * 0.4, 4680 * 28.3 * 0.8 * 0.4, 4740 * 28.3 * 0.8 * 0.4,
         5100 * 28.3 * 0.8 * 0.4, 5640 * 28.3 * 0.8 * 0.4, 5580 * 28.3 * 0.8 * 0.4, 6060 * 28.3 * 0.8 * 0.4,
         6180 * 28.3 * 0.8 * 0.4],
        [5520 * 28.3 * 0.8 * 0.2, 4920 * 28.3 * 0.8 * 0.2, 4680 * 28.3 * 0.8 * 0.2, 4740 * 28.3 * 0.8 * 0.2,
         5100 * 28.3 * 0.8 * 0.2, 5640 * 28.3 * 0.8 * 0.2, 5580 * 28.3 * 0.8 * 0.2, 6060 * 28.3 * 0.8 * 0.2,
         6180 * 28.3 * 0.8 * 0.2],
        [5520 * 28.3 * 0.8 * 0.4, 4920 * 28.3 * 0.8 * 0.4, 4680 * 28.3 * 0.8 * 0.4, 4740 * 28.3 * 0.8 * 0.4,
         5100 * 28.3 * 0.8 * 0.4, 5640 * 28.3 * 0.8 * 0.4, 5580 * 28.3 * 0.8 * 0.4, 6060 * 28.3 * 0.8 * 0.4,
         6180 * 28.3 * 0.8 * 0.4]]
    max_load = [
        [5520 * 28.3 * 1.2 * 0.4, 4920 * 28.3 * 1.2 * 0.4, 4680 * 28.3 * 1.2 * 0.4, 4740 * 28.3 * 1.2 * 0.4,
         5100 * 28.3 * 1.2 * 0.4, 5640 * 28.3 * 1.2 * 0.4, 5580 * 28.3 * 1.2 * 0.4, 6060 * 28.3 * 1.2 * 0.4,
         6180 * 28.3 * 1.2 * 0.4],
        [5520 * 28.3 * 1.2 * 0.2, 4920 * 28.3 * 1.2 * 0.2, 4680 * 28.3 * 1.2 * 0.2, 4740 * 28.3 * 1.2 * 0.2,
         5100 * 28.3 * 1.2 * 0.2, 5640 * 28.3 * 1.2 * 0.2, 5580 * 28.3 * 1.2 * 0.2, 6060 * 28.3 * 1.2 * 0.2,
         6180 * 28.3 * 1.2 * 0.2],
        [5520 * 28.3 * 1.2 * 0.4, 4920 * 28.3 * 1.2 * 0.4, 4680 * 28.3 * 1.2 * 0.4, 4740 * 28.3 * 1.2 * 0.4,
         5100 * 28.3 * 1.2 * 0.4, 5640 * 28.3 * 1.2 * 0.4, 5580 * 28.3 * 1.2 * 0.4, 6060 * 28.3 * 1.2 * 0.4,
         6180 * 28.3 * 1.2 * 0.4]]
    return [min_load, max_load]


def build_gas_network():
    [wey, pack] = get_weymouth_matrix()
    [min_load, max_load] = get_load_min_max()
    system_info = {
        'index': 0,
        'T': 9
    }
    gas_node_info = {
        'gas_node_num': 7,
        'node_pressure_min': [75, 140, 100, 70, 150, 160, 100],
        'node_pressure_max': [120, 170, 150, 120, 200, 240, 140],
        'gas_well_num': 2,
        'well_index': [6, 5],  # [0,0,4,5]
        'well_output_min': [0, 0],
        'well_output_max': [5300 * 28.3, 6000 * 28.3],
        'gas_load_num': 3,
        'load_index': [0, 2, 3],
        'gas_load_min': min_load,
        'gas_load_max': max_load,
        'gen_gas_num': 2,
        'gen_gas_index': [3, 0],
        'gen_gas_min': [0, 0],
        'gen_gas_max': [100, 100],
        'gen_gas_efficiency': [0.48, 0.45],
    }

    # gas node information
    gas_line_info = {
        'weymouth': wey,  # for easy, it should contain all line(include active line)
        'gas_line_num': 6,
        'gas_line_start_point': [1, 1, 5, 4, 6, 3],  # gas flow out
        'gas_line_end_point': [0, 4, 4, 2, 3, 1],  # gas flow in
        'gas_line_pack_coefficient': pack,
        'gas_line_pack_initial': 200 * 600,
        'gas_line_active': [1, 5],
        'gas_flow_in_max': [],  # unused
        'gas_flow_out_max': [],  # unused
        'compressor_num': 2,
        'compressor_start_point': [1, 3],
        'compressor_end_point': [4, 1],
        'compressor_coefficient': [2.5, 2.5],
        'compressor_max_flow': [2000000, 2000000],
        'compressor_energy_consumption': [0.03, 0.04],
    }
    # gas line information
    gas_1 = GasNet(system_info, gas_line_info, gas_node_info)
    return gas_1


def start():
    gas_1 = build_gas_network()
    gas_1.build_base()


if __name__ == '__main__':
    start()
