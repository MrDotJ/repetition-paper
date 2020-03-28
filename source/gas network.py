import gurobipy as gurobi
import numpy as np


class PowerNet:
    def __init__(self, system_info, node_info, line_info):
        self.index = system_info['index']
        self.T = system_info['T']

        self.gen_num = node_info['gen_num']
        self.gen_index = node_info['gen_index']
        self.gen_power_min = node_info['gen_power_min']
        self.gen_power_max = node_info['gen_power_max']
        self.gen_react_min = node_info['gen_react_min']
        self.gen_react_max = node_info['gen_react_max']
        self.bus_num = node_info['bus_num']
        self.bus_voltage_min = node_info['bus_voltage_min']
        self.bus_voltage_max = node_info['bus_voltage_max']
        self.load_num = node_info['load_num']
        self.load_index = node_info['load_index']
        self.load_power_min = node_info['load_power_min']
        self.load_power_max = node_info['load_power_max']
        self.load_react_min = node_info['load_react_min']
        self.load_react_max = node_info['load_react_max']

        self.line_num = line_info['line_num']
        self.line_current_capacity = line_info['line_current_capacity']
        self.line_start_point = line_info['line_start_point']
        self.line_end_point = line_info['line_end_point']
        self.line_resistance = line_info['line_resistance']
        self.line_reactance = line_info['line_reactance']
        # model
        self.model = gurobi.Model()
        self.power_gen = None
        self.react_gen = None
        self.voltage_square = None
        self.line_current_square = None
        self.line_power_flow = None
        self.line_react_flow = None
        self.power_load = None
        self.react_load = None

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
            name='line_power_flow')
        self.line_react_flow = self.model.addVars(
            self.line_num, self.T,
            name='line_react_flow')

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
                    lhs=sum(Power[:, time]) + sum(Power_In[:, time] - resistance[:, time] * Current_In[:, time]),
                    rhs=sum(Power_Load[:, time]) + sum(Power_Out[:, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='power_balance')
                self.model.addConstr(
                    lhs=sum(React[:, time]) + sum(React_In[:, time] - reactance[:, time] * Current_In[:, time]),
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
                    lhs=self.voltage_square[end_point, time] - self.voltage_square[start_point, time],
                    rhs=impedance_square * self.line_current_square[i, time] -
                        2 * (resistance * self.line_power_flow[i, time] + reactance * self.line_react_flow[i, time]),
                    sense=gurobi.GRB.EQUAL,
                    name='voltage_drop')
                self.model.addConstr(
                    lhs=self.line_power_flow[i, time] * self.line_power_flow[i, time] +
                        self.line_react_flow[i, time] * self.line_react_flow[i, time],
                    rhs=self.line_current_square[i, time] * self.voltage_square[start_point, time],
                    sense=gurobi.GRB.LESS_EQUAL,
                    name='flow_relax')

        self.model.setObjective(0)
        self.model.optimize()


def getPowerNet():
    system_info = {
        'index': 0,
        'T': 9
    }
    node_info = {
        'gen_num': 3,
        'gen_index': [0, 0, 5],
        'gen_power_min': [0, 0, 0],
        'gen_power_max': [10, 0.8, 1.2],
        'gen_react_min': [0, 0, 0],
        'gen_react_max': [1, 0.8, 1.2],
        'bus_num': 12,
        'bus_voltage_min': [1.2] * 12,
        'bus_voltage_max': [0.8] * 12,
        'load_num': 8,
        'load_index': [2, 3, 4, 7, 8, 9, 10, 11],
        'load_power_min': [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1],
        'load_power_max': [0.15, 0.25, 0.25, 0.15, 0.35, 0.15, 0.25, 0.15],
        'load_react_min': [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1],
        'load_react_max': [0.15, 0.25, 0.25, 0.15, 0.35, 0.15, 0.25, 0.15]
    }
    line_info = {
        'line_num': 11,
        'line_current_capacity': [1000000] * 11,
        'line_start_point': [0, 1, 0, 3, 0, 5, 8, 5, 6, 5, 6],
        'line_end_point': [1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11],
        'line_resistance': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'line_reactance': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    return PowerNet(system_info, node_info, line_info)


def start():
    powerNet1 = getPowerNet()
    powerNet1.build_model()


if __name__ == '__main__':
    start()
