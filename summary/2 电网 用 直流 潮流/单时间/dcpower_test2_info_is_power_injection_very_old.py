import gurobipy as gurobi
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
# remove angle!
# change the basic node to micro grid
# first two node
price = [[0, 1],
         [1, 0]]  # 你要注意这都是买的价格
g_tao = 10
g_lam = [0, 0]
g_lam_index = [[0],
               [0]]
g_power_exchange = [[0],  # 0 -- 1 => 2 => inside - outside
                    [0]]  # 1 -- 0 => 2 => inside - outside
g_connection = [[1],
                [0]]
g_link = 1
player_num = 2


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
        # list for max exchange for every line
        self.connection_exchange_max = connection_info['connection_exchange_max']
        # node_information
        self.node_info = node_info
        self.X_B1 = []
        # model
        self.model = gurobi.Model()
        self.load_vars = []
        self.gene_vars = []
        self.power_injections = []
        self.power_injections_outside = []
        self.object_basic = None
        self.object_addition = None
        self.object = None

        # old value
        self.old_value = [0] * (len(self.connection_area))

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
        array_B_1 = np.array(self.X_B1)
        array_B_1 = array_B_1[1:, 1:]    # set the 0 node as the reference node
        self.X_B1 = inv(array_B_1).tolist()

    def build_model(self):
        global price
        self.node_count = len(self.X_raw)
        index = self.index
        self.build_B()

        # start build model
        objects = []
        constrains = []
        # construct model
        # add variables
        # 每一个节点既有负荷又有generator
        # for每个节点添加负荷变量，发电变量，内部节点的注入功率变量，以及联系
        for node in range(self.node_count):
            self.load_vars.append(self.model.addVar(
                lb=self.node_info[node]['min_load'],
                ub=self.node_info[node]['max_load'],
                obj=0,
                name='Region_' + str(self.index) + '_node_' + str(node) + '_load'))
            self.gene_vars.append(self.model.addVar(
                lb=self.node_info[node]['min_power'],
                ub=self.node_info[node]['max_power'],
                obj=0,
                name='Region_' + str(self.index) + '_node_' + str(node) + '_power'))
            self.power_injections.append(
                self.model.addVar(
                    lb=-1 * gurobi.GRB.INFINITY,
                    ub=gurobi.GRB.INFINITY,
                    name='Region_' + str(self.index) +
                                       '_node_' + str(node) + '_power_injection')
            )
        # 功率注入的等式
        for node in range(self.node_count):
            self.model.addConstr(
                lhs=-1 * self.load_vars[node] + self.gene_vars[node],
                rhs=self.power_injections[node],
                sense=gurobi.GRB.EQUAL,
                name='power_injection' + str(node))

        # 外部注入功率变量
        for area in range(len(self.connection_area)):
            self.power_injections_outside.append(
                self.model.addVar(lb=-1 * self.connection_exchange_max[area],
                                  ub=self.connection_exchange_max[area],
                                  name='power_injection_area' + str(self.index) + '_' + str(area)))


        # 注意正负
        for node in range(self.node_count):
            objects.append(
                self.node_info[node]['load_coeff'] *
                (self.load_vars[node] - self.node_info[node]['load_ref']) *
                (self.load_vars[node] - self.node_info[node]['load_ref']))  # 舒适成本
            objects.append(
                self.node_info[node]['power_coeff_a'] * self.gene_vars[node] * self.gene_vars[node] +
                self.node_info[node]['power_coeff_b'] + self.gene_vars[node] +
                self.node_info[node]['power_coeff_c'])  # 发电成本
        for conn in range(len(self.connection_area)):
            objects.append(price[self.index][self.connection_area[conn]] *
                           self.power_injections_outside[conn])  # 购电成本

        # add constrain - power balance
        power_balance = self.model.addConstr(
            lhs=gurobi.quicksum(self.load_vars),
            rhs=gurobi.quicksum(self.gene_vars) + gurobi.quicksum(self.power_injections_outside),
            sense=gurobi.GRB.EQUAL,
            name='Region_' + str(index) + '_power_balance'
        )
        constrains.append(power_balance)

        # add object
        self.object_basic = sum(objects)

    def update_model(self, tao):  #
        # ex_i 是从小到大所有向i的数据的list，比如说是[01, 21, 31, 51], 相对应的 变量也应该是[10 12 13 15]
        # construct dual object
        # get lam for i
        lams = []  # one line => two node equal => four inequality
        for i in g_lam_index[self.index]:  # 获取用户i所对应的全部lam 索引
            lams.append(g_lam[i])
            lams.append(g_lam[i + 1])

        duals = []
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            duals.append(self.power_injections_outside[i] +
                         g_power_exchange[connect_to][g_connection[connect_to].index(self.index)])  #
            duals.append(-1 * self.power_injections_outside[i] -
                         g_power_exchange[connect_to][g_connection[connect_to].index(self.index)])

        dual_addition = sum([a * b for a, b in zip(duals, lams)])

        # construct norm object
        norm_addition = gurobi.quicksum((self.power_injections_outside[i] - self.old_value[i]) *
                                        (self.power_injections_outside[i] - self.old_value[i])
                                        for i in range(len(self.connection_area)))

        # set object
        self.object_addition = dual_addition + \
                               tao / 2 * norm_addition
        self.object = self.object_basic + self.object_addition
        self.model.setObjective(self.object)

    def optimize_model(self):  # calculate the response
        self.model.optimize()
        exchange_value = []
        for i in range(len(self.connection_area)):
            exchange_value.append(self.power_injections_outside[i].getAttr('X'))
        return exchange_value

    def set_old_value(self, old):  # 01 02 03 04
        for i in range(len(self.old_value)):
            self.old_value[i] = old[i]


class playerNp1:
    def __init__(self):
        self.old_value = [0] * (g_link * 2)

    def optimize(self, tao):  # [[01 02] [10 12] [20 21]]
        model = gurobi.Model()
        gx = []
        for i in range(len(g_connection)):  # per area
            for connect_to in g_connection[i]:  # per area - area
                if i < connect_to:
                    gx.append(g_power_exchange[i][g_connection[i].index(connect_to)] +
                              g_power_exchange[connect_to][g_connection[connect_to].index(i)])
                    # 我得到了第i个建筑到第k个建筑的流量，其中第k个建筑在第i个建筑中的索引是index
                    gx.append(-1 * g_power_exchange[i][g_connection[i].index(connect_to)] -
                              g_power_exchange[connect_to][g_connection[connect_to].index(i)])

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
        model.optimize()
        dual_value = []
        for i in range(len(gx)):
            dual_value.append(duals[i].getAttr('X'))
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
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0]
    ]
    node_info_0 = [
        # min_load = 9   max_load = 18  ref_load = 13.5
        # min_power = 1  max_power = 17   power-gap = +3.5
        {
            'min_load': 0,
            'max_load': 25,
            'min_power': 0,
            'max_power': 5,
            'load_coeff': 1,
            'load_ref': 15,
            'power_coeff_a': 0.01,
            'power_coeff_b': 0.1,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 10,
            'load_ref': 0,
            'power_coeff_a': 0.1,     # 2 * a * x + b  => 3
            'power_coeff_b': 1,
            'power_coeff_c': 1,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 0,
            'load_ref': 0,
            'power_coeff_a': 0,
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 10,
            'load_ref': 0,
            'power_coeff_a': 0,       # 2 * a + b = 2
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 10,
            'load_ref': 0,
            'power_coeff_a': 0,
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
    ]
    connection_info_0 = {
        'connection_index': [1],
        'connection_x': [1],
        'connection_area': [1],
        'connection_exchange_max': [100]
    }
    player0_info = {
        'index': 0,
        'X_raw': X_raw_0,
        'node_info': node_info_0,
        'connection_info': connection_info_0
    }

    X_raw_1 = [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0]
    ]
    node_info_1 = [
        # min_load = 11   max_load = 19   load_ref = 14.5
        # min_power = 5   max_power = 12
        {
            'min_load': 1,
            'max_load': 30,
            'min_power': 0,
            'max_power': 45,
            'load_coeff': 1,
            'load_ref': 20,
            'power_coeff_a': 0.01,
            'power_coeff_b': 0.1,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 0,
            'load_ref': 4,
            'power_coeff_a': 0.1,
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 0,
            'load_ref': 0,
            'power_coeff_a': 0,
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 0,
            'load_ref': 0,
            'power_coeff_a': 0,
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
        {
            'min_load': 0,
            'max_load': 0,
            'min_power': 0,
            'max_power': 0,
            'load_coeff': 0,
            'load_ref': 0,
            'power_coeff_a': 0.1,
            'power_coeff_b': 0,
            'power_coeff_c': 0,
        },
    ]
    connection_info_1 = {
        'connection_index': [1],
        'connection_x': [1],
        'connection_area': [0],
        'connection_exchange_max': [100]
    }
    player1_info = {
        'index': 1,
        'X_raw': X_raw_1,
        'node_info': node_info_1,
        'connection_info': connection_info_1,
    }

    player1 = getPlayer(player0_info)
    player2 = getPlayer(player1_info)
    playerN1 = playerNp1()
    return [player1, player2, playerN1]


##########################################################
# main process flow


def calculate_NE():
    global g_lam
    count_best_response = 0
    while count_best_response < 30:
        for i, player in enumerate(g_players):
            # get the data for the player i
            player.update_model(g_tao)  # 填充x_i 以及lam_i
            player_i_result = player.optimize_model()
            g_power_exchange[i] = player_i_result.copy()
            # update the lam_dual variable
            g_lam = g_playerN1.optimize(g_tao).copy()
            # update the response
        count_best_response = count_best_response + 1


def set_oldValue():
    for i, player in enumerate(g_players):
        player.set_old_value(g_power_exchange[i].copy())
    g_playerN1.set_old_value(g_lam.copy())


def start():
    global g_power_exchange
    result_plt = []
    result_plt1 = []
    result_plt2 = []
    # initial
    for player in g_players:
        player.build_model()
    # start the outer loop
    outer_loop_count = 0
    while outer_loop_count < 150:
        # give xn, lam_n, calculate the equilibrium
        calculate_NE()
        # 现在我们得到了一个新的NE，我们应该把这个NE设为参照值
        set_oldValue()
        outer_loop_count = outer_loop_count + 1
        result_plt.append(g_power_exchange[0][0])
        result_plt1.append(g_power_exchange[1][0])
        result_plt2.append(g_power_exchange[0][0] + g_power_exchange[1][0])
        # set all value in g_ex to zero
        g_power_exchange = [[0] * len(sublist) for sublist in g_power_exchange]
    plt.plot(result_plt, label='0->1')
    plt.plot(result_plt1, '.', label='1->0')
    plt.plot(result_plt2, '*', label='diff')
    plt.legend(loc='best')
    plt.show()
    #plt.savefig('x-node-micro-grid.svg')


if __name__ == '__main__':
    all_players = factory()
    g_players = all_players[:player_num]
    g_playerN1 = all_players[player_num]
    start()
