import gurobipy as gurobi
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
# simply the model
# change the basic node to micro grid
# first two node
# change ...
price = [[5, 5],
         [5, 5]]  # 你要注意这都是买的价格
g_tao = 10000
g_lam = [10, 11, 12, 13]
#         0     1
g_lam_index = [[0, 2],
               [2, 0]]
g_angles = [[0, 1],  # 0 -- 1 => 2 => inside - outside
            [2, 3]]  # 1 -- 0 => 2 => inside - outside
g_connection = [[1],
                [0]]
g_link = 1
player_num = 2
injection = [0, 0]


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
        self.load_vars = []
        self.gene_vars = []
        self.power_injections = []
        self.angles_inside = []
        self.angles_outside = []
        self.power_injections = []
        self.power_injections_outside = []
        self.object_basic = None
        self.object_addition = None
        self.object = None
        self.free_var = None
        # old value
        self.old_value = [0] * (len(self.connection_area) * 2)

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
                self.model.addVar(lb=-10000, ub=10000, name='Region_' + str(self.index) +
                                                            '_node_' + str(node) + '_power_injection'))
        # 功率注入的等式
        for node in range(self.node_count):
            self.model.addConstr(
                lhs=-1 * self.load_vars[node] + self.gene_vars[node],
                rhs=self.power_injections[node],
                sense=gurobi.GRB.EQUAL,
                name='power_injection' + str(node))
        # 添加内部/外部相角变量以及外部注入功率变量，**相角的范围是多少？**
        self.angles_inside = self.model.addVars(self.node_count, lb=-3.14, ub=3.14, name='angle_inside')
        self.angles_outside = self.model.addVars(len(self.connection_area), lb=-3.14, ub=3.14, name='angle_outside')
        # 外部功率注入
        self.power_injections_outside = \
            self.model.addVars(len(self.connection_area),
                               lb=[-1 * m_max for m_max in self.connection_exchange_max],
                               ub=self.connection_exchange_max,
                               name='Region_' + str(self.index) + '_power_injection_outside')
        # 外部功率注入直流潮流
        for conn in range(len(self.connection_area)):
            self.model.addConstr(self.power_injections_outside[conn] ==
                                 (self.angles_outside[conn] - self.angles_inside[self.connection_index[conn]])
                                 / self.connection_x[conn])  # the B is B^ not B^^
        # x should satisfy the max-power-flow
        # 内部线路直流潮流的计算公式 except the reference node
        external_injection_pos = 0
        for row in range(self.node_count - 1):
            if row in self.connection_index:  # if the index has connection with outside area
                self.model.addConstr(self.angles_inside.prod(self.X_B1[row]) ==
                                     self.power_injections[row] + self.power_injections_outside[external_injection_pos])
                external_injection_pos = external_injection_pos + 1
            else:  # else this index does not have external injection
                self.model.addConstr(
                    lhs=self.angles_inside.prod(self.X_B1[row]),
                    rhs=self.power_injections[row],
                    sense=gurobi.GRB.EQUAL)
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
        # global g_angles
        # global g_connection
        # global g_lam
        lams = []  # one line => two node equal => four inequality
        for i in g_lam_index[self.index]:  # 获取用户i所对应的全部lam 索引
            lams.append(g_lam[i])  # 0-0                # index_i less
            lams.append(g_lam[i + 1])  # 0-0            # index_i big
        duals = []
        if self.index == 1000:
            self.free_var = 0
        else:
            self.free_var = self.model.addVar(lb=-3.14, ub=3.14, name='free_var')
            #self.free_var = self.model.addVar(lb=0, ub=0, name='free_var')
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            #    o--------o
            #   this     that
            # ************************************************
            # this var
            thisvar_inside = self.angles_inside[self.connection_index[i]]  # 与 connect_to 区域相连的线路的内部的相角
            thisvar_outside = g_angles[connect_to][g_connection[connect_to].index(self.index) * 2 + 1]
            # that var
            thatvar_inside = self.angles_outside[i]
            thatvar_outside = g_angles[connect_to][g_connection[connect_to].index(self.index) * 2]
            # ***********************************************
            duals.append(thisvar_inside + self.free_var - thisvar_outside)
            duals.append(-1 * thisvar_inside - self.free_var + thisvar_outside)
            duals.append(thatvar_inside + self.free_var - thatvar_outside)
            duals.append(-1 * thatvar_inside - self.free_var + thatvar_outside)
        dual_addition = sum([a * b * 100 for a, b in zip(duals, lams)])
        # construct norm object
        norm_addition = 0
        for i in range(len(self.connection_area)):
            connect_to = self.connection_area[i]  # [2,3,6] 表示连接到的区域
            thisvar_inside = self.angles_inside[self.connection_index[i]]  # 与 connect_to 区域相连的线路的内部的相角
            thatvar_inside = self.angles_outside[i]
            thisvar_outside = g_angles[connect_to][g_connection[connect_to].index(self.index) * 2 + 1]
            thatvar_outside = g_angles[connect_to][g_connection[connect_to].index(self.index) * 2]
            norm_addition = norm_addition + \
                            (thisvar_inside + self.free_var - self.old_value[2 * i]) * \
                            (thisvar_inside + self.free_var - self.old_value[2 * i]) + \
                            (thatvar_inside + self.free_var - self.old_value[2 * i + 1]) * \
                            (thatvar_inside + self.free_var - self.old_value[2 * i + 1])
            if self.index != 0:
                dual_addition = dual_addition + \
                                10 * ((thisvar_inside + self.free_var) - thisvar_outside) * \
                                ((thisvar_inside + self.free_var) - thisvar_outside) + \
                                10 * ((thatvar_inside + self.free_var) - thatvar_outside) * \
                                ((thatvar_inside + self.free_var) - thatvar_outside)
                # select the best reference point
        # add difference
        self.object_addition = dual_addition + \
                               tao / 2 * norm_addition
        self.object = self.object_basic + self.object_addition
        self.model.setObjective(self.object)

    def optimize_model(self):  # calculate the response
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        injection[self.index] = self.power_injections_outside[0].getAttr('X')
        exchange_value = []
        if self.index == 0:
            freevar = 0
        else:
            freevar = self.free_var.getAttr('X')
        for i in range(len(self.connection_area)):
            thisvar_inside = self.angles_inside[self.connection_index[i]]
            thatvar_inside = self.angles_outside[i]
            exchange_value.append(thisvar_inside.getAttr('X') + freevar)
            exchange_value.append(thatvar_inside.getAttr('X') + freevar)
        return exchange_value

    def set_old_value(self, old):  # 01 02 03 04
        for i in range(len(self.old_value)):
            self.old_value[i] = old[i]


class playerNp1:
    def __init__(self):
        self.old_value = [0] * (g_link * 4)

    def optimize(self, tao):  # [[01 02] [10 12] [20 21]]
        global g_angles
        model = gurobi.Model()
        gx = []
        for i in range(len(g_connection)):  # per area
            for connect_to in g_connection[i]:  # per area - area
                if i < connect_to:  # just 0-1 no 1-0
                    # inside - outside
                    # for region i
                    #    o ----------- o
                    #   this          that
                    thisvar_inside = g_angles[i][2 * g_connection[i].index(connect_to)]
                    thisvar_outside = g_angles[connect_to][2 * g_connection[connect_to].index(i) + 1]
                    thatvar_inside = g_angles[i][2 * g_connection[i].index(connect_to) + 1]
                    thatvar_outside = g_angles[connect_to][2 * g_connection[connect_to].index(i)]
                    gx.append(thisvar_inside - thisvar_outside)
                    gx.append(-1 * thisvar_inside + thisvar_outside)
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
            dual_value.append(duals[i].getAttr('X') )
        #print(dual_value)
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
        [0, 0.1, 0],
        [0.1, 0, 0.1],
        [0, 0.1, 0]
    ]
    node_info_0 = [  # 8 - 12 - 10      3 - 6
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
        },
        {
            'min_load': 2,
            'max_load': 3,
            'min_power': 2,
            'max_power': 5,
            'load_coeff': 3,
            'load_ref': 2.5,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
        },

    ]
    connection_info_0 = {
        'connection_index': [1],
        'connection_x': [0.1],
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
        [0, 0.1, 0],
        [0.1, 0, 0.1],
        [0, 0.1, 0]
    ]
    node_info_1 = [  # 2 - 3 - 1.5      1 - 18
        {
            'min_load': 1,
            'max_load': 1,
            'min_power': 1,
            'max_power': 10,
            'load_coeff': 1,
            'load_ref': 1,
            'power_coeff_a': 0.1,
            'power_coeff_b': 1,
            'power_coeff_c': 1,
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
        },
    ]
    connection_info_1 = {
        'connection_index': [1],
        'connection_x': [0.1],
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
    while count_best_response < 40:
        # TODO: maybe 30 is a little small
        for i, player in enumerate(g_players):
            # get the data for the player i
            player.update_model(g_tao)  # 填充x_i 以及lam_i
            player_i_result = player.optimize_model()
            g_angles[i] = player_i_result.copy()
        # update the lam_dual variable
        g_lam = g_playerN1.optimize(g_tao).copy()
        # update the response
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
    # initial
    for player in g_players:
        player.build_model()
    # start the outer loop
    outer_loop_count = 0
    while outer_loop_count < 300:
        # give xn, lam_n, calculate the equilibrium
        calculate_NE()
        # 现在我们得到了一个新的NE，我们应该把这个NE设为参照值
        set_oldValue()
        outer_loop_count = outer_loop_count + 1
        # result_plt.append(g_angles[0][0])
        # result_plt1.append(g_angles[1][0])
        # result_plt2.append(g_angles[0][0] - g_angles[1][0])
        result_plt.append(injection[0])
        result_plt1.append(injection[1])
        result_plt2.append(injection[0] + injection[1])
        # result_plt2.append(g_lam[0] + g_lam[1])
        # set all value in g_ex to zero
        g_angles = [[0] * len(sublist) for sublist in g_angles]
    plt.plot(result_plt, label='0->1')
    plt.plot(result_plt1, '.', label='1->0')
    plt.plot(result_plt2, '*', label='diff')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('x-node-micro-grid.svg')


if __name__ == '__main__':
    all_players = factory()
    g_players = all_players[:player_num]
    g_playerN1 = all_players[player_num]
    start()
