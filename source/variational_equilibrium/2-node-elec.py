import gurobipy as gurobi
import matplotlib.pyplot as plt
# success! 2 - node
#   price   [[00 01 02 03]
#            [10 11 12 13]
#            [20 21 22 23]
#            [30 31 32 33]]     其中，ii均为0
#   g_ex  [[01 02 03]
#          [10  12 13]
#          [20 21  23]
#          [30 31 32 ]]
#   g_lam  [01+10 02+20 03+30 12+21 13+31 23+32] * 2
#   g_lam_index =
#   g_connection = [[1 2 3]
#                  [0 2 3]
#                  [0 1 3]
#                  [0 1 2]]
price = [[0, 1], [1, 0]]
g_tao = 10
g_ex = [[0], [0]]
g_lam = [0, 0]  # 01 01 02 02 12 12
g_lam_index = [[0], [0]]
g_connection = [[1], [0]]
g_link = 2


class Player:
    def __init__(self, index, demand_ref, supply_max, demand_max, exchange_max,
                 connection, supply_a, supply_b, demand_a):
        self.index = index
        self.demand_ref = demand_ref
        self.supply_max = supply_max
        self.demand_max = demand_max
        self.exchange_max = exchange_max
        self.connection = connection
        self.supply_a = supply_a
        self.supply_b = supply_b
        self.demand_a = demand_a
        # model
        self.model = gurobi.Model()
        self.demand = None
        self.supply = None
        self.exchange = None
        self.object_basic = None
        self.object_addition = None
        self.object = None
        # old value
        self.old_value = [0] * len(self.connection)

    def build_model(self):
        global price
        index = self.index
        objects = []
        constrains = []
        # construct model
        self.supply = self.model.addVar(lb=0, ub=self.supply_max, name='supply' + str(index))
        self.demand = self.model.addVar(lb=0, ub=self.demand_max, name='demand' + str(index))
        self.exchange = self.model.addVars(len(self.connection), lb=-1 * self.exchange_max,
                                           ub=self.exchange_max, name='exchange' + str(index))
        # construct object
        supply_cost = self.supply_a * self.supply + self.supply_b * self.supply * self.supply
        objects.append(supply_cost)
        load_cost = self.demand_a * (self.demand - self.demand_ref) * (self.demand - self.demand_ref)
        objects.append(load_cost)
        exchange_cost = 0
        pos = 0
        for i in self.connection:
            exchange_cost = exchange_cost + price[index][i] * self.exchange[pos]
            pos = pos + 1
        objects.append(-1 * exchange_cost)
        # add constrain
        power_balance = self.model.addConstr(
            lhs=self.supply,
            rhs=self.demand + gurobi.quicksum(self.exchange[i]
                                              for i in range(len(self.connection))),
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
        lams = []
        for i in g_lam_index[self.index]:       # 获取用户i所对应的全部lam 索引
            lams.append(g_lam[i])
            lams.append(g_lam[i + 1])
        duals = []
        for i in range(len(self.connection)):  #   4  [40 41 43 46]  => [0 1 3 6] => [0][4] [1][4] [3][4] [6][4]
            # i:3 ==> self.connection[i]:6 ==> g_connection[6].index(self.index)
            connect_to = self.connection[i]     # 比如说 i= 3 那么链接中的第四个是6的话，connect_to = 6 then g_ex[6][0] is 6-to-3
            duals.append(self.exchange[i] + g_ex[connect_to][g_connection[connect_to].index(self.index)])  # [6] => [4]
            duals.append(-self.exchange[i] - g_ex[connect_to][g_connection[connect_to].index(self.index)])
        dual_addition = sum([a * b for a, b in zip(duals, lams)])
        # construct norm object
        norm_addition = gurobi.quicksum((self.exchange[i] - self.old_value[i]) *
                                        (self.exchange[i] - self.old_value[i])
                                        for i in range(len(self.connection)))
        self.object_addition = dual_addition + \
                               tao / 2 * norm_addition
        self.object = self.object_basic + self.object_addition
        self.model.setObjective(self.object)

    def optimize_model(self):  # calculate the response
        self.model.optimize()
        exchange_value = []
        for i in range(len(self.connection)):
            exchange_value.append(self.exchange[i].getAttr('X'))
        return exchange_value

    def set_old_value(self, old):  # 01 02 03 04
        pos = 0
        for i in range(len(self.connection)):  # 0 1 2 3 4
            self.old_value[i] = old[pos]
            pos = pos + 1


class playerNp1:
    def __init__(self):
        self.old_value = [0] * g_link

    def optimize(self, tao):  # [[01 02] [10 12] [20 21]]
        model = gurobi.Model()
        gx = []
        for i in range(len(g_connection)):  # build_i
            for connect_to in g_connection[i]:  # 0_1
                if i < connect_to:
                    gx.append(g_ex[i][g_connection[i].index(connect_to)] +
                              g_ex[connect_to][g_connection[connect_to].index(i)])
                    # 我得到了第i个建筑到第k个建筑的流量，其中第k个建筑在第i个建筑中的索引是index
                    gx.append(-1 * g_ex[i][g_connection[i].index(connect_to)] -
                              g_ex[connect_to][g_connection[connect_to].index(i)])
        duals = model.addVars(len(gx))
        # duals * gx
        dual_express = gurobi.quicksum(
            duals[i] * gx[i] for i in range(len(gx)))
        norm_express = gurobi.quicksum(
            (duals[i] - self.old_value[i]) * (duals[i] - self.old_value[i])
            for i in range(len(gx)))
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
    instance = Player(
        player_info['index'],
        player_info['demand_ref'],
        player_info['supply_max'],
        player_info['demand_max'],
        player_info['exchange_max'],
        player_info['connection'],
        player_info['supply_a'],
        player_info['supply_b'],
        player_info['demand_a'])
    return instance


def factory():
    player1_info = {
        'index': 0,
        'demand_ref': 15,
        'supply_max': 5,
        'demand_max': 25,
        'exchange_max': 50,
        'connection': [1],
        'supply_a': 0.1,
        'supply_b': 0.01,
        'demand_a': 1,
    }
    player2_info = {
        'index': 1,
        'demand_ref': 20,
        'supply_max': 45,
        'demand_max': 30,
        'exchange_max': 50,
        'connection': [0],
        'supply_a': 0.1,
        'supply_b': 0.01,
        'demand_a': 1,
    }
    player1 = getPlayer(player1_info)
    player2 = getPlayer(player2_info)
    playerN1 = playerNp1()
    return [player1, player2, playerN1]


def norm_ex(list1, list2):
    sum_value = 0
    for i in range(len(list1)):
        sum_value = sum_value + (list1[i] - list2[i]) * (list1[i] - list2[i])
    return sum_value


def calculate_NE():
    global g_lam
    count_best_response = 0
    while count_best_response < 30:
        for i, player in enumerate(g_players):
            # get the data for the player i
            player.update_model(g_tao)  # 填充x_i 以及lam_i
            player_i_result = player.optimize_model()
            g_ex[i] = player_i_result.copy()
            # update the lam_dual variable
            g_lam = g_playerN1.optimize(g_tao)
            # update the response
        count_best_response = count_best_response + 1



def set_oldValue():
    for i, player in enumerate(g_players):
        player.set_old_value(g_ex[i])
    g_playerN1.set_old_value(g_lam)


def start():
    global g_ex
    result_plt = []
    result_plt1 = []
    result_plt2 = []
    # initial
    for player in g_players:
        player.build_model()
    # start the outer loop
    outer_loop_count = 0
    while outer_loop_count < 100:
        # give xn, lam_n, calculate the equilibrium
        calculate_NE()
        # 现在我们得到了一个新的NE，我们应该把这个NE设为参照值
        set_oldValue()
        outer_loop_count = outer_loop_count + 1
        result_plt.append(g_ex[0][0])
        result_plt1.append(g_ex[1][0])
        result_plt2.append(g_ex[0][0] + g_ex[1][0])
        g_ex = [[0], [0]]
    plt.plot(result_plt, label='0->1')
    plt.plot(result_plt1,'.', label='1->0')
    plt.plot(result_plt2,'*', label='difference')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('2-node-elec.svg')

if __name__ == '__main__':
    all_players = factory()
    g_players = all_players[:2]
    g_playerN1 = all_players[2]
    start()