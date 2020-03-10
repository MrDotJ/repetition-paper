import gurobipy as gurobi
import matplotlib.pyplot as plt


class Player:
    def __init__(self, index, demand_ref, supply_max, demand_max, exchange_max,
                 connection_n, supply_a, supply_b, demand_a):
        self.index = index
        self.demand_ref = demand_ref
        self.supply_max = supply_max
        self.demand_max = demand_max
        self.exchange_max = exchange_max
        self.connection_n = connection_n
        self.connection = []
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
        self.old_value = [0] * self.connection_n

    def build_model(self):
        global price
        index = self.index
        objects = []
        constrains = []
        # construct model
        self.supply = self.model.addVar(lb=0, ub=self.supply_max, name='supply' + str(index))
        self.demand = self.model.addVar(lb=0, ub=self.demand_max, name='demand' + str(index))
        self.exchange = self.model.addVars(self.connection_n, lb=-1 * self.exchange_max,
                                           ub=self.exchange_max, name='exchange' + str(index))
        # construct object
        supply_cost = self.supply_a * self.supply + self.supply_b * self.supply * self.supply
        objects.append(supply_cost)
        load_cost = self.demand_a * (self.demand - self.demand_ref) * (self.demand - self.demand_ref)
        objects.append(load_cost)
        exchange_cost = gurobi.quicksum(price[index][i] * self.exchange[i]
                                        for i in range(self.connection_n) if i != index)
        objects.append(exchange_cost)
        # add constrain
        power_balance = self.model.addConstr(
            lhs=self.supply,
            rhs=self.demand + gurobi.quicksum(self.exchange[i]
                                              for i in range(self.connection_n) if i != index),
            sense=gurobi.GRB.EQUAL,
            name='Region_' + str(index) + '_power_balance'
        )
        constrains.append(power_balance)
        # add object
        self.object_basic = sum(objects)

    def update_model(self, ex_i, lam_i, tao):  # 00 01 02 03 04 [10 20 30 40]
        # 10 11 12 13 14 [01 21 31 41]
        # construct dual object
        duals = []
        pos = 0
        for i in range(self.connection_n):  # 2: i => [0 1 ]
            if i != self.index:  # exchange[1]    i=1
                duals.append(self.exchange[i] + ex_i[pos])  # ex_i[1] ex_i[2]
                pos = pos + 1
        dual_addition = sum([a * b for a, b in zip(duals, lam_i)])
        # construct norm object
        norm_addition = gurobi.quicksum((self.exchange[i] - self.old_value[i]) *
                                        (self.exchange[i] - self.old_value[i])
                                        for i in range(self.connection_n) if i != self.index)
        self.object_addition = dual_addition + \
                               tao / 2 * norm_addition
        self.object = self.object_basic + self.object_addition
        self.model.setObjective(self.object)

    def optimize_model(self):  # calculate the response
        self.model.optimize()
        exchange_value = []
        for i in range(self.connection_n):
            if i != self.index:
                exchange_value.append(self.exchange[i].getAttr('X'))
        # [e01 e02]   [e10 e12]   [e20 e21]
        return exchange_value

    def set_old_value(self, old):   # 01 02 03 04
        pos = 0
        for i in range(self.connection_n):  # 0 1 2 3 4
            if i != self.index:
                self.old_value[i] = old[pos]
                pos = pos + 1


class playerNp1:
    def __init__(self):
        self.old_value = [0] * 2

    def optimize(self, exchange, connection, tao):  # [[01 02] [10 12] [20 21]]
        model = gurobi.Model()
        gx = []
        for i in range(len(exchange)):  # 2
            for j in connection[i]:  # [1]
                if j >= i:
                    gx.append(exchange[i][j] + exchange[j][i])
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
        player_info['connection_n'],
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
        'connection_n': 2,
        'supply_a': 0.1,
        'supply_b': 0.1,
        'demand_a': 1,
    }
    player2_info = {
        'index': 1,
        'demand_ref': 20,
        'supply_max': 45,
        'demand_max': 30,
        'exchange_max': 50,
        'connection_n': 2,
        'supply_a': 0.1,
        'supply_b': 0.1,
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


if __name__ == '__main__':
    price = [[0, 1], [1, 0]]
    g_tao = 1000
    g_ex = [0] * 2  # 01 02 10 12 20 21
    g_ex_list = []
    g_lam = [0]   # 01 01 02 02 12 12
    g_connection = [[1], [0]]
    all_players = factory()
    g_players = all_players[:2]
    g_playerN1 = all_players[2]
    for __player in g_players:
        __player.build_model()
    outer_loop_count = 0
    while outer_loop_count < 10:
        # give xn, lam_n, calculate the equilibrium
        count_best_response = 0
        while count_best_response < 100:
            g_ex_old = g_ex
            for __i, __player in enumerate(g_players):
                # get the data for the player i
                x__i = []
                lam__i = []
                if __i == 0:
                    x__i = [0, g_ex[1]]
                    lam__i = [g_lam[0]]
                elif __i == 1:
                    x__i = [g_ex[0], 0]
                    lam__i = [g_lam[0]]
                __player.update_model(x__i, lam__i, g_tao)
                player_i_result = __player.optimize_model()
                # update the lam_dual variable
                g_lam = g_playerN1.optimize([[0, g_ex[0]], [g_ex[1], 0]], g_connection, g_tao)
                # update the response
                g_ex[__i] = player_i_result[0]
            count_best_response = count_best_response + 1
            g_ex_new = g_ex
            g_ex_list.append(norm_ex(g_ex_new, g_ex_old))
        plt.plot(g_ex_list)  # ok
        # now we get the equilibrium
        # update xn, lam_n
        for __i, __player in enumerate(g_players):
            __player.set_old_value([g_ex[__i]])
        g_playerN1.set_old_value(g_lam)
        # re_calculate
        outer_loop_count = outer_loop_count + 1
