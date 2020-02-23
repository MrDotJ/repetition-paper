import gurobipy as gurobi
import matplotlib.pyplot as plt
import copy


class baseBlock:
    def __init__(self, build_index, cycle, power_reference, demand_price, supply_price, connection, node_count,
                 PV_output, chp_MIN, chp_MAX, gas_price, chp_convert_to_power, exchange_power_limit):
        self.build_index = build_index
        self.cycle = cycle
        self.power_reference = power_reference
        self.demand_price = demand_price
        self.supply_price = supply_price
        self.connection = connection
        self.node_count = node_count
        self.exchange_power_limit = exchange_power_limit
        # about generator
        self.PV = PV_output
        self.chp_MIN = chp_MIN
        self.chp_MAX = chp_MAX
        self.gas_price = gas_price
        self.chp_convert_to_power = chp_convert_to_power
        # variable about model
        self.model = gurobi.Model()
        self.power = None
        self.demand = None
        self.supply = None
        self.exchange_vars = None
        self.gas_consume_chp = None
        self.object = None
        self.object_basic = None
        self.object_addition = None

    def optimize_base(self, price):
        objects = []
        build_index = self.build_index
        # base load
        self.power = self.model.addVars(self.cycle, name='load_' + str(build_index))
        load_cost = gurobi.quicksum(
            (self.power[i] - self.power_reference[i]) * (self.power[i] - self.power_reference[i])
            for i in range(self.cycle))
        objects.append(load_cost)

        # demand and supply
        self.demand = self.model.addVars(self.cycle, name='demand_' + str(build_index))
        self.supply = self.model.addVars(self.cycle, name='supply_' + str(build_index))
        demand_cost = self.demand_price * gurobi.quicksum(self.demand) - \
                      self.supply_price * gurobi.quicksum(self.supply)
        objects.append(demand_cost)  # note that we dont add constrain for the power exchange

        # exchange with each other
        exchange_tuple_index = gurobi.tuplelist([('exchange', build_index, i, j)
                                                 for i in self.connection for j in range(self.cycle)])
        self.exchange_vars = self.model.addVars(exchange_tuple_index,
                                                lb=-1 * self.exchange_power_limit,
                                                ub=self.exchange_power_limit, name='exchange')
        exchange_cost = gurobi.quicksum(self.exchange_vars['exchange', build_index, i, j] * price[build_index][i]
                                        for i in self.connection
                                        for j in range(self.cycle))
        objects.append(-1 * exchange_cost)

        # CHP generator
        # if the chp generate too much power, it has to sell it to the power system, the money earned will
        # be influenced by the price of the power price: BASED ON GRID PRICE
        self.gas_consume_chp = self.model.addVars(self.cycle, lb=self.chp_MIN, ub=self.chp_MAX,
                                                  name='chp_consume_' + str(self.build_index))
        chp_cost = self.gas_price * gurobi.quicksum(self.gas_consume_chp)
        objects.append(chp_cost)

        # system balance
        for i in range(self.cycle):
            self.model.addConstr(
                lhs=self.power[i] - self.PV[i] - self.gas_consume_chp[i] * self.chp_convert_to_power +
                gurobi.quicksum(self.exchange_vars['exchange', build_index, j, i] for j in self.connection),
                rhs=self.demand[i] - self.supply[i],
                sense=gurobi.GRB.EQUAL,
                name='Region_' + str(build_index) + '_power_Balance')
        ########################
        # add object
        self.object_basic = sum(objects)
        ###########################

    def update_model(self, aux, dual, tao):
        addition = 0
        for i in self.connection:
            for j in range(self.cycle):
                addition += ((self.exchange_vars['exchange', self.build_index, i, j] -
                              aux[self.build_index][i][j] +
                              dual[self.build_index][i][j] / tao) *
                             (self.exchange_vars['exchange', self.build_index, i, j] -
                              aux[self.build_index][i][j] +
                              dual[self.build_index][i][j] / tao))
        self.object_addition = tao / 2 * addition
        self.object = self.object_basic + self.object_addition
        self.model.setObjective(self.object)

    def optimize_model(self, f):
        self.model.optimize()
        f.write('\nobj: ' + str(self.model.getAttr('ObjVal')) + '\n')
        load_value = []
        for i in range(self.cycle):
            load_value.append(self.power[i].getAttr('X'))
        # f.write('load value \n')
        # f.write('\n'.join(str(load_value)))
        # f.write('\n')
        result = []
        for i in range(self.node_count):
            per_build = []
            if i in self.connection:
                for j in range(self.cycle):
                    per_build.append(self.exchange_vars['exchange', self.build_index, i, j].getAttr('X'))
            else:
                per_build = [0] * self.cycle
            result.append(per_build)
        print('##############\n')
        print(result)
        print('##############\n')
        return result
        # [[0,0,0,...],[x,x,x,...],[x,x,x,...],[x,x,x,...]]

    def plot(self):
        fig1 = plt.figure()
        load_value = []
        exchange_list = []
        generator = self.PV
        retail_list = []
        # make information
        for i in range(self.cycle):
            load_value.append(self.power[i].getAttr('X'))
        for build in self.connection:
            temp_list = []
            for time in range(self.cycle):
                temp_list.append(self.exchange_vars['exchange', self.build_index, build, time].getAttr('X'))
            exchange_list.append(temp_list)
        for i in range(self.cycle):
            retail_list.append(self.demand[i].getAttr('X') -
                               self.supply[i].getAttr('X'))
        # plot
        subplot1 = fig1.add_subplot(1, 1, 1)
        # plot load var
        subplot1.plot(load_value, 'k', label='load')
        subplot1.plot(self.power_reference, 'k*', label='load_ref')
        # plot exchange var
        color_list = ['b--', 'g--', 'r--']
        for i in range(len(self.connection)):
            subplot1.plot(exchange_list[i], color_list[i], label='exchange' + str(self.connection[i] + 1))
        # plot generator
        subplot1.plot(generator, 'k+', label='generator')
        # plot retail
        subplot1.plot(retail_list, 'k-.', label='retail_list')
        # plot additional information
        subplot1.legend(loc='best')
        subplot1.set_xlabel('Time/h')
        subplot1.set_ylabel('Power')
        subplot1.set_title('build-' + str(self.build_index + 1) + ' Power-Time')
        fig1.savefig('build' + str(self.build_index + 1) + '.svg')


def update_aux_variable(exchange, dual, tao, cycle, node_count, connection_info):
    model = gurobi.Model()
    aux = model.addVars(node_count, node_count, cycle, lb=-1 * gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
    obj = gurobi.quicksum((exchange[i][j][h] - aux[i, j, h] + dual[i][j][h] / tao) *
                          (exchange[i][j][h] - aux[i, j, h] + dual[i][j][h] / tao)
                          for i in range(node_count)
                          for j in connection_info[i]
                          for h in range(cycle))
    model.addConstrs(aux[i, j, h] + aux[j, i, h] <= 0.01
                     for i in range(node_count)
                     for j in connection_info[i]
                     for h in range(cycle))
    model.addConstrs(aux[i, j, h] + aux[j, i, h] >= -0.01
                     for i in range(node_count)
                     for j in connection_info[i]
                     for h in range(cycle))
    model.setObjective(tao / 2 * obj)
    model.optimize()
    result = [[[0] * cycle] * node_count] * node_count
    for i in range(node_count):
        for j in connection_info[i]:
            for h in range(cycle):
                result[i][j][h] += aux[i, j, h].getAttr('X')
    return result


def update_dual_variable(dual, tao, aux, exchange, cycle, connection, connection_info):
    for i in range(connection):
        for j in connection_info[i]:
            if j > i:
                for h in range(cycle):
                    dual[i][j][h] += tao * (exchange[i][j][h] - aux[i][j][h])


# build_index, cycle,  power_reference, demand_price, supply_price, connection, node_count, \
#               PV_output, chp_MIN, chp_MAX, gas_price, chp_convert_to_power
def BASIC(basic_info, gener_info):
    instance = baseBlock(
        basic_info['build_index'],
        basic_info['cycle'],
        basic_info['power_reference'],
        basic_info['grid_buy_price'],
        basic_info['grid_sell_price'],
        basic_info['connection'],
        basic_info['node_count'],

        gener_info['PV_output'],
        gener_info['chp_MIN'],
        gener_info['chp_MAX'],
        gener_info['gas_price'],
        gener_info['chp_convert_to_power'],

        basic_info['exchange_power_limit'],
    )
    return instance


def factory():
    build_1_basic_info = {
        'build_index': 0,
        'power_reference': [26, 26, 28, 29, 30, 31, 32, 32, 33],
        'grid_buy_price': 0.55,
        'grid_sell_price': 0.01,  # change we can invest the response-demand
        'node_count': 5,
        'connection': [1],
        'exchange_power_limit': 100,
        'cycle': 9,
    }
    build_1_gener_info = {
        'PV_output': [3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.5, 3.0],
        'chp_MIN': 0,  # 5,
        'chp_MAX': 0,  # 15,
        'gas_price': 0.1,
        'chp_convert_to_power': 1
    }
    build_2_basic_info = {
        'build_index': 1,
        'power_reference': [20, 20, 20, 20, 20, 20, 20, 20, 20],
        'grid_buy_price': 0.55,
        'grid_sell_price': 0.01,  # change we can invest the response-demand
        'node_count': 5,
        'connection': [0, 2, 3],
        'exchange_power_limit': 100,
        'cycle': 9,
    }
    build_2_gener_info = {
        'PV_output': [20, 40, 55, 60, 70, 50, 40, 35, 30],
        'chp_MIN': 0,
        'chp_MAX': 0,
        'gas_price': 0.1,
        'chp_convert_to_power': 1
    }
    build_3_basic_info = {
        'build_index': 2,
        'power_reference': [15, 17, 18, 18, 17, 19, 16, 15, 15],
        'grid_buy_price': 0.55,
        'grid_sell_price': 0.01,  # change we can invest the response-demand
        'node_count': 5,
        'connection': [1, 4],
        'exchange_power_limit': 100,
        'cycle': 9,
    }
    build_3_gener_info = {
        'PV_output': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'chp_MIN': 0,
        'chp_MAX': 0,  # 30,
        'gas_price': 0.1,
        'chp_convert_to_power': 1
    }
    build_4_basic_info = {
        'build_index': 3,
        'power_reference': [26, 26, 28, 29, 30, 31, 32, 32, 33],
        'grid_buy_price': 0.55,
        'grid_sell_price': 0.01,  # change we can invest the response-demand
        'node_count': 5,
        'connection': [1],
        'exchange_power_limit': 100,
        'cycle': 9,
    }
    build_4_gener_info = {
        'PV_output': [6.0, 7.0, 10.0, 9.0, 10.0, 8.0, 7.0, 6.0, 6.0],
        'chp_MIN': 0,  # 5,
        'chp_MAX': 0,  # 15,
        'gas_price': 0.1,
        'chp_convert_to_power': 1
    }
    build_5_basic_info = {
        'build_index': 4,
        'power_reference': [20, 18, 16, 13, 15, 14, 17, 18, 20],
        'grid_buy_price': 0.55,
        'grid_sell_price': 0.01,  # change we can invest the response-demand
        'node_count': 5,
        'connection': [2],
        'exchange_power_limit': 100,
        'cycle': 9,
    }
    build_5_gener_info = {
        'PV_output': [2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 3.0],
        'chp_MIN': 0,
        'chp_MAX': 0,
        'gas_price': 0.1,
        'chp_convert_to_power': 1
    }
    build1 = BASIC(build_1_basic_info, build_1_gener_info)
    build2 = BASIC(build_2_basic_info, build_2_gener_info)
    build3 = BASIC(build_3_basic_info, build_3_gener_info)
    build4 = BASIC(build_4_basic_info, build_4_gener_info)
    build5 = BASIC(build_5_basic_info, build_5_gener_info)
    return [build1, build2, build3, build4, build5]


def start_go():
    f = open("test.txt", 'a')
    node_count = 5
    cycle = 9
    connection_info = [
        [1],
        [0, 2, 3],
        [1, 4],
        [1],
        [2]
    ]
 #   '''
    price = [[0.2, 0.2, 0.2, 0.2, 0.2],
             [0.2, 0.2, 0.2, 0.2, 0.2],
             [0.2, 0.2, 0.2, 0.2, 0.2],
             [0.2, 0.2, 0.2, 0.2, 0.2],
             [0.2, 0.2, 0.2, 0.2, 0.2]]
    '''
    price = [[0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]]'''
    # 按照福利最大来说，应该是所有的本地机组满发，社会福利最大，成本最小
    # 按照竞争均衡，可以淘汰低效机组，促进高效机组？
    g_exchange = [[[0 for _ in range(cycle)] for _ in range(node_count)] for _ in range(node_count)]
    dual = [[[0 for _ in range(cycle)] for _ in range(node_count)] for _ in range(node_count)]
    exchange_old = [[[0 for _ in range(cycle)] for _ in range(node_count)] for _ in range(node_count)]
    tao = 0.03
    aux = [[[0 for _ in range(cycle)] for _ in range(node_count)] for _ in range(node_count)]
    terminals = []
    builds = factory()
    for build in builds:
        build.optimize_base(price)
    iterator_count = 1
    f.write('optimize start....')
    while iterator_count < 80:
        f.write('\n\n\niterator' + str(iterator_count) + '\n')
        iterator_count = iterator_count + 1
        # S1
        for i, build in enumerate(builds):
            build.update_model(aux, dual, tao)
            exchange = build.optimize_model(f)
            g_exchange[i] = exchange  # update exchange
        f.write('\n exchange: \n')
        f.write("\n".join(str(item) for item in g_exchange))
        # S2                                # update aux
        aux = update_aux_variable(g_exchange, dual, tao, cycle, node_count, connection_info)
        # S3                                # update dual
        update_dual_variable(dual, tao, aux, g_exchange, cycle, node_count, connection_info)

        f.write('\n aux: \n')
        f.write("\n".join(str(item) for item in aux))
        f.write('\n dual: \n')
        f.write("\n".join(str(item) for item in dual))
        f.write('\n')
        normal = 0
        for i, page in enumerate(g_exchange):
            for j, col in enumerate(page):
                for h, value in enumerate(col):
                    normal += (g_exchange[i][j][h] - exchange_old[i][j][h]) * \
                              (g_exchange[i][j][h] - exchange_old[i][j][h])
        terminals.append(normal)
        exchange_old = copy.deepcopy(g_exchange)

    for build in builds:
        build.plot()
    f.close()
    fig2 = plt.figure()
    subplot2 = fig2.add_subplot(1, 1, 1)
    subplot2.plot(terminals)
    subplot2.set_title('norm(difference of exchange per iterator)')
    subplot2.set_xlabel('iterator')
    subplot2.set_ylabel('difference')
    fig2.savefig('difference''build' + '.svg')
    plt.show()


if __name__ == '__main__':
    start_go()
