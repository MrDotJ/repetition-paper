import gurobipy as gurobi
import matplotlib.pyplot as plt
import numpy as np
import copy

class buildSVAC:
    def __init__(self, temperature, C, R, M, alpha, nu, charge_coefficient, discharge_coefficient, SoC, charge_limit,
                 discharge_limit, charge_cost, discharge_cost, demand_price, supply_price, storage_capacity,
                 storage_initial, build_index, ess_enable, connection_n, exchange_power_limit, cycle, generator):
        ################
        SVAC_json = {
            'build_index': build_index,
            'T_outside': temperature,
            'C': C,
            'R': R,
            'M': M,
            'alpha': alpha,
            'nu': nu,
            'grid_buy_price': demand_price,
            'grid_sell_price': supply_price,
            'connection_n': connection_n,
            'exchange_power_limit': exchange_power_limit,
            'generator' : generator
        }
        SVAC_json.keys()
        self.build_index = build_index
        self.T_outside = temperature
        self.C = C
        self.R = R
        self.M = M
        self.alpha = alpha
        self.nu = nu
        self.demand_price = demand_price
        self.supply_price = supply_price
        self.connection_n = connection_n
        self.exchange_power_limit = exchange_power_limit
        self.cycle = cycle
        #################
        ########################################
        ESS = {
            'ess_enable': ess_enable,
            'charge_coefficient': charge_coefficient,
            'charge_power_limit': charge_limit,
            'charge_cost': charge_cost,
            'discharge_coefficient': discharge_coefficient,
            'discharge_power_limit': discharge_limit,
            'discharge_cost': discharge_cost,
            'storage_capacity': storage_capacity,
            'storage_initial': storage_initial,
            'SoC': SoC
        }
        ESS.keys()
        self.ess_enable = ess_enable
        self.charge_coefficient = charge_coefficient
        self.discharge_coefficient = discharge_coefficient
        self.SoC = SoC
        self.charge_limit = charge_limit
        self.discharge_limit = discharge_limit
        self.storage_initial = storage_initial
        self.storage_capacity = storage_capacity
        self.charge_cost = charge_cost
        self.discharge_cost = discharge_cost
        ###############################################
        ##############################################
        self.generator = generator  # we assume it is known now
        #########################################
        # model
        self.model = gurobi.Model()
        self.charges = [0] * self.cycle
        self.discharges = [0] * self.cycle
        self.storage = [0] * self.cycle
        self.power = None
        self.tempe = None
        self.demand = None
        self.supply = None
        self.exchange_vars = None
        self.object_basic = 0
        self.object_addition = 0
        self.object = 0

    def optimize_base(self):
        global price
        build_index = self.build_index
        ess_enable = self.ess_enable
        objects = []  # list contain every part of object
        constrains = []  # list contain every constrain
        # add to the global model
        # base load
        self.power = self.model.addVars(self.cycle, name="load_" + str(build_index) + "_SVAC")
        self.tempe = self.model.addVars(self.cycle, name="tempe_" + str(build_index) + "_SVAC")
        for i in range(self.cycle):
            # 0---23
            if i == 0:
                expr = self.tempe[0] - self.tempe[self.cycle - 1] + 1 / (self.C * self.R) * \
                       (self.tempe[self.cycle - 1] - self.T_outside[0] + self.nu * self.R * self.power[0])
            else:
                expr = self.tempe[i] - self.tempe[i - 1] + 1 / (self.C * self.R) * \
                       (self.tempe[i - 1] - self.T_outside[i] + self.nu * self.R * self.power[i])
            build_constrain = self.model.addConstr(expr, sense=gurobi.GRB.EQUAL, rhs=0,
                                                   name="tempe-power_" + str(build_index) + '_SVAC')
            constrains.append(build_constrain)
        build_cost = self.M - self.alpha * \
                     gurobi.quicksum((self.tempe[i] - 24) * (self.tempe[i] - 24) for i in range(self.cycle))
        objects.append(-1 * build_cost)
        #############################
        if ess_enable:
            # energy storage system
            self.charges = self.model.addVars(self.cycle, lb=0, ub=self.charge_limit,
                                              name='charges_' + str(build_index) + '_build')
            self.discharges = self.model.addVars(self.cycle, lb=0, ub=self.discharge_limit,
                                                 name='discharges_' + str(build_index) + '_build')
            self.storage = self.model.addVars(self.cycle, lb=(1 - self.SoC) * self.storage_capacity,
                                              ub=self.SoC * self.storage_capacity,
                                              name='storage_' + str(build_index) + '_build')
            for i in range(self.cycle - 1):
                # 0--22
                time = i + 1
                # 1--23
                ess_constrain = self.model.addConstr(lhs=self.storage[time],
                                                     rhs=self.storage[time - 1] +
                                                         self.charge_coefficient * self.charges[time] -
                                                         self.discharge_coefficient * self.discharges[time],
                                                     sense=gurobi.GRB.EQUAL,
                                                     name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                         time) + "_SVAC")
                constrains.append(ess_constrain)
            ess_constrain = self.model.addConstr(lhs=self.storage[0],
                                                 rhs=self.storage_initial * self.storage_capacity +
                                                     self.charge_coefficient * self.charges[0] -
                                                     self.discharge_coefficient * self.discharges[0],
                                                 sense=gurobi.GRB.EQUAL,
                                                 name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                     0) + '_SVAC')
            constrains.append(ess_constrain)
            ess_constrain = self.model.addConstr(lhs=self.storage[self.cycle - 1],
                                                 rhs=self.storage_initial * self.storage_capacity,
                                                 sense=gurobi.GRB.GREATER_EQUAL,
                                                 name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                     self.cycle - 1) + '(addition)' + '_SVAC')
            constrains.append(ess_constrain)
            # depreciation charge
            storage_cost = gurobi.quicksum(self.charges) * self.charge_cost + \
                           gurobi.quicksum(self.discharges) * self.discharge_cost
            objects.append(storage_cost)
        ####################
        # demand and supply with power system
        self.demand = self.model.addVars(self.cycle, name='demand_' + str(build_index))
        self.supply = self.model.addVars(self.cycle, name='supply_' + str(build_index))
        demand_cost = self.demand_price * gurobi.quicksum(self.demand) - \
                      self.supply_price * gurobi.quicksum(self.supply)
        objects.append(demand_cost)
        ############################
        # exchange with other buildings
        exchange_tuple_index = gurobi.tuplelist([('exchange', build_index, i, j)
                                                 for i in range(self.connection_n) for j in range(self.cycle)])
        self.exchange_vars = self.model.addVars(exchange_tuple_index,
                                                lb=-1 * self.exchange_power_limit,
                                                ub=self.exchange_power_limit, name='exchange')
        # g_exchange.append(self.exchange_vars)
        exchange_cost = gurobi.quicksum(self.exchange_vars['exchange', build_index, i, j] * price[build_index][i]
                                        for i in range(self.connection_n) if i != self.build_index
                                        for j in range(self.cycle))
        objects.append(-1 * exchange_cost)
        exchange_constrain = self.model.addConstrs(
            self.exchange_vars['exchange', build_index, i, j] == 0
            for i in range(self.connection_n) if i == build_index
            for j in range(self.cycle))
        constrains.append(exchange_constrain)
        ##############################
        # system balance
        for i in range(self.cycle):
            power_balance = self.model.addConstr(
                lhs=self.power[i] - self.discharges[i] + self.charges[i] - self.generator[i] +
                gurobi.quicksum(self.exchange_vars['exchange', build_index, j, i]
                                    for j in range(self.connection_n) if j != build_index),
                rhs=self.demand[i] - self.supply[i],
                sense=gurobi.GRB.EQUAL,
                name='Region_' + str(build_index) + '_power_Balance')
            constrains.append(power_balance)
        ########################
        # add object
        self.object_basic = sum(objects)
        ###########################

    def update_model(self, aux, dual, tao):
        addition = 0
        for i in range(self.connection_n):
            for j in range(self.cycle):
                if i != self.build_index:
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
        f.write('\n'.join(str(value) for value in load_value))
        f.write('\n')
        result = []
        for i in range(self.connection_n):
            per_build = []
            for j in range(self.cycle):
                per_build.append(self.exchange_vars['exchange', self.build_index, i, j].getAttr('X'))
            result.append(per_build)
        return result
        # [[0,0,0,...],[x,x,x,...],[x,x,x,...],[x,x,x,...]]

    def plot(self):
        fig1 = plt.figure()
        load_value = []
        exchange_list = []
        generator = self.generator
        retail_list = []
        for i in range(self.cycle):
            load_value.append(self.power[i].getAttr('X'))
        for connection in range(self.connection_n):
            if connection != self.build_index:
                temp_list = []
                for time in range(self.cycle):
                    temp_list.append(self.exchange_vars['exchange', self.build_index, connection, time].getAttr('X'))
                exchange_list.append(temp_list)
        for i in range(cycle):
            retail_list.append(self.demand[i].getAttr('X') -
                               self.supply[i].getAttr('X'))
        subplot1 = fig1.add_subplot(1, 1, 1)
        subplot1.plot(load_value, 'k', label='load')
        color_list = ['b--', 'g--', 'r--']
        for i in range(self.connection_n - 1):
            subplot1.plot(exchange_list[i], color_list[i], label='exchange' + str(i))
        subplot1.plot(generator, 'k+', label='generator')
        subplot1.plot(retail_list, 'k-.', label='retail_list')
        subplot1.legend(loc='best')
        subplot1.set_xlabel('Time/h')
        subplot1.set_ylabel('Power')
        subplot1.set_title('build-' + str(self.build_index) + ' Power-Time')
        fig1.savefig('build' + str(self.build_index) + '.svg')

def SVAC(basic_info, storage_system):
    instance = buildSVAC(
        basic_info['T_outside'],
        basic_info['C'],
        basic_info['R'],
        basic_info['M'],
        basic_info['alpha'],
        basic_info['nu'],
        storage_system['charge_coefficient'],
        storage_system['discharge_coefficient'],
        storage_system['SoC'],
        storage_system['charge_power_limit'],
        storage_system['discharge_power_limit'],
        storage_system['charge_cost'],
        storage_system['discharge_cost'],
        basic_info['grid_buy_price'],
        basic_info['grid_sell_price'],
        storage_system['storage_capacity'],
        storage_system['storage_initial'],
        basic_info['build_index'],
        storage_system['ess_enable'],
        basic_info['connection_n'],
        basic_info['exchange_power_limit'],
        basic_info['cycle'],
        basic_info['generator']
    )
    #######################
    # add the build to the model

    return instance


class buildSEA:
    def __init__(self, load_ref, D, load_min, load_max, M, beta, charge_coefficient, discharge_coefficient, SoC,
                 charge_limit, discharge_limit, charge_cost, discharge_cost, demand_price, supply_price,
                 storage_capacity, storage_initial, build_index, ess_enable, connection_n,
                 exchange_power_limit, cycle, generator):
        SEA_json = {
            'build_index': build_index,
            'load_min': load_min,
            'load_max': load_max,
            'load_ref': load_ref,
            'grid_buy_price': demand_price,
            'grid_sell_price': supply_price,
            'M': M,
            'D': D,
            'beta': beta,
            'exchange_power_limit': exchange_power_limit,
            'cycle': cycle,
            'connection_n' : connection_n,
            'generator' : generator
        }
        SEA_json.keys()
        self.connection_n = connection_n
        self.build_index = build_index
        self.D = D
        self.load_min = load_min
        self.load_max = load_max
        self.M = M
        self.beta = beta
        self.load_ref = load_ref
        self.demand_price = demand_price
        self.supply_price = supply_price
        self.exchange_power_limit = exchange_power_limit
        self.cycle = cycle
        ########################################
        ESS = {
            'ess_enable': ess_enable,
            'charge_coefficient': charge_coefficient,
            'charge_power_limit': charge_limit,
            'charge_cost': charge_cost,
            'discharge_coefficient': discharge_coefficient,
            'discharge_power_limit': discharge_limit,
            'discharge_cost': discharge_cost,
            'storage_capacity': storage_capacity,
            'storage_initial': storage_initial,
            'SoC': SoC
        }
        ESS.keys()
        self.charge_coefficient = charge_coefficient
        self.discharge_coefficient = discharge_coefficient
        self.SoC = SoC
        self.charge_limit = charge_limit
        self.discharge_limit = discharge_limit
        self.storage_initial = storage_initial
        self.storage_capacity = storage_capacity
        self.charge_cost = charge_cost
        self.discharge_cost = discharge_cost
        self.ess_enable = ess_enable
        #############################
        self.generator = generator  # we assume it is known now
        self.object_sum = 0
        #############################
        self.model = gurobi.Model()
        ################
        self.charges = [0] * self.cycle
        self.discharges = [0] * self.cycle
        self.storage = [0] * self.cycle
        ################
        self.loads = None
        self.demand = None
        self.supply = None
        self.exchange_vars = None
        self.object_basic = 0
        self.object_addition = 0
        self.object = 0

    def optimize_base(self):
        global price
        build_index = self.build_index
        ess_enable = self.ess_enable
        objects = []  # list contain every part of object
        constrains = []  # list contain every constrain
        # add to the global model
        # base load
        self.loads = self.model.addVars(self.cycle, lb=self.load_min, ub=self.load_max, name='loads')
        load_punish = self.M - gurobi.quicksum((self.loads[i] - self.load_ref[i]) * (self.loads[i] - self.load_ref[i])
                                               for i in range(self.cycle))
        self.model.addConstr(lhs=gurobi.quicksum(self.loads),
                             rhs=self.D,
                             sense=gurobi.GRB.EQUAL,
                             name='powerBalance')

        self.model.setObjective(-1 * load_punish)
        #############################
        if ess_enable:
            # energy storage system
            self.charges = self.model.addVars(self.cycle, lb=0, ub=self.charge_limit,
                                              name='charges_' + str(build_index) + '_build')
            self.discharges = self.model.addVars(self.cycle, lb=0, ub=self.discharge_limit,
                                                 name='discharges_' + str(build_index) + '_build')
            self.storage = self.model.addVars(self.cycle, lb=(1 - self.SoC) * self.storage_capacity,
                                              ub=self.SoC * self.storage_capacity,
                                              name='storage_' + str(build_index) + '_build')
            for i in range(self.cycle - 1):
                # 0--22
                time = i + 1
                # 1--23
                ess_constrain = self.model.addConstr(lhs=self.storage[time],
                                                     rhs=self.storage[time - 1] +
                                                         self.charge_coefficient * self.charges[time] -
                                                         self.discharge_coefficient * self.discharges[time],
                                                     sense=gurobi.GRB.EQUAL,
                                                     name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                         time) + "_SVAC")
                constrains.append(ess_constrain)
            ess_constrain = self.model.addConstr(lhs=self.storage[0],
                                                 rhs=self.storage_initial * self.storage_capacity +
                                                     self.charge_coefficient * self.charges[0] -
                                                     self.discharge_coefficient * self.discharges[0],
                                                 sense=gurobi.GRB.EQUAL,
                                                 name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                     0) + '_SVAC')
            constrains.append(ess_constrain)
            ess_constrain = self.model.addConstr(lhs=self.storage[self.cycle - 1],
                                                 rhs=self.storage_initial * self.storage_capacity,
                                                 sense=gurobi.GRB.GREATER_EQUAL,
                                                 name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                     self.cycle - 1) + '(addition)' + '_SVAC')
            constrains.append(ess_constrain)
            # depreciation charge
            storage_cost = gurobi.quicksum(self.charges) * self.charge_cost + \
                           gurobi.quicksum(self.discharges) * self.discharge_cost
            objects.append(storage_cost)
        ####################
        # demand and supply with power system
        self.demand = self.model.addVars(self.cycle, name='demand_' + str(build_index))
        self.supply = self.model.addVars(self.cycle, name='supply_' + str(build_index))
        demand_cost = self.demand_price * gurobi.quicksum(self.demand) - \
                      self.supply_price * gurobi.quicksum(self.supply)
        objects.append(demand_cost)
        ############################
        # exchange with other buildings
        exchange_tuple_index = gurobi.tuplelist([('exchange', build_index, i, j)
                                                 for i in range(self.connection_n) for j in range(self.cycle)])
        self.exchange_vars = self.model.addVars(exchange_tuple_index,
                                                lb=-1 * self.exchange_power_limit,
                                                ub=self.exchange_power_limit, name='exchange')
        # g_exchange.append(self.exchange_vars)
        exchange_cost = gurobi.quicksum(self.exchange_vars['exchange', build_index, i, j] * price[build_index][i]
                                        for i in range(self.connection_n) if i != self.build_index
                                        for j in range(self.cycle))
        objects.append(-1 * exchange_cost)
        exchange_constrain = self.model.addConstrs(
            self.exchange_vars['exchange', build_index, i, j] == 0
            for i in range(self.connection_n) if i == build_index
            for j in range(self.cycle))
        constrains.append(exchange_constrain)
        ##############################
        # system balance
        for i in range(self.cycle):
            power_balance = self.model.addConstr(
                lhs=self.loads[i] - self.discharges[i] + self.charges[i] - self.generator[i] +
                gurobi.quicksum(self.exchange_vars['exchange', build_index, j, i]
                                for j in range(self.connection_n) if j != build_index),
                rhs=self.demand[i] - self.supply[i],
                sense=gurobi.GRB.EQUAL,
                name='Region_' + str(build_index) + '_power_Balance')
            constrains.append(power_balance)
        ########################
        # add object
        self.object_basic = sum(objects)
        ###########################

    def update_model(self, aux, dual, tao):
        addition = 0
        for i in range(self.connection_n):
            for j in range(self.cycle):
                if i != self.build_index:
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
            load_value.append(self.loads[i].getAttr('X'))
        f.write('\n'.join(str(value) for value in load_value))
        f.write('\n')
        result = []
        for i in range(self.connection_n):
            per_build = []
            for j in range(self.cycle):
                per_build.append(self.exchange_vars['exchange', self.build_index, i, j].getAttr('X'))
            result.append(per_build)
        return result

    def plot(self):
        fig1 = plt.figure()
        load_value = []
        exchange_list = []
        generator = self.generator
        retail_list = []
        for i in range(self.cycle):
            load_value.append(self.loads[i].getAttr('X'))
        for connection in range(self.connection_n):
            if connection != self.build_index:
                temp_list = []
                for time in range(self.cycle):
                    temp_list.append(self.exchange_vars['exchange', self.build_index, connection, time].getAttr('X'))
                exchange_list.append(temp_list)
        for i in range(cycle):
            retail_list.append(self.demand[i].getAttr('X') -
                               self.supply[i].getAttr('X'))
        subplot1 = fig1.add_subplot(1, 1, 1)
        subplot1.plot(load_value, 'k', label='load')
        color_list = ['b--', 'g--', 'r--']
        for i in range(self.connection_n - 1):
            subplot1.plot(exchange_list[i], color_list[i], label='exchange' + str(i))
        subplot1.plot(generator, 'k+', label='generator')
        subplot1.plot(retail_list, 'k-.', label='retail_list')
        subplot1.legend(loc='best')
        subplot1.set_xlabel('Time/h')
        subplot1.set_ylabel('Power')
        subplot1.set_title('build-' + str(self.build_index) + ' Power-Time')
        fig1.savefig('build' + str(self.build_index) + '.svg')

def SEA(basic_info, storage_system):
    instance = buildSEA(
        basic_info['load_ref'],
        basic_info['D'],
        basic_info['load_min'],
        basic_info['load_max'],
        basic_info['M'],
        basic_info['beta'],
        storage_system['charge_coefficient'],
        storage_system['discharge_coefficient'],
        storage_system['SoC'],
        storage_system['charge_power_limit'],
        storage_system['discharge_power_limit'],
        storage_system['charge_cost'],
        storage_system['discharge_cost'],
        basic_info['grid_buy_price'],
        basic_info['grid_sell_price'],
        storage_system['storage_capacity'],
        storage_system['storage_initial'],
        basic_info['build_index'],
        storage_system['ess_enable'],
        basic_info['connection_n'],
        basic_info['exchange_power_limit'],
        basic_info['cycle'],
        basic_info['generator']
    )
    return instance


class buildFCS:
    def __init__(self, lam, load_min, load_max, charge_coefficient, discharge_coefficient, SoC,
                 charge_limit, discharge_limit, charge_cost, discharge_cost, demand_price, supply_price,
                 storage_capacity, storage_initial, build_index, ess_enable, cycle, exchange_power_limit,
                 connection_n, generator):
        FCS_json = {
            'build_index': build_index,
            'lam': lam,
            'load_min': load_min,
            'load_max': load_max,
            'grid_buy_price': demand_price,
            'grid_sell_price': supply_price,
            'connection_n': connection_n,
            'cycle': cycle,
            'exchange_power_limit': exchange_power_limit,
            'generator' : generator
        }
        FCS_json.keys()
        self.connection_n = connection_n
        self.build_index = build_index
        self.lam = lam
        self.load_min = load_min
        self.load_max = load_max
        self.demand_price = demand_price
        self.supply_price = supply_price
        self.cycle = cycle
        self.exchange_power_limit = exchange_power_limit
        ########################################
        ESS = {
            'ess_enable': ess_enable,
            'charge_coefficient': charge_coefficient,
            'charge_power_limit': charge_limit,
            'charge_cost': charge_cost,
            'discharge_coefficient': discharge_coefficient,
            'discharge_power_limit': discharge_limit,
            'discharge_cost': discharge_cost,
            'storage_capacity': storage_capacity,
            'storage_initial': storage_initial,
            'SoC': SoC
        }
        ESS.keys()
        self.charge_coefficient = charge_coefficient
        self.discharge_coefficient = discharge_coefficient
        self.SoC = SoC
        self.charge_limit = charge_limit
        self.discharge_limit = discharge_limit
        self.storage_initial = storage_initial
        self.storage_capacity = storage_capacity
        self.charge_cost = charge_cost
        self.discharge_cost = discharge_cost
        self.ess_enable = ess_enable
        #############################
        self.generator = generator  # we assume it is known now
        self.object_sum = 0
        ############################
        #############################
        self.model = gurobi.Model()
        ################
        self.charges = [0] * self.cycle
        self.discharges = [0] * self.cycle
        self.storage = [0] * self.cycle
        ################
        self.loads = None
        self.demand = None
        self.supply = None
        self.exchange_vars = None
        self.object_basic = 0
        self.object_addition = 0
        self.object = 0

    def optimize_base(self):
        global price
        build_index = self.build_index
        ess_enable = self.ess_enable
        objects = []  # list contain every part of object
        constrains = []  # list contain every constrain
        # add to the global model
        # base load
        # base load
        self.loads = self.model.addVars(self.cycle, lb=self.load_min, ub=self.load_max, name='load')
        load_cost = self.lam * (gurobi.quicksum(self.loads) )#-
                                # 1 / 2 * gurobi.quicksum(self.loads[i] * self.loads[i] for i in range(self.cycle)))
        objects.append(-1 * load_cost)
        #############################
        if ess_enable:
            # energy storage system
            self.charges = self.model.addVars(self.cycle, lb=0, ub=self.charge_limit,
                                              name='charges_' + str(build_index) + '_build')
            self.discharges = self.model.addVars(self.cycle, lb=0, ub=self.discharge_limit,
                                                 name='discharges_' + str(build_index) + '_build')
            self.storage = self.model.addVars(self.cycle, lb=(1 - self.SoC) * self.storage_capacity,
                                              ub=self.SoC * self.storage_capacity,
                                              name='storage_' + str(build_index) + '_build')
            for i in range(self.cycle - 1):
                # 0--22
                time = i + 1
                # 1--23
                ess_constrain = self.model.addConstr(lhs=self.storage[time],
                                                     rhs=self.storage[time - 1] +
                                                         self.charge_coefficient * self.charges[time] -
                                                         self.discharge_coefficient * self.discharges[time],
                                                     sense=gurobi.GRB.EQUAL,
                                                     name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                         time) + "_SVAC")
                constrains.append(ess_constrain)
            ess_constrain = self.model.addConstr(lhs=self.storage[0],
                                                 rhs=self.storage_initial * self.storage_capacity +
                                                     self.charge_coefficient * self.charges[0] -
                                                     self.discharge_coefficient * self.discharges[0],
                                                 sense=gurobi.GRB.EQUAL,
                                                 name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                     0) + '_SVAC')
            constrains.append(ess_constrain)
            ess_constrain = self.model.addConstr(lhs=self.storage[self.cycle - 1],
                                                 rhs=self.storage_initial * self.storage_capacity,
                                                 sense=gurobi.GRB.GREATER_EQUAL,
                                                 name='energyStorage_build_' + str(build_index) + '_time_' + str(
                                                     self.cycle - 1) + '(addition)' + '_SVAC')
            constrains.append(ess_constrain)
            # depreciation charge
            storage_cost = gurobi.quicksum(self.charges) * self.charge_cost + \
                           gurobi.quicksum(self.discharges) * self.discharge_cost
            objects.append(storage_cost)
        ####################
        # demand and supply with power system
        self.demand = self.model.addVars(self.cycle, name='demand_' + str(build_index))
        self.supply = self.model.addVars(self.cycle, name='supply_' + str(build_index))
        demand_cost = self.demand_price * gurobi.quicksum(self.demand) - \
                      self.supply_price * gurobi.quicksum(self.supply)
        objects.append(demand_cost)
        ############################
        # exchange with other buildings
        exchange_tuple_index = gurobi.tuplelist([('exchange', build_index, i, j)
                                                 for i in range(self.connection_n) for j in range(self.cycle)])
        self.exchange_vars = self.model.addVars(exchange_tuple_index,
                                                lb=-1 * self.exchange_power_limit,
                                                ub=self.exchange_power_limit, name='exchange')
        # g_exchange.append(self.exchange_vars)
        exchange_cost = gurobi.quicksum(self.exchange_vars['exchange', build_index, i, j] * price[build_index][i]
                                        for i in range(self.connection_n) if i != self.build_index
                                        for j in range(self.cycle))
        objects.append(-1 * exchange_cost)
        exchange_constrain = self.model.addConstrs(
            self.exchange_vars['exchange', build_index, i, j] == 0
            for i in range(self.connection_n) if i == build_index
            for j in range(self.cycle))
        constrains.append(exchange_constrain)
        ##############################
        # system balance
        for i in range(self.cycle):
            power_balance = self.model.addConstr(
                lhs=self.loads[i] - self.discharges[i] + self.charges[i] - self.generator[i] +
                gurobi.quicksum(self.exchange_vars['exchange', build_index, j, i]
                                    for j in range(self.connection_n) if j != build_index),
                rhs=self.demand[i] - self.supply[i],
                sense=gurobi.GRB.EQUAL,
                name='Region_' + str(build_index) + '_power_Balance')
            constrains.append(power_balance)
        ########################
        # add object
        self.object_basic = sum(objects)
        ###########################

    def update_model(self, aux, dual, tao):
        addition = 0
        for i in range(self.connection_n):
            for j in range(self.cycle):
                if i != self.build_index:
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
        f.write('\nobj ' + str(self.model.getAttr('ObjVal')) + '\n')
        load_value = []
        for i in range(self.cycle):
            load_value.append(self.loads[i].getAttr('X'))
        f.write('\n'.join(str(value) for value in load_value))
        f.write('\n')
        result = []
        for i in range(self.connection_n):
            per_build = []
            for j in range(self.cycle):
                per_build.append(self.exchange_vars['exchange', self.build_index, i, j].getAttr('X'))
            result.append(per_build)
        return result

    def plot(self):
        fig1 = plt.figure()
        load_value = []
        exchange_list = []
        generator = self.generator
        retail_list = []
        for i in range(self.cycle):
            load_value.append(self.loads[i].getAttr('X'))
        for connection in range(self.connection_n):
            if connection != self.build_index:
                temp_list = []
                for time in range(self.cycle):
                    temp_list.append(self.exchange_vars['exchange', self.build_index, connection, time].getAttr('X'))
                exchange_list.append(temp_list)
        for i in range(cycle):
            retail_list.append(self.demand[i].getAttr('X') -
                               self.supply[i].getAttr('X'))
        subplot1 = fig1.add_subplot(1, 1, 1)
        subplot1.plot(load_value, 'k', label='load')
        color_list = ['b--', 'g--', 'r--']
        for i in range(self.connection_n - 1):
            subplot1.plot(exchange_list[i], color_list[i], label='exchange' + str(i))
        subplot1.plot(generator, 'k+', label='generator')
        subplot1.plot(retail_list, 'k-.', label='retail_list')
        subplot1.legend(loc='best')
        subplot1.set_xlabel('Time/h')
        subplot1.set_ylabel('Power')
        subplot1.set_title('build-' + str(self.build_index) + ' Power-Time')
        fig1.savefig('build' + str(self.build_index) + '.svg')

def FCS(basic_info, storage_system):
    instance = buildFCS(
        basic_info['lam'],
        basic_info['load_min'],
        basic_info['load_max'],
        storage_system['charge_coefficient'],
        storage_system['discharge_coefficient'],
        storage_system['SoC'],
        storage_system['charge_power_limit'],
        storage_system['discharge_power_limit'],
        storage_system['charge_cost'],
        storage_system['discharge_cost'],
        basic_info['grid_buy_price'],
        basic_info['grid_sell_price'],
        storage_system['storage_capacity'],
        storage_system['storage_initial'],
        basic_info['build_index'],
        storage_system['ess_enable'],
        basic_info['cycle'],
        basic_info['exchange_power_limit'],
        basic_info['connection_n'],
        basic_info['generator']
    )
    return instance


def factory():
    svac_info_1 = {
        'build_index': 0,
        'T_outside': [26, 26, 28, 29, 30, 31, 32, 32, 33],
        'C': 3.3,
        'R': 1.35,
        'M': 50,
        'alpha': 1,
        'nu': 0.185,
        'grid_buy_price': 0.25,
        'grid_sell_price': 0.1,
        'connection_n': 4,
        'exchange_power_limit': 100,
        'cycle': 9,
        'generator' : [3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.5, 3.0]
    }
    ess_info_1 = {
        'ess_enable': 1,
        'charge_coefficient': 0.94,
        'charge_power_limit': 80,
        'charge_cost': 0.01,
        'discharge_coefficient': 1.06,
        'discharge_power_limit': 80,
        'discharge_cost': 0.01,
        'storage_capacity': 340,
        'storage_initial': 0.17647,
        'SoC': 0.88235
    }
    ess_info_2 = {
        'ess_enable': 0,
        'charge_coefficient': 0.94,
        'charge_power_limit': 80,
        'charge_cost': 0.01,
        'discharge_coefficient': 1.06,
        'discharge_power_limit': 80,
        'discharge_cost': 0.01,
        'storage_capacity': 340,
        'storage_initial': 0.17647,
        'SoC': 0.88235
    }
    sea_info_1 = {
        'build_index': 1,
        'load_min': 10,
        'load_max': 40,
        'load_ref': [20, 30, 20, 30, 20, 30, 10, 20, 20],
        'grid_buy_price': 0.25,
        'grid_sell_price': 0.1,
        'M': 50,
        'D': 200,
        'beta': 0.1,
        'exchange_power_limit': 100,
        'cycle': 9,
        'generator' : [4.2, 6.0, 7.0, 7.5, 7.8, 7.5, 6.0, 4.0, 2.0],
        'connection_n' : 4
    }
    sea_info_2 = {
        'build_index': 2,
        'load_min': 10,
        'load_max': 40,
        'load_ref': [20, 40, 30, 40, 30, 30, 10, 20, 20],
        'grid_buy_price': 0.25,
        'grid_sell_price': 0.1,
        'M': 50,
        'D': 240,
        'beta': 0.1,
        'exchange_power_limit': 100,
        'cycle': 9,
        'connection_n' : 4,
        'generator': [5.2, 7.0, 8.3, 9.0, 9.2, 8.2, 7.6, 5.4, 3.0]
    }
    fcs_info_3 = {
        'build_index': 3,
        'lam': 3,
        'load_min': 15,
        'load_max': 35,
        'grid_buy_price': 0.25,
        'grid_sell_price': 0.1,
        'connection_n': 4,
        'cycle': 9,
        'exchange_power_limit': 100,
        'generator': [60, 80, 100 , 110, 112, 100, 90, 60, 40]
    }
    svac1 = SVAC(svac_info_1, ess_info_1)
    sea1 = SEA(sea_info_1, ess_info_2)
    sea2 = SEA(sea_info_2, ess_info_2)
    fcs1 = FCS(fcs_info_3, ess_info_2)
    return [svac1, sea1, sea2, fcs1]


def update_aux_variable(exchange, dual, tao, cycle, connection):
    model = gurobi.Model()
    aux = model.addVars(connection, connection, cycle, lb=-1 * gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
    obj = gurobi.quicksum((exchange[i][j][h] - aux[i, j, h] + dual[i][j][h] / tao) *
                          (exchange[i][j][h] - aux[i, j, h] + dual[i][j][h] / tao)
                          for i in range(connection)
                          for j in range(connection) if j != i
                          for h in range(cycle))
    model.addConstrs(aux[i, j, h] + aux[j, i, h] == 0
                     for i in range(connection)
                     for j in range(connection) if j != i
                     for h in range(cycle))
    model.setObjective(tao / 2 * obj)
    model.optimize()
    result = [[[aux[i, j, h].getAttr('X') for h in range(cycle)] for j in range(connection)] for i in range(connection)]
    return result


def update_dual_variable(dual, tao, aux, exchange, cycle, connection):
    for i in range(connection):
        for j in range(connection):
            for h in range(cycle):
                dual[i][j][h] += tao * (exchange[i][j][h] - aux[i][j][h])


if __name__ == '__main__':
    f = open("test.txt", 'a')
    connection_n = 4
    cycle = 9
    price = [[0.08, 0.08, 0.08, 0.08],
             [0.08, 0.08, 0.08, 0.08],
             [0.08, 0.08, 0.08, 0.08],
             [0.08, 0.08, 0.08, 0.08]]
    price = [[0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
    g_exchange = [[[0 for i in range(9)] for j in range(4)] for h in range(4)]
    dual = [[[0 for i in range(9)] for j in range(4)] for h in range(4)]
    exchange_old = [[[0 for i in range(9)] for j in range(4)] for h in range(4)]
    tao = 0.03
    aux = [[[0 for i in range(9)] for j in range(4)] for h in range(4)]
    terminals = []
    builds = factory()
    for build in builds:
        build.optimize_base()
    iterator_count = 1
    while iterator_count < 20:
        f.write('\n\n\niterator' + str(iterator_count) + '\n')
        iterator_count = iterator_count + 1
        # S1
        for i, build in enumerate(builds):
            build.update_model(aux, dual, tao)
            exchange = build.optimize_model(f)
            g_exchange[i] = exchange
        f.write('\n exchange: \n')
        f.write("\n".join(str(item) for item in g_exchange))
        # S2
        aux = update_aux_variable(g_exchange, dual, tao, cycle, connection_n)
        # S3
        update_dual_variable(dual, tao, aux, g_exchange, cycle, connection_n)
        f.write('\n aux: \n')
        f.write("\n".join(str(item) for item in aux))
        f.write('\n dual: \n')
        f.write("\n".join(str(item) for item in dual))
        f.write('\n')
        normal = 0
        for i, row in enumerate(g_exchange) :
            for j, col in enumerate(row):
                for h, value in enumerate(col):
                    normal += (g_exchange[i][j][h] - exchange_old[i][j][h]) *\
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

    print(terminals)