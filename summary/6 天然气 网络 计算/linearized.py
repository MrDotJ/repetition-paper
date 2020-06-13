import gurobipy as gurobi
from gurobipy import abs_

model = gurobi.Model()

x = model.addVar(lb=-5, ub=5)
y = model.addVar(lb=-5, ub=5)
z = model.addVar()

model.addConstr(x*x + y*y == z*z + 10)
model.setObjective(x*x)

model.Params.Nonconvex = 2
model.optimize()
