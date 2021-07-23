import math


class Function:
    def __init__(self, normal, derived):
        self.normal = normal
        self.derived = derived


# ACTIVATION FUNCTIONS
step_act_f = Function(lambda x: (0 if x <= 0 else 1),
                      lambda x: 0)

sigmoid_act_f = Function(lambda x: 1 / (1 + math.e**-x),
                         lambda x: math.e**-x / (1 + math.e**-x)**2)


# COST FUNCTIONS
linear_cost_f = Function(lambda expected, res: expected - res,
                         lambda expected, res: 1)

mean_squared_error_cost_f = Function(lambda expected, res: (expected - res)**2,
                                     lambda expected, res: 2 * (expected - res))
