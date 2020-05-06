
X_raw_0 = [
    [0, 0.1, 0, 0, 0],
    [0.1, 0, 0.1, 0, 0.1],
    [0, 0.1, 0, 0.1, 0],
    [0, 0, 0.1, 0, 0],
    [0, 0.1, 0, 0, 0]
]
node_info_0 = [
    {
        'min_load': 0,
        'max_load': 30,
        'min_power': 0,
        'max_power': 15,
        'load_coeff': 10,
        'load_ref': 20,
        'power_coeff_a': 0.1,
        'power_coeff_b': 2.5,
        'power_coeff_c': 0,
        'gen_ramp_up': 50,
        'gen_ramp_down': 50
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 0,
        'load_ref': 0,
        'power_coeff_a': 1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 0,
        'load_ref': 1,
        'power_coeff_a': 1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 10.1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 10.1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },

]  # 9 - 13 - 11      5 - 10
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
    [0, 0.1, 0, 0, 0],
    [0.1, 0, 0.1, 0, 0.1],
    [0, 0.1, 0, 0.1, 0],
    [0, 0, 0.1, 0, 0],
    [0, 0.1, 0, 0, 0]
]
node_info_1 = [
    {
        'min_load': 0,
        'max_load': 25,
        'min_power': 0,
        'max_power': 40,
        'load_coeff': 10,
        'load_ref': 20,
        'power_coeff_a': 0.1,
        'power_coeff_b': 2,
        'power_coeff_c': 0,
        'gen_ramp_up': 50,
        'gen_ramp_down': 50
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 4,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 2,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 1,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
]  # 8 - 10 - 9      4 - 19
connection_info_1 = {
    'connection_index': [0, 2, 3],
    'connection_x': [0.1, 0.1, 0.1],
    'connection_area': [0, 2, 3],
    'connection_exchange_max': [100, 100, 100]
}
player1_info = {
    'index': 1,
    'X_raw': X_raw_1,
    'node_info': node_info_1,
    'connection_info': connection_info_1,
}

X_raw_2 = [
    [0, 0.1, 0, 0, 0],
    [0.1, 0, 0.1, 0, 0.1],
    [0, 0.1, 0, 0.1, 0],
    [0, 0, 0.1, 0, 0],
    [0, 0.1, 0, 0, 0]
]
node_info_2 = [
    {
        'min_load': 0,
        'max_load': 25,
        'min_power': 0,
        'max_power': 20,
        'load_coeff': 5,
        'load_ref': 15,
        'power_coeff_a': 0.1,
        'power_coeff_b': 3,
        'power_coeff_c': 0,
        'gen_ramp_up': 50,
        'gen_ramp_down': 50
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 4,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 3,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },

]  # 9 - 13 - 11      5 - 8
connection_info_2 = {
    'connection_index': [1, 4],
    'connection_x': [0.1, 0.1],
    'connection_area': [1, 4],
    'connection_exchange_max': [100, 100]
}
player2_info = {
    'index': 2,
    'X_raw': X_raw_2,
    'node_info': node_info_2,
    'connection_info': connection_info_2
}

X_raw_3 = [
    [0, 0.1, 0, 0, 0],
    [0.1, 0, 0.1, 0, 0.1],
    [0, 0.1, 0, 0.1, 0],
    [0, 0, 0.1, 0, 0],
    [0, 0.1, 0, 0, 0]
]
node_info_3 = [
    {
        'min_load': 0,
        'max_load': 25,
        'min_power': 0,
        'max_power': 10,
        'load_coeff': 5,
        'load_ref': 20,
        'power_coeff_a': 0.1,
        'power_coeff_b': 2,
        'power_coeff_c': 0,
        'gen_ramp_up': 50,
        'gen_ramp_down': 50
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 4,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 3,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },

]  # 9 - 13 - 11      5 - 8
connection_info_3 = {
    'connection_index': [1],
    'connection_x': [0.1],
    'connection_area': [1],
    'connection_exchange_max': [100]
}
player3_info = {
    'index': 3,
    'X_raw': X_raw_3,
    'node_info': node_info_3,
    'connection_info': connection_info_3
}

X_raw_4 = [
    [0, 0.1, 0, 0, 0],
    [0.1, 0, 0.1, 0, 0.1],
    [0, 0.1, 0, 0.1, 0],
    [0, 0, 0.1, 0, 0],
    [0, 0.1, 0, 0, 0]
]
node_info_4 = [
    {
        'min_load': 0,
        'max_load': 25,
        'min_power': 0,
        'max_power': 10,
        'load_coeff': 5,
        'load_ref': 20,
        'power_coeff_a': 0.1,
        'power_coeff_b': 3,
        'power_coeff_c': 0,
        'gen_ramp_up': 50,
        'gen_ramp_down': 50
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 4,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 3,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },
    {
        'min_load': 0,
        'max_load': 0,
        'min_power': 0,
        'max_power': 0,
        'load_coeff': 1,
        'load_ref': 0,
        'power_coeff_a': 0.1,
        'power_coeff_b': 1,
        'power_coeff_c': 0,
        'gen_ramp_up': 5,
        'gen_ramp_down': 5
    },

]  # 9 - 13 - 11      5 - 8
connection_info_4 = {
    'connection_index': [2],
    'connection_x': [0.1],
    'connection_area': [2],
    'connection_exchange_max': [100]
}
player4_info = {
    'index': 4,
    'X_raw': X_raw_4,
    'node_info': node_info_4,
    'connection_info': connection_info_4
}

namejqy = 'jqy'
