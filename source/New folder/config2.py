import itertools
price = [[0, 6, 6, 6, 6],
         [6, 0, 6, 6, 6],
         [6, 6, 0, 6, 6],
         [6, 6, 6, 0, 6],
         [6, 6, 6, 6, 0]]
# [[6.68847635620013, -1.5043844856390587],
#  [-7.530935327521067, -2.349528058932746],
#  [0.6367732484834399, 1.577324838479214]]
#                       [0]
#                        o
#                  (2)  / \  (0)
#                      /   \
#                (3)  /     \  (1)
#                 [2]o-------o[1]
#                     (5)  (4)
#               0 - 1         2 - 3
#  (0)[0](1)(2)(3)[1](4)(5)
#   o ----- o ------- o
#           |         |
#       [2] |     [3] |
#           o         o
#          (6)       (7)
T = 5
g_tao = 100
# g_lam = [0, 1,   2, 3,   4, 5,   6, 7,   8, 9,   10, 11]
# #       (0)     (1)     (2)     (3)     (4)     (5)
g_lam = [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T + [0, 1] * T
#       (0)          (1)          (2)          (3)          (4)          (5)          (6)          (7)

# g_lam_index = [[0, 1, 2, 3,   4, 5, 6, 7],
# #               1 - 0         4   - 5
#                [3, 2, 1, 0,   8, 9, 10, 11],
# #               3 - 2         5 - 4
#                [7, 6, 5, 4,   11, 10, 9, 8]]
g_lam_index = [
                list(itertools.chain(*[[2*t, 2*t+1, 2*T+2*t, 2*T+2*t+1] for t in range(T)])),
# for time t all time t for line[0]   [(0)t=0,          (1)t=0         ] + [(0)t=1, (1)t=1] + ... [(0)t=T-1, (1)t=T-1]

               list(itertools.chain(*[[2*T+2*t+1, 2*T+2*t, 2*t+1, 2*t] for t in range(T)])) +
#      [0]                           [(1)                     (0)    ]
               list(itertools.chain(*[[4*T+2*t, 4*T+2*t+1, 8*T+2*t, 8*T+2*t+1] for t in range(T)])) +
#      [1]                            [(2)                    (4)
               list(itertools.chain(*[[6*T+2*t, 6*T+2*t+1, 12*T+2*t, 12*T+2*t+1] for t in range(T)])),
#      [2]                            [(3)                      (6)

               list(itertools.chain(*[[8*T+2*t+1, 8*T+2*t, 4*T+2*t+1, 4*T+2*t] for t in range(T)])) +
#      [1]                            [(4)                  (2)
               list(itertools.chain(*[[10*T+2*t, 10*T+2*t+1, 14*T+2*t, 14*T+2*t+1] for t in range(T)])),
#      [3]                            [(5)                      (7)

               list(itertools.chain(*[[12*T+2*t+1, 12*T+2*t, 6*T+2*t+1, 6*T+2*t] for t in range(T)])),
#      [2]                            [(6)                  (3)

               list(itertools.chain(*[[14*T+2*t+1, 14*T+2*t, 10*T+2*t+1, 10*T+2*t] for t in range(T)]))]
#      [3]                            [(7)                  (5)

# g_angles = [[0, 1, 2, 3],
#             [1, 0, 2, 3],
#             [3, 2, 2, 1]]
# g_angles = [[[0] * T, [1] * T, [2] * T, [3] * T],
#             [[1] * T, [0] * T, [2] * T, [3] * T],
#             [[3] * T, [2] * T, [2] * T, [1] * T]]
g_angles = [[[0] * T, [1] * T],
            [[1] * T, [0] * T, [2] * T, [4] * T, [3] * T, [6] * T],
            [[4] * T, [2] * T, [5] * T, [7] * T],
            [[6] * T, [3] * T],
            [[7] * T, [5] * T]]
g_connection = [[1],
                [0, 2, 3],
                [1, 4],
                [1],
                [2]]
# injection = [[0] * T,
#              [0, 0, 0] * T,
#              [0, 0] * T,
#              [0] * T,
#              [0] * T]
injection = [[], [], [], [], []]
g_link = 4
player_num = 5


def sub_norm(a, b):
    # return np.linalg.norm(np.array(a) - np.array(b))
    norm = 0.0
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(T):
                norm = norm + (a[i][j][k] - b[i][j][k]) * (a[i][j][k] - b[i][j][k])
    return norm


def reset(a):
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(T):
                a[i][j][k] = 0
