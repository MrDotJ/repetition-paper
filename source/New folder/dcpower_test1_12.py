import matplotlib.pyplot as plt
from DCpower.config import *
from DCpower.config3 import *


def factory():
    player1 = getPlayer(player0_info)
    player2 = getPlayer(player1_info)
    player3 = getPlayer(player2_info)
    player4 = getPlayer(player3_info)
    player5 = getPlayer(player4_info)
    playerN1 = playerNp1()
    return [player1, player2, player3, player4, player5, playerN1]


def calculate_NE():
    global g_lam
    count_best_response = 0
    g_angles_old = 0
    while count_best_response < 10:
        # TODO: maybe 30 is a little small
        g_angles_old = copy.deepcopy(g_angles)
        for i, player in enumerate(g_players):
            # get the data for the player i
            player.update_model(g_tao)  # 填充x_i 以及lam_i
            player_i_result = player.optimize_model()
            g_angles[i] = player_i_result.copy()
        # update the lam_dual variable
        g_lam = g_playerN1.optimize(g_tao).copy()
        # update the response
        if sub_norm(g_angles_old, g_angles) < 0.001:
            print(count_best_response)
            break
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
    result_plt3 = []
    # initial
    for player in g_players:
        player.build_model()
    # start the outer loop
    outer_loop_count = 0
    while outer_loop_count < 300:
        print(outer_loop_count)
        # give xn, lam_n, calculate the equilibrium
        calculate_NE()
        # 现在我们得到了一个新的NE，我们应该把这个NE设为参照值
        set_oldValue()
        outer_loop_count = outer_loop_count + 1
        # result_plt.append(g_angles[0][0])
        # result_plt1.append(g_angles[1][0])
        # result_plt2.append(g_angles[0][0] - g_angles[1][0])
        result_plt.append(injection[0][0][0])
        result_plt1.append(injection[1][0][0])
        result_plt2.append(injection[0][0][0] + injection[1][0][0])
        # result_plt3.append(injection[2][0][0] + injection[0][1][0])
        # result_plt2.append(g_lam[0] + g_lam[1])
        # set all value in g_ex to zero
        if outer_loop_count != 600:
            # reset(g_angles)
            namejqy

    plt.plot(result_plt, label='0->1')
    plt.plot(result_plt1, '-r', label='1->0')
    # plt.plot(result_plt2, '-g', label='diff')
    # plt.plot(result_plt3, '*b', label='diff')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('x-node-micro-grid.svg')


if __name__ == '__main__':
    all_players = factory()
    g_players = all_players[:player_num]
    g_playerN1 = all_players[player_num]
    start()
