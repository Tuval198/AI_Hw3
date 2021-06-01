import argparse
from GameWrapper import GameWrapper
import os, sys
import utils
import players.CompetePlayer
import itertools
import csv

ref_params = {"maxVision": 7,
              "ccWeight": 20,
              "possessionWeight": 1,
              "potentialScoreWeight": 0.5, }

maxVision_param = [6, 7, 8]
ccWeight_param = [3, 20, 50]
possessionWeightParam = [0.1, 1, 10]
potentialScoreWeight_param = [0.1, 0.5, 2]

param_grid = list(itertools.product(maxVision_param, ccWeight_param, possessionWeightParam, potentialScoreWeight_param))


def play_game(board_name, game_time, h_params):
    board = utils.get_board_from_csv(board_name)
    player_1 = sys.modules["players.CompetePlayer"].Player(game_time=game_time,
                                                           penalty_score=300,
                                                           heuristic_params=ref_params)
    player_2 = sys.modules["players.CompetePlayer"].Player(game_time=game_time,
                                                           penalty_score=300,
                                                           heuristic_params=h_params)

    game = GameWrapper(board[0], board[1], board[2], player_1=player_1, player_2=player_2,
                       terminal_viz=True,
                       print_game_in_terminal=False,
                       time_to_make_a_move=game_time,
                       game_time=game_time,
                       penalty_score=300,
                       )
    return game.start_game()


def main():
    for params in param_grid:
        h_params = {"maxVision": params[0],
                    "ccWeight": params[1],
                    "possessionWeight": params[2],
                    "potentialScoreWeight": params[3], }

        print(h_params)
        print("##############################################################################################")

        win_count = 0
        param_score = 0
        print("-----------default_board--------------")
        p1_score, p2_score = play_game("default_board.csv", 21, h_params)
        param_score += p2_score - p1_score
        if p1_score < p2_score:
            win_count += 1
        print("-----------MyBoard--------------")
        p1_score, p2_score = play_game("MyBoard.csv", 180, h_params)
        param_score += p2_score - p1_score
        if p1_score < p2_score:
            win_count += 1
        print("-----------1 board--------------")
        p1_score, p2_score = play_game("1.csv", 90, h_params)
        param_score += p2_score - p1_score
        if p1_score < p2_score:
            win_count += 1

        print("-----------2 board--------------")
        p1_score, p2_score = play_game("2.csv", 60, h_params)
        param_score += p2_score - p1_score
        if p1_score < p2_score:
            win_count += 1

        print("-----------5 board--------------")
        p1_score, p2_score = play_game("5.csv", 60, h_params)
        param_score += p2_score - p1_score
        if p1_score < p2_score:
            win_count += 1

        f = open("param_grid_res.csv", "a")
        print(params)
        print(win_count)
        f.write("{},{},{}\n".format(params, param_score, win_count))
        f.close()
        print("##############################################################################################")


main()
