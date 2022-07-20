'''
    File name: Main_SARSA.py
    Author: Eosandra Grund
    Date created: 16.07.2022
    Date last modified: 20.07.2022
    Python Version: 3.10.4
'''

import matplotlib.pyplot as plt
import numpy as np
import SARSAn
import Grid
import Gridworlds

if __name__ == "__main__":

    # create world
    which_gridworld = 0 # between 0 and 4 unless you add some more gridworlds to Gridworlds.Gridworlds.GRIDWORLD list
    world = Grid.Gridworld(Gridworlds.Gridworlds.GRIDWORLD[which_gridworld])
    world.visualize()

    # create n-step SARSA Agent
    player = SARSAn.SARSAn(gridworld=world, n=10, epsilon=0.5, decreasing_epsilon = True, gamma = 0.99, alpha = 0.3, visualize_policy = False, visualize_grid = True)
    # or
    # create Monte Carlo without Exploring Starts Agent
    # player = SARSAn.SARSAn(gridworld=world,n = np.inf, epsilon= 0.3, alpha = 1)

    # learn
    player.start(episodes = 50, evaluation = True)

    # in case you want to save all the plots you created in a picture
    #plt.savefig("Figure_SARSA_policy_returns.png")