import matplotlib.pyplot as plt
import numpy as np
import SARSAn
import Grid
import Gridworlds

if __name__ == "__main__":

    # create world
    # world = Grid.Gridworld() # default is 0
    which_gridworld = 1 # between 0 and 4 unless you add some more gridworlds to Gridworlds.Gridworlds.GRIDWORLD list
    world = Grid.Gridworld(Gridworlds.Gridworlds.GRIDWORLD[which_gridworld])
    world.visualize()

    # create n-step SARSA Agent
    player = SARSAn.SARSAn(gridworld=world,n=10,epsilon=0.5,decreasing_epsilon = False,gamma = 0.99,alpha = 0.3,visualize_policy = False, visualize_grid = True)
    # or
    # create Monte Carlo without Exploring Starts Agent
    # player = SARSAn(gridworld=world,n = np.inf, epsilon= 0.05,alpha = 1)

    # learn
    player.start(episodes = 50,evaluation = True)

    # in case you want to save all the plots you created in a picture
    #plt.savefig("Figure_SARSA_policy_returns.png")