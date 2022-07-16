import matplotlib.pyplot as plt
import numpy as np
import SARSAn
import Grid

#if __name__ == "__main__":

# go in interactive mode  
plt.ion() 
# create world
world = Grid.Gridworld()
# create Monte Carlo without Exploring Starts Agent
# player = SARSAn(gridworld=world,n = np.inf,alpha = 1, epsilon= 0.05)

# create n-step SARSA Agent
player = SARSAn.SARSAn(gridworld=world,n=10,epsilon=0.5,gamma = 0.99,alpha = 0.3,visualize_policy = False, visualize_grid = False)
# show visualizatin without blocking the caluclations
plt.show(block=False)

# learn
player.start(2)