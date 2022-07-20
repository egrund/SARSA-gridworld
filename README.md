# SARSA-gridworld
This is an implementation of n-step SARSA an a gridworld to be solved by it

The Gridworld
-------------
The class Gridworld is implemented in the File Grid.py. The Constructor gets a dictionary with the layout. There are some hardcoded gridworld dictionaries in the Gridworlds.py file(access via class variable GRIDWORLD[index]), to use, but you can also create your own ones. 
A Gridworld has a starting state, a terminal state (with the positive rewards of 10), some other negative rewards and barriers. Possible actions are up, down, left and right. 

State transition function: In the environment you take the given action with the probabiliy 1- epsilon (a parameter) and a random action with probability epsilon. 

Reward function: 
	- 10 for the terminal state
	- other rewards as user inputs
	- -0.5 for invalid moves (against barriers or outside of the gridworld)
	- -0.1 for every move (if no other reward)

It can be visualized via the consol, but because it is always printed new, it is best to execute it in a terminal so the old prints can be removed and it stays in the same place.

The Agent
-----------
The Agent is in the SARSAn.py file. It is an implementation of the reinforcement-learning algorithm n-step SARSA and can also do 1-step SARSA and MonteCarlo. 
It uses an epsilon-greedy policy with the possibility of it decreasing over time (set decreasing_epsilon to True).

More Information about n-step SARSA: https://towardsdatascience.com/introduction-to-reinforcement-learning-rl-part-7-n-step-bootstrapping-6c3006a13265

If you set visualize_policy to False, the q-values will be visualized as a matplotlib heatmap showing all state-action values.
Start the learning process with the start method. It gets the amount of episodes you want to do and if you want an evaluation (list and plot of the total return and steps per episode). 

Create an MonteCarlo approach by setting n to np.inf, and alpha to 1. 

