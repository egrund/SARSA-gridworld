'''
    File name: Gridworlds.py   
    Author: Eosandra Grund   
    Date created: 18.07.2022   
    Date last modified: 20.07.2022   
    Python Version: 3.10.4 
'''

class Gridworlds:
    """
    Contains some default gridworld for creating a Grid object.   
    
    ### Visualization in comments:   
        s = starting state   
        X = barrier   
        int = value / reward of the state   
        10 = terminal state   
        
    ### keys in the dictionary:   
        x_dim (int>0) : x dimension of gridworld    
        y_dim (int>0) : y dimension of gridworld   
        epsilon (0<float<1) : for epsilon-greedy state transition function   
        start [x,y] = starting state of agent for each episode   
        terminal [x,y] = terminal state with a positive reward   
        neg_rewards [[x,y,reward],[x,y,reward],...] = list of fields with negative rewards   
        barrier [[x,y],[x,y],...] = list of fields that are barriers   
    """
    
    GRIDWORLD0 = {
        "x_dim" : 5,
        "y_dim" : 5,
        "epsilon" : 0.1,
        "start" : [0,0],
        "terminal" : [4,0],
        "neg_reward" : [[0,4,-1],[2,1,-1],[4,4,-1]],
        "barrier" :[[1,2],[2,0],[2,3]]
    }
    """ GRIDWORLD0
    #  s  0  X  0 10      |
    #  0  0 -1  0  0      V
    #  0  X  0  0  0      y
    #  0  0  X  0  0
    # -1  0  0  0 -1
    -> x dimension
    """
    
    GRIDWORLD1 = {
        "x_dim" : 3,
        "y_dim" : 3,
        "epsilon" : 0.1,
        "start" : [0,2],
        "terminal" : [2,2],
        "neg_reward" : [],
        "barrier" :[[1,2],[1,1]]
    }
    """GRIDWORLD1
    #  0  0  0     |
    #  0  X  0     V
    #  s  X  10    y
    -> x dimension
    """

    GRIDWORLD2 = {
        "x_dim" : 8,
        "y_dim" : 8,
        "epsilon" : 0.1,
        "start" : [0,0],
        "terminal" : [6,6],
        "neg_reward" : [[0,4,-1],[5,6,-1],[5,7,-1]],
        "barrier" :[[1,1],[2,2],[3,3],[4,4],[5,5],[2,4],[3,0],[4,2],[5,1],[6,1]]
    }
    """GRIDWORLD2
       #  s  0  0  X  0  0  0  0
       #  0  X  0  0  0  X  X  0 
       #  0  0  X  0  X  0  0  0 
       #  0  0  0  X  0  0  0  0 
       # -1  0  X  0  X  0  0  0 
       #  0  0  0  0  0  X  0  0 
       #  0  0  0  0  0 -1 10  0 
       #  0  0  0  0  0 -1  0  0 
        -> x dimension
    """

    GRIDWORLD3 = {
        "x_dim" : 3,
        "y_dim" : 3,
        "epsilon" : 0.1,
        "start" : [0,2],
        "terminal" : [2,0],
        "neg_reward" : [[0,0,-3]],
        "barrier" :[]
    }
    """GRIDWORLD3
    # -3  0 10        |
    #  0  0  0        V
    #  s  0  0        y
    -> x dimension
    """
    
    GRIDWORLD4 = {
        "x_dim" : 4,
        "y_dim" : 4,
        "epsilon" : 0.1,
        "start" : [1,1],
        "terminal" : [2,0],
        "neg_reward" : [[1,2,-1]],
        "barrier" :[[1,0],[2,1],[2,2]]
    }
    """GRIDWORLD4
    #  0  X 10  0      |
    #  0  s  X  0      V
    #  0 -1  X  0      y
    #  0  0  0  0
    -> x dimension
    """
    
    GRIDWORLD = [GRIDWORLD0, GRIDWORLD1, GRIDWORLD2, GRIDWORLD3, GRIDWORLD4]
    """to access the gridworlds via number (list of all the gridworlds)"""