'''
    File name: Grid.py  
    Author: Eosandra Grund  
    Date created: 20.04.2022  
    Date last modified: 20.07.2022  
    Python Version: 3.10.4 
'''

import numpy as np
import os
import Gridworlds

def clearConsole():
    '''
    clears the console
    '''
    command = 'clear'
    # If Machine is running on Windows, use cls
    if os.name in ('nt', 'dos'): 
        command = 'cls'
    os.system(command)

# try new output type
class Gridworld:
    """
    This methods creates a gridworld for a RL algorithm

    ### Reward function:   
        10 for the terminal state  
        user input for neg reward field  
        -0.5 for invalid move (against barrier or outside of the gridworld)    
        -0.1 for each move where you do not get any other reward  

    ### State transition function:     
        take random action with probabilitiy epsilon  
        take given action with probabilitiy 1- epsilon  

    ### Attributes:     
        x_dim (int>0) : x dimension of gridworld   
        y_dim (int>0) : y dimension of gridworld   
        epsilon (0<float<1) : for epsilon-greedy state transition function   
        agent (list) : [y,x] coordinates of the current agent   
        initial_agent (list) : [y,x] coordinates of the starting state   
        terminal (list) : [y,x] coordinates of the terminal state   
        action (list) : list of all the possible actions in order as strings   
        world (2D list) : [y][x] with values for all states being int for rewards and np.NaN for barriers   
    """
    
    def __init__(self,gridworld = Gridworlds.Gridworlds.GRIDWORLD1):
        """
        Initializes a gridworld with all parameters   
        gives one positive reward in the terminal state    
        all states as [y,x] in the code, in constructor parameters as [x,y] for convenience
        
        create an empty grid world default hardcoded some examples in Gridworlds 
        e.g. GRIDWORLD0 : 
        
         s  0  X  0 10   
         0  0 -1  0  0   
         0  X  0  0  0   
         0  0  X  0  0   
        -1  0  0  0 -1   
        
        ### Keys in the dictionary gridworld:   
            x_dim (int>0) : x dimension of gridworld  
            y_dim (int>0) : y dimension of gridworld  
            epsilon (0<float<1) : for epsilon-greedy state transition function  
            start [x,y] = starting state of agent for each episode  
            terminal [x,y] = terminal state with a positive reward  
            neg_rewards [[x,y,reward],[x,y,reward],...] = list of fields with negative rewards  
            barrier [[x,y],[x,y],...] = list of fields that are barriers  
        """
        
        self.x_dim = gridworld["x_dim"]
        self.y_dim = gridworld["x_dim"]
        self.epsilon = gridworld["epsilon"]
        self.agent = gridworld["start"].copy()
        self.agent.reverse() # [y,x]
        self.initial_agent = self.agent # needed for reset
        self.terminal = gridworld["terminal"].copy()
        self.terminal.reverse()# [y,x]
        
        self.action = ['up', 'down' , 'left' , 'right']
        
        # create empty gridworld
        world =np.zeros(shape=(self.y_dim,self.x_dim))
        
        # put terminal
        world[self.terminal[0],self.terminal[1]] = 10 # [y,x]
        
        # put negative rewards in gridworld
        for r in gridworld["neg_reward"]:
            world[r[1],r[0]] = r[2]
            
        # put barrier in gridworld
        for b in gridworld["barrier"]:
            world[b[1],b[0]] = np.NaN
        
        self.world = world
        
    # getter and setter
    def getXdim(self):
        return self.x_dim
    
    def getYdim(self):
        return self.y_dim
    
    def getActions(self):
        return self.action
    
    def getTerminal(self):
        return self.terminal # [y,x]
    
    def getState(self):
        return self.agent # [y,x]
    
    # methods
    def isValid(self,x,y):
        """
        checks whether coordinates x,y are in the gridworld and not on a barrier
        """
        # check whether in the Gridworld
        if(x>=0 and x<self.x_dim and y>= 0 and y<self.y_dim):
            # check whether the state is a barrier
            if not np.array_equal(self.world[y,x], np.NaN, equal_nan=True):
                return True
        return False
    
    def inTerminal(self):
        """
        checks whether the current agent is in the terminal state
        """
        if self.agent == self.terminal: # [y,x]
            return True
        return False
        
    def reset(self):
        """
        resets the gridworld to its initial state
        """
        self.agent = self.initial_agent
        return self.initial_agent
    
    def step(self, action):
        """
        applies the state transition dynamics and reward dynamics 
        based on the state of the environment and the action argument
        Arguments: 
            action int : [0,1,2,3] for ['up', 'down' , 'left' , 'right'] =
            
        returns: 
            the new state
            reward of this step
            a boolean indication whether this state is terminal
        """
        
        # state transition policy
        # check whether action or for epsilon random other one
        take_greedy_action = np.random.choice([True,False],p=[1-self.epsilon, self.epsilon])
        
        # take random action
        if (not take_greedy_action):
            action = np.random.choice(len(self.action))
            
        # get new place after action
        y,x = self.agent[0], self.agent[1] # [y,x]
        if action == 0: # up
            y -= 1
        elif action == 1: # down
            y += 1
        elif action == 2: # left
            x -= 1
        elif action == 3: # right
            x += 1
           
        # basic reward in case no other for each step
        reward = -0.1 # for eachs step done
        
        # check if action is valid, then do action
        if self.isValid(x,y):
            self.agent = [y,x]
            reward = self.world[self.agent[0],self.agent[1]]
        else:
            # invalid
            reward = -0.5
    
        return self.agent, reward , self.inTerminal()
        
        
    def visualize(self):
        """
        visualizes the current state
        """
        
        clearConsole()
        print("")
        
        for y in range(self.y_dim):
            
            # left side y values
            firstLine = "    ||"
            thisLine = ("      " + str(y) + " ||")[-6:]
            nextLine =  "____||"
            
            for x in range(self.x_dim):
                val = self.world[y,x]
                
                # print one field if agent is there or if barrier on bottom line
                if ([y,x]==self.agent):
                    nextLine += "__A__|"
                elif np.array_equal(val, np.NaN, equal_nan=True): # if it is a barrier
                    nextLine += "XXXXX|"
                else:
                    nextLine +=  "_____|"
                    
                # print vlaues on field in middle line and first line
                if (val==0.0): # if 
                    firstLine += "     |"
                    thisLine += "     "
                elif np.array_equal(val, np.NaN, equal_nan=True): # if it is a barrier
                    firstLine += "XXXXX|"
                    thisLine += "XXXXX"
                else: # if it has a reward
                    firstLine += "     |"
                    thisLine += ( "     " + str(int(val)) )[-5:]
                thisLine += "|"
                
            print(firstLine)
            print(thisLine)
            print(nextLine)
            
        # at the bottom of the print, pritn x coordinates
        topLine = "____||"
        line = "    ||"
        middleLine = "    ||"
        for x in range(self.x_dim):
            line += "     |"
            middleLine += ("      " + str(x) + " |")[-6:]
            topLine += "_____|"
            
        print(topLine)
        print(line)
        print(middleLine)
        print(line)
        print("")

        
        
