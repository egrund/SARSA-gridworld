'''
    File name: SARSAn.py   
    Author: Eosandra Grund   
    Date created: 23.04.2022   
    Date last modified: 20.07.2022   
    Python Version: 3.10.4  
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import Grid

# with pseudocode from book
class SARSAn:
    """
    Implements Tabular n-step SARSA to solve the Gridworld   

    Policy: epsilon-greedy policy   

    remove visualizations for better performance   

    ### Attributes:    
        gridworld = Gridworld objekt : the environment, we are going to learn   
        n (int > 0) : amounts of steps   
        epsilon (0<= float <= 1) = for the epsilon-greedy policy   
        decreasing_epsilon (bool) = if true decreasing epsilon after each episode, so we do more exploration at the beginning and more exploitation at the end   
        gammma (0<= float <= 1) = discount for future rewards    
        alpha (0<= float <= 1) = stepsize (learning rate)   
        visualize_policy (bool) = if the policy should be visualized after each episode with pyplot   
        visualize_grid (bool) = if the grid should be visualized after each step     
        q (np.array(shape(len(action) , y , x))) = the q-values (state-action values)   

    """
    
    def __init__(self,gridworld,n=10,epsilon=0.5,decreasing_epsilon = False,gamma = 0.99,alpha = 0.3,visualize_policy = False,visualize_grid = True):
        """
        ### Arguments:   
            gridworld = Gridworld objekt : the environment, we are going to learn   
            n (int > 0) : amounts of steps   
            epsilon (0<= float <= 1) = for the epsilon-greedy policy   
            decreasing_epsilon (bool) = if true decreasing epsilon after each episode, so we do more exploration at the beginning and more exploitation at the end   
            gammma (0<= float <= 1) = discount for future rewards   
            alpha (0<= float <= 1) = stepsize (learning rate)   
            visualize_policy (bool) = if the policy should be visualized after each episode with pyplot   
            visualize_grid (bool) = if the grid should be visualized after each step   
        """
        
        self.gridworld = gridworld
        self.n = n
        self.epsilon = epsilon # e-greedy policy, with decreasing epsilon
        self.gamma = gamma # discount for future rewards
        self.alpha = alpha # step size (learning rate)
        self.visualize_policy = visualize_policy # because learning is slow when visualized
        self.visualize_grid = visualize_grid
        self.decreasing_epsilon = decreasing_epsilon
        
        # take values from gridworld
        
        # initialize policy q : len(action) * y * x
        self.q = np.random.normal(size=(len(self.gridworld.getActions()),self.gridworld.getYdim(),self.gridworld.getXdim()),scale=0.2)

        # make terminal state 0
        terminal = self.gridworld.getTerminal()
        self.q[:,terminal[0],terminal[1]] = [0,0,0,0] # [y,x] here
        
        # prepare for visualization
        if self.visualize_policy:
            # go in interactive mode  
            plt.ion() 
            # show visualizatin without blocking the caluclations
            plt.show(block=False)

            self.fig, self.axes = plt.subplots(3,3, num ='SARSAn State')
            for ax in self.axes.flat:
                ax.axis('off')
            self.visualize()
             
    def policy(self,state):
        """ gives back an action according to the given state and current policy   

        ### Attributes: 
            state [x,y] = starting state   
        ### return:    
            action_index (int 0- len(action)) = index of action   
        """
        
        # calculate best action after policy
        action_index = np.argmax(self.q[:,state[0],state[1]])
        
        # check whether greedy or random
        greedy = np.random.choice([True,False],p=[1-self.epsilon, self.epsilon])
        if not greedy: # get random action
            action_index = np.random.choice(range(len(self.gridworld.getActions())))
        return action_index
        
    def episode(self,e = "manually"):
        """ creates one episode of the n-Step SARSA algorithm   

        ### Attributes:     
            e = When using the Start method, to print which episode we are in   
        """
        
        # reset the environment gridworld and initialize states array
        state = np.array([self.gridworld.reset()],dtype=int) # [[y,x],[y,x],...]
        
        # initialize n, a (actions trajectory) and reward (reward trajectory)
        n = self.n # n-step SARSA
        action = np.array([self.policy(state[0])], dtype=int)
        reward = np.empty(shape=(0), dtype=int)
        
        t = 0 # in which step the agent is
        t_update = 0 # where we are updating the policy, because always behind t (tau in formula)
        terminal_state_index = np.inf # where the Terminal state in the episode is, if we found it (T in formula)
        
        # for calculating average return
        steps = 0
        returns = 0
        
        # print Gridworld and episode
        if self.visualize_grid:
            self.gridworld.visualize()
            print("Epsiode:",e)        
        
        at_terminal = False                       
        while(not at_terminal):
                            
            # make step and observe newState and reward
            s, r, at_terminal = self.gridworld.step(action[t])     
            returns += r       
            steps+=1 # one step done
            # selection next action
            action = np.append(action,[self.policy(s)],axis=0)
            
            # remember state and reward for later policy updates
            state = np.append(state,[s],axis=0) # state[t+1]
            reward = np.append(reward,[r],axis=0) # reward[t]

            # print Gridworld and episode
            if self.visualize_grid:
                self.gridworld.visualize()
                print("Epsiode:",e)        
            
            if at_terminal:
                terminal_state_index = t+1
            
            #t += 1 # already next so t_update + n = t + 1
            # update the estimates

            visited_states = [] # here enough because only for MC

            while at_terminal or t_update + n <= t :

                # we do not want to update the terminal state
                if np.mean(np.equal(state[t_update],np.array(self.gridworld.getTerminal()))) == 1:
                    break
                
                
                # implement first visited check for Monte Carlo (Is this the first time we are here in this episode)
                if(n == np.inf):
                    for visited_state in visited_states:
                        if np.mean(np.equal(state[t_update],visited_state)) == 1:
                            break
                visited_states.append(state[t_update])

                # calcualte value for n steps or until the terminal if found
                mc_estimate = np.sum([self.gamma**(i-t_update) * reward[i] for i in range(t_update,min(t_update+n,terminal_state_index))])
                future_estimate = 0

                if t_update+n < terminal_state_index: # if we are not yet at the terminals state
                    # calculate the estimate after n
                    future_estimate =  self.gamma**n * self.q[action[t_update+n],state[t_update+n][0],state[t_update+n][1]] # y,x

                estimate = mc_estimate + future_estimate
                    
                # improve policy
                self.q[action[t_update],state[t_update][0],state[t_update][1]] += self.alpha * (estimate - self.q[action[t_update],state[t_update][0],state[t_update][1]] )               
            
                t_update += 1                                 
                
            t += 1

        # end of one episode
        print("Epsiode:",e)  

        self.gridworld.reset()
        
        if self.visualize_policy:
            self.visualize()  

        average_return = returns / steps
        
        return average_return, returns, steps
            
        
    def visualize(self):
        """ visualizes the current policy """
            
        for i, action in enumerate(self.gridworld.getActions()):
            ax = self.axes.flat[2*i + 1]
            ax.cla() # add axis
            ax.set(title = action)
            ax.set_xticks(np.arange(self.gridworld.getXdim()))
            ax.set_yticks(np.arange(self.gridworld.getYdim()))
            ax.imshow(self.q[i,:,:], interpolation='None')
            for y in range(self.q.shape[1]): 
                for x in range(self.q.shape[2]): 
                    text = ax.text(x, y, "{:.1f}".format(self.q[i,y,x],1),
                       ha="center", va="center", color="black", fontsize=8)
                    plt.setp(text, path_effects=[
        PathEffects.withStroke(linewidth=1, foreground="w")])

        self.fig.suptitle("Policy ",fontsize=18)
        plt.draw()
        pylab.pause(1.e-6) # important, do not delete
       
    def start(self,episodes=10,evaluation = True):
        ''' Starts the Learning Process and does episodes amounds of episodes   

        ### Arguments:    
            episodes (int >=1 ) = the amount of episodes to do     
            evaluation (bool) = if you want a list and plot of the total return and steps per episodes at the end (only works if visualize_policy == False)   
        '''

        # to save values for each episode
        average_return = np.array(range(episodes),dtype=np.float64)
        returns = np.array(range(episodes))
        steps = np.array(range(episodes))
        
        # do the learning
        for e in range(episodes):
            average_return[e], returns[e], steps[e] = self.episode(e+1)

            # calculate new epsilon, should be 0 at the end
            if self.decreasing_epsilon:
                self.epsilon -= self.epsilon / episodes

        # visualizing only works if not visualize_poliy
        if evaluation: 
            # print statistic average return
            for e in range(episodes):
                print("Episode",("       " + str(e+1))[-7:],"; Average Return: ", (str(average_return[e]) + "                 ")[:10], "; Return: ", ("        " + str(returns[e]) )[-10:], "; Steps: ", ("        " + str(steps[e]) )[-10:])

            plt.plot(returns, label = "Total Returns")
            plt.plot(steps, label = "Steps")
            plt.legend()
            plt.show()