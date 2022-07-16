#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.pylab as pylab
import Grid

# with pseudocode from book
class SARSAn:
    """
    Implements Tabular n-step SARSA to solve the Gridworld
    """
    
    def __init__(self,gridworld,n=10,epsilon=0.5,gamma = 0.99,alpha = 0.3,visualize_always = False):
        """
        Arguments:
            gridworld = Gridworld objekt : the environment, we are going to learn
            n (int > 0) : amounts of steps
            epsilon (0<= float <= 1) = for the e-greedy policy
            gammma (0<= float <= 1) = discount for future rewards
            alpha (0<= float <= 1) = stepsize (learning rate)
            visualize_always (bool) = if we visuale the policy only after each episode (False) or after each change (True) (might be time intensive)
        """
        
        self.gridworld = gridworld
        self.n = n
        self.epsilon = epsilon # e-greedy policy, with decreasing epsilon
        self.gamma = gamma # discount for future rewards
        self.alpha = alpha # step size (learning rate)
        self.visualize_always = visualize_always # because learning is slow when visualized
        
        # take values from gridworld
        
        # initialize policy q : len(action) * y * x
        self.q = np.random.normal(size=(len(self.gridworld.getActions()),self.gridworld.getYdim(),self.gridworld.getXdim()),scale=0.2)

        # make terminal state 0
        terminal = self.gridworld.getTerminal()
        self.q[:,terminal[0],terminal[1]] = [0,0,0,0] # [y,x] here
        
        # prepare for visualization
        self.fig, self.axes = plt.subplots(3,3, num ='SARSAn State')
        for ax in self.axes.flat:
            ax.axis('off')
        self.visualize()
             
    def policy(self,state):
        """
        gives back an action according to the given state and current policy
        Attributes: 
            state [x,y] = starting state
        return: 
            action_index (int 0- len(action)) = index of action
        """
        
        # calculate best action after policy
        action_index = np.argmax(self.q[:,state[0],state[1]])
        
        # check whether greedy or random
        greedy = np.random.choice([True,False],p=[1-self.epsilon, self.epsilon])
        if not greedy: # get random action
            action_index = np.random.choice(range(len(self.gridworld.getActions())))
        return action_index
        
    def episode(self,e="manually"):
        """
        creates one episode of the n-Step SARSA algorithm
        Attributes: 
            e = When using the Start method, to print which episode we are in
        """
        
        # reset the environment gridworld and initialize s
        state = np.array([self.gridworld.reset()],dtype=int) # [[y,x],[y,x]]
        
        # initialize n, a (actions trajectory) and reward (reward trajectory)
        n = self.n # n-step SARSA
        a = np.array([self.policy(state[0])], dtype=int)
        reward = np.empty(shape=(0), dtype=int)
        
        t = 0 # in which step the agent is
        tau = 0 # where we are updating the policy, because always behind t
        big_T = np.inf # where the Terminal state in the episode is
        
        # safe amount of steps needed for ending table of start()
        steps = 0
        
        # print Gridworld and episode
        self.gridworld.visualize()
        print("Epsiode:",e)        
        
        at_terminal = False                           
        while(not at_terminal):
                            
            # make step and observe newState and reward
            s, r, at_terminal = self.gridworld.step(a[t])                
            steps+=1 # one step done
            # selection next action
            a = np.append(a,[self.policy(s)],axis=0)
            
            # remember state and reward for later policy updates
            state = np.append(state,[s],axis=0) # state[t+1]
            reward = np.append(reward,[r],axis=0) # reward[t]

            # print Gridworld and episode
            self.gridworld.visualize()
            print("Epsiode:",e)        
            
            if at_terminal:
                big_T = t+1
            
            #t += 1 # already next so tau + n = t + 1
            # update the estimates

            visited_states = []

            while at_terminal or tau + n <= t :

                # we do not want to update the terminal state
                if np.mean(np.equal(state[tau],np.array(self.gridworld.getTerminal()))) == 1:
                    break
                
                
                # implement first visited implementation
                for visited_state in visited_states:
                    if np.mean(np.equal(state[tau],visited_state)) == 1:
                        break

                visited_states.append(state[tau])

                # calcualte value for n steps or until the terminal if found
                G = np.sum([self.gamma**(i-tau) * reward[i] for i in range(tau,min(tau+n,big_T))])
                
                if tau+n < big_T: # if we are not yet at the terminals tate
                    # calculate the estimate after n
                    G += self.gamma**n * self.q[a[tau+n],state[tau+n][0],state[tau+n][1]] # y,x
                    
                # improve policy
                self.q[a[tau],state[tau][0],state[tau][1]] += self.alpha * (G - self.q[a[tau],state[tau][0],state[tau][1]] )               
            
                # next step, update next values
                tau += 1
                # after each update only visualize new if wanted
                if self.visualize_always:
                    self.visualize()                      
                
            t += 1
        # reset the gridworld for next round
        self.gridworld.reset()
        
        # at the end of one episode always visualize the policy
        self.visualize()
        
        
        return steps
            
        
    def visualize(self):
        """
        visualizes the current policy
        """
            
        for i, action in enumerate(self.gridworld.getActions()):
            ax = self.axes.flat[2*i + 1]
            ax.cla() # add axis
            ax.set(title = action) # set title
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
        

       
    def start(self,episodes=10):
        '''
        Starts the Learning Process and does episodes amounds of episodes        
        '''
        
        steps = np.array(range(episodes)) # to safe how many steps per episode
        
        # do the learning
        for e in range(episodes):
            steps[e] = self.episode(e+1)
           
        # print statistic how many steps needed per episode
        for e in range(episodes):
            print("Episode",("       " + str(e+1))[-7:],"; Steps: ", ("          " + str(steps[e]))[-10:])
        
        
if __name__ == "__main__":
    # go in interactive mode  
    plt.ion() 
    # create world
    world = Grid.Gridworld()
    # create Monte Carlo without Exploring Starts Agent
    # player = SARSAn(gridworld=world,n = np.inf,alpha = 1, epsilon= 0.05)

    # create n-step SARSA Agent
    player = SARSAn(gridworld=world,n=10,epsilon=0.5,gamma = 0.99,alpha = 0.3)
    # show visualizatin without blocking the caluclations
    plt.show(block=False)
    
    # learn
    player.start(100)