# Policy-based methods



![](images\policy_based_methods.png)

Policy-based methods tries to skip the step of learning the expected value function

____

### 1. Optimization method

#### 1.1 Gradient Ascent:

**Gradient ascent** is similar to gradient descent.  

- Gradient descent steps in the **direction opposite the gradient**, since it wants to minimize a function.  
- Gradient ascent is otherwise identical, except we step in the **direction of the gradient**, to reach the maximum.

While we won't cover gradient-based methods in this lesson, you'll explore them later in the course!



#### 1.2 Local Minima

In the video above, you learned that **hill climbing** is a relatively simple algorithm that the agent can use to gradually improve the weights θ\thetaθ in its policy network while interacting with the environment.

Note, however, that it's **not** guaranteed to  always yield the weights of the optimal policy.  This is because we can  easily get stuck in a local maximum.  In this lesson, you'll learn about some policy-based methods that are less prone to this.



#### 1.3 Beyond Hill Climbing

In the previous video, you learned about the hill climbing algorithm. 

We denoted the expected return by $J$.  Likewise, we used $\theta$ to refer to the weights in the policy network. Then, since $\theta$ encodes the policy, which influences how much reward the agent will likely receive, we know that $J$ is a function of $\theta$. 

Despite the fact that we have no idea what that function $J = J(\theta)$ looks like, the *hill climbing* algorithm helps us determine the value of $\theta$ that maximizes it.  Watch the video below to learn about some improvements you can make to the hill climbing algorithm!  

*Note*: We refer to the general class of approaches that find $\arg\max_{\theta}J(\theta)$ through randomly perturbing the most recent best estimate as **stochastic policy search**.  Likewise, we can refer to $J$ as an **objective function**, which just refers to the fact that we'd like to *maximize* it!



**Steepest Ascent**

![](images\Steepest_Ascent.png)

Create a few set of weights and validate each. Then pick the best of the policies and continue.

The problem is that we can get stuck in a suboptimal solution. But there are some modifications that can help mitigate this:

* Random restart
* Simulated annealing
* adaptive noise

**Simulated annealing:**

It's a predefined scheduling of how much noise we apply on each step. Starting with a large noise parameter and lower it as we train.

![](images\Simulated_annealing.png)

But this can still leads us to a suboptimal solution.

**Adaptive Noise:**

It's reduces noise when our model improves, but when we are stuck it starts to explore (increment the noise parameter) until we find an alternative or better solution. This can be seen in the picture below.

![](D:\dev\learning\DRL-project\DRL-course\notes\images\adaptive_noise.png)

CODE Agent  ( called policy)

~~~~python
class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4*np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
        
    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x)/sum(np.exp(x))
    
    def act(self, state):
        probs = self.forward(state)
        #action = np.random.choice(2, p=probs) # option 1: stochastic policy pi(s,a) = P[a|s]
        action = np.argmax(probs)              # option 2: deterministic policy pi: s->a 
        return action
~~~~

CODE training:

~~~~python
def hill_climbing(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
    """Implementation of hill climbing with adaptive noise scaling.
        
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    best_w = policy.w
    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

        if R >= best_R: # found better weights
            best_R = R
            best_w = policy.w
            # Lowering the exploration
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape) 
        else: # did not find better weights
            # Increase the exploration
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            policy.w = best_w
            break
        
    return scores
            
scores = hill_climbing()
~~~~



#### 1.4 Black-box Optimization

All of the algorithms that you’ve learned about in this lesson can be classified as **black-box optimization** techniques.  

**Black-box** refers to the fact that in order to find the value of $\theta$ that maximizes the function $J = J(\theta)$, we need only be able to estimate the value of $J$ at any potential value of $\theta$.  

That is, both hill climbing and steepest ascent hill climbing don't  know that we're solving a reinforcement learning problem, and they do  not care that the function we're trying to maximize corresponds to the  expected return.  

These algorithms only know that for each value of $\theta$, there's a corresponding **number**.  We know that this **number** corresponds to the return obtained by using the policy corresponding to $\theta$ to collect an episode, but the algorithms are not aware of this.  To the algorithms, the way we evaluate $\theta$ is considered a black box, and they don't worry about the details.  The algorithms only care about finding the value of $\theta$ that will maximize the number that comes out of the black box.

There is a couple of other black-box optimization techniques, to include the **cross-entropy method** and **[evolution strategies](https://blog.openai.com/evolution-strategies/)**.

**Cross-entropy method:**

As in the **steepest ascent Hill Climbing method**  we generate a couple of neighbouring policies at each iteration, but instead of only take the best policy, we take the average of a percentage of the policies.

CODE Agent - (look for evaluate function):

~~~~python
class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.shape[0]
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        
    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data
        
    # Run the game with a set of weights
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

~~~~

CODE  training: 

~~~~python
def cem(n_iterations=500, 
        max_t=1000,
        gamma=1.0,
        print_every=10,
        pop_size=50,
        elite_frac=0.2,
        sigma=0.5):
    """PyTorch implementation of a cross-entropy method.
        
    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    # Init best weight to something random
    best_weight = sigma*np.random.randn(agent.get_weights_dim())
	
    for i_iteration in range(1, n_iterations+1):
        # Create population size amount of weights
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        # Test agent on each weight.
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
		# Find the N best
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        # Compute mean
        best_weight = np.array(elite_weights).mean(axis=0)
		# Test the best weights ( For updating the weight on agent and progression save)
        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)
    	# Save current best    
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        # Test if done
        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    return scores

scores = cem()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
~~~~



**Evolution strategies:**

Another approach is to look at the return that was collected by each candidate policy. The best policy will be a weighted sum of all of these, where policies that got higher return, are given more say or get a higher weight.

This idea originally comes from the idea of biological evolution, where the idea is that the most successful individuals in the policy population, will have the most influence on the next generation or iteration.

That said, evolution strategies are just another black box optimization techniques

### 2. Why Policy-based methods 

Well in policy-based method we don't need to find a expected value, hence we need no value function to approximate.

* We can have truly stochastic behaviour, meaning we get a probability of taking a certain action so two action can have the same.
  * Aliased states - 

* Policy-based methods are better when it comes to Continous action space, since finding the optimal value would be a optimization problem in it self.

### 3. Summary

##### Policy-Based Methods

------

- With **value-based methods**, the agent uses its  experience with the environment to maintain an estimate of the optimal  action-value function.  The optimal policy is then obtained from the  optimal action-value function estimate.
- **Policy-based methods** directly learn the optimal policy, without having to maintain a separate value function estimate.

##### Policy Function Approximation

------

- In deep reinforcement learning, it is common to represent the policy with a neural network.  
  - This network takes the environment state as ***input\***.  
  - If the environment has discrete actions, the ***output\*** layer has a node for each possible action and contains the probability that the agent should select each possible action.
- The weights in this neural network are initially set to random  values.  Then, the agent updates the weights as it interacts with (*and learns more about*) the environment.

##### More on the Policy

------

- Policy-based methods can learn either stochastic or deterministic  policies, and they can be used to solve environments with either finite  or continuous action spaces.

##### Hill Climbing

------

- **Hill climbing** is an iterative algorithm that can be used to find the weights θ\thetaθ for an optimal policy.
- At each iteration, 
  - We slightly perturb the values of the current best estimate for the weights θbest\theta_{best}θbest, to yield a new set of weights.  
  - These new weights are then used to collect an episode.  If the new weights $\theta_{new}$resulted in higher return than the old weights, then we set $\theta_{best} \leftarrow \theta_{new}$.

##### Beyond Hill Climbing

------

- **Steepest ascent hill climbing** is a variation of  hill climbing that chooses a small number of neighboring policies at  each iteration and chooses the best among them.
- **Simulated annealing** uses a pre-defined schedule to  control how the policy space is explored, and gradually reduces the  search radius as we get closer to the optimal solution.
- **Adaptive noise scaling** decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.

##### More Black-Box Optimization

------

- The **cross-entropy method** iteratively suggests a  small number of neighboring policies, and uses a small percentage of the best performing policies to calculate a new estimate.
- The **evolution strategies** technique considers the  return corresponding to each candidate policy.  The policy estimate at  the next iteration is a weighted sum of all of the candidate policies,  where policies that got higher return are given higher weight.  

##### Why Policy-Based Methods?

------

- There are three reasons why we consider policy-based methods:
  1. **Simplicity**: Policy-based methods directly get to  the problem at hand (estimating the optimal policy), without having to  store a bunch of additional data (i.e., the action values) that may not  be useful.
  2. **Stochastic policies**: Unlike value-based methods, policy-based methods can learn true stochastic policies.
  3. **Continuous action spaces**: Policy-based methods are well-suited for continuous action spaces.

____

