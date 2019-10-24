# Deep Q-Network

#### 1. Introduction

Q-learning is a form of Temporal Difference (TD) learning



For the following improvements to Deep Q-Network a more detailed description can be found here:

https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/

____

#### 2. Experience Replay

Experience replays helps us address one type of correlation. That is between consecutive experience tupels

When the agent interacts with the environment, the sequence of  experience tuples can be highly correlated (connected to eachother).  The naive Q-learning  algorithm that learns from each of these experience tuples in sequential  order runs the risk of getting swayed by the effects of this  correlation, or in other words priotizing the actions that we have seen the most.

By instead keeping track of a **replay buffer** and using **experience replay** to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The **replay buffer** contains a collection of experience tuples (SSS, AAA, RRR, S′S'S′).  The tuples are gradually added to the buffer as we are interacting with the environment.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as **experience replay**.   In addition to breaking harmful correlations, experience replay allows  us to learn more from individual tuples multiple times, recall rare  occurrences, and in general make better use of our experience.

TLDR; So instead of learning sequentail, we are keeping a buffer of <S,A,R,S'> pairs and learn from these in a random order. We can learn multiple times from each pair, and by doing so we can priotize seeing the rare cases more often than the normal pairs.

____

#### 3. Fixed Q-Targets

There is another kind of correlation that Q-learning is susceptible to.

In Q-Learning, we **update a guess with a guess**, and this can potentially lead to harmful correlations.  To avoid this, we can update the parameters $w$ in the network $\hat{q}$ to better approximate the action value corresponding to state $S$ and action $A$ with the following update rule:

<img src="images\Q-learning_update_rule.jpg" style="zoom: 80%;" />

where $w^-$ are the weights of a separate target network that are not changed during the learning step, and $(S, A, R, S')$ is an experience tuple.

The main idea is to use two separarete networks with identical architectures. 
Lets call it target Q-Network and primary Q-Network, where the target Q-Network is updated less often to have a stable (fixed) target $\hat{q} (S',a ,w^-)$.

____

#### 4. Double DQN

Deep Q-Learning [tends to overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) action values.  [Double Q-Learning](https://arxiv.org/abs/1509.06461) has been shown to work well in practice to help with this. 



____

#### 5. Prioritized Experience Replay

Deep Q-Learning samples experience transitions *uniformly* from a replay memory.  [Prioritized experienced replay](https://arxiv.org/abs/1511.05952) is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability. 



Implementation: https://github.com/Ullar-Kask/TD3-PER/tree/master/Pytorch/src

____

#### 6. Duelling DQN

Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values *for each action*.  However, by replacing the traditional Deep Q-Network (DQN) architecture with a [dueling architecture](https://arxiv.org/abs/1511.06581), we can assess the value of each state, without having to learn the effect of each action.



____

#### 7. Algorithm Deep Q-network (With Experience replay and fixed Q-target)

**7.1 Agent - Functionality**

How we initialize the agent, this can of cause be done in sooo many ways, but here is an example:

 ~~~~python
def __init__(self, state_size, action_size, seed):
    """Initialize an Agent object.
        
	Params
	======
		state_size (int): dimension of each state
		action_size (int): dimension of each action
		seed (int): random seed
	"""
	self.state_size = state_size
	self.action_size = action_size
	self.seed = random.seed(seed)
		
	# ------------------- Q-Network ------------------- #
	# We need two QNetworks, one for choosing action (local)  
	# and one for displaying a fixed target (target). 
	self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
	self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
	# Choosing an optimizer for updating the weights.
	self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

	# Replay memory (will be shown in later sections)
	self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
	# Initialize time step (for updating every UPDATE_EVERY steps)
	self.t_step = 0
 ~~~~

Next we can see how we step and how we choose an action:

~~~~python
def step(self, state, action, reward, next_state, done):
	# Save experience in replay memory
	self.memory.add(state, action, reward, next_state, done)
        
	# Learn every UPDATE_EVERY time steps.
	self.t_step = (self.t_step + 1) % UPDATE_EVERY
	if self.t_step == 0:
		# If enough samples are available in memory, get random subset and learn
		if len(self.memory) > BATCH_SIZE:
			experiences = self.memory.sample()
			self.learn(experiences, GAMMA)

def act(self, state, eps=0.):
	"""Returns actions for given state as per current policy.
        
	Params
	======
		state (array_like): current state
		eps (float): epsilon, for epsilon-greedy action selection
	"""
	state = torch.from_numpy(state).float().unsqueeze(0).to(device)
	self.qnetwork_local.eval()
	with torch.no_grad():
		action_values = self.qnetwork_local(state)
	self.qnetwork_local.train()

	# Epsilon-greedy action selection
	if random.random() > eps:
		return np.argmax(action_values.cpu().data.numpy())
	else:
		return random.choice(np.arange(self.action_size))
~~~~



**7.2 Agent - Update network**

Here we see an example on how we can update the network from a batch of experience.

~~~~python
def learn(batch, GAMMA):
    """
    Update value parameters using given batch of experience tuples.
    Params
    ======
      experiences(batch) (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor
	"""
    # Unpack batch into 5 tensors
	states, actions, rewards, next_states, dones = batch
    
    # ------------------- Compute loss and minimize ------------------- #
    
    # Compute target Q-value, returns a vector of max Q_value for each next_state.
    Qval_next_state = Qn_target(next_states).detach()
    # Find the best Q-value for each next_state returns the max value in a list and the position of the max value
    max_Qval_nState, max_Qval_indice = Qval_next_state.max(1)
    # Unsqueezing the tensor to make a vector and not a list.
    max_Qval_nState = max_Qval_nState.unsqueeze(1)
    
    # Finding the target value for current state
    Q_Target_val = rewards + (GAMMA * max_Qval_nState * (1-dones))
    
    # Find the actual Q-value we got from this state:
    Qvals = Qn_local(states)
    # Find the Q-value for the given action we took:
    Qvals_done = Qvals.gather(1, actions)
    
    # Finding the loss between the what we done and what our target was:
    loss = loss_func(Qvals_done, Qvals_target)
    
    #### Minimize loss (update weights) ####
    # Let the gradient be reset
    self.optimizer.zero_grad()
    # Compute backpropergations:
    loss.backward()
    # Optimize and update:
    self.optimizer.step()
    
    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
~~~~


Soft update means we are learning "tau" from the new network and keep 1-"tau" from the old.

~~~~python
def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
   """
	for target_param, local_param in zip(target_model.parameters(), 															local_model.parameters()):
		target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
~~~~



**7.3 Experience Replay**

This is an implementation of the Experience Replay, we call it the replay buffer.

~~~~python
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", 
			field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)
        
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        
        dones = torch.from_numpy(
            np.vstack(
                [e.done for e in experiences if e is not None]
            ).astype(np.uint8)
        ).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
~~~~

**7.5 Main function**

```python
def dqn(agent, n_episodes=500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action 							   selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores
```