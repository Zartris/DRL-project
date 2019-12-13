# Actor Critic Methods

- **Value based methods** (Q-learning, Deep Q-learning): where we learn a value function **that will map each state action pair to a value.** Thanks to these methods, we find the best action to take for each state — the action with the biggest value. This works well when you have a finite set of actions.
- **Policy based methods** (REINFORCE with Policy Gradients): where we directly optimize the policy without using a value function. This is useful when the action space is continuous or stochastic. The main problem is finding a good score function to compute how good a policy is. We **use total rewards of the episode.**

https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/

____

The actor-critic method is a combination of value-based and policy-based.
We are training a network **Actor** as a policy-based method to decide in a given state what the corresponding distribution over the action space. Another network is made, the **Critic** which is a value-based network, that will learn to evaluate the state value function $V_\pi$ using TD estimation. Using this **Critic** we will calculate the advantage function and train the **Actor** using this value.

![](images\actor_critic_basic.png)

**One training step in the Actor-critic cycle:**

1. Given a state:

   * $\pi(a|s;\theta_\pi) =$ **Actor**(state)
   * best_action = max($\pi(a|s;\theta_\pi)$)
   * reward, state_next = env(state, best_action)
   * (s,a,r,s') = (state, best_action, reward, state_next)

2. Given reward, state_next and discount value $\gamma$:

   * $\text{value}_{state}$ = $V(s,\theta_v)$ = **Critic** (s)

   * $\text{value}_{Nstate}$ = $V(s',\theta_v)$ = **Critic** (s')
   * train **Critic** (reward+$\gamma V(s'; \theta_v)$)

3. Given state, action, reward, $\text{value}_{state}$, $\text{value}_{Nstate}$ and discount value $\gamma$

   * Advantage(state,action) = $\text{reward} + \gamma V(s';\theta_v) - V(s;\theta_v)$
   * train **Actor** (Advantage(state,action))



### 1. A3C: Asynchronous Advantage Actor-critic

#### 1.1 N-step bootstrapping

#### 1.2 Parallel Training instead of replay buffer

The main reason that we needed the replay buffer was so we could decorrelate experienced tuples and sequentailly stroing them for later processing. We would then randomly (or weighted) select small mini batches of experiences without each experience being correlated.

A3C replaces the replay buffer with parallel training, by creating multiple instances of the environment and agent, and run them all at the same time. Our agent will receive mini batches of the correlated experiences juist as we need. The samples will be decorrelated because agents will likely be experiencing different states at any given time.

![](images\A3C_parallel.png)

On top of that, this way of training allows us to use on-policy learning in our learning algorithm, which is often associated with more stable learning.

#### 1.3 Asynchronous



#### 1.4 Off-policy vs On-policy

https://leimao.github.io/blog/RL-On-Policy-VS-Off-Policy/

### 2. A2C: Advantage Actor-critic

You may be wondering what the asynchronous part in A3C is about? Recall, Asynchronous Advantage Actor-Critic. 

Let me explain. A3C accumulates gradient updates and applies those updates asynchronously to a global neuronetwork (trainable network). Each agent in simulation does this at its own time.
So, the agents(contains a local network) use a local copy of the network (uses the weight of the trained network) to collect experience, calculate, and accumulate gradient updates across multiple time steps,
and then they apply these gradients to a global network asynchronously. A synchronous here means that each agent will update the network on its own.

There is no synchronization between the agents which also means that the weights an agent is using might
be different from the weights in use by another agent at any given time.
There is a synchronous implementation of A3C called Advantage Actor-Critic, A2C.

![](images\A2C_sync.png)

A2C has some extra bit of code that synchronizes all agents as seen in the above picture.
It waits for all agents to finish a segment of interaction with its copy of the environment,
and then updates the network all at once, before sending the updated weights back to all agents.
A2C is arguably simpler to implement, yet it gives pretty much the same result,
and allegedly in some cases performs even better. (When????)

A3C is most easily train on a CPU, while A2C is more straightforward to extend to a GPU implementation.

#### 2.1 Code

The code can be found here, (Later i will write it here)

https://github.com/ShangtongZhang/DeepRL (Adopt the codestyle aswell, very pretty)

### 3. GAE: Generalized Advantage Estimation

There is another way for estimating expected returns called the lambda return.
To givev a intuition of how it works, say after you try N-step bootstrapping you realize
that numbers of N larger than one often perform better, but it's still hard to tell what the number should be. Should it be a two, three, six or something else?

To make the decision even more difficult, in some problems small numbers of n are better while in others,
large numbers of n are better. How do you get this right?

The idea of the lambda return is to create a mixture of all n-step bootstrapping estimates out once.

Lambda is a hyperparameter used for waiting the combination of each n-step estimate to the lambda return.

![](images\GAE.png)



Say you set $\lambda = 0.5$ .
The contribution to the $\lambda$ return would be a combination of all N-step returns weighted by the exponentially decaying factor across the different n-step estimates.

Notice how the weight depends on the value of lambda you set and it decays exponentially at the rate of that value. So, for calculating the lambda return for state s at time step **t**, we would use all N-step returns and multiply each of the n-step return by the currents bonding weight.
Then add them all up. Remember, that sum will be the lambda return for state **s** at time step **t**. 

![](images\GAE2.png)

Interestingly, when lambda is set to zero, the two-step, three-step and all n-step return other than the one step return, will be equal to zero. So, the lambda return when lambda is set to zero will be equivalent to the td estimate.

If your lambda is set to one all N-step return other than the infinite step return will be equal to zero.
So, the lambda return when lambda is set to one, will be equivalent to the Monte Carlo estimate.

Again, a number between zero and one gives a mixture of all n-step bootstrapping estimate.

Generalized Advantage Estimation, GAE, is a way to train the critic with this lambda return.
You can fit the advantage function just like in A3C and A2C or using a mixture of n-step bootstrapping estimates.
It's important to highlight that this type of return can be combined with virtually any policy-based method.
In fact, in the paper that introduce GAE, TRPO was that policy-based method used.
By using this type of estimation, this algorithm, TRPO plus GAE trains very quickly because multiple value functions are spread around on every time step due to the lambda return star estimate.

[*GAE paper*](https://arxiv.org/abs/1506.02438)

### 4. DDPG: Deep Deterministic Policy Gradient

DDPG is a different kind of actor-critic method. In fact, it could be seen as approximate DQN,
instead of an actual actor critic. The reason for this is that the critic in DDPG, is used to approximate the maximizer over the Q values of the next state, and not as a learned baseline, as we have seen so far.

Though, this is still a very important algorithm and it is good to discuss it in more detail.
One of the limitations of the DQN agent is that it is not straightforward to use in continuous action spaces.
Imagine a DQN network that takes inner state and outputs the action value function. 

**For example**, for two action, say, up and down:
Q(s,"up") gives you the estimated expected value for selecting the up action in state s, say minus 2.18.
Q(s, "down"), gives you the estimated expected value for selecting the down action in state s, say 8.45.

To find the max action value function for this state, you just calculate the max of these values. Pretty easy.
It's very easy to do a max operation in this example because this is a discrete action space.
Even if you had more actions say a left, a right, a jump and so on, you still have a discrete action space.
Even if it was high dimensional with many, many more actions, it would still be doable.

But why do you need an action with continuous range? 
How do you get the value of a continuous action with this architecture?

Say you want the jump action to be continuous, a variable between one and 100 centimeters.
How do you find the value of jump, say 50 centimeters.

This is one of the problems DDPG solves.
In DDPG, we use two deep neural networks. We can call one the **actor** and the other the **critic**.
Nothing new to this point.

![](images\DDPG_arch.png)

Now, the **actor** here is used to approximate the optimal policy deterministically.
That means we want to always output the best believed action for any given state.
This is unlike a *stochastic policies* in which we want the policy to learn a probability distribution over the actions.

![](images\DDPG_actor_arch.png)

In DDPG, we want the believed best action every single time we query the actor network.
That is a deterministic policy.
The **actor** is basically learning the argmax a $Q(s,a)$, which is the best action.
The **critic** learns to evaluate the optimal action value function by using the actors best believed action.
Again, we use this **actor**, which is an approximate maximizer, to calculate a new target value for training the action value function, much in the way DQN does.

![](images\DDPG_arch2.png)

In the image above we see the Critic use the target value (or the action value since we are in a contiuse action space) as input, to compute the action value.

Two other interesting aspects of DDPG are first, the use of a replay buffer, and second, the soft updates to the target networks.

Since we already know how the replay buffer part works we are going to skip the explaination here, but go to Deep_Q_Network for an explaination. Ijust wanted to mention that DDPG uses a replay buffer.
But the soft updates are a bit different (as we have seen aswell), in DQN, you have two copies of your network weights, the regular and the target network.

In the Atari paper in which DQN was introduced, the target network is updated every 10,000 time steps.
You simply copy the weights of your regular network into your target network. That is the target network is fixed for 10,000 time steps and then he gets a big update (We call this Hard-Update).

In DDPG, you have two copies of your network weights for each network, a regular for the actor, and a regular for the critic, and a target for the actor, and a target for the critic.

But in DDPG, the target networks are updated using a soft updates strategy.
A soft update strategy consists of slowly blending your regular network weights with your target network weights. So, every time step you make your target network be 99.99 percent of
your target network weights and only a 0.01 percent of your regular network weights.
You are slowly mix in your regular network weights into your target network weights.

Recall, the regular network is the most up today network because it's their one where training, while the target network is the one we use for prediction to stabilize strain.

In practice, you'll get faster convergence by using this update strategy, and in fact, this way for updating the target network weights can be used with other algorithms that use target networks including DQN.

![](images\soft_updates.png)

#### 4.1 DDPG Continues space:

##### 4.1.1 DDPG AGENT:

~~~~python 
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
~~~~



**Noise function to explore:**

~~~~python
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


~~~~



**Replay buffer**

~~~~python
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
~~~~



##### 4.1.2 Network architecture

~~~~python
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
~~~~



##### 4.1.3 Training

~~~~python
def Training(n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores

scores = Training()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
~~~~





### 5. Twin Delayed Deep Deterministic Policy gradient network

https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

paper: 

