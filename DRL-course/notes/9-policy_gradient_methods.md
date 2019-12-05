# Policy Gradient Methods

Policy gradient methods are a subset of the policy-based methods. Here we use the gradient decent method to optimize the correct action.

In general Policy based methods is very similar to supervied learning here is why and what the differences is.

##### What are Policy Gradient Methods?

- **Policy-based methods** are a class of algorithms that search directly for the optimal policy, without simultaneously  maintaining value function estimates.
- **Policy gradient methods** are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent.
- We can represent the policy with a neural network, where our goal is to find the weights $\theta$ of the network that maximize expected return.

##### The Big Picture:

- The policy gradient method will iteratively amend the policy network weights to:
  - make (state, action) pairs that resulted in positive return more likely, and
  - make (state, action) pairs that resulted in negative return less likely.

##### Problem Setup:

- A **trajectory** $\tau$ is a state-action sequence $s_0, a_0, \ldots, s_H, a_H, s_{H+1}$.
- In this lesson, we will use the notation $R(\tau)$ to refer to the return (rewards) corresponding to trajectory $\tau$.
- Our goal is to find the weights $\theta$ of the policy network to maximize the **expected return** $U(\theta) := \sum_\tau \mathbb{P}(\tau;\theta)R(\tau)$.  

![](D:/dev/learning/DRL-project/DRL-course/notes/images/understand_expected_return.png)

#### 1. Supervised vs reinforcement learning

Here is for future reading: http://karpathy.github.io/2016/05/31/rl/



#### 2. REINFORCE method

we learned that our goal is to find the values of the weights $\theta$ in the neural network that maximize the expected return $U$

$U(\theta) = \sum_\tau P(\tau;\theta)R(\tau)$

where $\tau$ is an arbitrary trajectory. One way to determine the value of $\theta$ that maximizes this function is through **gradient ascent**. This algorithm is closely related to **gradient descent**, where the differences are that:

- gradient descent is designed to find the **minimum** of a function, whereas gradient ascent will find the **maximum**, and
- gradient descent steps in the direction of the **negative gradient**, whereas gradient ascent steps in the direction of the **gradient**.

Our update step for gradient ascent appears as follows:

$\theta \leftarrow \theta + \alpha \nabla_\theta U(\theta)$

where $\alpha$ is the step size that is generally allowed to decay over time. Once we know how to calculate or estimate this gradient, we can repeatedly apply this update step, in the hopes that $\theta$ converges to the value that maximizes $U(\theta)$.

- The pseudocode for REINFORCE is as follows:
  1. Use the policy $\pi_\theta$ to collect $m$ trajectories $\{ \tau^{(1)}, \tau^{(2)}, \ldots, \tau^{(m)}\}$ with horizon $H$.  We refer to the $i$-th trajectory as $\tau^{(i)} = (s_0^{(i)}, a_0^{(i)}, \ldots, s_H^{(i)}, a_H^{(i)}, s_{H+1}^{(i)})$.
  2. Use the trajectories to estimate the gradient $\nabla_\theta U(\theta)$: $\nabla_\theta U(\theta) \approx \hat{g} := \frac{1}{m}\sum_{i=1}^m \sum_{t=0}^{H}  \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) R(\tau^{(i)})$
  3. Update the weights of the policy: $\theta \leftarrow \theta + \alpha \hat{g}$
  4. Loop over steps 1-3.

Here is the formula for computing the gradient ascent, and it is simple because we have some assumption that have to be met: $\tau$ corresponds to 1 whole episode and we compute the gradient ascent on only one trajectory.

![](D:/dev/learning/DRL-project/DRL-course/notes/images/simple_gradient_ascent.png)

So now that this is explained we can look at the one without assumption:
![](D:/dev/learning/DRL-project/DRL-course/notes/images/not_so_simple_gradient_ascent.png)

Here we sum over the $m$ trajectories we have collected and compute the gradient ascent for each visited pair of state action pair. Then we take the average of the gradient found.

##### Derivation:

- We derived the **likelihood ratio policy gradient**: $\nabla_\theta U(\theta) = \sum_\tau \mathbb{P}(\tau;\theta)\nabla_\theta \log \mathbb{P}(\tau;\theta)R(\tau)$
- We can approximate the gradient above with a sample-weighted average: $\nabla_\theta U(\theta) \approx \frac{1}{m}\sum_{i=1}^m \nabla_\theta \log \mathbb{P}(\tau^{(i)};\theta)R(\tau^{(i)}) $.
- We calculated the following: $\nabla_\theta \log \mathbb{P}(\tau^{(i)};\theta) = \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta (a_t^{(i)}|s_t^{(i)})$

##### Code for REINFORCE

Code agent:

~~~~python
env = gym.make('CartPole-v0')
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        # Create model
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
       	# Categorical creates a distribution
        m = Categorical(probs)
        # Get an action based on the distribution above.
        action = m.sample()
        return action.item(), m.log_prob(action)
~~~~

Code Training:

~~~~python
policy = Policy().to(device)
# Using adam optimizer
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # Compute 
        policy_loss = []
        for log_prob in saved_log_probs:
            # The minus is for computing the gradient ascent
            # Since θ←θ + α∇θ U(θ) is the gradient ascent update
            # But the optimizer is doing θ←θ - α∇θ U(θ) : Descenting
            # but by adding the minus to the U(θ) we fix this.
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum() # U(θ) = ∑_τ(P(τ;θ)R(τ))
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores
    
scores = reinforce()
~~~~



____

#### 3. Proximal Policy Optimization

State-of-the-art RL algorithms contain many important tweaks in  addition to simple value-based or policy-based methods. One of these key improvements is called **Proximal Policy Optimization (PPO)** -- also  closely related to **Trust Region Policy Optimization (TRPO)**. It has  allowed faster and more stable learning. From developing agile robots,  to creating expert level gaming AI, PPO has proven useful in a wide  domain of applications, and has become part of the standard toolkits in  complicated learning environments.

Here we will first review the most basic policy gradient  algorithm -- REINFORCE, and discuss issues associated with the  algorithm. We will get an in-depth understanding of why these problems  arise, and find ways to fix them. The solutions will lead us to **PPO**. Our lesson will focus on learning the intuitions behind why and how **PPO**  improves learning, and implement it to teach a computer to play  Atari-Pong, using only the pixels as input (see video below). Let's dive in!

*The idea of PPO was published by the team at OpenAI, and you can read their paper through this [link](https://arxiv.org/abs/1707.06347)*

#### 3.1 Breife recap of REINFORCE

Here, we briefly review key ingredients of the REINFORCE algorithm. 

REINFORCE works as follows: First, we initialize a random policy $\pi_\theta(a;s)$, and using the policy we collect a trajectory -- or a list of (state, actions, rewards) at each time step:

$s_1, a_1, r_1, s_2, a_2, r_2, \dots$

Second, we compute the total reward of the trajectory $R=r_1+r_2+r_3+\dots$, and compute an estimate the gradient of the expected reward, $g$:
$$
g = R \sum_t \nabla_\theta \log\pi_\theta(a_t|s_t) =R \sum_t \frac{d}{d_\theta} \log\pi_\theta(a_t|s_t)
$$
Third, we update our policy using gradient ascent with learning rate $\alpha$:
$$
\theta \leftarrow \theta + \alpha g
$$
The process then repeats.



What are the main problems of REINFORCE? There are three issues:

1. The update process is very **inefficient**! We run the policy once, update once, and then throw away the trajectory.
2. The gradient estimate ggg is very **noisy**. By chance the collected trajectory may not be representative of the policy.
3. There is no clear **credit assignment**. A  trajectory may contain many good/bad actions and whether these actions  are reinforced depends only on the final total output.

In the following concepts, we will go over ways to improve the  REINFORCE algorithm and resolve all 3 issues. All of the improvements  will be utilized and implemented in the PPO algorithm.

#### 3.2 Noise Reduction

The way we optimize the policy is by maximizing the average rewards $U(\theta)$. To do that we use stochastic gradient ascent. Mathematically, the  gradient is given by an average over all the possible trajectories, 
$$
\nabla_\theta U(\theta) =  \overbrace{\sum_\tau P(\tau; \theta)}^{ \begin{matrix} \scriptsize\textrm{average over}\\ \scriptsize\textrm{all trajectories} \end{matrix} } \underbrace{\left( R_\tau \sum_t \nabla_\theta \log \pi_\theta(a_t^{(\tau)}|s_t^{(\tau)}) \right)}_{ \textrm{only one is sampled} }
$$
There could easily be well over millions of trajectories for simple problems, and infinite for continuous problems.

For practical purposes, we simply take one trajectory to compute the  gradient, and update our policy. So a lot of times, the result of a  sampled trajectory comes down to chance, and doesn't contain that much  information about our policy. How does learning happen then? The hope is that after training for a long time, the tiny signal accumulates.

The easiest option to reduce the noise in the gradient is to simply  sample more trajectories! Using distributed computing, we can collect  multiple trajectories in parallel, so that it won’t take too much time.  Then we can estimate the policy gradient by averaging across all the  different trajectories
$$
\left. \begin{matrix} s^{(1)}_t, a^{(1)}_t, r^{(1)}_t\\[6pt] s^{(2)}_t, a^{(2)}_t, r^{(2)}_t\\[6pt] s^{(3)}_t, a^{(3)}_t, r^{(3)}_t\\[6pt] \vdots   \end{matrix} \;\; \right\}\!\!\!\! \rightarrow g = \frac{1}{N}\sum_{i=1}^N  R_i \sum_t\nabla_\theta \log \pi_\theta(a^{(i)}_t | s^{(i)}_t)
$$

#### 3.3 Rewards Normalization

There is another bonus for running multiple trajectories: we can  collect all the total rewards and get a sense of how they are  distributed.

In many cases, the distribution of rewards shifts as learning  happens. Reward = $1$ might be really good in the beginning, but really  bad after $1000$ training episode. 

Learning can be improved if we normalize the rewards, where μ\muμ is the mean, and σ\sigmaσ the standard deviation.
$$
R_i \leftarrow \frac{R_i -\mu}{\sigma} \qquad \mu = \frac{1}{N}\sum_i^N R_i \qquad \sigma = \sqrt{\frac{1}{N}\sum_i (R_i - \mu)^2}
$$
(when all the $R_i$ are the same, $\sigma =0$, we can set all the normalized rewards to $0$ to avoid numerical problems)

This batch-normalization technique is also used in many other  problems in AI (e.g. image classification), where normalizing the input  can improve learning.

Intuitively, normalizing the rewards roughly corresponds to picking  half the actions to encourage/discourage, while also making sure the  steps for gradient ascents are not too large/small.

#### 3.4 Credit Assignment

Going back to the gradient estimate, we can take a closer look at the total reward RRR, which is just a sum of reward at each step $R=r_1+r_2+\dots+r_{t-1}+r_t+\dots$
$$
g=\sum_t (...+r_{t-1}+r_{t}+...)\nabla_{\theta}\log \pi_\theta(a_t|s_t)
$$
Let’s think about what happens at time-step ttt. Even before an action is decided, the agent has already received all the rewards up until step $t-1$. So we can think of that part of the total reward as the reward from the past. The rest is denoted as the future reward. 
$$
(\overbrace{...+r_{t-1}}^{\cancel{R^{\rm past}_t}}+ \overbrace{r_{t}+...}^{R^{\rm future}_t})
$$
Because we have a Markov process, the action at time-step $t$ can only affect the future reward, so the past reward shouldn’t be contributing to the policy gradient.  So to properly assign credit to the action $a_t$, we should ignore the past reward. So a better policy gradient would simply have the future reward as the coefficient. 
$$
g=\sum_t R_t^{\rm future}\nabla_{\theta}\log \pi_\theta(a_t|s_t)
$$

#### 



**Notes on Gradient Modification**

You might wonder, why is it okay to just change our gradient?  Wouldn't that change our original goal of maximizing the expected  reward?

It turns out that mathematically, ignoring past rewards might change  the gradient for each specific trajectory, but it doesn't change the **averaged** gradient. So even though the gradient is different during training, on  average we are still maximizing the average reward. In fact, the  resultant gradient is less noisy, so training using future reward should speed things up!
$$

$$
