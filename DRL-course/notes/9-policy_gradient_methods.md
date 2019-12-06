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

#### 3.5 Code so far:

**Agent / policy**

~~~~python
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16
        
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to 
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
~~~~

**Compute the loss:**

~~~~python
# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):

    # Compute the discounted reward
    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

    ratio = new_probs/old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(ratio*rewards + beta*entropy)

~~~~

**Training Code:**

~~~~python
from parallelEnv import parallelEnv
import numpy as np
# WARNING: running through all 800 episodes will take 30-45 minutes

# training loop max iterations
episode = 500
# episode = 800

# widget bar to display progress
!pip install progressbar
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
beta = .01
tmax = 320

# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories Hence play an episode
    old_probs, states, actions, rewards = \
        pong_utils.collect_trajectories(envs, policy, tmax=tmax)
        
    total_rewards = np.sum(rewards, axis=0)

    # this is the SOLUTION!
    # use your own surrogate function
    L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    
    # L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    del L
        
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta*=.995
    
    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    
    # display some progress every 20 iterations
    if (e+1)%20 ==0 :
        print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
        print(total_rewards)
        
    # update progress widget bar
    timer.update(e+1)
    
timer.finish()
~~~~



#### 3.6 Importance sampling

**Policy Update in REINFORCE**

Let’s go back to the REINFORCE algorithm. We start with a policy, $\pi_\theta$, then using that policy, we generate a trajectory (or multiple ones to reduce noise) $(s_t, a_t, r_t)$. Afterward, we compute a policy gradient, $g$, and update $\theta' \leftarrow \theta + \alpha g$.

At this point, the trajectories we’ve just generated are simply  thrown away. If we want to update our policy again, we would need to  generate new trajectories once more, using the updated policy.

You might ask, why is all this necessary? It’s because we need to  compute the gradient for the current policy, and to do that the  trajectories need to be representative of the current policy.

But this sounds a little wasteful. What if we could somehow recycle  the old trajectories, by modifying them so that they are representative  of the new policy? So that instead of just throwing them away, we  recycle them!

Then we could just reuse the recycled trajectories to compute  gradients, and to update our policy, again, and again. This would make  updating the policy a lot more efficient.  So, how exactly would that work?

**Importance sampling:**

This is where importance sampling comes in. Let’s look at the trajectories we generated using the policy $ \pi_\theta$. It had a probability $P(\tau;\theta)$, to be sampled.

Now Just by chance, the same trajectory can be sampled under the new policy, with a different probability $P(\tau;\theta')$

Imagine we want to compute the average of some quantity, say $f(\tau)$. We could simply generate trajectories from the new policy, compute  $f(\tau)$ and average them. 

Mathematically, this is equivalent to adding up all the $f(\tau)$, weighted by a probability of sampling each trajectory under the new policy. 
$$
\sum_\tau P(\tau;\theta') f(\tau)
$$
Now we could modify this equation, by multiplying and dividing by the same number, $P(\tau;\theta)$ and rearrange the terms.
$$
\sum_\tau \overbrace{P(\tau;\theta)}^{ \begin{matrix} \scriptsize \textrm{sampling under}\\ \scriptsize \textrm{old policy } \pi_\theta \end{matrix} }  \overbrace{\frac{P(\tau;\theta')}{P(\tau;\theta)}}^{ \begin{matrix} \scriptsize  \textrm{re-weighting}\\ \scriptsize \textrm{factor} \end{matrix} } f(\tau)
$$
It doesn’t look we’ve done much. But written in this way, we can  reinterpret the first part as the coefficient for sampling under the old policy, with an extra re-weighting factor, in addition to just  averaging.

Intuitively, this tells us we can use old trajectories for computing  averages for new policy, as long as we add this extra re-weighting  factor, that takes into account how under or over–represented each  trajectory is under the new policy compared to the old one.

The same tricks are used frequently across statistics, where the  re-weighting factor is included to un-bias surveys and voting  predictions. 

**The re-weighting factor:**

Now Let’s a closer look at the re-weighting factor. 
$$
\frac{P(\tau;\theta')}{P(\tau;\theta)} =\frac {\pi_{\theta'}(a_1|s_1)\, \pi_{\theta'}(a_2|s_2)\, \pi_{\theta'}(a_3|s_3)\,...} {\pi_\theta(a_1|s_1) \, \pi_\theta(a_2|s_2)\, \pi_\theta(a_2|s_2)\, ...}
$$
Because each trajectory contains many steps, the probability contains a chain of products of each policy at different time-step.

This formula is a bit complicated. But there is a bigger problem.  When some of policy gets close to zero, the re-weighting factor can  become close to zero, or worse, close to 1 over 0 which diverges to  infinity.

When this happens, the re-weighting trick becomes unreliable. So, In  practice, we want to make sure the re-weighting factor is not too far  from 1 when we utilize importance sampling

#### 3.7 Proximal Policy Optimization

**Re-weighting the Policy Gradient**

Suppose we are trying to update our current policy, $\pi_{\theta'}$. To do that, we need to estimate a gradient, $g$. But we only have trajectories generated by an older policy $\pi_{\theta}$. How do we compute the gradient then?

Mathematically, we could utilize importance sampling. The answer just what a normal policy gradient would be, times a re-weighting factor $P(\tau;\theta')/P(\tau;\theta)$:


$$
g=\frac{P(\tau; \theta')}{P(\tau; \theta)}\sum_t  \frac{\nabla_{\theta'} \pi_{\theta'}(a_t|s_t)}{\pi_{\theta'}(a_t|s_t)}R_t^{\rm future}
$$


We can rearrange these equations, and the re-weighting factor is just the product of all the policy across each step -- I’ve picked out the  terms at time-step $t$ here. We can cancel some terms, but we're still left with a product of the policies at different times, denoted by ".........".


$$
g=\sum_t \frac{...\, \cancel{\pi_{\theta'}(a_t|s_t)} \,...} {...\,\pi_{\theta}(a_t|s_t)\,...} \, \frac{\nabla_{\theta'} \pi_{\theta'}(a_t|s_t)}{\cancel{\pi_{\theta'}(a_t|s_t)}}R_t^{\rm future}
$$


Can we simplify this expression further? This is where proximal  policy comes in. If the old and current policy is close enough to each  other, all the factors inside the "........." would be pretty close to 1, and then we can ignore them.

Then the equation simplifies:


$$
g=\sum_t \frac{\nabla_{\theta'} \pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}R_t^{\rm future}
$$


It looks very similar to the old policy gradient. In fact, if the  current policy and the old policy is the same, we would have exactly the vanilla policy gradient. But remember, this expression is different  because we are comparing two *different* policies



**The Surrogate Function**

Now that we have the approximate form of the gradient, we can think  of it as the gradient of a new object, called the surrogate function:


$$
g=\nabla_{\theta'} L_{\rm sur}(\theta', \theta) \\
L_{\rm sur}(\theta', \theta)= \sum_t \frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}R_t^{\rm future}
$$


So using this new gradient, we can perform gradient ascent to update  our policy -- which can be thought as directly maximize the surrogate  function. 

But there is still one important issue we haven’t addressed yet. If  we keep reusing old trajectories and updating our policy, at some point  the new policy might become different enough from the old one, so that  all the approximations we made could become invalid. 

We need to find a way make sure this doesn’t happen.

**The Policy/Reward Cliff**

What is the problem with updating our policy and ignoring the fact  that the approximations are not valid anymore? One problem is it could  lead to a really bad policy that is very hard to recover from. Let's see how:



[![img](https://video.udacity-data.com/topher/2018/September/5b9a9625_policy-reward-cliff/policy-reward-cliff.png)](https://classroom.udacity.com/nanodegrees/nd893/parts/286e7d2c-e00c-4146-a5f2-a490e0f23eda/modules/089d6d51-cae8-4d4b-84c6-9bbe58b8b869/lessons/e6ae0022-3c68-479a-8111-e264de5e34ab/concepts/4a6fa90a-eeb1-4412-ac1b-35d6ca4f3809#)



Say we have some policy parameterized by $\pi_{\theta'}$ (shown on the left plot in black), and with an average reward function (shown on the right plot in black). 

The current policy is labelled by the red text, and the goal is to  update the current policy to the optimal one (green star). To update the policy we can compute a surrogate function $L_{\rm sur}$  (dotted-red curve on right plot). So $L_{\rm sur}$ approximates the reward pretty well around the current policy. But far  away from the current policy, it diverges from the actual reward.

If we continually update the policy by performing gradient ascent, we might get something like the red-dots. The big problem is that at some  point we hit a cliff, where the policy changes by a large amount. From  the perspective of the surrogate function, the average reward is really  great. But the actually average reward is really bad! 

What’s worse, the policy is now stuck in a deep and flat bottom, so  that future updates won’t be able to bring the policy back up! we are  now stuck with a really bad policy.

How do we fix this? Wouldn’t it be great if we can somehow stop the  gradient ascent so that our policy doesn’t fall off the cliff?

**Clipped Surrogate Function**



[![img](https://video.udacity-data.com/topher/2018/September/5b9a99cd_clipped-surrogate/clipped-surrogate.png)](https://classroom.udacity.com/nanodegrees/nd893/parts/286e7d2c-e00c-4146-a5f2-a490e0f23eda/modules/089d6d51-cae8-4d4b-84c6-9bbe58b8b869/lessons/e6ae0022-3c68-479a-8111-e264de5e34ab/concepts/4a6fa90a-eeb1-4412-ac1b-35d6ca4f3809#)



Here’s an idea: what if we just flatten the surrogate function (blue curve)? What would policy update look like then?

So starting with the current policy (blue dot), we apply gradient  ascent. The updates remain the same, until we hit the flat plateau. Now  because the reward function is flat, the gradient is zero, and the  policy update will stop!

Now, keep in mind that we are only showing a 2D figure with one θ′\theta'θ′ direction. In most cases, there are thousands of parameters in a  policy, and there may be hundreds/thousands of high-dimensional cliffs  in many different directions. We need to apply this clipping  mathematically so that it will automatically take care of all the  cliffs. 

**Clipped Surrogate Function**

Here's the formula that will automatically flatten our surrogate function to avoid all the cliffs:
$$
L^{\rm clip}_{\rm sur} (\theta',\theta)= \sum_t \min\left\{ \frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}R_t^{\rm future} ,  {\rm clip}_\epsilon\!\! \left( \frac{\pi_{\theta'}(a_t|s_t)} {\pi_{\theta}(a_t|s_t)} \right) R_t^{\rm future} \right\}
$$
Now let’s dissect the formula by looking at one specific term in the sum, and set the future reward to 1 to make things easier.



[![img](https://video.udacity-data.com/topher/2018/September/5b9a9d58_clipped-surrogate-explained/clipped-surrogate-explained.png)](https://classroom.udacity.com/nanodegrees/nd893/parts/286e7d2c-e00c-4146-a5f2-a490e0f23eda/modules/089d6d51-cae8-4d4b-84c6-9bbe58b8b869/lessons/e6ae0022-3c68-479a-8111-e264de5e34ab/concepts/4a6fa90a-eeb1-4412-ac1b-35d6ca4f3809#)



We start with the original surrogate function (red), which involves the ratio $\pi_{\theta'}(a_t|s_t)/\pi_\theta(a_t|s_t)$. The black dot shows the location where the current policy is the same as the old policy $(\theta'=\theta)$

We want to make sure the two policy is similar, or that the ratio is close to $1$. So we choose a small $\epsilon$ (typically 0.1 or 0.2), and apply the ${\rm clip}$ function to force the ratio to be within the interval $[1-\epsilon,1+\epsilon]$ (shown in purple).

Now the ratio is clipped in two places. But we only want to clip the  top part and not the bottom part. To do that, we compare this clipped  ratio to the original one and take the minimum (show in blue). This then ensures the clipped surrogate function is always less than the original surrogate function $L_{\rm sur}^{\rm clip}\le L_{\rm sur}$, so the clipped surrogate function gives a more conservative "reward".

(*the blue and purple lines are shifted slightly for easier viewing*)