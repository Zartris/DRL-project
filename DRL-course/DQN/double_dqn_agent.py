import torch
import torch.nn.functional as F

from dqn_agent import Agent

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DoubleDQNAgent(Agent):
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        "*** YOUR CODE HERE ***"
        # Getting the max action of local network (using weights w)
        max_actions_Snext_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Getting the Q-value for these actions (using weight w^-)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_actions_Snext_local)

        # Compute Q targets for current states (TD-target)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
