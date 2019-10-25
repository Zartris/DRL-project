import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn_agent import DQNAgent
from experience_replay import PrioritizedReplayBuffer


class DoubleDQNAgent(DQNAgent):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BUFFER_SIZE = (2 ** 20)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

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
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)


class DoubleDQNAgentPER(DQNAgent):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BUFFER_SIZE = (2 ** 20)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network

    def __init__(self, state_size, action_size, seed, continues=False):
        super().__init__(state_size, action_size, seed)
        # Override replaybuffer to use priority replay
        self.memory = PrioritizedReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, seed, self.device)
        self.continues = continues

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        error = reward
        self.memory.add((state, action, reward, next_state, done), error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                self.learn(self.GAMMA)

    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        idxs, experiences, is_weights = self.memory.sample()
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        if self.continues:
            actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self.device)
        else:
            actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        is_weights = torch.from_numpy(is_weights).float().to(self.device)

        "*** YOUR CODE HERE ***"
        # Getting the max action of local network (using weights w)
        max_actions_Snext_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Getting the Q-value for these actions (using weight w^-)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_actions_Snext_local)

        # Compute Q targets for current states (TD-target)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
        self.memory.batch_update(idxs, errors)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        # loss = self.my_weighted_mse(Q_expected, Q_targets, is_weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    @staticmethod
    def my_weighted_mse(Q_expected, Q_targets, is_weights):
        """Custom loss function that takes into account the importance-sampling weights."""
        td_error = Q_expected - Q_targets
        weighted_squared_error = is_weights * td_error * td_error
        return torch.sum(weighted_squared_error) / torch.numel(weighted_squared_error)

    def compute_error(self, state, action, reward, next_state, done):
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        action = torch.as_tensor(action).to(self.device)
        val, max_actions_Snext_local = self.qnetwork_local(next_state).detach().max(0)

        # Getting the Q-value for these actions (using weight w^-)
        Q_targets_next = self.qnetwork_target(next_state).detach()[max_actions_Snext_local]

        # Compute Q targets for current states (TD-target)
        Q_targets = reward + (self.GAMMA * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state)[action]

        error = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
        self.qnetwork_local.train()
        self.qnetwork_target.train()
        return error
