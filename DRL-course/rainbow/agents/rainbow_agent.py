import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from replay_buffers.prioritized_experience_replay import PrioritizedReplayBuffer


class RainbowAgent:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 state_size, action_size, models, seed, continues=False,
                 BUFFER_SIZE=(2 ** 20), BATCH_SIZE=64, GAMMA=0.995, TAU=1e-3, LR=5e-4, UPDATE_MODEL_EVERY=4,
                 UPDATE_TARGET_EVERY=1000, use_soft_update=False,
                 PER_e=0.01, PER_a=.6, PER_b=.4, PER_bi=0.001, PER_aeu=3
                 ):
        # seed for comparison
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Hyper parameters:
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        self.UPDATE_MODEL_EVERY = UPDATE_MODEL_EVERY
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY
        self.use_soft_update = use_soft_update
        self.t_step = 0

        # Double DQN:
        self.DDQN = models[0].to(self.device)
        self.DDQN_target = models[1].to(self.device)
        self.continues = continues
        self.optimizer = optim.Adam(self.DDQN.parameters(), lr=self.lr)

        # Priority Experience Replay:
        self.memory_tree = PrioritizedReplayBuffer(capacity=self.buffer_size, batch_size=self.batch_size,
                                                   seed=self.seed, epsilon=PER_e, alpha=PER_a, beta=PER_b,
                                                   beta_increase=PER_bi, absolute_error_upper=PER_aeu)

    def step(self, state, action, reward, next_state, done):
        """Saves learning experience in memory tree and decides if it is time to update models.

        Params
        ======
            state: The state we are moving from.
            action: What action we took.
            reward: The reward from going from state to the next state
            next_state: The state we end up in.
            done: If the game terminated after this step.
        """
        # Save experience in replay memory
        error = reward
        self.memory_tree.add((state, action, reward, next_state, done), error)

        # Learn every UPDATE_EVERY time steps.
        if self.t_step == self.t_step % self.UPDATE_MODEL_EVERY:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory_tree) > self.batch_size:
                self.learn()
                if self.use_soft_update:
                    self.soft_update(self.DDQN, self.DDQN_target, self.tau)

        if not self.use_soft_update and self.t_step % self.UPDATE_TARGET_EVERY == 0:
            self.update_target_model()

        # Update t_step:
        self.t_step += 1
        if self.t_step % self.UPDATE_MODEL_EVERY == 0 and self.t_step % self.UPDATE_TARGET_EVERY == 0:
            self.t_step = 0

    def act(self, state, eps=0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.DDQN.eval()
        with torch.no_grad():
            action_values = self.DDQN(state)
        self.DDQN.train()

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.

                Params
                ======
                    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
                    gamma (float): discount factor
                """
        idxs, experiences, is_weights = self.memory_tree.sample()
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

        # Getting the max action of local network (using weights w)
        max_actions = self.DDQN(next_states).detach().max(1)[1].unsqueeze(1)

        # Getting the Q-value for these actions (using weight w^-)
        Q_targets_next = self.DDQN_target(next_states).detach().gather(1, max_actions)

        # Compute Q targets for current states (TD-target)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.DDQN(states).gather(1, actions)

        errors = torch.abs(Q_expected - Q_targets).detach().cpu()
        self.memory_tree.update_memory_tree(idxs, errors)

        # Compute loss
        loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        # loss = self.my_weighted_mse(Q_expected, Q_targets, is_weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """ Hard update model parameters.
            Copying the current weights from DDQN to the target model.
        """
        self.DDQN_target.load_state_dict(self.DDQN.state_dict())

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
