import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from replay_buffers.experience_replay import ReplayBuffer
from replay_buffers.prioritized_experience_replay import PrioritizedReplayBuffer


class RainbowAgent:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 state_size, action_size, models, use_noise, seed, continues=False,
                 BUFFER_SIZE=(2 ** 20), BATCH_SIZE=64, GAMMA=0.99, TAU=1e-3, LR=5e-4, UPDATE_MODEL_EVERY=4,
                 UPDATE_TARGET_EVERY=1000, use_soft_update=False, priority_method="reward", per=True,
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
        self.priority_method = priority_method
        self.per = per
        self.t_step = 0

        # Double DQN or QN:
        self.use_noise = use_noise
        self.model = models[0].to(self.device)
        self.model_target = models[1].to(self.device)
        self.continues = continues
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Priority Experience Replay:
        if self.per:
            self.memory = PrioritizedReplayBuffer(capacity=self.buffer_size, batch_size=self.batch_size,
                                                  seed=self.seed, epsilon=PER_e, alpha=PER_a, beta=PER_b,
                                                  beta_increase=PER_bi, absolute_error_upper=PER_aeu)
        else:
            self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed, self.device)

        # plotting:
        self.losses = []

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
        if self.per:
            if self.priority_method == "None":
                error = None
            elif self.priority_method == "error":
                error = self.compute_error(state, action, reward, next_state, done)
            else:
                error = reward
            self.memory.add((state, action, reward, next_state, done), error)
        else:
            self.memory.add(state, action, reward, next_state, done)
        # Update t_step:
        self.t_step += 1
        if self.t_step % self.UPDATE_MODEL_EVERY == 0 and self.t_step % self.UPDATE_TARGET_EVERY == 0:
            self.t_step = 0

        # Learn every UPDATE_EVERY time steps.
        if self.t_step % self.UPDATE_MODEL_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                loss = self.learn()
                self.losses.append(loss)
                if self.use_soft_update:
                    self.soft_update(self.model, self.model_target, self.tau)

        if not self.use_soft_update and self.t_step % self.UPDATE_TARGET_EVERY == 0:
            self.update_target_model()

    def act(self, state, eps=0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # TODO: Question for reviewers, should i disable noise here? hence put it on eval mode
        #       I do see a faster growth in avg score in shorter training period,
        #       but this might be from less exploring, hence might be bad in the longer run.
        #       The noise is applied to the sampled memories under model update,
        #       so this is why we might not need it here?
        self.model.eval()
        with torch.no_grad():
            action_values = self.model.forward(state)
        self.model.train()
        # Epsilon-greedy action selection
        if np.random.random() > eps or self.use_noise:
            return np.argmax(action_values.detach().cpu().numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.

                Params
                ======
                    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
                    gamma (float): discount factor
                """
        if self.per:
            idxs, experiences, is_weights = self.memory.sample()
            is_weights = torch.from_numpy(is_weights).float().to(self.device)
            states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
            if self.continues:
                actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(
                    self.device)
            else:
                actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(
                    self.device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(
                self.device)
            dones = torch.from_numpy(
                np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
                self.device)
        else:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

        # Getting the max action of local network (using weights w)
        max_actions = self.model.forward(next_states).detach().max(1)[1].unsqueeze(1)

        # Getting the Q-value for these actions (using weight w^-)
        Q_targets_next = self.model_target.forward(next_states).detach().gather(1, max_actions)

        # Compute Q targets for current states (TD-target)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.model(states).gather(1, actions)

        if self.per:
            errors = torch.abs(Q_expected - Q_targets).detach().cpu()
            self.memory.update_memory_tree(idxs, errors)
            loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Compute loss
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # todo::Consider resetting noise here
        self.model.reset_noise()
        self.model_target.reset_noise()
        return loss.item()

    def compute_error(self, state, action, reward, next_state, done):
        self.model.eval()
        self.model_target.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            action = torch.as_tensor(action).to(self.device)
            val, max_actions_Snext_local = self.model_target(next_state).detach().max(0)

            # Getting the Q-value for these actions (using weight w^-)
            Q_targets_next = self.model_target(next_state).detach()[max_actions_Snext_local]

            # Compute Q targets for current states (TD-target)
            Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))

            # Get expected Q values from local model
            Q_expected = self.model(state)[action]

            error = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
        self.model.train()
        self.model_target.train()
        return error

    def update_target_model(self):
        """ Hard update model parameters.
            Copying the current weights from DDQN to the target model.
        """
        self.model_target.load_state_dict(self.model.state_dict())

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
