import numpy as np
import torch
from torch import optim

from projects.p1_navigation.replay_buffers.per_nstep import PerNStep


class RainbowAgent:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 models: list,
                 replay_buffer: PerNStep,
                 seed: int = None,
                 BATCH_SIZE: int = 64,
                 GAMMA: float = 0.99,
                 TAU: float = 1e-3,
                 LR: float = 5e-4,
                 UPDATE_MODEL_EVERY: int = 4,
                 UPDATE_TARGET_EVERY: int = 1000,
                 use_soft_update: bool = False,
                 priority_method: str = "reward",
                 # PER
                 PER_learn_start: int = 0,
                 PER_eps: float = 1e-6,
                 # N-step
                 n_step: int = 3,
                 # Distributed
                 atom_size: int = 0,
                 support=None,
                 v_max: int = 200,
                 v_min: int = 0):
        # seed for comparison
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # Hyper parameters:
        self.state_size = state_size  # Not used only for debugging
        self.action_size = action_size  # Not used only for debugging
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        self.UPDATE_MODEL_EVERY = UPDATE_MODEL_EVERY
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY
        self.use_soft_update = use_soft_update
        self.priority_method = priority_method
        self.t_step = 0

        # PER:
        self.learn_start = PER_learn_start
        self.PER_eps = PER_eps
        # Double DQN or QN:
        self.model = models[0].to(self.device)
        self.model_target = models[1].to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1.5e-4)

        # N-step:
        self.n_step = n_step

        # Priority Experience Replay:
        self.memory = replay_buffer

        # Distributional aspect:
        # The support for the value distribution. Set to 51 for C51
        self.atom_size = atom_size
        # Break the range of rewards into 51 uniformly spaced values (support)
        self.v_max = v_max
        self.v_min = v_min
        self.support = support

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
        if self.priority_method == "None":
            error = None
        elif self.priority_method == "error":
            error = self.compute_error(state, action, reward, next_state, done)
        else:
            error = reward
        self.memory.add(state, action, reward, next_state, done, error)

        # Filling memory
        if self.learn_start != 0:
            self.learn_start -= 1
            if self.learn_start % 1000 == 0:
                print("\tFilling memory: \t{0}".format(self.learn_start, end="\r"))
            return
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
            print("\tTarget model updated")

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_values = self.model.forward(state)
        return np.argmax(action_values.detach().cpu().numpy())

    def learn(self):
        """Update value parameters using given batch of experience tuples.

                Params
                ======
                    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
                    gamma (float): discount factor
                """
        # PER sampling:
        idxs, experiences, is_weights = self.memory.sample()

        # Compute the error or loss:
        errors = self._compute_loss(experiences)

        # Prepare weights:
        is_weights = torch.from_numpy(is_weights).float().to(self.device)

        # Compute loss - Per: weighted loss computed and the priorities are updated
        loss = torch.mean(is_weights * errors)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities:
        self.memory.update_memory_tree(idxs, errors.detach().cpu())

        # Resetting the noise of the model
        self.model.reset_noise()
        self.model_target.reset_noise()
        return loss.item()  # Return loss for logging

    def _compute_loss(self, experiences):
        """ Computing the loss from the categorical result
        """
        # TODO::Check if gamma should be gamme = gamma ** n_steps
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        # TODO: Check if no_grad() here:
        with torch.no_grad():
            # Double DQN
            next_actions = self.model(next_states).argmax(1)
            actions = actions.reshape(-1)  # Flatten
            next_q_dist = self.model_target.get_distribution(next_states)
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]

            # Compute the projection of Tˆz onto the support {z}
            t_z = rewards + (1 - dones) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            b = (t_z - self.v_min) / delta_z  # b_j ∈ [0, batch_size − 1]
            lower = b.floor().long()
            upper = b.ceil().long()

            # Distribute probability of Tˆz
            offset = (
                torch.linspace(
                    start=0,
                    end=(self.batch_size - 1) * self.atom_size,
                    steps=self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_q_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                dim=0,
                index=(lower + offset).view(-1),
                source=(next_q_dist * (upper.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                dim=0,
                index=(upper + offset).view(-1),
                source=(next_q_dist * (b - lower.float())).view(-1)
            )

        q_dist = self.model.get_distribution(states)
        log_p = torch.log(q_dist[range(self.batch_size), actions])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss  # Cross-entropy loss

    def compute_error(self, state, action, reward, next_state, done):
        """ Compute the error between model and model_target given one experience
        """
        # Set to eval to avoid backpropergation:
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
