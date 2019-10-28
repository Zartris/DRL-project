import torch

from replay_buffers.prioritized_experience_replay import PrioritizedReplayBuffer


class RainbowAgent:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 state_size, action_size, seed, models, continues=False,
                 BUFFER_SIZE=(2 ** 20), BATCH_SIZE=64, GAMMA=0.995, TAU=1e-3, LR=5e-4, UPDATE_EVERY=4
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.DDQN = models[0].to(self.device)
        self.DDQN_target = models[1].to(self.device)
        self.continues = continues

        # Hyper parameters:
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        self.UPDATE_EVERY = UPDATE_EVERY

        # Priority Experience Replay:
        self.memory_tree = PrioritizedReplayBuffer(capacity=self.buffer_size)

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state, eps=0):
        pass

    def learn(self, gamma):
        pass
