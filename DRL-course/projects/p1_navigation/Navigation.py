from pathlib import Path

import numpy as np
import torch
from unityagents import UnityEnvironment

from projects.p1_navigation.agents.rainbow_agent import RainbowAgent
from projects.p1_navigation.models.models import NoisyDDQN, DDQN, DistributedNoisyDDQN
from projects.p1_navigation.replay_buffers.per_nstep import PerNStep
from projects.p1_navigation.train import train
from projects.p1_navigation.utils import log

if __name__ == '__main__':
    # take test seed:
    use_test_seed = False
    test_seed = np.random.randint(low=1, high=1000)

    # Seed values from now on.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Game values
    game = "Banana.exe"
    env = UnityEnvironment(file_name=game, seed=test_seed if use_test_seed else seed, no_graphics=False)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # game info:
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    general_info = log.create_general_info("*general info:*", game, seed, state_size, action_size)

    # Model parameters:
    std_init = 0.2
    model_info = log.create_model_info("*model info:*", std_init)

    # PER:
    BUFFER_SIZE = (2 ** 20)  # The space we use to store Experiences
    BATCH_SIZE = 512  # Amount of replays we train on each update.
    RB_method = "nstep_per"  # Choice of replay buffer: nstep_per, (per, replay_buffer)=not_implemented
    PER_e = 0.01  # Epsilon
    PER_a = 0.6  # Alpha
    PER_b = 0.4  # Beta init
    PER_bi = 0.00001  # Beta increase is the increase in taking the most prioritiesed replays
    PER_aeu = 100  # Absolute error upper is the max priority a replay can have
    PER_learn_start = 0  # Used to populated the sumtree with replays
    n_step = 8  # Used in the n-step implementation for choosing how many sequent replays we use.
    per_info = log.create_per_info("*per_info:*", BUFFER_SIZE, BATCH_SIZE,
                                   RB_method, PER_e, PER_a, PER_b, PER_bi, PER_aeu, PER_learn_start, n_step)

    # Double DQN agent
    GAMMA = 0.99  # Future discount value
    TAU = 1e-3  # Amount we update the target model each update session (use_soft_update=True)
    LR = 0.00005  # The learning rate of the model
    opt_eps = 1.5e-4  # Adam epsilon (more info)
    UPDATE_MODEL_EVERY = 10  # The amount of steps between model updates
    UPDATE_TARGET_EVERY = 8000  # The amount of steps between target updates (use_soft_update=Flase)
    use_soft_update = True  # Wether we are updating the model using soft updates or copying model weights over.
    priority_method = "reward"  # Initialised priorities (reward, none=max_val, error=compute_error)

    # Distributed
    atom_size = 51  # Number of atoms
    v_max = 200  # Max value for supprt
    v_min = 0  # Min value for support

    agent_info = log.create_agent_info("*agent info:*", GAMMA, TAU, LR, opt_eps,
                                       UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY, use_soft_update, priority_method,
                                       atom_size, v_max, v_min)

    # Training parameters:
    episodes = 500  # Number of training episodes
    evaluation_interval = 200  # Indicating how often we evaluate the current agent.
    max_t = 1000  # The max number of steps before going into new episode (not used)
    train_info = log.create_train_info("*train_info:*", episodes, evaluation_interval, max_t)
    # Create plot
    plot = True

    # Title of plot:
    title = "model: NoisyDueling, agent: rainbow, NSTEP_PER-" + priority_method + "Update: "
    if use_soft_update:
        title += "soft"
    else:
        title += "hard"

    # Create data folder:
    base_dir = Path("saved", "test0")
    counter = 0
    while base_dir.exists():
        counter += 1
        base_dir = Path("saved", "test" + str(counter))
    base_dir.mkdir(parents=True)
    file = str(Path(base_dir, "model_test.md"))
    print(file)
    save_file = str(Path(base_dir, "rainbow_checkpoint.pth"))
    save_image = str(Path(base_dir, "plot.png"))

    # Write hyperparameters to file
    with open(file, "a+") as f:
        f.write("\n# " + str(title) + "\n\n")
        f.write(general_info + "\n")
        f.write(model_info + "\n")
        f.write(agent_info + "\n")
        f.write(per_info + "\n")
        f.write(train_info + "\n\n")
        f.write("\n## train data: \n\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    support = torch.linspace(v_min, v_max, atom_size).to(device)
    # Create models
    models = (DistributedNoisyDDQN(state_size, action_size, std_init=std_init, support=support,
                                   atom_size=atom_size, seed=seed),
              DistributedNoisyDDQN(state_size, action_size, std_init=std_init, support=support,
                                   atom_size=atom_size, seed=seed))
    # Create N-step PER buffer
    replay_buffer = PerNStep(BUFFER_SIZE,
                             BATCH_SIZE,
                             state_size=state_size,
                             seed=seed,
                             epsilon=PER_e,
                             alpha=PER_a,
                             beta=PER_b,
                             beta_increase=PER_bi,
                             absolute_error_upper=PER_aeu,
                             n_step=n_step,
                             gamma=GAMMA)

    agent = RainbowAgent(state_size,
                         action_size,
                         models,
                         replay_buffer,
                         seed=seed,
                         BATCH_SIZE=BATCH_SIZE,
                         GAMMA=GAMMA,
                         TAU=TAU,
                         LR=LR,
                         UPDATE_MODEL_EVERY=UPDATE_MODEL_EVERY,
                         UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY,
                         use_soft_update=use_soft_update,
                         priority_method=priority_method,
                         PER_learn_start=PER_learn_start,
                         PER_eps=PER_e,
                         n_step=n_step,
                         atom_size=atom_size,
                         support=support,
                         v_max=v_max,
                         v_min=v_min
                         )

    train(agent, brain_name, env, file=file, save_img=save_image, save_file=save_file, n_episodes=episodes,
          evaluation_interval=evaluation_interval, plot=plot, plot_title=title)
