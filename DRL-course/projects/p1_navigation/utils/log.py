def create_model_info(name, std_init):
    m_info = str(name) + "\n"
    m_info += "\tstd_init:" + str(std_init) + "\n"
    return m_info


def create_train_info(name, episodes, evaluation_interval, max_t):
    t_info = str(name) + "\n"
    t_info += "\tepisodes:" + str(episodes) + "\n"
    t_info += "\tevaluation_interval:" + str(evaluation_interval) + "\n"
    t_info += "\tmax_t: " + str(max_t) + "\n"
    return t_info


def create_general_info(name, game, seed, state_size, action_size):
    g_info = str(name) + "\n"
    g_info += "\tgame:" + str(game) + "\n"
    g_info += "\tseed: " + str(seed) + "\n"
    g_info += "\tstate_size: " + str(state_size) + "\n"
    g_info += "\taction_size: " + str(action_size) + "\n"
    return g_info


def create_per_info(name, BUFFER_SIZE, BATCH_SIZE,
                    RB_method, PER_e, PER_a, PER_b, PER_bi, PER_aeu, PER_learn_start, n_step):
    per_info = str(name) + "\n"
    per_info += "\tRB_method:" + str(RB_method) + "\n"
    per_info += "\tBUFFER_SIZE:" + str(BUFFER_SIZE) + "\n"
    per_info += "\tBATCH_SIZE:" + str(BATCH_SIZE) + "\n"
    per_info += "\tPER_e: " + str(PER_e) + "\n"
    per_info += "\tPER_a: " + str(PER_a) + "\n"
    per_info += "\tPER_b: " + str(PER_b) + "\n"
    per_info += "\tPER_bi: " + str(PER_bi) + "\n"
    per_info += "\tPER_aeu: " + str(PER_aeu) + "\n"
    per_info += "\tPER_learn_start " + str(PER_learn_start) + "\n"
    per_info += "\tn_step " + str(n_step) + "\n"
    return per_info


def create_agent_info(name: str,
                      GAMMA: float,
                      TAU: float,
                      LR: float,
                      opt_eps: float,
                      UPDATE_MODEL_EVERY: int,
                      UPDATE_TARGET_EVERY: int,
                      use_soft_update: bool,
                      priority_method: str,
                      atom_size: int,
                      v_max: int,
                      v_min: int) -> str:
    agent_info = str(name) + "\n"
    agent_info += "\tAgent: rainbow\n"
    agent_info += "\tGAMMA: " + str(GAMMA) + "\n"
    agent_info += "\tTAU: " + str(TAU) + "\n"
    agent_info += "\tLR: " + str(LR) + "\n"
    agent_info += "\topt_eps: " + str(opt_eps) + "\n"
    agent_info += "\tUPDATE_MODEL_EVERY: " + str(UPDATE_MODEL_EVERY) + "\n"
    agent_info += "\tUPDATE_TARGET_EVERY: " + str(UPDATE_TARGET_EVERY) + "\n"
    agent_info += "\tuse_soft_update: " + str(use_soft_update) + "\n"
    agent_info += "\tpriority_method: " + str(priority_method) + "\n"
    agent_info += "\tatom_size: " + str(atom_size) + "\n"
    agent_info += "\tv_max: " + str(v_max) + "\n"
    agent_info += "\tv_min: " + str(v_min) + "\n"
    return agent_info
