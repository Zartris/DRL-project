
# model: NoisyDueling, agent: rainbow, NSTEP_PER-rewardUpdate: soft

*general info:*
	game:Banana.exe
	seed: 0
	state_size: 37
	action_size: 4

*model info:*
	std_init:0.2

*agent info:*
	Agent: rainbow
	GAMMA: 0.99
	TAU: 0.001
	LR: 5e-05
	opt_eps: 0.00015
	UPDATE_MODEL_EVERY: 10
	UPDATE_TARGET_EVERY: 8000
	use_soft_update: True
	priority_method: reward
	atom_size: 51
	v_max: 1
	v_min: -1

*per_info:*
	RB_method:nstep_per
	BUFFER_SIZE:1048576
	BATCH_SIZE:512
	PER_e: 0.01
	PER_a: 0.6
	PER_b: 0.4
	PER_bi: 1e-05
	PER_aeu: 100
	PER_learn_start 0
	n_step 8

*train_info:*
	episodes:500
	evaluation_interval:200
	max_t: 1000



## train data: 

	Episode 100	Average Score: 1.11
