
# model: NoisyDueling, agent: rainbow, PER-none, Update: hard

*general info:*
	game:Banana.exe
	seed: 0
	state_size: 37
	action_size: 4

*agent info:*
	Agent: rainbow
	continues: False
	BUFFER_SIZE: 262144
	BATCH_SIZE: 124
	GAMMA: 0.99
	TAU: 0.0001
	LR: 0.0005
	opt_eps: 0.00015
	UPDATE_MODEL_EVERY: 4
	UPDATE_TARGET_EVERY: 1000
	use_soft_update: False
	priority_method: none

*per_info:*
	use_per:True
	PER_e: 0.01
	PER_a: 0.6
	PER_b: 0.4
	PER_bi: 0.001
	PER_aeu: 1
	PER_learn_start 20000

*train_info:*
	episodes:2000
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

