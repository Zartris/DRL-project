
# model: NoisyDueling, agent: rainbow, PER-none, Update: hard

*general info:*
	game:Banana.exe
	seed: 0
	state_size: 37
	action_size: 4

*agent info:*
	Agent: rainbow
	continues: False
	BUFFER_SIZE: 16384
	BATCH_SIZE: 64
	GAMMA: 0.99
	TAU: 0.0001
	LR: 0.0001
	UPDATE_MODEL_EVERY: 2
	UPDATE_TARGET_EVERY: 1000
	use_soft_update: False
	priority_method: none

*per_info:*
	use_per:True
	PER_e: 1e-05
	PER_a: 0.6
	PER_b: 0.4
	PER_bi: 0.0001
	PER_aeu: 2

*train_info:*
	episodes:800
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

	Episode 100	Average Score: 2.84
	Episode 200	Average Score: 3.46
