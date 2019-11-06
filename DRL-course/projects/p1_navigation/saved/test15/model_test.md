
# model: NoisyDueling, agent: rainbow, PER-none, Update: hard

*general info:*
	game:Banana.exe
	seed: 0
	state_size: 37
	action_size: 4

*agent info:*
	Agent: rainbow
	continues: False
	BUFFER_SIZE: 8192
	BATCH_SIZE: 124
	GAMMA: 0.99
	TAU: 0.0001
	LR: 0.0005
	UPDATE_MODEL_EVERY: 6
	UPDATE_TARGET_EVERY: 1000
	use_soft_update: False
	priority_method: none

*per_info:*
	use_per:True
	PER_e: 0.01
	PER_a: 0.6
	PER_b: 0.4
	PER_bi: 0.001
	PER_aeu: 2

*train_info:*
	episodes:2000
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

	Episode 100	Average Score: 0.42
	Episode 200	Average Score: 5.16
	Episode 300	Average Score: 8.89
	Episode 400	Average Score: 9.69
	Episode 500	Average Score: 10.21
	Episode 600	Average Score: 9.44
