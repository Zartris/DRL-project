d
# model: NoisyDueling, agent: rainbow, PER-reward, Update: hard

*general info:*
	game:Banana.exe
	seed: 0
	state_size: 37
	action_size: 4

*agent info:*
	Agent: rainbow
	continues: False
	BUFFER_SIZE: 1048576
	BATCH_SIZE: 32
	GAMMA: 0.99
	TAU: 0.0001
	LR: 0.0005
	opt_eps: 0.00015
	UPDATE_MODEL_EVERY: 4
	UPDATE_TARGET_EVERY: 8000
	use_soft_update: False
	priority_method: reward

*per_info:*
	use_per:True
	PER_e: 0.01
	PER_a: 0.6
	PER_b: 0.4
	PER_bi: 1e-05
	PER_aeu: 1
	PER_learn_start 0

*train_info:*
	episodes:2000
	evaluation_interval:200
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

	Episode 100	Average Score: 0.94
	Episode 200	Average Score: 4.14
	Episode 300	Average Score: 7.10
	Episode 400	Average Score: 10.21
	Episode 500	Average Score: 9.95
	Episode 600	Average Score: 7.69
	Episode 700	Average Score: 11.61
	Episode 800	Average Score: 12.43
	Episode 900	Average Score: 11.83
	Episode 1000	Average Score: 11.57
