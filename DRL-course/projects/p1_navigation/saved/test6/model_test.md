
# model: NoisyDueling, agent: rainbow, PER-reward, Update: hard

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
	UPDATE_MODEL_EVERY: 4
	UPDATE_TARGET_EVERY: 1000
	use_soft_update: False
	priority_method: reward

*per_info:*
	use_per:True
	PER_e: 0.01
	PER_a: 0.6
	PER_b: 0.4
	PER_bi: 0.001
	PER_aeu: 1

*train_info:*
	episodes:400
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

	Episode 100	Average Score: 3.56
	Episode 200	Average Score: 10.03
	Episode 300	Average Score: 10.90
	Episode 400	Average Score: 10.66


best score: 20.0 at eps: 184
## test result: 

