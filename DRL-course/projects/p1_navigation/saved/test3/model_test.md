
# model: NoisyDueling, agent: rainbow, PER-reward, Update: hard

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
	episodes:800
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

	Episode 100	Average Score: 5.44
	Episode 200	Average Score: 7.42
	Episode 300	Average Score: 8.83
	Episode 400	Average Score: 10.46
	Episode 500	Average Score: 11.17
	Episode 600	Average Score: 13.13
	Episode 700	Average Score: 12.50
	Episode 800	Average Score: 10.71


best score: 24.0 at eps: 503
## test result: 

	Episode 100	Average Score: 11.43
