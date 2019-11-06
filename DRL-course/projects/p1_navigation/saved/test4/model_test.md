
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
	PER_e: 1e-06
	PER_a: 0.2
	PER_b: 0.6
	PER_bi: 1e-05
	PER_aeu: 2

*train_info:*
	episodes:800
	max_t: 1000
	eps_start: 1.0
	eps_end: 0.01
	eps_decay: 0.995



## train data: 

	Episode 100	Average Score: 5.46
	Episode 200	Average Score: 9.75
	Episode 300	Average Score: 9.14
	Episode 400	Average Score: 11.09
	Episode 500	Average Score: 11.42
	Episode 600	Average Score: 10.53
	Episode 700	Average Score: 11.41
	Episode 800	Average Score: 11.47


best score: 23.0 at eps: 455
## test result: 

	Episode 100	Average Score: 12.87
