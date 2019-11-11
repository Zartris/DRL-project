
# model: NoisyDueling, agent: rainbow, PER-reward, Update: soft

*general info:*
	game:Banana.x86_64
	seed: 0
	state_size: 37
	action_size: 4

*agent info:*
	Agent: rainbow
	continues: False
	BUFFER_SIZE: 1048576
	BATCH_SIZE: 32
	GAMMA: 0.99
	TAU: 0.001
	LR: 5e-05
	opt_eps: 0.00015
	UPDATE_MODEL_EVERY: 4
	UPDATE_TARGET_EVERY: 8000
	use_soft_update: True
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

	Episode 100	Average Score: 1.87
	Episode 200	Average Score: 5.58
	Episode 300	Average Score: 10.44
	Episode 400	Average Score: 13.30
	Episode 500	Average Score: 13.93
	Episode 600	Average Score: 13.70
	Episode 700	Average Score: 13.41
	Episode 800	Average Score: 14.58
	Episode 900	Average Score: 14.90
	Episode 1000	Average Score: 13.60
	Episode 1100	Average Score: 13.23
	Episode 1200	Average Score: 14.07
	Episode 1300	Average Score: 13.79
	Episode 1400	Average Score: 13.72
	Episode 1500	Average Score: 13.86
	Episode 1600	Average Score: 12.92
	Episode 1700	Average Score: 13.82
	Episode 1800	Average Score: 14.47
	Episode 1900	Average Score: 14.23
	Episode 2000	Average Score: 14.16

## test result: 


	train_episode: 200	 Average Score over 100 episodes: 7.16
	train_episode: 400	 Average Score over 100 episodes: 11.26
	train_episode: 600	 Average Score over 100 episodes: 9.41
	train_episode: 800	 Average Score over 100 episodes: 12.13
	train_episode: 1000	 Average Score over 100 episodes: 11.84
	train_episode: 1200	 Average Score over 100 episodes: 11.14
	train_episode: 1400	 Average Score over 100 episodes: 11.64
	train_episode: 1600	 Average Score over 100 episodes: 12.3
	train_episode: 1800	 Average Score over 100 episodes: 10.72
	train_episode: 2000	 Average Score over 100 episodes: 13.52

best score: 26.0 at eps: 888