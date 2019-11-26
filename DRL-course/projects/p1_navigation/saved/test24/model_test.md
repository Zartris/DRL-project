
# model: NoisyDueling, agent: rainbow, NSTEP_PER-rewardUpdate: soft

*general info:*
	game: Banana.exe
	seed: 0
	state_size: 37
	action_size: 4

*model info:*
	std_init:0.2

*agent info:*
	Agent: rainbow
	GAMMA: 0.95
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
	PER_aeu: 1
	PER_learn_start 0
	n_step 20

*train_info:*
	episodes:1000
	evaluation_interval:100
	max_t: 1000

## train data: 

	Episode 100	Average Score: 2.69
	Episode 200	Average Score: 10.46
	Episode 300	Average Score: 12.47
	Episode 400	Average Score: 12.96
	Episode 500	Average Score: 13.21
	Episode 600	Average Score: 12.86
	Episode 700	Average Score: 13.30
	Episode 800	Average Score: 13.24
	Episode 900	Average Score: 13.42
	Episode 1000	Average Score: 13.49

## test result: 

	train_episode: 100	 Average Score over 100 episodes: 9.86
	train_episode: 200	 Average Score over 100 episodes: 9.72
	train_episode: 300	 Average Score over 100 episodes: 13.16
	train_episode: 400	 Average Score over 100 episodes: 12.6
	train_episode: 500	 Average Score over 100 episodes: 12.69
	train_episode: 600	 Average Score over 100 episodes: 13.32
	train_episode: 700	 Average Score over 100 episodes: 13.34
	train_episode: 800	 Average Score over 100 episodes: 13.55
	train_episode: 900	 Average Score over 100 episodes: 13.2
	train_episode: 1000	 Average Score over 100 episodes: 13.22

best score: 21.0 at eps: 276