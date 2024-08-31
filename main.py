def make_env(gym_id, seed, idx, capture_video, run_name):
    def env_create():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.unwrapped.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_create


def rollout(envs, agent, num_steps, gamma = 0.99, gae_lambda = 0.95):
    
    
    observations = torch.zeros((num_steps , num_envs) + envs.single_observation_space.shape, dtype=torch.float32)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, dtype=torch.int32)
    rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    values = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    dones = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    truncs = torch.zeros((num_steps, num_envs), dtype=torch.float32)

    next_obs = torch.Tensor(envs.reset()[0])
    # next_done = torch.zeros(num_envs)
    # next_trunc = torch.zeros(num_envs)
    
    for step in range(num_steps):
        observations[step] = next_obs
        #dones[step] = next_done
        #truncs[step] = next_trunc
        
        with torch.no_grad():
            action, logprob, _, value = agent.get_action(next_obs)
            
        actions[step] = action
        values[step] = value.view(-1)
        logprobs[step] = logprob
        
        next_obs, reward, next_done, next_trunc, _ = envs.step(action.cpu().numpy())
        next_obs = torch.Tensor(next_obs)
        rewards[step] = torch.tensor(reward)
        
    with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
    
    return observations, actions, returns, values, advantages, logprobs, dones, truncs
        
    
