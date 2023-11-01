from gym.envs.registration import register

register(
    id='Jackal-v0',
    entry_point='utils.jackal_env.env:Env',
    max_episode_steps=2000,
    reward_threshold=10000.0,
)

register(
    id='Doggo-v0',
    entry_point='utils.doggo_env.env:Env',
    max_episode_steps=2000,
    reward_threshold=10000.0,
)

register(
    id='Turtlebot-easy-v0',
    entry_point='utils.turtlebot_env.env:Env',
    max_episode_steps=2000,
    reward_threshold=10000.0,
)

register(
    id='Turtlebot-medium-v0',
    entry_point='utils.turtlebot_env.env:Env',
    max_episode_steps=2000,
    reward_threshold=10000.0,
)

register(
    id='Turtlebot-hard-v0',
    entry_point='utils.turtlebot_env.env:Env',
    max_episode_steps=2000,
    reward_threshold=10000.0,
)