import gym
import numpy as np
import gym_goal
import gym_platform
import gym_hybrid
from time import sleep


def goal_env_test():
    env = gym.make('Goal-v0')

    # Check the space
    print(f"########## action ##########")
    print(f'action_space : {env.action_space}')
    print(env.action_space.spaces[0])
    print(env.action_space.spaces[1].spaces[0], env.action_space.spaces[1].spaces[1], env.action_space.spaces[1].spaces[2])
    print(f'discrete_action_space: {env.action_space.spaces[0].n}, \
        continuous_action_space: {env.action_space.spaces[1].spaces[0].shape[0] + env.action_space.spaces[1].spaces[1].shape[0] + env.action_space.spaces[1].spaces[2].shape[0]}')

    action_config = {'disc_act_spaces': env.action_space.spaces[0].n, \
                'cont_act_spaces': np.array([env.action_space.spaces[1].spaces[i].shape[0] for i in range(env.action_space.spaces[0].n)]), \
                'disc_act_max'   : 0, \
                'disc_act_min'   : 0, \
                'cont_act_max'   : np.array([env.action_space.spaces[1].spaces[i].high for i in range(env.action_space.spaces[0].n)]).ravel(), \
                'cont_act_min'   : np.array([env.action_space.spaces[1].spaces[i].low for i in range(env.action_space.spaces[0].n)]).ravel()}

    print(f"action_config: {action_config}")

    print(f"\n########## observation ##########")
    print(f'observation_space : {env.observation_space}')
    print(f'observation_space.spaces[0] : {env.observation_space.spaces[0]}')
    print(f'observation_space.spaces[1] : {env.observation_space.spaces[1]}')
    print(f'observation_space.spaces[0].shape : {env.observation_space.spaces[0].shape}')
    print(f'observation_space.spaces[1].shape : {env.observation_space.spaces[1].shape}')

    # Check the value, rendering
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    print(f'action: {action}')
    print(f'obs: {obs}')
    print(f'reward: {reward}')
    print(f'done: {done}')
    print(f'info: {info}')

    obs = np.append(obs[:-1][0],obs[-1])
    flat_obs = obs

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def platform_env_test():
    env = gym.make('Platform-v0')

    # Check the space
    print(f"########## action ##########")
    print(f'action_space : {env.action_space}')
    print(env.action_space.spaces[0])
    print(env.action_space.spaces[1].spaces[0], env.action_space.spaces[1].spaces[1], env.action_space.spaces[1].spaces[2])
    print(f'discrete_action_space: {env.action_space.spaces[0].n}, \
        continuous_action_space: {env.action_space.spaces[1].spaces[0].shape[0] + env.action_space.spaces[1].spaces[1].shape[0] + env.action_space.spaces[1].spaces[2].shape[0]}')

    action_config = {'disc_act_spaces': env.action_space.spaces[0].n, \
                'cont_act_spaces': np.array([env.action_space.spaces[1].spaces[i].shape[0] for i in range(env.action_space.spaces[0].n)]), \
                'disc_act_max'   : 0, \
                'disc_act_min'   : 0, \
                'cont_act_max'   : np.concatenate([env.action_space.spaces[1].spaces[i].high for i in range(env.action_space.spaces[0].n)]).ravel(), \
                'cont_act_min'   : np.concatenate([env.action_space.spaces[1].spaces[i].low for i in range(env.action_space.spaces[0].n)]).ravel()}

    print(f"action_config: {action_config}")

    print(f"\n########## observation ##########")

    print(f'observation_space : {env.observation_space}')
    print(f'observation_space.spaces[0] : {env.observation_space.spaces[0]}')
    print(f'observation_space.spaces[1] : {env.observation_space.spaces[1]}')
    print(f'observation_space.spaces[0].shape : {env.observation_space.spaces[0].shape}')
    print(f'observation_space.spaces[1].shape : {env.observation_space.spaces[1].shape}')

    # Check the value, rendering
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    print(f'action: {action}')
    print(f'obs: {obs}')
    print(f'reward: {reward}')
    print(f'done: {done}')
    print(f'info: {info}')

    obs = np.append(obs[:-1][0],obs[-1])
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def move_env_test():
    env = gym.make('Moving-v0')
    # env = gym.make(
    #     'Moving-v0', 
    #     seed=0, 
    #     max_turn=1,
    #     max_acceleration=1.0, 
    #     delta_t=0.001, 
    #     max_step=500, 
    #     penalty=0.01
    # )

    # Check the space
    print(f"########## action ##########")
    print(f'action_space : {env.action_space}')
    print(env.action_space.spaces[0])
    print(env.action_space.spaces[1].shape)
    print(f'discrete_action_space: {env.action_space.spaces[0].n}, \
        continuous_action_space: {env.action_space.spaces[1].shape[0]}')

    action_config = {'disc_act_spaces': env.action_space.spaces[0].n, \
                'cont_act_spaces': np.array([1, 1], dtype=np.float32), \
                'disc_act_max'   : 0, \
                'disc_act_min'   : 0, \
                'cont_act_max'   : np.concatenate([env.action_space.spaces[1].high]).ravel(), \
                'cont_act_min'   : np.concatenate([env.action_space.spaces[1].low]).ravel()}

    print(f"action_config: {action_config}")

    print(f"\n########## observation ##########")
    print(f'observation_space : {env.observation_space}')
    print(f'observation_space.spaces.shape : {env.observation_space.shape}')

    # Check the value, rendering
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    print(f'action: {action}')
    print(f'obs: {obs}')
    print(f'reward: {reward}')
    print(f'done: {done}')
    print(f'info: {info}')

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


#Todo
def hard_goal_env_test():
    env = gym.make('Platform-v0')

    # Check the space
    print(f'action_space : {env.action_space}')
    print(env.action_space.spaces[0])
    print(env.action_space.spaces[1].spaces[0], env.action_space.spaces[1].spaces[1], env.action_space.spaces[1].spaces[2])
    print(f'observation_space : {env.observation_space}')
    print(f'observation_space.spaces[0] : {env.observation_space.spaces[0]}')
    print(f'observation_space.spaces[1] : {env.observation_space.spaces[1]}')
    print(f'observation_space.spaces[0].shape : {env.observation_space.spaces[0].shape}')
    print(f'observation_space.spaces[1].shape : {env.observation_space.spaces[1].shape}')

    # Check the value, rendering
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    print(f'action: {action}')
    print(f'obs: {obs}')
    print(f'reward: {reward}')
    print(f'done: {done}')
    print(f'info: {info}')

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


#Todo
def hard_move_env_test():
    env = gym.make('Platform-v0')

    # Check the space
    print(f'action_space : {env.action_space}')
    print(env.action_space.spaces[0])
    print(env.action_space.spaces[1].spaces[0], env.action_space.spaces[1].spaces[1], env.action_space.spaces[1].spaces[2])
    print(f'observation_space : {env.observation_space}')
    print(f'observation_space.spaces[0] : {env.observation_space.spaces[0]}')
    print(f'observation_space.spaces[1] : {env.observation_space.spaces[1]}')
    print(f'observation_space.spaces[0].shape : {env.observation_space.spaces[0].shape}')
    print(f'observation_space.spaces[1].shape : {env.observation_space.spaces[1].shape}')

    # Check the value, rendering
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    print(f'action: {action}')
    print(f'obs: {obs}')
    print(f'reward: {reward}')
    print(f'done: {done}')
    print(f'info: {info}')

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


#Todo
def domestic_env_test():
    env = gym.make('Platform-v0')

    # Check the space
    print(f'action_space : {env.action_space}')
    print(env.action_space.spaces[0])
    print(env.action_space.spaces[1].spaces[0], env.action_space.spaces[1].spaces[1], env.action_space.spaces[1].spaces[2])
    print(f'observation_space : {env.observation_space}')
    print(f'observation_space.spaces[0] : {env.observation_space.spaces[0]}')
    print(f'observation_space.spaces[1] : {env.observation_space.spaces[1]}')
    print(f'observation_space.spaces[0].shape : {env.observation_space.spaces[0].shape}')
    print(f'observation_space.spaces[1].shape : {env.observation_space.spaces[1].shape}')

    # Check the value, rendering
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    print(f'action: {action}')
    print(f'obs: {obs}')
    print(f'reward: {reward}')
    print(f'done: {done}')
    print(f'info: {info}')

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


if __name__ == "__main__":
    """
    Env
    1: Goal, 2: Platform 3: Catch point
    4: Hard Goal, 5: Hard move, 6: Domestic
    """

    env_switch = 1

    if env_switch == 1:
        goal_env_test()
    elif env_switch == 2:
        platform_env_test()
    elif env_switch == 3:
        move_env_test()
    elif env_switch == 4: # Todo
        hard_goal_env_test()
    elif env_switch == 5: # Todo
        hard_move_env_test()
    elif env_switch == 6:
        domestic_env_test()
    else:
        raise ValueError('Please check the env switch. You could use switch in {1, 2, ... , 6}')