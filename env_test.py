import gym
import numpy as np
import gym_platform

def goal_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def platform_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def platform_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def catch_point_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def hard_goal_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def hard_move_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

    obs = np.array(obs)
    flat_obs = obs.reshape(-1)

    print('original : ', np.shape(obs))
    print('flat_obs : ', np.shape(flat_obs))


def domestic_env_test():
    env = gym.make('Platform-v0')

    print('action_space : ', env.action_space)
    print(env.action_space.spaces[0].n)
    print(env.action_space.spaces[1][0])
    print('observation_space : ', env.observation_space)
    print('observation_space[0] : ', env.observation_space[0])
    print('observation_space[1] : ', env.observation_space[1])
    print('observation_space[0].shape : ', env.observation_space[0].shape)
    print('observation_space[1].shape : ', env.observation_space[1].shape)

    obs = env.reset()

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

    env_switch = 7

    if env_switch == 1:
        goal_env_test()
    elif env_switch == 2:
        platform_env_test()
    elif env_switch == 3:
        catch_point_env_test()
    elif env_switch == 4:
        hard_goal_env_test()
    elif env_switch == 5:
        hard_move_env_test()
    elif env_switch == 6:
        domestic_env_test()
    else:
        raise ValueError('Please check the env switch. You could use switch in {1, 2, ... , 6}')