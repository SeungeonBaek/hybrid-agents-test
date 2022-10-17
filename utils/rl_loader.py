import gym


class RLLoader():
    def __init__(self, env_config, agent_config):
        self.env_config = env_config
        self.agent_config = agent_config

    def env_loader(self): # Todo
        if self.env_config['env_name'] == 'LunarLanderContinuous-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'LunarLanderContinuous-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'LunarLanderContinuous-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'LunarLanderContinuous-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'LunarLanderContinuous-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'domestic':
            env = gym.make('nota-its-v0')
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape

        return env, obs_space, act_space

    def agent_loader(self):
        if self.agent_config['agent_name'] == 'Q-PAMDP':
            if False:
                pass
            else:
                from agents.Q_PAMDP import Agent

        elif self.agent_config['agent_name'] == 'PA-DDPG':
            if False:
                pass
            else:
                from agents.PA_DDPG import Agent

        elif self.agent_config['agent_name'] == 'P-DQN':
            if False:
                pass
            else:
                from agents.P_DQN import Agent

        elif self.agent_config['agent_name'] == 'MP-DQN':
            if False:
                pass
            else:
                from agents.MP_DQN import Agent

        elif self.agent_config['agent_name'] == 'HPPO':
            if False:
                pass
            else:
                from agents.HPPO import Agent

        else:
            raise ValueError('Please try to set the correct Agent')

        return Agent