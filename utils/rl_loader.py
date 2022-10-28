import gym

"""

"""


class RLLoader():
    def __init__(self, env_config, agent_config):
        self.env_config = env_config
        self.agent_config = agent_config

    def env_loader(self):
        if self.env_config['env_name'] == 'Goal':
            env = gym.make('Goal-v0')
            obs_space = observation_space.spaces[0].shape + 1 #((obs), time)
            discrete_act_space = env.action_space.spaces[0].n
            continuous_act_space = env.action_space.spaces[1].spaces[0].shape[0] + env.action_space.spaces[1].spaces[1].shape[0] + env.action_space.spaces[1].spaces[2].shape[0]

        elif self.env_config['env_name'] == 'Platform':
            env = gym.make('Platform-v0')
            obs_space = observation_space.spaces[0].shape + 1 #((obs), time)
            discrete_act_space = env.action_space.spaces[0].n
            continuous_act_space = env.action_space.spaces[1].spaces[0].shape[0] + env.action_space.spaces[1].spaces[1].shape[0] + env.action_space.spaces[1].spaces[2].shape[0]

        elif self.env_config['env_name'] == 'Move':
            env = gym.make('Moving-v0')
            obs_space = env.observation_space.shape
            discrete_act_space = env.action_space.spaces[0].n
            continuous_act_space = env.action_space.spaces[1].shape[0]

        elif self.env_config['env_name'] == 'Hard-Move': # Todo
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            discrete_act_space = env.action_space.spaces[0].n
            continuous_act_space = env.action_space.spaces[1].shape[0]

        elif self.env_config['env_name'] == 'Hard-Goal': # Todo
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            discrete_act_space = env.action_space.spaces[0].n
            continuous_act_space = env.action_space.spaces[1].shape[0]

        elif self.env_config['env_name'] == 'domestic': # Todo
            env = gym.make('nota-its-v0')
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape

        return env, obs_space, discrete_act_space, continuous_act_space

    def agent_loader(self):
        if self.agent_config['agent_name'] == 'Q-PAMDP':
            if False:
                pass
            else:
                from agents.Q_PAMDP import Agent

        elif self.agent_config['agent_name'] == 'PA-DDPG':
            if self.agent_config['extension']['name'] == 'Double':
                from agents.PA_DDPG import Agent
            elif self.agent_config['extension']['name'] == 'Variation1':
                from agents.Target7 import Agent
            elif self.agent_config['extension']['name'] == 'Variation2':
                from agents.Target8 import Agent
            else:
                from agents.PA_DDPG import Agent

        elif self.agent_config['agent_name'] == 'P-DQN':
            if self.agent_config['extension']['name'] == 'Double':
                from agents.P_DQN import Agent
            elif self.agent_config['extension']['name'] == 'Variation1':
                from agents.Target11 import Agent
            elif self.agent_config['extension']['name'] == 'Variation2':
                from agents.Target12 import Agent
            else:
                from agents.P_DQN import Agent

        elif self.agent_config['agent_name'] == 'MP-DQN':
            if self.agent_config['extension']['name'] == 'Double':
                from agents.MP_DQN import Agent
            elif self.agent_config['extension']['name'] == 'Variation1':
                from agents.Target15 import Agent
            elif self.agent_config['extension']['name'] == 'Variation2':
                from agents.Target16 import Agent
            else:
                from agents.MP_DQN import Agent

        elif self.agent_config['agent_name'] == 'HPPO':
            if False:
                pass
            else:
                from agents.HPPO import Agent

        elif self.agent_config['agent_name'] == 'HHQN':
            if False:
                pass
            else:
                from agents.HHQN import Agent

        elif self.agent_config['agent_name'] == 'Target25':
            if False:
                pass
            else:
                from agents.Target25 import Agent

        elif self.agent_config['agent_name'] == 'Target29':
            if False:
                pass
            else:
                from agents.Target29 import Agent

        else:
            raise ValueError('Please try to set the correct Agent')

        return Agent