
 # Todo
def env_agent_config(env_switch, agent_switch):
    if env_switch == 1:
        env_config = {'env_name': 'Goal', 'seed': 777, 'render': False, 'max_step': 500, 'max_episode': 50000}
    elif env_switch == 2:
        env_config = {'env_name': 'Platform', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 3:
        env_config = {'env_name': 'Catch point', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 4:
        env_config = {'env_name': 'Hard Goal', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 5:
        env_config = {'env_name': 'Hard move', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 6:
        env_config = {'env_name': 'TrafficSignalControl', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    else:
        raise ValueError('Please try to correct env_switch')

    # Q-PAMDP
    if agent_switch == 1:
        from agent_config import Q_PAMDP_Vanilla_agent_config
        agent_config = Q_PAMDP_Vanilla_agent_config

    # PA-DDPG
    elif agent_switch == 5:
        from agent_config import PA_DDPG_Vanilla_agent_config
        agent_config = PA_DDPG_Vanilla_agent_config

    # P-DQN
    elif agent_switch == 9:
        from agent_config import P_DQN_Vanilla_agent_config
        agent_config = P_DQN_Vanilla_agent_config

   # MP-DQN
    elif agent_switch == 13:
        from agent_config import MP_DQN_Vanilla_agent_config
        agent_config = MP_DQN_Vanilla_agent_config

    # HPPO
    elif agent_switch == 17:
        from agent_config import HPPO_Vanilla_agent_config
        agent_config = HPPO_Vanilla_agent_config

    # HHQN
    elif agent_switch == 21:
        from agent_config import HHQN_Vanilla_agent_config
        agent_config = HHQN_Vanilla_agent_config

    else:
        raise ValueError('Please try to correct agent_switch')

    return env_config, agent_config