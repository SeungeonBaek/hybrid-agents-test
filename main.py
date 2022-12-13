import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter

from agent_env_config import env_agent_config

from utils.rl_logger import RLLogger
from utils.rl_loader import RLLoader

from utils.state_logger import StateLogger


def pad_action(act, act_param, action_config):
    params = np.array([np.zeros((i,), dtype=np.float32) for i in action_config['cont_act_spaces']])
    params[act][:] = act_param

    return (act, params)


def main(env_config: Dict,
         agent_config: Dict,
         rl_confing: Dict,
         rl_custom_config: Dict,
         result_path:str,
         rl_logger: RLLogger,
         rl_loader: RLLoader,
         state_logger: StateLogger):
    # Env
    env, env_obs_space, env_act_space, action_config = rl_loader.env_loader()
    env_name = env_config['env_name']
    print(f"env_name : {env_name}, obs_space : {env_obs_space}, act_space : {env_act_space},\nact_config : {action_config}")

    act_space = env_act_space

    if len(env_obs_space) > 1:
        for space in env_obs_space:
            obs_space *= space
    else:
        obs_space = env_obs_space[0]

    max_step = env_config['max_step']

    # Agent
    RLAgent = rl_loader.agent_loader()
    Agent = RLAgent(agent_config, obs_space, action_config)

    if rl_custom_config['use_learned_model']:
        if os.name == 'nt':
            Agent.load_models(path=result_path + "\\" + str(rl_custom_config['learned_model_score']) + "_model")
        elif os.name == 'posix':
            Agent.load_models(path=result_path + "/" + str(rl_custom_config['learned_model_score']) + "_model")
    else:
        pass

    agent_name = agent_config['agent_name']
    extension_name = Agent.extension_name
    print(f"agent_name: {agent_name}, extension_name: {extension_name}")

    # csv logging
    if rl_confing['csv_logging']:
        state_logger.initialize_memory(env_config['max_episode'], max_step, act_space)

    total_step = 0
    max_score = 0

    for episode_num in range(1, env_config['max_episode']):
        episode_score = 0
        episode_step = 0
        done = False

        prev_obs = None
        prev_action = None
        prev_log_policy = None
        episode_rewards = []

        obs = env.reset()
        if env_name == 'Goal' or env_name == 'Platform':
            obs = np.append(obs[:-1][0],obs[-1])
        elif env_name == 'Move':
            obs = obs

        obs = np.array(obs)
        obs = obs.reshape(-1)

        action = None

        while not done:
            if env_config['render']:
                env.render()
            episode_step += 1
            total_step += 1

            if agent_config['agent_name'] == 'HPPO':
                act, act_param, all_action_parameters, log_policy = Agent.action(obs)
            else:
                act, act_param, all_action_parameters = Agent.action(obs)
            
            action = pad_action(act, act_param, action_config)
            
            if env_name == 'Platform':
                obs, reward, done, _ = env.step(action)
                reward = reward * 10
                obs = np.append(obs[:-1][0], obs[-1]/200)

            elif env_name == 'Goal':
                obs, reward, done, _ = env.step(action)
                obs = np.append(obs[:-1][0], obs[-1])

            elif env_name == 'Move':
                obs, reward, done, _ = env.step(action)

            obs = np.array(obs)
            obs = obs.reshape(-1)

            action = (act, all_action_parameters)

            episode_score += reward
            episode_rewards.append(reward)

            # Save_xp
            if episode_step >= 2:
                if agent_config['agent_name'] == 'HPPO':
                    Agent.save_xp(prev_obs, obs, reward, prev_action, prev_log_policy, done)
                else:
                    Agent.save_xp(prev_obs, obs, reward, prev_action, done)

            prev_obs = obs
            prev_action = action
            if agent_config['agent_name'] == 'HPPO':
                prev_log_policy = log_policy

            if episode_step >= env_config['max_step']:
                done = True
                continue
            
            rl_logger.step_logging(Agent)

            if rl_config['csv_logging']:
                state_logger.step_logger(episode_num=episode_num, episode_step=episode_step, origin_obs=None, obs=obs, action_values=None, action=action)

        env.close()

        rl_logger.episode_logging(Agent, episode_score, episode_step, episode_num, episode_rewards, obs[-1])

        if rl_confing['csv_logging']:
            state_logger.episode_logger(episode_score, episode_step, obs[-1])
            state_logger.save_data(episode_num)

        if episode_score > max_score:
            if os.name == 'nt':
                pass
                # Agent.save_models(path=result_path + "\\", score=round(episode_score, 3))
            elif os.name == 'posix':
                pass
                # Agent.save_models(path=result_path + "/", score=round(episode_score, 3))
            max_score = episode_score

        if episode_num % 50 == 0:
            print(f"epi_num : {episode_num}, epi_step : {episode_step}, score : {episode_score}, mean_reward : {episode_score/episode_step}")
        
    env.close()


if __name__ == '__main__':
    """
    Env
    1: Goal, 2: Platform 3: Move
    4: Hard Goal, 5: Hard move, 6: Domestic

    Agent
    1:  Q-PAMDP,  2: 
    5:  PA-DDPG,  6: PA-TDDPG,  7: PA-DDPG variation 1,  8: PA-DDPG variation 2
    9:  P-DQN,   10: P-TDDQN,  11: P-DQN variation 1,   12: P-DQN variation 2
    13: MP-DQN,  14: MP-TDDQN, 15: MP-DQN variation 1,  16: MP-DQN variation 2
    17: HPPO,    18: 
    21: HHQN,    22: 
    """
    
    env_switch = 2
    agent_switch = 9

    env_config, agent_config = env_agent_config(env_switch, agent_switch)
    
    rl_config = {'csv_logging': False, 'wandb': False, 'tensorboard': True}
    rl_custom_config = {'use_learned_model': False, 'learned_model_score': 59.009}

    parent_path = str(os.path.abspath(''))
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}/{agent_config['extension']['name']}_result/" + time_string

    if os.name == 'nt':
        data_save_path = parent_path + f"\\results\\{env_config['env_name']}\\{agent_config['agent_name']}_{agent_config['extension']['name']}_result\\" + time_string + '\\'
    elif os.name == 'posix':
        data_save_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}_{agent_config['extension']['name']}_result/" + time_string + '/'

    summary_writer = SummaryWriter(result_path+'/tensorboard/')
    if rl_config['wandb'] == True:
        import wandb
        wandb_session = wandb.init(project="RL-test-2", job_type="train", name=time_string)
    else:
        wandb_session = None

    rl_logger = RLLogger(env_config, agent_config, rl_config, summary_writer, wandb_session)
    rl_loader = RLLoader(env_config, agent_config)

    state_logger = StateLogger(env_config, agent_config, rl_config, data_save_path)

    main(env_config, agent_config, rl_config, rl_custom_config, result_path, rl_logger, rl_loader, state_logger)