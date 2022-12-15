

class RLLogger():
    def __init__(self, env_config, agent_config, rl_config, summary_writer = None, wandb_session = None):
        self.env_config = env_config
        self.agent_config = agent_config
        self.rl_config = rl_config
        self.summary_writer = summary_writer
        self.wandb_session = wandb_session

    def step_logging(self, Agent):
        if self.rl_config['tensorboard'] == True:
            self.step_logging_tensorboard(Agent)
        if self.rl_config['wandb'] == True:
            self.step_logging_wandb(Agent)

    def episode_logging(self, Agent, episode_score, episode_step, episode_num, episode_rewards, env_inner_step):
        if self.rl_config['tensorboard'] == True:
            self.episode_logging_tensorboard(Agent, episode_score, episode_step, episode_num, episode_rewards, env_inner_step)
        if self.rl_config['wandb'] == True:
            self.episode_logging_wandb(Agent, episode_score, episode_step, episode_num, episode_rewards, env_inner_step)

    def step_logging_tensorboard(self, Agent):
        # update
        if self.agent_config['agent_name'] == 'Q-PAMDP':
            if self.agent_config['extension']['name'] == 'Double': # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation1':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation2':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            else: # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()

        elif self.agent_config['agent_name'] == 'PA-DDPG':
            if self.agent_config['extension']['name'] == 'Double': # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation1':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation2':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            else: # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()

        elif self.agent_config['agent_name'] == 'P-DQN':
            if self.agent_config['extension']['name'] == 'Double': # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation1':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation2':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            else: # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()

        elif self.agent_config['agent_name'] == 'MP-DQN':
            if self.agent_config['extension']['name'] == 'Double': # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation1':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            elif self.agent_config['extension']['name'] == 'Variation2':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
            else: # Todo
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()

        elif self.agent_config['agent_name'] == 'HPPO':
            updated, std, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update()

        elif self.agent_config['agent_name'] == 'HHQN':
            updated, std, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update()

        elif self.agent_config['agent_name'] == 'Target25':
            updated, std, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update()

        elif self.agent_config['agent_name'] == 'Target29':
            updated, std, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update()

        # logging
        if self.agent_config['agent_name'] == 'Q-PAMDP':
            if updated:
                self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)

                if self.agent_config['extension']['name'] == 'Double':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation1':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation2':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                else:
                    pass

        elif self.agent_config['agent_name'] == 'PA-DDPG':
            if updated: # Todo
                self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)

                if self.agent_config['extension']['name'] == 'Double':  # Todo
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation1':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation2':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                else:
                    pass

        elif self.agent_config['agent_name'] == 'P-DQN':
            if updated:  # Todo
                self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)

                if self.agent_config['extension']['name'] == 'Double':  # Todo
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation1':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation2':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                else:
                    pass

        elif self.agent_config['agent_name'] == 'MP-DQN':
            if updated:  # Todo
                self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)

                if self.agent_config['extension']['name'] == 'Double':  # Todo
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation1':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                elif self.agent_config['extension']['name'] == 'Variation2':
                    self.summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
                else:
                    pass

        elif self.agent_config['agent_name'] == 'HPPO':
            if updated:
                self.summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_1_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Critic_2_loss', critic_2_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Alpha_loss', alpha_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_2_value', critic_2_value, Agent.update_step)

        elif self.agent_config['agent_name'] == 'HHQN':
            if updated:
                self.summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_1_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Critic_2_loss', critic_2_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Alpha_loss', alpha_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_2_value', critic_2_value, Agent.update_step)

        elif self.agent_config['agent_name'] == 'Target25':
            if updated:
                self.summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_1_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Critic_2_loss', critic_2_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Alpha_loss', alpha_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_2_value', critic_2_value, Agent.update_step)

        elif self.agent_config['agent_name'] == 'Target29':
            if updated:
                self.summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_1_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Critic_2_loss', critic_2_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('01_Loss/Alpha_loss', alpha_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
                self.summary_writer.add_scalar('02_Critic/Critic_2_value', critic_2_value, Agent.update_step)

    def step_logging_wandb(self, Agent):
        if self.agent_config['agent_name'] == 'Q-PAMDP':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update()

        elif self.agent_config['agent_name'] == 'PA-DDPG':
            updated, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

        elif self.agent_config['agent_name'] == 'P-DQN':
            updated, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

        elif self.agent_config['agent_name'] == 'MP-DQN':
            updated, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

        elif self.agent_config['agent_name'] == 'HPPO':
            updated, alpha_loss, actor_loss, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

        if self.agent_config['agent_name'] == 'Q-PAMDP':
            if updated:
                self.wandb_session.log({
                    "01_Loss/Critic_loss": critic_loss,
                    "01_Loss/Actor_loss": actor_loss,
                    '02_Critic/Target_Q_mean': trgt_q_mean, 
                    '02_Critic/Critic_value': critic_value
                }, step=self.Agent.update_step)
        elif self.agent_config['agent_name'] == 'PA-DDPG':
            if updated:
                self.wandb_session.log({
                    "01_Loss/Critic_loss": critic_loss,
                    "01_Loss/Actor_loss": actor_loss,
                    '02_Critic/Target_Q_mean': trgt_q_mean,
                    '02_Critic/Critic_1_value': critic_1_value,
                    '02_Critic/Critic_2_value': critic_2_value,
                }, step=self.Agent.update_step)
        elif self.agent_config['agent_name'] == 'P-DQN':
            if updated:
                self.wandb_session.log({
                    "01_Loss/Critic_loss": critic_loss,
                    '02_Critic/Target_Q_mean': trgt_q_mean, 
                    '02_Critic/Critic_value': critic_value
                }, step=self.Agent.update_step)
        elif self.agent_config['agent_name'] == 'MP-DQN':
            if updated:
                self.wandb_session.log({
                    "01_Loss/Critic_loss": critic_loss,
                    "01_Loss/Actor_loss": actor_loss,
                    '02_Critic/Target_Q_mean': trgt_q_mean, 
                    '02_Critic/Critic_value': critic_value
                }, step=self.Agent.update_step)
        elif self.agent_config['agent_name'] == 'HPPO':
            if updated:
                self.wandb_session.log({
                    "01_Loss/Critic_1_loss": critic_1_loss,
                    "01_Loss/Critic_1_loss": critic_2_loss,
                    "01_Loss/Actor_loss": actor_loss,
                    "01_Loss/Alpha_loss": alpha_loss,
                    '02_Critic/Target_Q_mean': trgt_q_mean, 
                    '02_Critic/Critic_1_value': critic_1_value,
                    '02_Critic/Critic_2_value': critic_2_value
                }, step=self.Agent.update_step)

    def episode_logging_tensorboard(self, Agent, episode_score, episode_step, episode_num, episode_rewards, env_inner_step):
        if self.agent_config['agent_name'] == 'P-DQN': # Todo
            pass

        self.summary_writer.add_scalar('00_Episode/Score', episode_score, episode_num)
        self.summary_writer.add_scalar('00_Episode/Average_reward', episode_score/episode_step, episode_num)
        self.summary_writer.add_scalar('00_Episode/Steps', episode_step, episode_num)

        if self.env_config['env_name'] == 'Platform' or self.env_config['env_name'] == 'Goal':
            self.summary_writer.add_scalar('00_Episode/Env_step', env_inner_step * 200, episode_num)

        elif self.env_config['env_name'] == 'Move':
            pass

        else:
            pass

        self.summary_writer.add_histogram('Reward_histogram', episode_rewards, episode_num)

    def episode_logging_wandb(self, Agent, episode_score, episode_step, episode_num, episode_rewards):
        if self.agent_config['agent_name'] == 'P-DQN':
            pass

        self.wandb_session.log({
            '00_Episode/Average_reward': episode_score/episode_step,
            "00_Episode/Score": episode_score,
            '00_Episode/Steps': episode_step,
            "episode_num": episode_num
        })

        histogram = self.wandb.Histogram(episode_rewards)
        self.wandb_session.log({"reward_hist": histogram})