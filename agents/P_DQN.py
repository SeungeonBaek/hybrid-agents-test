from typing import Dict, Union, Any, Tuple
from numpy.typing import NDArray

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization

from utils.prioritized_memory_numpy import PrioritizedMemory
from utils.replay_buffer import ExperienceMemory

from agents.ICM_model import ICM_model
from agents.RND_model import RND_target, RND_predict


class ContinuousActor(Model):
    def __init__(self,
                 continuous_act_space: int)-> None:
        super(ContinuousActor,self).__init__()
        self.continuous_act_space = continuous_act_space

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.0005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)

        self.l2 = Dense(128, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)

        self.l3 = Dense(64, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_ln = LayerNormalization(axis=-1)

        self.mu = Dense(self.continuous_act_space, activation='tanh')

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        l1 = self.l1(state)
        l1_ln = self.l1_ln(l1)

        l2 = self.l2(l1_ln)
        l2_ln = self.l2_ln(l2)

        l3 = self.l3(l2_ln)
        l3_ln = self.l3_ln(l3)

        mu = self.mu(l3_ln)

        return mu


class DiscreteActor(Model):
    def __init__(self,
                 discrete_act_spac:int)-> None:
        super(DiscreteActor,self).__init__()
        self.discrete_act_spac = discrete_act_spac

        self.initializer = initializers.glorot_normal()
        self.regularizer = regularizers.l2(l=0.0005)

        self.l1 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)

        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)

        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_ln = LayerNormalization(axis=-1)

        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4_ln = LayerNormalization(axis=-1)

        self.value = Dense(self.discrete_act_spac, activation = None)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        l1 = self.l1(state)
        l1_ln = self.l1_ln(l1)

        l2 = self.l2(l1_ln)
        l2_ln = self.l2_ln(l2)

        l3 = self.l3(l2_ln)
        l3_ln = self.l3_ln(l3)

        l4 = self.l4(l3_ln)
        l4_ln = self.l4_ln(l4)

        value = self.value(l4_ln)

        return value


class Agent:
    """
    input argument: obs_space, disc_act_space, cont_act_space, action_config, agent_config

    action_config: disc_act_spaces, cont_act_spaces, disc_act_max, disc_act_min, cont_act_max, cont_act_min

    agent_config: agent_name, gamma, epsilon_init, epsilon_final, epsilon_decaying_rate, tau, batch_size,\
                  warm_up, gaussian_std, noise_clip, noise_reduce_rate, lr_disc_actor, lr_cont_actor,\
                  use_PER, buffer_size, reward_normalize
    """
    def __init__(self,
                 agent_config: Dict,
                 obs_space: int,
                 action_config: Dict):
        self.agent_config = agent_config
        self.action_config = action_config
        self.name = self.agent_config['agent_name']

        self.obs_space = obs_space
        self.disc_act_space = self.action_config['disc_act_spaces'] 
        self.cont_act_space = self.action_config['cont_act_spaces'].sum()
        print(f'obs_space: {self.obs_space}, act_space: {self.disc_act_space}, cont_act_space: {self.cont_act_space}')

        self.disc_act_spaces = self.action_config['disc_act_spaces']
        self.cont_act_spaces = self.action_config['cont_act_spaces']
        self.disc_act_max = self.action_config['disc_act_max']
        self.disc_act_min = self.action_config['disc_act_min']
        self.cont_act_max = self.action_config['cont_act_max']
        self.cont_act_min = self.action_config['cont_act_min']

        print(f'act_spaces: {self.disc_act_spaces}, cont_act_spaces: {self.cont_act_spaces}')

        self.gamma = self.agent_config['gamma']
        self.discrete_tau = self.agent_config['discrete_tau']
        self.continuous_tau = self.agent_config['continuous_tau']

        if self.agent_config['use_PER']:
            self.replay_buffer = PrioritizedMemory(self.agent_config['buffer_size'])
        else:
            self.replay_buffer = ExperienceMemory(self.agent_config['buffer_size'])
        self.batch_size = self.agent_config['batch_size']

        self.update_call_step = 0
        self.update_step = 0
        self.update_freq = self.agent_config['update_freq']
        self.target_update_freq = self.agent_config['target_update_freq']

        self.warm_up = self.agent_config['warm_up']

        self.lr_disc_actor = self.agent_config['lr_disc_actor']
        self.lr_cont_actor = self.agent_config['lr_cont_actor']

        # network config
        self.disc_actor_main   = DiscreteActor(self.disc_act_space)
        self.disc_actor_target = DiscreteActor(self.disc_act_space)
        self.disc_actor_target.set_weights(self.disc_actor_main.get_weights())
        self.disc_actor_opt_main = Adam(self.lr_disc_actor)
        self.disc_actor_main.compile(optimizer=self.disc_actor_opt_main)
        
        self.cont_actor_main   = ContinuousActor(self.cont_act_space)
        self.cont_actor_target = ContinuousActor(self.cont_act_space)
        self.cont_actor_target.set_weights(self.cont_actor_main.get_weights())
        self.cont_actor_opt_main = Adam(self.lr_cont_actor)
        self.cont_actor_main.compile(optimizer=self.cont_actor_opt_main)

        # extension config
        self.extension_config = self.agent_config['extension']
        self.extension_name = self.extension_config['name']

        self.std = self.extension_config['gaussian_std']
        self.min_std = self.extension_config['min_std']
        self.noise_clip = self.extension_config['noise_clip']
        self.noise_reduce_rate = self.extension_config['noise_reduction_rate']

        self.epsilon = self.extension_config['epsilon']
        self.epsilon_decaying_rate = self.extension_config['epsilon_decaying_rate']
        self.min_epsilon = self.extension_config['min_epsilon']

        if self.extension_config['use_Twin_Delay']:
            pass

        if self.extension_name == 'ICM':
            self.icm_update_freq = self.extension_config['icm_update_freq']

            self.icm_lr = self.extension_config['icm_lr']
            self.icm_feature_dim = self.extension_config['icm_feature_dim']
            # self.icm = ICM_model(self.obs_space, self.act_space, self.icm_feature_dim) # Todo
            self.icm_opt = Adam(self.icm_lr)

        elif self.extension_name == 'RND':
            self.rnd_update_freq = self.extension_config['rnd_update_freq']

            self.rnd_lr = self.extension_config['rnd_lr']
            # self.rnd_target = RND_target(self.obs_space, self.act_space) # Todo
            # self.rnd_predict = RND_predict(self.obs_space, self.act_space) # Todo
            self.rnd_opt = Adam(self.rnd_lr)

        elif self.extension_name == 'NGU':
            self.icm_lr = self.extension_config['ngu_lr']

    def action(self, obs)-> Tuple[NDArray, NDArray, NDArray]:
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print('in action, obs: ', np.shape(np.array(obs)), obs)
        cont_actions = self.cont_actor_main(obs)
        # print('in action, mu: ', np.shape(np.array(mu)), mu)

        # epsilon-greedy
        random_val = np.random.rand()
        if self.update_step > self.warm_up:
            if random_val < self.epsilon:
                disc_action = np.random.choice(self.disc_act_space)

            else:
                q_value = self.disc_actor_main(tf.concat([obs, cont_actions], axis=1))
                disc_action = np.argmax(q_value.numpy())
            
            self.epsilon = self.epsilon * self.epsilon_decaying_rate
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
        else:
            disc_action = np.random.choice(self.disc_act_space)

        cont_actions = cont_actions.numpy()[0]
        offset_start = np.array([self.cont_act_spaces[i] for i in range(disc_action)], dtype=int).sum()
        offset_end   = offset_start + self.cont_act_spaces[disc_action]

        ## gaussian action noise
        if self.update_step > self.warm_up:
            if self.cont_act_spaces[disc_action] == 0:
                pass
            else:
                std = tf.convert_to_tensor([self.std]*self.cont_act_spaces[disc_action], dtype=tf.float32)
                dist = tfp.distributions.Normal(loc=tf.zeros(shape=(self.cont_act_spaces[disc_action]), dtype=tf.float32), scale=std)
                noises = tf.squeeze(dist.sample())
                noises = noises.numpy()
                self.std = self.std * self.noise_reduce_rate
                if self.std < self.min_std:
                    self.std = self.min_std

                cont_actions[offset_start:offset_end] += np.clip(noises, - self.noise_clip, self.noise_clip)
        else:
            pass
        
        if self.cont_act_spaces[disc_action] == 0:
            chosen_cont_action = []
        else:
            chosen_cont_action = np.clip(cont_actions[offset_start:offset_end], -1, 1)
            chosen_cont_action = (chosen_cont_action + 1)*0.5
            chosen_cont_action = (self.cont_act_max[disc_action] - self.cont_act_min[disc_action]) * chosen_cont_action + self.cont_act_min[disc_action]

        return disc_action, chosen_cont_action, cont_actions

    def update_target(self)-> None:
        if self.discrete_tau == None:
            disc_actor_weights = self.disc_actor_main.get_weights()
            self.disc_actor_target.set_weights(disc_actor_weights)
        else:
            disc_actor_weights = []
            disc_actor_targets = self.disc_actor_target.weights
            
            for idx, weight in enumerate(self.disc_actor_main.get_weights()):
                disc_actor_weights.append(weight * self.discrete_tau + disc_actor_targets[idx] * (1 - self.discrete_tau))
            self.disc_actor_target.set_weights(disc_actor_weights)

        if self.continuous_tau == None:
            cont_actor_weithgs = self.cont_actor_main.get_weights()
            self.cont_actor_target.set_weights(cont_actor_weithgs)
        else:
            cont_actor_weithgs = []
            cont_actor_targets = self.cont_actor_target.weights
            
            for idx, weight in enumerate(self.cont_actor_main.get_weights()):
                cont_actor_weithgs.append(weight * self.continuous_tau + cont_actor_targets[idx] * (1 - self.continuous_tau))
            self.cont_actor_target.set_weights(cont_actor_weithgs)


    def update(self)-> None:
        self.update_call_step += 1

        if (self.replay_buffer._len() < self.batch_size) or (self.update_call_step % self.update_freq != 0):
            return False, 0.0, 0.0, 0.0, 0.0

        updated = True
        self.update_step += 1

        disc_actor_loss_val, target_q_val, current_q_val = 0.0, 0.0, 0.0

        if self.agent_config['use_PER']:
            states, next_states, rewards, actions, dones, idxs, is_weight = self.replay_buffer.sample(self.batch_size)

            if self.agent_config['reward_normalize']:
                rewards = np.asarray(rewards)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.convert_to_tensor(actions, dtype = tf.float32)
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)
            is_weight = tf.convert_to_tensor(is_weight, dtype=tf.float32)

        else:
            states, next_states, rewards, actions, dones = self.replay_buffer.sample(self.batch_size)
            disc_actions, cont_actions = actions[:,0], actions[:, 1:]

            if self.agent_config['reward_normalize']:
                rewards = np.asarray(rewards)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.convert_to_tensor(actions, dtype = tf.float32)
            disc_actions = tf.convert_to_tensor(disc_actions, dtype = tf.float32)
            cont_actions = tf.convert_to_tensor(cont_actions, dtype = tf.float32)
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)

        # print(f"states : {states.shape}")
        # print(f"next_states : {next_states.shape}")
        # print(f"rewards : {rewards.shape}")
        # print(f"actions : {actions.shape}")
        # print(f"disc_actions : {disc_actions.shape}")
        # print(f"cont_actions : {cont_actions.shape}")
        # print(f"dones : {dones.shape}")

        ## cont_actor update
        cont_actor_variable = self.cont_actor_main.trainable_variables
        with tf.GradientTape() as tape_cont_actor:
            tape_cont_actor.watch(cont_actor_variable)

            new_cont_actions = self.cont_actor_main(states)
            # print(f"new_cont_actions : {new_cont_actions.shape}")

            cont_actor_loss = -self.disc_actor_main(tf.concat([states, new_cont_actions],1))
            # print('cont_actor_loss : {}'.format(cont_actor_loss.numpy().shape))

            cont_actor_loss = tf.math.reduce_sum(cont_actor_loss, axis=1)
            # print('cont_actor_loss : {}'.format(cont_actor_loss.numpy().shape))

            cont_actor_loss = tf.math.reduce_mean(cont_actor_loss)
            # print('cont_actor_loss : {}'.format(cont_actor_loss.numpy().shape))

        grads_actor, _ = tf.clip_by_global_norm(tape_cont_actor.gradient(cont_actor_loss, cont_actor_variable), 0.5)
        # grads_actor = tape_actor.gradient(cont_actor_loss, self.actor_main.trainable_variables)        
        self.cont_actor_opt_main.apply_gradients(zip(grads_actor, cont_actor_variable))

        cont_actor_loss_val = cont_actor_loss.numpy()

        ## dist_actor update
        disc_actor_variable = self.disc_actor_main.trainable_variables
        with tf.GradientTape() as tape_disc_actor:
            tape_disc_actor.watch(disc_actor_variable)

            # target
            target_cont_action = self.cont_actor_target(next_states)
            # print(f"target_cont_action: {target_cont_action.shape}")
            current_q_next = self.disc_actor_main(tf.concat([next_states, target_cont_action], axis=1))
            # print(f"current_q_next: {current_q_next.shape}")
            next_action = tf.argmax(current_q_next, axis=1)
            # print(f"next_action: {next_action.shape}")
            indices = tf.stack([range(self.batch_size), next_action], axis=1)
            # print(f"indices: {indices.shape}")

            target_q_next = tf.cond(tf.convert_to_tensor(self.extension_config['use_DDQN'], dtype=tf.bool),\
                    lambda: tf.gather_nd(params=self.disc_actor_target(tf.concat([next_states, target_cont_action], axis=1)), indices=indices), \
                    lambda: tf.reduce_max(self.disc_actor_target(tf.concat([next_states, target_cont_action], axis=1)), axis=1))
            # print(f"[states, cont_action]: {tf.concat([next_states,target_cont_action], 1).shape}")
            # print(f"target_q_next: {target_q_next.shape}")

            target_q = rewards + self.gamma * target_q_next * (1.0 - tf.cast(dones, dtype=tf.float32))
            target_q = tf.stop_gradient(target_q)
            # print(f"target_q : {target_q.shape}")

            current_q = self.disc_actor_main(tf.concat([states, cont_actions], axis=1))
            # print(f"current_q : {current_q.shape}")
            action_one_hot = tf.one_hot(tf.cast(disc_actions, tf.int32), self.disc_act_space)
            # print(f"action_one_hot: {action_one_hot.shape}")
            current_q = tf.reduce_sum(tf.multiply(current_q, action_one_hot), axis=1)
            # print(f"current_q: {current_q.shape}")

            disc_actor_loss = tf.subtract(current_q, target_q)
            disc_actor_huber_loss = tf.where(tf.less(tf.math.abs(disc_actor_loss), 1.0), 1/2 * tf.math.square(disc_actor_loss), 1.0 * tf.abs(disc_actor_loss) - 1.0 * 1/2)
            # print(f"disc_actor_loss : {disc_actor_loss.shape}")
            # print(f"disc_actor_huber_loss : {disc_actor_huber_loss.shape}")
            
            critic_losses = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool),\
                                lambda: tf.multiply(is_weight, disc_actor_huber_loss),\
                                lambda: disc_actor_huber_loss)
            # print(f"critic_losses : {critic_losses.shape}")

            critic_loss = tf.math.reduce_mean(critic_losses)
            # print(f"critic_loss : {critic_loss.shape}")
            
        grads_disc_actor, _ = tf.clip_by_global_norm(tape_disc_actor.gradient(critic_loss, disc_actor_variable), 0.5)
        self.disc_actor_opt_main.apply_gradients(zip(grads_disc_actor, disc_actor_variable))

        disc_actor_loss_val = tf.math.reduce_mean(disc_actor_loss).numpy()
        target_q_val  = tf.math.reduce_mean(target_q).numpy()
        current_q_val = tf.math.reduce_mean(current_q).numpy()


        if self.update_step % self.target_update_freq == 0:
            self.update_target()

        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_errors[i].numpy())

        return updated, cont_actor_loss_val, disc_actor_loss_val, target_q_val, current_q_val

    def save_xp(self, state: NDArray, next_state: NDArray, reward: float, action: NDArray, done: bool)-> None:
        # Store transition in the replay buffer.
        if self.agent_config['use_PER']:
            state_tf = tf.convert_to_tensor([state], dtype = tf.float32)
            action_tf = tf.convert_to_tensor([action], dtype = tf.float32)
            next_state_tf = tf.convert_to_tensor([next_state], dtype = tf.float32)
            target_action_tf = self.actor_target(next_state_tf)
            # print('state_tf: {}'.format(state_tf))
            # print('action_tf: {}'.format(action_tf))
            # print('next_state_tf: {}'.format(next_state_tf))
            # print('target_action_tf: {}'.format(target_action_tf))

            target_q_next_1 = tf.squeeze(self.critic_target_1(tf.concat([next_state_tf,target_action_tf], 1)), 1)
            target_q_next_2 = tf.squeeze(self.critic_target_2(tf.concat([next_state_tf,target_action_tf], 1)), 1)
            target_q_next = tf.math.minimum(target_q_next_1, target_q_next_2)
            # print('target_q_next_1: {}'.format(target_q_next_1))
            # print('target_q_next_2: {}'.format(target_q_next_2))
            # print('target_q_next: {}'.format(target_q_next))
            
            current_q_1 = tf.squeeze(self.critic_main_1(tf.concat([state_tf,action_tf], 1)), 1)
            current_q_2 = tf.squeeze(self.critic_main_2(tf.concat([state_tf,action_tf], 1)), 1)
            # print('current_q_1: {}'.format(current_q_1))
            # print('current_q_2: {}'.format(current_q_2))
            
            target_q = reward + self.gamma * target_q_next * (1.0 - tf.cast(done, dtype=tf.float32))
            # print('target_q: {}'.format(target_q))
            
            td_errors_1 = target_q - current_q_1
            td_errors_2 = target_q - current_q_2
            # print('td_errors_1: {}'.format(td_errors_1))
            # print('td_errors_2: {}'.format(td_errors_2))

            td_error = (0.5 * tf.math.square(td_errors_1) + 0.5 * tf.math.square(td_errors_2))
            # print('td_error: {}'.format(td_error))

            td_error = td_error.numpy()
            # print('td_error: {}'.format(td_error))

            self.replay_buffer.add(td_error[0], (state, next_state, reward, action, done))
        else:
            disc_action, cont_action = action
            
            # print(f'state: {state}')
            # print(f'next_state: {next_state}')
            # print(f'reward: {reward}')
            # print(f'action: {action}')
            # print(f'disc_action: {disc_action}')
            # print(f'cont_action: {cont_action}')
            # print(f'concat_action: {np.concatenate(([disc_action],np.squeeze(cont_action))).ravel()}')
            # print(f'done: {done}')

            self.replay_buffer.add((state, next_state, reward, np.concatenate(([disc_action], np.squeeze(cont_action))).ravel(), done))

    def load_models(self, path: str):
        print('Load Model Path : ', path)
        self.actor_main.load_weights(path, "_actor_main")
        self.actor_target.load_weights(path, "_actor_target")
        self.critic_main_1.load_weights(path, "_critic_main_1")
        self.critic_main_2.load_weights(path, "_critic_main_2")
        self.critic_target_1.load_weights(path, "_critic_target_1")
        self.critic_target_2.load_weights(path, "_critic_target_2")

    def save_models(self, path: str, score: float):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor_main.save_weights(save_path, "_actor")
        self.actor_target.save_weights(save_path, "_actor_target")
        self.critic_main_1.save_weights(save_path, "_critic_main_1")
        self.critic_main_2.save_weights(save_path, "_critic_main_2")
        self.critic_target_1.save_weights(save_path, "_critic_target_1")
        self.critic_target_2.save_weights(save_path, "_critic_target_2")
