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


class ContinuousActor(Model):
    def __init__(self,
                 continuous_act_space: int):
        super(ContinuousActor,self).__init__()
        self.continuous_act_space = continuous_act_space

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.0005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.mu = Dense(self.continuous_act_space, activation='tanh')

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        l1 = self.l1(state)
        l1_ln = self.l1_ln(l1)
        l2 = self.l2(l1_ln)
        l2_ln = self.l2_ln(l2)
        mu = self.mu(l2_ln)

        return mu


class DiscreteActor(Model):
    def __init__(self,
                 discrete_act_spac:int):
        super(DiscreteActor,self).__init__()
        self.discrete_act_spac = discrete_act_spac

        self.initializer = initializers.glorot_normal()
        self.regularizer = regularizers.l2(l=0.0005)

        self.l1 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.value = tf.keras.layers.Dense(self.discrete_act_spac, activation = None)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        l1 = self.l1(state)
        l1_ln = self.l1_ln(l1)
        l2 = self.l2(l1)
        l2_ln = self.l2_ln(l2)
        value = self.value(l2_ln)

        return value


class Agent: # Todo
    """
    input argument: obs_space, disc_act_space, cont_act_space, action_config, agent_config

    action_config: disc_act_spaces, cont_act_spaces, disc_act_max, disc_act_min, cont_act_max, cont_act_min

    agent_config: agent_name, gamma, epsilon_init, epsilon_final, epsilon_reduce_rate, tau, batch_size,\
                  warm_up, gaussian_std, noise_clip, noise_reduce_rate, lr_disc_actor, lr_cont_actor,\
                  use_PER, buffer_size, reward_normalize
    """
    def __init__(self, obs_space, action_config, agent_config):
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
        self.tau = self.agent_config['tau']

        if self.agent_config['use_PER']:
            self.replay_buffer = PrioritizedMemory(self.agent_config['buffer_size'])
        else:
            self.replay_buffer = ExperienceMemory(self.agent_config['buffer_size'])
        self.batch_size = self.agent_config['batch_size']

        self.epsilon_init = self.agent_config['epsilon_init']
        self.epsilon_final = self.agent_config['epsilon_final']
        self.epsilon_reduce_rate = self.agent_config['epsilon_reduce_rate']
        self.epsilon = copy.deepcopy(self.epsilon_init)

        self.std = self.agent_config['gaussian_std']
        self.noise_clip = self.agent_config['noise_clip']
        self.reduce_rate = self.agent_config['noise_reduce_rate']
        self.gradient_steps = 0
        self.critic_steps = 0
        self.warm_up = self.agent_config['warm_up']

        self.lr_disc_actor = self.agent_config['lr_disc_actor']
        self.lr_cont_actor = self.agent_config['lr_cont_actor']

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

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print('in action, obs: ', np.shape(np.array(obs)), obs)
        cont_actions = self.cont_actor_main(obs)
        # print('in action, mu: ', np.shape(np.array(mu)), mu)

        # epsilon-greedy
        rnd = np.random.rand()
        if rnd < self.epsilon:
            disc_action = np.random.choice(self.disc_act_space)

        else:
            q_value = self.disc_actor_main(tf.concat([obs,cont_actions], 1))
            disc_action = np.argmax(q_value.numpy())

        cont_actions = cont_actions.numpy()[0]
        offset_start = np.array([self.cont_act_spaces[i] for i in range(disc_action)], dtype=int).sum()
        offset_end   = offset_start + self.cont_act_spaces[disc_action]

        ## gaussian action noise
        if self.gradient_steps > self.warm_up:
            std = tf.convert_to_tensor([self.std]*self.cont_act_spaces[disc_action], dtype=tf.float32)
            dist = tfp.distributions.Normal(loc=tf.zeros(shape=(self.cont_act_spaces[disc_action]), dtype=tf.float32), scale=std)
            noises = tf.squeeze(dist.sample())
            noises = noises.numpy()
            self.std = self.std * self.reduce_rate
            self.epsilon = self.epsilon * self.epsilon_reduce_rate
            if self.epsilon < self.epsilon_final:
                self.epsilon = self.epsilon_final
            cont_actions[offset_start:offset_end] += np.clip(noises, - self.noise_clip, self.noise_clip)
        else:
            pass

        chosen_cont_action = np.clip(cont_actions[offset_start:offset_end], -1, 1)

        return disc_action, chosen_cont_action, cont_actions

    def update_target(self):
        disc_actor_weights = []
        disc_actor_targets = self.disc_actor_target.weights
        
        for idx, weight in enumerate(self.disc_actor_main.weights):
            disc_actor_weights.append(weight * self.tau + disc_actor_targets[idx] * (1 - self.tau))
        self.disc_actor_target.set_weights(disc_actor_weights)
        
        cont_actor_weithgs = []
        cont_actor_targets = self.cont_actor_target.weights
        
        for idx, weight in enumerate(self.cont_actor_main.weights):
            cont_actor_weithgs.append(weight * self.tau + cont_actor_targets[idx] * (1 - self.tau))
        self.cont_actor_target.set_weights(cont_actor_weithgs)

    def update(self):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0, 0.0

        self.gradient_steps += 1

        updated = True
        self.gradient_steps += 1
        self.critic_steps += 1

        actor_loss_val, criitic_loss1_val, ciritic_loss2_val = 0.0, 0.0, 0.0
        target_q_val, current_q_1_val, current_q_2_val = 0.0, 0.0, 0.0

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
            # print('states : {}'.format(states.numpy().shape), states)
            # print('next_states : {}'.format(next_states.numpy().shape), next_states)
            # print('rewards : {}'.format(rewards.numpy().shape), rewards)
            # print('actions : {}'.format(actions.numpy().shape), actions)
            # print('dones : {}'.format(dones.numpy().shape), dones)
            # print('is_weight : {}'.format(is_weight.numpy().shape), is_weight)

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

        disc_actor_variable = self.disc_actor_main.trainable_variables
        with tf.GradientTape() as tape_disc_actor:
            tape_disc_actor.watch(disc_actor_variable)
            target_cont_action = self.cont_actor_target(next_states)
            # print('next_states : {}'.format(next_states.numpy().shape))
            # print('target_action : {}'.format(target_cont_action.numpy().shape))
            # print('before squeeze q_next : {}'.format(self.disc_actor_target(tf.concat([next_states,target_cont_action], 1)).numpy().shape))

            target_q_next = self.disc_actor_target(tf.concat([next_states,target_cont_action], 1))
            # print('target_q_next : {}'.format(target_q_next.numpy().shape))

            target_q = tf.add(tf.expand_dims(rewards, axis=1), tf.multiply(self.gamma, tf.multiply(target_q_next, tf.expand_dims(tf.subtract(1.0, tf.cast(dones, dtype=tf.float32)),axis=1))))
            # print('target_q : {}'.format(target_q.numpy().shape))

            # print('before squeeze current_q : {}'.format(self.disc_actor_target(tf.concat([states,cont_actions], 1)).numpy().shape))
            current_q = self.disc_actor_main(tf.concat([states,cont_actions], 1))
            # print('current_q : {}'.format(current_q.numpy().shape))

            disc_actor_loss = tf.subtract(current_q, target_q)
            # print('disc_actor_loss : {}'.format(disc_actor_loss.numpy().shape))
            
            # (tf.abs(td_errors_1) + tf.abs(td_errors_2))/10 * is_weight
            td_errors = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool),\
                                lambda: tf.multiply(is_weight, tf.multiply(0.5, tf.math.square(disc_actor_loss))),\
                                lambda: tf.multiply(0.5, tf.math.square(disc_actor_loss)))
            # print('td_errors : {}'.format(td_errors.numpy().shape))

            td_error = tf.math.reduce_mean(td_errors)
            # print('td_error : {}'.format(td_error.numpy().shape))
            
        grads_disc_actor, _ = tf.clip_by_global_norm(tape_disc_actor.gradient(td_error, disc_actor_variable), 0.5)
        self.disc_actor_opt_main.apply_gradients(zip(grads_disc_actor, disc_actor_variable))

        disc_actor_loss_val = tf.math.reduce_mean(disc_actor_loss).numpy()
        target_q_val  = tf.math.reduce_mean(target_q).numpy()
        current_q_val = tf.math.reduce_mean(current_q).numpy()

        ## cont_actor update
        cont_actor_variable = self.cont_actor_main.trainable_variables
        with tf.GradientTape() as tape_cont_actor:
            tape_cont_actor.watch(cont_actor_variable)

            new_cont_actions = self.cont_actor_main(states)
            # print('new_policy_actions : {}'.format(new_policy_actions.numpy().shape))

            cont_actor_loss = -self.disc_actor_main(tf.concat([states, new_cont_actions],1))
            # print('actor_loss : {}'.format(actor_loss.numpy().shape))

            cont_actor_loss = tf.math.reduce_mean(cont_actor_loss)
            # print('actor_loss : {}'.format(actor_loss.numpy().shape))

        grads_actor, _ = tf.clip_by_global_norm(tape_cont_actor.gradient(cont_actor_loss, cont_actor_variable), 0.5)
        # grads_actor = tape_actor.gradient(cont_actor_loss, self.actor_main.trainable_variables)        
        self.cont_actor_opt_main.apply_gradients(zip(grads_actor, cont_actor_variable))

        cont_actor_loss_val = cont_actor_loss.numpy()

        self.update_target()

        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_errors[i].numpy())

        return updated, cont_actor_loss_val, disc_actor_loss_val, target_q_val, current_q_val

    def save_xp(self, state, next_state, reward, action, done):
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
            # print(f'concat_action: {np.concatenate(([disc_action],cont_action)).ravel()}')
            # print(f'done: {done}')
            
            self.replay_buffer.add((state, next_state, reward, np.concatenate(([disc_action],cont_action)).ravel(), done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.actor_main.load_weights(path, "_actor_main")
        self.actor_target.load_weights(path, "_actor_target")
        self.critic_main_1.load_weights(path, "_critic_main_1")
        self.critic_main_2.load_weights(path, "_critic_main_2")
        self.critic_target_1.load_weights(path, "_critic_target_1")
        self.critic_target_2.load_weights(path, "_critic_target_2")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor_main.save_weights(save_path, "_actor")
        self.actor_target.save_weights(save_path, "_actor_target")
        self.critic_main_1.save_weights(save_path, "_critic_main_1")
        self.critic_main_2.save_weights(save_path, "_critic_main_2")
        self.critic_target_1.save_weights(save_path, "_critic_target_1")
        self.critic_target_2.save_weights(save_path, "_critic_target_2")
