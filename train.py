import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
#from sympy.abc import epsilon

from PPO import PPO
from dqn_sumo_gym import Agent
#from gym_sumo.envs.RuleBasedDriverModel import Vehicle


################################### Training ###################################
def train():
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "sumo-v0"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = gym.make(env_name,render_mode="")

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        #取出action_space中的两个维度
        action_dim = env.action_space.nvec[0] *env.action_space.nvec[1]


        #action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim+1, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    AV_agent = Agent("Agent")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    collision_counts = 0
    # training loop
    epsilon = 0
    while time_step <= max_training_timesteps:

        state ,surrounding_vehicles, excepted_risk= env.reset()
        current_ep_reward = 0
        r_r = 0
        for t in range(1, max_ep_len+1):

            # select action with policy
            state  = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            AV_action = AV_agent.select_action(state)

            BV_action,BV_action_logprob,BV_state_val = ppo_agent.select_action(state)
            # BV_action = BV_action.numpy().flatten()

            BV_candidate_index = BV_action.item() % env.action_space.nvec[0]
            BV_action_index = BV_action.item() // env.action_space.nvec[0]
            Veh_id= surrounding_vehicles[BV_candidate_index]

            # BV_candidate_index = np.where(surrounding_vehicles == Veh_id)[0][0]
            final_actions = (AV_action.item(), Veh_id, BV_action_index, epsilon)
            observation, reward, done, other_information = env.step(final_actions)
            BV_reward = other_information["BV_reward"]
            surrounding_vehicles = other_information["surrounding_vehicles"]
            excepted_risk = other_information["excepted_risk"]
            total_risk = other_information["total_risk"]
            collision_flag = other_information["collision_flag"]
            Adversarial_flag = other_information["Adversarial_flag"]
            #other_information
            # saving reward and is_terminals




            time_step +=1
            if collision_flag:
                collision_counts += 1
            if Adversarial_flag:
                BV_reward = BV_reward + excepted_risk[BV_candidate_index]

                ppo_agent.buffer.states.append(state)
                ppo_agent.buffer.actions.append(BV_action)
                ppo_agent.buffer.logprobs.append(BV_action_logprob)
                ppo_agent.buffer.state_values.append(BV_state_val)
                ppo_agent.buffer.rewards.append(BV_reward)
                ppo_agent.buffer.is_terminals.append(done)

                # update PPO agent
                if collision_counts % update_timestep == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and collision_counts % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # log in logging file
                if collision_counts % log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if collision_counts % print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if collision_counts % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

            collision_rate = collision_counts/time_step
            epsilon = 1 / (1+np.exp(-1*(2 - collision_rate - (total_risk + 1e-3)/2)))



            current_ep_reward += reward
            #excepted_risk_last_time = excepted_risk

            r_r += reward
            if reward == -10:
                print(f'Collision: {reward}')
            reward = torch.tensor([reward], device=device)
            # done = terminated
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            AV_agent.memory.push(state, AV_action, next_state, reward)
            state = next_state

            AV_agent.learn_model()
            AV_agent.updateTargetNetwork()
            if (i_episode + 1) % 2 == 0:
                torch.save(AV_agent.policy_net.state_dict(), "models/model_test.pth")

            if done:
                AV_agent.episode_durations.append(r_r)
                AV_agent.plot_durations()
                env.closeEnvConnection()
                print(f'Episodes:{i_episode + 1}, Reward: {r_r}')
                break
            env.move_gui()


            # # update PPO agent
            # if time_step % update_timestep == 0:
            #     ppo_agent.update()
            #
            # # if continuous action space; then decay action std of ouput action distribution
            # if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            #
            # # log in logging file
            # if time_step % log_freq == 0:
            #
            #     # log average reward till last episode
            #     log_avg_reward = log_running_reward / log_running_episodes
            #     log_avg_reward = round(log_avg_reward, 4)
            #
            #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            #     log_f.flush()
            #
            #     log_running_reward = 0
            #     log_running_episodes = 0
            #
            # # printing average reward
            # if time_step % print_freq == 0:
            #
            #     # print average reward till last episode
            #     print_avg_reward = print_running_reward / print_running_episodes
            #     print_avg_reward = round(print_avg_reward, 2)
            #
            #     print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            #
            #     print_running_reward = 0
            #     print_running_episodes = 0
            #
            # # save model weights
            # if time_step % save_model_freq == 0:
            #     print("--------------------------------------------------------------------------------------------")
            #     print("saving model at : " + checkpoint_path)
            #     ppo_agent.save(checkpoint_path)
            #     print("model saved")
            #     print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            #     print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over


        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
