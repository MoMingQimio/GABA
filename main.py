
import os
from datetime import datetime
from itertools import count
import torch
import numpy as np
import math
import gym
from multiprocessing import Process

from PPO import PPO
from dqn_sumo_gym import Agent






def train(seed = 0):
    print("程序正在运行中，请勿关闭电脑")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    ####### initialize environment hyperparameters ######
    env_name = "sumo-v0"
    has_continuous_action_space = False  # continuous action space; else discrete
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)

    update_timestep = 1000 # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = int(seed)  # set random seed if required (0 = no random seed)

    env = gym.make(env_name, render_mode="")
    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        # 取出action_space中的两个维度
        action_dim = [env.action_space.nvec[0], env.action_space.nvec[1]]

        # action_dim = env.action_space.n

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
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num)+"_"+seed + ".csv"

    #### create new log file for each run
    av_dir = "runs"
    if not os.path.exists(av_dir):
        os.makedirs(av_dir)
    av_dir = av_dir + '/' + env_name + '/'
    if not os.path.exists(av_dir):
        os.makedirs(av_dir)

    av_run_num = 0
    current_num_files = next(os.walk(av_dir))[2]
    av_run_num = len(current_num_files)

    av_log_f_name = av_dir + "DQN_" + env_name + "_log_" + str(av_run_num) +"_"+seed+ ".csv"


    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    directory = "AV_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    av_checkpoint_path = directory + "AV_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    print("save checkpoint path : " + checkpoint_path)

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    ppo_agent = PPO(state_dim + 1, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std)
    AV_agent = Agent("Agent", env.action_space.nvec[1], state_dim + 1)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,collision_counts,adversarial_counts,reward\n')
    av_log_f = open(av_log_f_name, "w+")
    av_log_f.write(
        "episode,timestep,reward,if_collision,collision_rate,adversarial_counts,epsilon,av_speed_avg,av_total_risk_avg,av_distance,av_running_time,av_acc_avg,av_dece_avg,av_left_avg,av_right_avg\n")
    # av_log_f.write(
    #     '{},{},{},{},{},{},{},{},{},{},{},{},\n'.format(i_episode, time_step, r_r, if_collision, av_speed_avg,
    #                                                     av_total_risk_avg, av_distance, av_running_time, av_acc_avg,
    #                                                     av_dece_avg, av_left_avg, av_right_avg))
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 1

    av_log_reward = 0
    av_log_episodes = 0

    time_step = 0
    i_episode = 0

    collision_counts = 0
    adversarial_counts = 0
    collision_rate = 0
    # training loop
    epsilon = 0
    current_ep_reward = 0
    while time_step <= max_training_timesteps:

        state, surrounding_vehicles, excepted_risk = env.reset()

        r_r = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        av_speed = [0]
        av_acceleration = [0]
        av_deceleration = [0]
        av_left_change = [0]
        av_right_change = [0]
        av_total_risk = [0]
        av_distance = 0
        av_running_time = 0
        collision_flag = False

        for t in count():

            # select action with policy

            AV_action = AV_agent.select_action(state)

            BV_action, BV_action_logprob, BV_state_val = ppo_agent.select_action(state)
            # BV_action = BV_action.numpy().flatten()
            # print(f'AV_action:{AV_action},BV_action:{BV_action}')
            # BV_candidate_index = BV_action.item() % env.action_space.nvec[0]
            # BV_action_index = BV_action.item() // env.action_space.nvec[0]
            BV_candidate_index = BV_action[0]
            BV_action_index = BV_action[1]
            Veh_id = surrounding_vehicles[BV_candidate_index]

            # BV_candidate_index = np.where(surrounding_vehicles == Veh_id)[0][0]
            final_actions = (AV_action.item(), Veh_id, BV_action_index, epsilon)
            observation, reward, done, other_information = env.step(final_actions)
            BV_reward = other_information["BV_reward"]
            surrounding_vehicles = other_information["surrounding_vehicles"]
            excepted_risk = other_information["excepted_risk"]
            total_risk = other_information["total_risk"]
            collision_flag = other_information["collision_flag"]
            Adversarial_flag = other_information["Adversarial_flag"]


            av_speed.append(observation[1])
            if observation[2] > 0:
                av_acceleration.append(observation[2])
                # av_acc_counts += 1
            else:
                av_deceleration.append(observation[2])
                # av_dece_counts += 1
            if AV_action.item() == env.action_space.nvec[1] - 3:
                av_left_change.append(1)
                # av_left_counts += 1
            if AV_action.item() == env.action_space.nvec[1] - 2:
                av_right_change.append(1)
                # av_right_counts += 1
            av_total_risk.append(total_risk)

            time_step += 1

            if Adversarial_flag:
                adversarial_counts += 1
                BV_reward = BV_reward + excepted_risk[BV_candidate_index]

                ppo_agent.buffer.states.append(state)
                ppo_agent.buffer.actions.append(BV_action)
                ppo_agent.buffer.logprobs.append(BV_action_logprob)
                ppo_agent.buffer.state_values.append(BV_state_val)
                ppo_agent.buffer.rewards.append(BV_reward)
                ppo_agent.buffer.is_terminals.append(done)

                current_ep_reward += BV_reward
                if (adversarial_counts+1) % update_timestep == 0:
                    ppo_agent.update()

            if collision_flag:
                #print("Collision detected")
                collision_counts += 1
                current_ep_reward = round(current_ep_reward, 4)
                log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, collision_counts, adversarial_counts,
                                                      current_ep_reward))
                # print("BV--> Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                #                                                                               log_avg_reward))
                log_f.flush()
                if (collision_counts + 1) % save_model_freq == 0:
                    ppo_agent.save(checkpoint_path)
                current_ep_reward = 0

            r_r += reward
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

            if done:

                av_running_time = env.getRunningTime()
                av_distance = env.getDistance()


                env.closeEnvConnection()

                break

            env.move_gui()



        collision_rate = collision_counts / (i_episode + 1)
        # epsilon = 1 / (1+np.exp(-1*(2 - collision_rate - (total_risk + 1e-3)/2)))
        # epsilon = 1 / (1 + np.exp((np.array(collision_rate) * np.array(total_risk))))
        # other_information
        # saving reward and is_terminal
        # epsilon = 1 / (1 + math.exp(-1 * (1 / (collision_rate + 1e-3) / (total_risk + 1e-3))))
        # epsilon = 1 / (1 + math.exp(-1 * (1 / (collision_rate + 1e-3)))) sigmoid function, but not work
        epsilon = 1 - math.exp(0.05 * (1 - 1 / (collision_rate + 1e-3)))

        if (i_episode + 1) % 2 == 0:
            torch.save(AV_agent.policy_net.state_dict(), av_checkpoint_path)

        r_r = round(r_r, 4)
        if_collision = 0
        if collision_flag:
            if_collision = 1
        av_speed_avg = np.mean(av_speed)
        av_acc_avg = np.mean(av_acceleration)
        av_dece_avg = np.mean(av_deceleration)
        av_left_avg = np.sum(av_left_change) / (t + 1)
        av_right_avg = np.sum(av_right_change) / (t + 1)
        av_total_risk_avg = np.mean(av_total_risk)

        av_log_f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i_episode, time_step, r_r, if_collision,
                                                                                collision_rate, adversarial_counts,
                                                                                epsilon, av_speed_avg,
                                                                                av_total_risk_avg, av_distance,
                                                                                av_running_time, av_acc_avg,
                                                                                av_dece_avg, av_left_avg, av_right_avg))
        av_log_f.flush()
        # print(
        #     'Ep={}, Step={},R={},C_F={},C_R = {},Adv_c={},epsilon={},SPD={},Risk={},Dis={},T={},Acc={},Dec={},Left={},Right={}'.format(
        #         i_episode, time_step, r_r, if_collision, round(collision_rate, 4), adversarial_counts,
        #         round(epsilon, 4), round(av_speed_avg, 4),
        #         round(av_total_risk_avg, 4), round(av_distance, 4), round(av_running_time, 4), round(av_acc_avg, 4),
        #         round(av_dece_avg, 4), round(av_left_avg, 4), round(av_right_avg, 4)))

        i_episode += 1


    log_f.close()
    av_log_f.close()


    #print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")





def main():
	ra = [i for i in range(5)]
	proces = []
	for i in ra:
	    proc = Process(target=train, args=(str(i+1),))
	    proces.append(proc)
	    proc.start()

	# complete
	for proc in proces:
	    proc.join()

if __name__ == '__main__':
	main()