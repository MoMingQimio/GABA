import os
import gym
from gym import spaces
#import pygame
import numpy as np 
import sys



from gym_sumo.envs import env_config as c
from gym_sumo.envs.RuleBasedDriverModel import DriverModel as RB
#要使用该库，<SUMO_HOME>/tools 目录必须位于python加载路径上。 通常如下：
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci
#import sumolib
## add each lane density and mean_speed ##
state_space_list = ['lane_index', 'ego_speed', 'ego_acc', 'ego_heading_angle',
                            'ego_dis_to_leader', 'leader_speed', 'leader_acc',
                            'ego_dis_to_follower', 'follower_speed', 'follower_acc',
                            'dis_to_left_leader', 'left_leader_speed', 'left_leader_acc',
                            'dis_to_right_leader', 'right_leader_speed', 'right_leader_acc',
                            'dis_to_left_follower', 'left_follower_speed', 'left_follower_acc',
                            'dis_to_right_follower', 'right_follower_speed', 'right_follower_acc'
                            ]
for i in range(c.NUM_OF_LANES):
    state_space_list.append("lane_" + str(i) + "_mean_speed")
    state_space_list.append("lane_" + str(i) + "_density")
def creat_observation():

    #print(state_space_list)
    state_space_low = np.array([c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE, -c.HEADING_ANGLE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY
                                ])

    state_space_high = np.array([c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE, c.HEADING_ANGLE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_MAX_SPEED_LIMIT, c.MAX_LANE_DENSITY,
                                 c.RL_MAX_SPEED_LIMIT, c.MAX_LANE_DENSITY,
                                 c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                 c.RL_MAX_SPEED_LIMIT, c.MAX_LANE_DENSITY
                                 ])

    #obs = spaces.Tuple([spaces.Discrete(c.NUM_OF_LANES),spaces.Box(low=state_space_low,high=state_space_high,dtype=np.float64)])不改了，不太对
    obs = spaces.Box(low=state_space_low,high=state_space_high,dtype=np.float64)
    return obs

class SumoEnv(gym.Env):
    """docstring for SumoEnv"""
    metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None):
        super(SumoEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = creat_observation()

        ## class variable
        self.ego = c.EGO_ID
        self.num_of_lanes = c.NUM_OF_LANES
        self.max_speed_limit = c.RL_MAX_SPEED_LIMIT
        self.is_collided = False
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode =render_mode
        self.rbm = RB()
        print(self.render_mode)


    def _getInfo(self):
        return {"current_episode":0}

    def _startSumo(self):
        sumoBinary = "sumo"
        if self.render_mode=="human":
            sumoBinary = "sumo-gui"
        sumoCmd = [sumoBinary,
                   "-c", "sumo_networks/test.sumocfg",
                   "--lateral-resolution","3.8",
                   "--start", "true",
                   "--quit-on-end", "true",
                   "--no-warnings","True",
                   "--no-step-log", "True",
                   "--collision.mingap-factor", "0.0",
                   "--collision.action", "warn",
                   ]

        traci.start(sumoCmd)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.is_collided = False
        self._startSumo()
        self._warmup()
        #
        index = np.array(np.random.randint(0, self.num_of_lanes))
        obs = np.array(self.observation_space.sample())

        #将index和obs连接在一起
        obs =np.insert(obs, 0, index)

        info = self._getInfo()
        return obs, info

    def _getCloseLeader(self, leaders):
        if len(leaders) <= 0:
            return "", -1
        min_dis = float('inf')
        current_leader = None,
        for leader in leaders:
            leader_id, dis = leader
            if dis < min_dis:
                current_leader = leader_id
                min_dis = dis
        return (current_leader, min_dis)

    def _getLaneDensity(self):
        road_id = traci.vehicle.getRoadID(self.ego)
        density = []
        mean_speed = []
        for i in range(self.num_of_lanes):
            density.append(len(traci.lane.getLastStepVehicleIDs(road_id+"_"+str(i))))
            mean_speed.append(traci.lane.getLastStepMeanSpeed(road_id+"_"+str(i)))
        return density, mean_speed

    def _get_rand_obs(self):
        index = np.array(np.random.randint(0, self.num_of_lanes))
        obs = np.array(self.observation_space.sample())
        obs = np.insert(obs, 0, index)
        #obs = np.array(self.observation_space.sample())
        return obs

    def get_observation(self,veh_id):
        if not self.isVehRunninng(veh_id):
            return self._get_rand_obs() ,["","","","","",""]
        #Basic Information
        lane_index = traci.vehicle.getLaneIndex(veh_id)
        speed = traci.vehicle.getSpeed(veh_id)
        acceleration = traci.vehicle.getAccel(veh_id)
        heading_angle = (traci.vehicle.getAngle(veh_id) -90.0)/c.HEADING_ANGLE


        leader = traci.vehicle.getLeader(veh_id)
        if leader is not None:
            leader_id, l_distance = leader
        else:
            leader_id, l_distance = "", -1
        leader_speed = traci.vehicle.getSpeed(leader_id) if leader_id != "" else 0.01
        leader_acc = traci.vehicle.getAccel(leader_id) if leader_id != "" else -2.6
        left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(veh_id, blockingOnly=True))
        left_l_speed = traci.vehicle.getSpeed(left_leader) if left_leader != "" else 0.01
        left_l_acc = traci.vehicle.getAccel(left_leader) if left_leader != "" else -2.6

        right_leader, right_l_dis = self._getCloseLeader(traci.vehicle.getRightLeaders(veh_id, blockingOnly=True))
        right_l_speed = traci.vehicle.getSpeed(right_leader) if right_leader != "" else 0.01
        right_l_acc = traci.vehicle.getAccel(right_leader) if right_leader != "" else -2.6

        follower = traci.vehicle.getFollower(veh_id)
        if follower is not None:
            follower_id, f_distance = follower
        else:
            follower_id, f_distance = "", -1
        follower_speed = traci.vehicle.getSpeed(follower_id) if follower_id != "" else 0.01
        follower_acc = traci.vehicle.getAccel(follower_id) if follower_id != "" else -2.6

        left_follower, left_f_dis = self._getCloseLeader(traci.vehicle.getLeftFollowers(veh_id, blockingOnly=True))
        left_follower_speed = traci.vehicle.getSpeed(left_follower) if left_follower != "" else 0.01
        left_follower_acc = traci.vehicle.getAccel(left_follower) if left_follower != "" else -2.6

        right_follower, right_f_dis = self._getCloseLeader(traci.vehicle.getRightFollowers(veh_id, blockingOnly=True))
        right_follower_speed = traci.vehicle.getSpeed(right_follower) if right_follower != "" else 0.01
        right_follower_acc = traci.vehicle.getAccel(right_follower) if right_follower != "" else -2.6



        states = [lane_index, speed, acceleration,heading_angle,
                  l_distance, leader_speed, leader_acc,
                  left_l_dis, left_l_speed, left_l_acc,
                  right_l_dis, right_l_speed, right_l_acc,
                  f_distance, follower_speed, follower_acc,
                  left_f_dis, left_follower_speed, left_follower_acc,
                  right_f_dis, right_follower_speed, right_follower_acc]

        density, mean_speed = self._getLaneDensity()
        for i in range(self.num_of_lanes):
            states.append(density[i])
            states.append(mean_speed[i])
        observations = np.array(states)
        # surrounding vehicle id list
        surrounding_vehicles =[leader_id, left_leader, right_leader, follower_id, left_follower, right_follower]

        return observations , surrounding_vehicles



    def _get_observation(self):
        if not self._isEgoRunning():
            return self._get_rand_obs()

        ego_speed = traci.vehicle.getSpeed(self.ego)
        ego_acceleration = traci.vehicle.getAccel(self.ego)
        ego_leader = traci.vehicle.getLeader(self.ego)
        if ego_leader is not None:
            leader_id, distance = ego_leader
        else:
            leader_id, distance = "", -1
        l_speed = traci.vehicle.getSpeed(leader_id) if leader_id != "" else 0.01
        l_acc = traci.vehicle.getAccel(leader_id) if leader_id != "" else -2.6
        left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(self.ego, blockingOnly=True))
        left_l_speed = traci.vehicle.getSpeed(left_leader) if left_leader != "" else 0.01
        left_l_acc = traci.vehicle.getAccel(left_leader) if left_leader != "" else -2.6

        right_leader, right_l_dis = self._getCloseLeader(traci.vehicle.getRightLeaders(self.ego, blockingOnly=True))
        right_l_speed = traci.vehicle.getSpeed(right_leader) if right_leader != "" else 0.01
        right_l_acc = traci.vehicle.getAccel(right_leader) if right_leader != "" else -2.6

        states = [ego_speed, ego_acceleration, distance, l_speed, l_acc, left_l_dis, left_l_speed, left_l_acc,
            right_l_dis, right_l_speed, right_l_acc]
        density, mean_speed = self._getLaneDensity()
        for i in range(self.num_of_lanes):
            states.append(density[i])
            states.append(mean_speed[i])

        observations = np.array(states)
        return observations

    def _applyAction(self, action):
        if self._isEgoRunning() == False:
            return
        current_lane_index = traci.vehicle.getLaneIndex(self.ego)

        if action == 0:
            # do nothing: stay in the current lane
            pass
        elif action == 1:

            target_lane_index = min(current_lane_index+1, self.num_of_lanes-1)
            traci.vehicle.changeLane(self.ego, target_lane_index, 0.1)

        elif action == 2:

            target_lane_index = max(current_lane_index-1, 0)
            traci.vehicle.changeLane(self.ego, target_lane_index, 0.1)

        elif action == 3:
            traci.vehicle.setAcceleration(self.ego,0.2, 0.1)
        elif action == 4:
            traci.vehicle.setAcceleration(self.ego, -4.5, 0.1)


    def _collision_reward(self):
        collide_vehicles = traci.simulation.getCollidingVehiclesIDList()
        if self.ego in collide_vehicles:
            self.is_collided = True
            return -10
        return 0.0
    def _efficiency(self):
        speed = traci.vehicle.getSpeed(self.ego)
        if speed < 25.0:
            return (speed-25.0)/(self.max_speed_limit-25.0)
        if speed > self.max_speed_limit:
            return (self.max_speed_limit-speed)/(self.max_speed_limit-25.0)
        return speed/self.max_speed_limit
    def _lane_change_reward(self,action):
        if action == 1 or action == 2:
            return -1.0
        return 0

    def _reward(self, action):
        c_reward = self._collision_reward()
        if self.is_collided or self._isEgoRunning()==False:
            return c_reward
        return c_reward + self._efficiency() + self._lane_change_reward(action)



    def step(self, action):
        #print(action)
        self._applyAction(action)
        traci.simulationStep()
        reward = self._reward(action)
        #observation = self._get_observation()
        observation, surrounding_vehicles = self.get_observation(self.ego)
        done = self.is_collided or (self._isEgoRunning()==False)

        if done == False and traci.simulation.getTime() > 360:
            done = True
        return (observation, reward, done, {})

    def isVehRunninng(self, veh_id):
        """
            Check if a vehicle is running on any of the specified edges.

            Parameters:
            self (object): The instance of the class.
            veh_id (str): The ID of the vehicle to check.

            Returns:
            bool: True if the vehicle is running on any of the edges E0, E1, or E2, False otherwise.
        """
        v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
        v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
        v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
        if veh_id in v_ids_e0 or veh_id in v_ids_e1 or veh_id in v_ids_e2:
            return True
        return False



    def _isEgoRunning(self):
        v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
        v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
        v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
        if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
            return True
        return False

    def _warmup(self):
        while True:
            v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
            v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
            v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
            if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
                traci.vehicle.setLaneChangeMode(self.ego,0)
                #traci.vehicle.setSpeedMode(self.ego,0)
                return True
            traci.simulationStep()


    def closeEnvConnection(self):
        traci.close()

    def move_gui(self):

        if self.render_mode == "human":
            x, y = traci.vehicle.getPosition('av_0')
            traci.gui.setOffset("View #0",x-23.0,108.49)




    def risk_assessment(self,observation,surrounding_vehicles):


        state = dict(zip(state_space_list, observation))
        state['lane_index'] = int(state['lane_index'])
        SSM = [self.calc_safety_surrogate_metric(state['ego_speed'], state['leader_speed'], state['ego_dis_to_leader']),
               self.calc_safety_surrogate_metric(state['follower_speed'], state['ego_speed'],state['ego_dis_to_follower']),
               self.calc_safety_surrogate_metric(state['ego_speed'], state['left_leader_speed'],state['dis_to_left_leader']),
               self.calc_safety_surrogate_metric(state['ego_speed'], state['right_leader_speed'],state['dis_to_right_leader']),
               self.calc_safety_surrogate_metric(state['left_follower_speed'], state['ego_speed'],state['ego_dis_to_left_follower']),
               self.calc_safety_surrogate_metric(state['right_follower_speed'], state['ego_speed'],state['ego_dis_to_right_follower'])]




	def

    def calc_safety_surrogate_metric(self, ego_speed, leader_speed,dis_to_leader):
            """
            计算前车和后车的TTC
            :param ego_speed: 车辆速度（m/s）
            :param leader_speed: 前车速度（m/s）
            :param dis_to_leader: 与前车的距离（m）
            :return: TTC（秒）
            """
            # 检查距离和速度
            if ego_speed <= leader_speed:
                return float('inf')  # 如果后车速度不大于前车，TTC为无穷大

            ttc = dis_to_leader / (ego_speed - leader_speed)
            return ttc


    def _applyBVaction(self):
        pass



