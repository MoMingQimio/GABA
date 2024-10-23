NUM_OF_LANES = 4
LENGTH_OF_EDGE = 1000
RL_SENSING_RADIUS = 300.0
RL_MAX_SPEED_LIMIT = 33.528
RL_MIN_SPEED_LIMIT = 24.5872
RL_ACC_RANGE = 2.0
RL_DCE_RANGE = 4.0
HEADING_ANGLE = 90.0
MIN_LANE_DENSITY = 0
MAX_LANE_DENSITY = 100 #this is an assumption for the traffic

EGO_ID = "av_0"

ACC_INTERVAL = 0.4

#IDM & MOBIL Parameters
IDM_v0 = 29.7  #speedFactor
IDM_s0 = 63.9    #minimumGap
IDM_a = 1.2        #accelerationFactor
IDM_b = 2.0        #comfortableDeceleration
IDM_T = 2.0      #desiredTimeHeadway
IDM_delta = 4.0  #exponentDM模型中的加速度指数，通常表示车辆加速的非线性程度。delta 决定了车辆在速度接近目标速度时的减速曲线。
# IDM_TAU = 1.87   #驾驶员的反应时间，表示驾驶员对前方车辆或道路情况作出反应所需要的时间。在IDM模型中，它通常影响车辆的安全距离和跟车策略。
# IDM_ACTIONSTEPLENGTH = 0.4 #
#IDM_STEPPING = 0.1  #the internal step length (in s) when computing follow speed

MOBIL_left_Bias = 0.3
MOBIL_politeness = 0.3
MOBIL_change_threshold = 0.1



#Risk assignment parameters
SSM_D_THRESHOLD = 0.5
SSM_A_THRESHOLD = 3


SSM_UNCERTAINTY = 0.5

