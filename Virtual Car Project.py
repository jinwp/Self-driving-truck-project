import numpy as np
import math

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()

env = UnityEnvironment(file_name = 'Road3/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 10)

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]
def distance(x1, z1, x2, z2):
    dis_sqr = (x1 - x2)**2 + (z1 - z2)**2
    dis = math.sqrt(dis_sqr)
    return dis

while distance(cur_obs[0], cur_obs[2], cur_obs[3], cur_obs[5])>=4:
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    print("cur observations : ", decision_steps.obs[0][0,:])
    s1,s2,s3,s4,s5 = cur_obs[6],cur_obs[7],cur_obs[8],cur_obs[9],cur_obs[10]
    if distance(cur_obs[0], cur_obs[2], cur_obs[3], cur_obs[5])<=10:
        env.set_actions(behavior_name, np.array([[0,150,150]]))
    elif 6 < s5 < 9 and s1 < s3:
        if s2 > 19:
            for i in range(10):
                env.set_actions(behavior_name, np.array([[-1,-100,100]]))
            for i in range(10):
                env.set_actions(behavior_name, np.array([[0.5,74,75]]))
        elif s3 > 19 and s4>19:
            for i in range(10):
                env.set_actions(behavior_name, np.array([[-1,-100,100]]))
                env.step()
            for i in range(10):
                env.set_actions(behavior_name, np.array([[-0.5,90,90]]))
                env.step()
        else:
            env.set_actions(behavior_name, np.array([[0,-100,100]]))
    elif 6 < s5 < 9 and s3 < s1:
        env.set_actions(behavior_name, np.array([[0,100,-100]]))
    elif s5 > 14.5 and s3 < s1 and s2 < 2:
        env.set_actions(behavior_name, np.array([[-1,100,100]]))
    elif s5 < 7:
        if s1 + s2 > s3 +s4:
            env.set_actions(behavior_name, np.array([[0,-50,-100]]))
        elif s1 + s2 < s3 + s4:
            env.set_actions(behavior_name, np.array([[0,-100,-50]]))
    elif 16.97<s1<17.09 and 11.62<s2<11.68 and 5.25<s3<5.39 and s4 > 19.99 and 5.7<s5<17.9:
        for i in range(30):
             env.set_actions(behavior_name, np.array([[1,20,20]]))
             env.step()
    elif s1>19.99 and s5>19.99 and s3 < 10 and s4 > 19.99:
        for i in range(10):
             env.set_actions(behavior_name, np.array([[0.86,-81,-81]]))
             env.step()
    elif s3 > 19 and s5 > 19 and s1 <6:
        for i in range(10):
            env.set_actions(behavior_name, np.array([[-0.8,50,50]]))
            env.step()
    elif s1 > 19 and s5 > 19 and s3 < 6 and s2 < 10 and s4 < 10:
        for i in range(10):
            env.set_actions(behavior_name, np.array([[0.5,40,40]]))
            env.step()
    elif s2 < 2 and s4 < 2:
        env.set_actions(behavior_name, np.array([[0,-100,-100]]))
    elif s4 < 3:
        env.set_actions(behavior_name, np.array([[0,0,-150]]))
    elif s2 < 3:
        env.set_actions(behavior_name, np.array([[0,-150,0]]))
    elif s3 < 4:
        env.set_actions(behavior_name, np.array([[0,150,-150]]))
    elif s1 < 4:
        env.set_actions(behavior_name, np.array([[0,-150,150]]))
    elif s3 > 18 and s5 > 18 and s1 > 18 and s2 > 18 and s4 > 18:
        env.set_actions(behavior_name, np.array([[0,150,150]]))
    elif s4 > s2:
        env.set_actions(behavior_name, np.array([[-0.3,130,150]]))
    elif s2 > s4:
        env.set_actions(behavior_name, np.array([[0.3,150,130]]))
    else: 
        env.set_actions(behavior_name, np.array([[0,50,50]]))
    env.step()

env.close()