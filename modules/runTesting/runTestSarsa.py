import os
import time
import sys
import pandas as pd
import numpy as np
import csv
import math as m
import pickle

sys.path.append('../modules')

import reinforcementLearningTuned as rl


def run_S(sarsa_agent, agent_label, total_mc):
    mc_count = []
    overall_reward = []
    num_timesteps=[]
    runtime_arr = []
    num_collisions_arr = []
    
    for mc_num in range(0,total_mc):
        tic = time.perf_counter()
        sarsa_agent.environment.chooseStart()
        sum_reward = 0
        num_iterations = 0
        num_collisions = 0
        while not sarsa_agent.environment.at_finish:
            # Record the previous step's position
            prev_x = sarsa_agent.environment.cur_x
            prev_y = sarsa_agent.environment.cur_y

            # Record previous step's acceleration
            prev_a_x, prev_a_y = sarsa_agent.getAcceleration()
            prev_state_action_vector = [prev_x, prev_y, prev_a_x, prev_a_y]

            # Determine a new acceleration
            sarsa_agent.setAcceleration(prev_x, prev_y)

            # Get updated accelration, check for collision
            a_x, a_y = sarsa_agent.getAcceleration()

            # Update the new position
            x, y = sarsa_agent.environment.updatePosition(a_x, a_y)
            v_x = sarsa_agent.environment.cur_vx
            v_y = sarsa_agent.environment.cur_vy


            last_valid_x, last_valid_y = sarsa_agent.environment.detectCollision(prev_x, prev_y, x, y)
            sarsa_agent.environment.cur_x = last_valid_x
            sarsa_agent.environment.cur_y = last_valid_y

            if sarsa_agent.environment.collision_occurred:
                sarsa_agent.environment.resetPosition(last_valid_x, last_valid_y)
                num_collisions+=1

            reward = sarsa_agent.environment.returnRewardValue()

            # Complete the SARSA update equation 
            x = sarsa_agent.environment.cur_x
            y = sarsa_agent.environment.cur_y
            cur_state_action_vector = [x, y, a_x, a_y]

            # Record the reward for the agent
            sarsa_agent.recordReward(reward)

            sum_reward += reward
            num_iterations+=1

            if num_iterations%20000==0:
                print(x,y,prev_x,prev_y)
                print(f'At Iteration {num_iterations}')

        mc_count.append(mc_num)
        overall_reward.append(sum_reward)
        num_timesteps.append(num_iterations)
        sarsa_agent.environment.resetGame()
        toc = time.perf_counter()
        runtime = toc-tic
        runtime_arr.append(runtime)
        num_collisions_arr.append(num_collisions)
        if mc_num % 50 == 0:
            print(f'Monte Carlo Number: {mc_num}')
            print(f'Number of collisions: {num_collisions}')
            print(f'Runtime: {runtime:.03f} s\n')
        
    results_df = pd.DataFrame({'mc_count': mc_count, 'overall_reward': overall_reward, 'num_timesteps': num_timesteps, 'runtime': runtime_arr, 'num_collisions': num_collisions_arr})
    results_df.to_csv(f'../results/test-results_{agent_label}.csv')

    return sarsa_agent

# Create the agents, tracks
mc_num = 100

## Load agents from trained instances
#agent_label = 'l_s'
#with open(f'../trained_agents/{agent_label}.instance', 'rb') as agent_instance:
#    l_q = pickle.load(agent_instance)
#run_S(l_q, 'l-s_agent', mc_num)
#
#agent_label = 'o_s'
#with open(f'../trained_agents/{agent_label}.instance', 'rb') as agent_instance:
#    l_q = pickle.load(agent_instance)
#run_S(l_q, 'o-s_agent', mc_num)

agent_label = 'r_s_a'
with open(f'../trained_agents/{agent_label}.instance', 'rb') as agent_instance:
    l_q = pickle.load(agent_instance)
run_S(l_q, 'r-s-a_agent', mc_num)

l_q.environment.collision_procedure='b'
run_S(l_q, 'r-s-b_agent', mc_num)

