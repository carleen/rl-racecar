import modules
import pickle
import numpy as np
import time
from modules.Environment import *
from modules.QAgent import *
from modules.SAgent import *

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
            sarsa_agent.sarsaUpdateQTable(prev_state_action_vector, cur_state_action_vector, reward)

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
    results_df.to_csv(f'../results/{agent_label}_epsilon-5.csv')
    sarsa_agent.q_table.to_csv(f'../results/{agent_label}_qtable.csv')

    with open(f'../trained_agents/{agent_label}.instance', 'wb') as agent_instance_file:
        pickle.dump(sarsa_agent, agent_instance_file)


    return sarsa_agent

def run_Q(a, agent_label, num_mcs):
    mc_count = []
    overall_reward = []
    num_timesteps=[]
    timestep_arr=[]
    num_collisions_arr = []
    
    for mc_num in range(0,num_mcs):
        tic = time.perf_counter()
        a.environment.chooseStart()
        sum_reward = 0
        num_iterations = 0
        num_collisions = 0
        while not a.environment.at_finish:
            # Record the previous step's position
            prev_x = a.environment.cur_x
            prev_y = a.environment.cur_y

            # Record previous steps' acceleration
            prev_a_x, prev_a_y = a.getAcceleration()
            prev_state_action_vector = [prev_x, prev_y, prev_a_x, prev_a_y]
            
            # Determine a new acceleration
            a.setAcceleration(prev_x, prev_y)
            a_x, a_y = a.getAcceleration()
            
            # Update the nuew position
            x, y = a.environment.updatePosition(a_x, a_y)
            v_x = a.environment.cur_vx
            v_y = a.environment.cur_vy
    
            last_valid_x, last_valid_y = a.environment.detectCollision(prev_x, prev_y, x, y)
            a.environment.cur_x = last_valid_x
            a.environment.cur_y = last_valid_y 
    
            if a.environment.collision_occurred:
                a.environment.resetPosition(last_valid_x, last_valid_y)
                num_collisions+=1
    
            # Get the reward for the move
            reward = a.environment.returnRewardValue()
    
            # Record the reward for the agent
            a.recordReward(reward)
    
            sum_reward += reward
            num_iterations+=1

        mc_count.append(mc_num)
        overall_reward.append(sum_reward)
        num_timesteps.append(num_iterations)
        num_collisions_arr.append(num_collisions)
        a.getPolicyUpdate()
        a.environment.resetGame()
        toc= time.perf_counter()
        timestep_arr.append(toc-tic)
        runtime=toc-tic
        a.environment.at_finish=False
        if mc_num%50==0:
            print(f'Monte Carlo Number: {mc_num}')
            print(f'Number of collisions: {num_collisions}')
            print(f'Runtime: {runtime:.03f} s\n')

    results_df = pd.DataFrame({'mc_count': mc_count, 'overall_reward': overall_reward, 'num_timesteps': num_timesteps, 'runtime': timestep_arr, 'num_collisions': num_collisions_arr})
    results_df.to_csv(f'../results/{agent_label}.csv')
    a.q_table.to_csv(f'../results/{agent_label}_q-table.csv')
    # Pickle output of agent, to be used for testing portion

    with open(f'../trained_agents/{agent_label}.instance', 'wb') as agent_instance_file:
        pickle.dump(a, agent_instance_file)
    
    return a


if __name__=='__main__':

    '''
    Generate all combinations of tracks, labels
    '''
    track_list = ['L-track.txt', 'R-track.txt', 'Q-track.txt']

    num_mcs = 250
    
    '''
    For each track in the track list, train a Q-Learning agent
    on that specific track
    '''
    for track_name in track_list:
        agent_label = track_name.strip('.txt')

        track = Environment(f'./track_files/{track_name}')

        current_q_agent = QAgent(f'{agent_label}_QL', track) 
        current_q_agent.initializeTables()
        run_Q(current_q_agent, agent_label, num_mcs)

        current_s_agent = SAgent(f'{agent_label}_SARSA', track)
        current_s_agent.initializeTables()
        run_S(current_s_agent, agent_label, num_mcs)
        
