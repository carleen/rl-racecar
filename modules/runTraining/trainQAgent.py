from modules import *

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
		
# Create the agents, tracks
l_track_q = rl.Environment('../track_files/L-track.txt')
l_q = rl.QAgent('l_q', l_track_q)
l_q.initializeTables()
run_Q(l_q, 'l-q_agent')

r_track_q_a = rl.Environment('../track_files/R-track.txt')
l_r_a = rl.QAgent('r_q_a', r_track_q_a)
l_r_a.initializeTables()
run_Q(l_r_a, 'r-q-a_agent')

o_track_q = rl.Environment('../track_files/O-track.txt')
o_q = rl.QAgent('o_q', o_track_q)
o_q.initializeTables()
run_Q(o_q, 'o-q_agent')

r_track_q_b = rl.Environment('../track_files/R-track.txt')
l_r_b = rl.QAgent('r_q_b', r_track_q_b)
l_r_b.initializeTables()
run_Q(l_r_b, 'r-q-b_agent')

