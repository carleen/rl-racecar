# Import agent module
from modules.Agent import Agent

class SAgent(Agent):

    def __init__(self, label, environment):
        super().__init__(label, environment)
    
    def sarsaUpdateQTable(self, prev_state_action_vector, cur_state_action_vector, reward):
        # Get the state/action/reward, work backwards, update Q table for agent

        eta = 0.1 # Learning Rate
        gamma = 0.9 # Discount Factor
        
        # Previous state, action
        prev_x = prev_state_action_vector[0]
        prev_y = prev_state_action_vector[1]
        prev_a_x = prev_state_action_vector[2]
        prev_a_y = prev_state_action_vector[3]
        
        df_index = (self.environment.max_cols) * prev_y + prev_x
        action_str = f'[{prev_a_x}, {prev_a_y}]'
        if df_index >= 0 and df_index < len(self.q_table):
            prev_q = self.q_table.at[df_index, action_str]
        
        # Current state, action
        cur_x = cur_state_action_vector[0]
        cur_y = cur_state_action_vector[1]
        cur_a_x = cur_state_action_vector[2]
        cur_a_y = cur_state_action_vector[3]
        
        cur_df_index = (self.environment.max_cols) * cur_y + cur_x
        cur_action_str = f'[{cur_a_x}, {cur_a_y}]'
        
        if cur_df_index >= 0 and cur_df_index < len(self.q_table):
            cur_q = self.q_table.at[cur_df_index, cur_action_str]
            self.q_table.at[df_index, action_str] = prev_q + eta * (reward + gamma * cur_q - prev_q)
            
        if self.environment.at_finish:
            self.q_table.at[df_index, action_str] = (prev_q + eta * (reward - prev_q))


