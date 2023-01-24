# Import Agent Module
from modules.Agent import Agent


# Suppress warnings that are a result of current diasgreements between pandas/numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class QAgent(Agent):

    def __init__(self, label, environment):
        super().__init__(label, environment)
    
    def getPolicyUpdate(self):
        # Get the state/action/reward, work backwards, update Q table for agent
        reward_arr = self.reward_arr
        state_arr_x = self.state_arr_x
        state_arr_y = self.state_arr_y
        action_arr_x = self.action_arr_x
        action_arr_y = self.action_arr_y

        reward_arr.reverse()
        state_arr_x.reverse()
        state_arr_y.reverse()
        action_arr_x.reverse()
        action_arr_y.reverse()

        G = 0
        gamma = 0.9

        for ind in range(0, len(reward_arr)):
            r = reward_arr[ind]
            a_x = action_arr_x[ind]
            a_y = action_arr_y[ind]
            s_x = state_arr_x[ind]
            s_y = state_arr_y[ind]
            
            action_str = f'[{a_x}, {a_y}]'
            
            if s_x < 0 or s_y <0 or s_x >= self.environment.max_cols or s_y >= self.environment.max_rows:
                val_skipped = True
            else:
                df_index = self.environment.max_cols * s_y + s_x
    
                q_update = G*gamma + r
    
                self.n_table.at[df_index, action_str] = self.n_table.at[df_index, action_str] + 1
                n_value = self.n_table.at[df_index, action_str]
    
                G = q_update
    
                if n_value==1:
                    self.q_table.at[df_index, action_str] = self.q_table.at[df_index, action_str]
                else:
                    self.q_table.at[df_index, action_str] = self.q_table.at[df_index, action_str] + (1/n_value)*(G-self.q_table.at[df_index, action_str])
            

        self.updatePolicy()
        
    def updatePolicy(self):
        for cur_ind in range(0, len(self.q_table)):
            action_columns = self.q_table.columns.to_list()
            action_columns.remove('row_index')
            action_columns.remove('column_index')
            action_columns.remove('char_arr')

            action_df_only = self.q_table[action_columns]
            cur_row = action_df_only.loc[cur_ind]
            A_max = max(cur_row)
            max_probs = []
            for c in action_columns:
                cur_a = action_df_only.at[cur_ind, c]
                if cur_a >= A_max:
                    max_probs.append(c)

            for c in action_columns:
                if c in max_probs:
                    self.policy_table.at[cur_ind, c] = 1/len(max_probs)
                else:
                    self.policy_table.at[cur_ind, c] = 0
        
    def resetSARArrays(self):
        # Reset reward, state, and action arrays
        self.reward_arr = []
        self.state_arr_x = []
        self.state_arr_y = []
        self.action_arr_x = []
        self.action_arr_y = []
