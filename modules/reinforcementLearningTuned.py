import os
import sys
sys.path.append('../utilities')
import pandas as pd
import numpy as np
import csv
import plotly.express as px
import math as m


# Suppress warnings that are a result of current diasgreements between pandas/numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Environment:
    
    def __init__(self, track_file, collision_procedure='a'):
        '''
        Initializes state/action variables to be stored in enivronment.
        Sets the maximum allowed velocity in the environment to 5. 
        
        Inputs:
            - track_file: name of track.txt file to be used
            - collision_procedure (optional, str): will either reset
            car to last valid position before collision (a), or to 
            start position (b)
        '''
        self.label = track_file.strip('.txt')
        self.track_file = track_file
        self.collision_procedure = collision_procedure
        
        self.max_rows = 0
        self.max_cols = 0
        
        self.environment_df = self.initializeEnvironmentTable(track_file)
        
        self.cur_x = 0
        self.cur_y = 0
        self.prev_x = 0
        self.prev_y = 0
        self.cur_vx = 0
        self.cur_vy = 0
        self.cur_ax = 0
        self.cur_ay = 0
        
        self.max_velocity = 5
        
        self.collision_occurred = False
        
        self.at_finish = False
        

    def initializeEnvironmentTable(self, track_file):
        '''
        Creates the environment table that is used for character look-up
        '''
        row_ind_arr = []
        col_ind_arr = []
        char_arr = []

        with open(track_file) as f:
            # Get the number of rows/columns in the track
            row_col_count = f.readline().strip('\n').split(',')

            num_rows = int(row_col_count[0])
            num_cols = int(row_col_count[1])
            
            self.max_rows = num_rows
            self.max_cols = num_cols

            # Initialize row index
            row_ind = 0

            for line in f.readlines():
                line = line.strip('\n')

                # Initialize column index
                col_ind = 0
                for c in line:
                    row_ind_arr.append(row_ind)
                    col_ind_arr.append(col_ind)
                    char_arr.append(c)
                    col_ind +=1
                row_ind+=1

        a_x_neg_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_pos_arr = np.zeros(num_rows*num_cols)
        
        a_x_zero_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_pos_arr = np.zeros(num_rows*num_cols)
        
        a_x_pos_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_pos_arr = np.zeros(num_rows*num_cols)


        environment_table = {'row_index': row_ind_arr,
                            'column_index': col_ind_arr,
                            'char_arr': char_arr,
                            '[-1, -1]': a_x_neg_y_neg_arr,
                            '[-1, 0]': a_x_neg_y_zero_arr,
                            '[-1, 1]': a_x_neg_y_pos_arr,
                             
                            '[0, -1]': a_x_zero_y_neg_arr,
                            '[0, 0]': a_x_zero_y_zero_arr,
                            '[0, 1]': a_x_zero_y_pos_arr,
                             
                            '[1, -1]': a_x_pos_y_neg_arr,
                            '[1, 0]': a_x_pos_y_zero_arr,
                            '[1, 1]': a_x_pos_y_pos_arr}
        
        environment_df = pd.DataFrame(environment_table)

        return environment_df
    
    def chooseStart(self):
        '''
        Randomizes start position of car. 
        '''
        e_df = self.environment_df
        start_choice = e_df[e_df.char_arr=='S'].sample() # Limit to valid positions
        self.cur_x = start_choice.column_index.to_list()[0]
        self.cur_y = start_choice.row_index.to_list()[0]
        self.cur_vx = 0
        self.cur_vy = 0
        
        self.start_x = self.cur_x
        self.start_y = self.cur_y

    def getPosition(self):
        '''
        Returns state position
        '''
        return self.cur_x, self.cur_y
        
    def updatePosition(self, a_x, a_y):
        '''
        Updates the position of the vehicle based by an
        acceleration determined by the agent. 
            - stores previous velocity
            - calculates new velocity
            - checks that new velocity does not exceed "speed limit"
        '''
        prev_vx = self.cur_vx
        prev_vy = self.cur_vy
        
        new_vx = prev_vx + a_x
        new_vy = prev_vy + a_y
        
        if abs(new_vx)<=5:
            self.cur_vx = new_vx
        if abs(new_vy)<=5:
            self.cur_vy = new_vy
        
        cur_x = self.cur_x + self.cur_vx
        cur_y = self.cur_y + self.cur_vy
        
        return cur_x, cur_y
        
    def getSymbolAtGridLocation(self, col_ind, row_ind):
        '''
        Returns the symbol at a given col, row index
        '''
        df = self.environment_df
        row_df_ind = np.array(df.index[df['row_index']==row_ind].tolist())
        col_df_ind = np.array(df.index[df['column_index']==col_ind].tolist())
        df_ind = np.intersect1d(row_df_ind, col_df_ind)

        ch = ''

        if any(df_ind):
            ind = df_ind[0]
            ch = df.iloc[ind].char_arr

        return ch
    
    def bresenhamLine(self, x0, y0, x1, y1):
        '''
        Bresenham's line algorithm, generated by chatGPT. 
        
        Input:
            - x0, y0: start points
            - x1, y1: finish points
            
        Output:
            - points: array of points that are included in the path
        '''
        points = []
        # Check if slope is greater than 1 or less than -1
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        # Swap start and end points if necessary
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = abs(y1 - y0)
        err = dx / 2
        ystep = -1 if y0 > y1 else 1
        y = y0
        for x in range(x0, x1 + 1):
            if steep:
                points.append((y, x))
            else:
                points.append((x, y))
            err -= dy
            if err < 0:
                y += ystep
                err += dx
        return points
    
    def detectCollision(self, x0, y0, x1,y1):
        points = self.bresenhamLine(x0, y0, x1, y1)
        
        # Check the path to make sure it's in the right order
        start = (x0, y0)
        end = (x1, y1)
        
        start_ind = points.index(start)
        end_ind = points.index(end)

        # Reverse path to begin with x0,y0, end with x1,y1
        if start_ind > end_ind:
            points = points[end_ind:start_ind+1]
            points.reverse()
        else:
            points = points[start_ind:end_ind+1]
        
        # Initialize last_valid_x and last_valid_y to x0,y0
        last_valid_x = x0
        last_valid_y = y0
        
        # Walk through all of the points. If the points are
        # outside of the max row/col values, or if the point
        # is a wall (#), then 
        at_finish = False
        collision_occurred = False
        
        for grid_point in points:
            grid_x = grid_point[0]
            grid_y = grid_point[1]
            
            if grid_x <= 0 or grid_y <=0:
                collision_occurred = True
            
            elif grid_x >= self.max_cols or grid_y >= self.max_rows:
                collision_occurred = True
                
            else: # Valid point on game board, so check the signal
                sym = self.getSymbolAtGridLocation(grid_x, grid_y)
                if sym == 'F':
                    at_finish = True
                    if not collision_occurred:
                        last_valid_x = grid_x
                        last_valid_y = grid_y
                elif sym == '#':
                    if not at_finish:
                        collision_occurred = True
                else:
                    last_valid_x = grid_x
                    last_valid_y = grid_y
        
        if not collision_occurred:
            self.at_finish = at_finish
        else:
            self.collision_occurred = collision_occurred
        
        return last_valid_x, last_valid_y
    
    def setCollisionProcedure(self, procedure):
        self.collision_procedure = procedure
    
    def resetPosition(self, x, y):
        '''
        Resets vehicle location in case of a collision
        '''
        if self.collision_procedure == 'a':
            self.cur_x = x
            self.cur_y = y
        else:
            self.chooseStart()
        self.cur_vx = 0
        self.cur_vy = 0    
        self.collision_occurred = False
        
            
    def returnRewardValue(self):
        '''
        Returns reward
        '''
        reward_value = -1
        if self.at_finish:
            reward_value = 25
        return reward_value
    
    def resetGame(self):
        self.at_finish = False
        self.collision_occurred = False

class Agent:


    def __init__(self, label, environment):
        '''
        Parent class, both the Q-Learning and SARSA agents inherit 
        methods/properties from this class
        '''
        self.label = label
        self.environment = environment

        self.accel_arr = [-1,0,1]

        self.a_x = 0
        self.a_y = 0

        self.reward_arr = []
        self.state_arr_x = []
        self.state_arr_y = []
        self.action_arr_x = []
        self.action_arr_y = []

    def setAcceleration(self, s_x, s_y):
        '''
        Determines the acceleration for the agent using the following
        steps:
            - gets a table of possible actions
            - finds the maximum q value for the current state, action pair 
            - randomly selects one of the chosen actions
            - sets epsilon of 0.2, generates a random number between (0,1)
            - if b is greater than epsilon, use the selected action;
            otherwise, randomly choose an action

        Additional step allows for random failure to accelerate:
            - select random number between (0,1)
            - if value is greater than 0.2, the car fails to 
            accelerate
        '''
        action_columns = self.q_table.columns.to_list()
        action_columns.remove('row_index')
        action_columns.remove('column_index')
        action_columns.remove('char_arr')

        cur_ind = (self.environment.max_cols) * s_y + s_x
        
        action_df_only = self.q_table[action_columns]
        cur_row = action_df_only.loc[cur_ind]
        A_max = max(cur_row)
        max_probs = []
        for c in action_columns:
            cur_a = action_df_only.at[cur_ind, c]
            if cur_a >= A_max:
                max_probs.append(c)


        # Select random action
        np.random.shuffle(max_probs)
        select_col = max_probs[0]

        epsilon = 0.05
        b = np.random.uniform()

        if b > epsilon:
            action = select_col
        else:
            np.random.shuffle(action_columns)
            action = action_columns[0]
        final_a_x = 0
        final_a_y = 0
        for a_x in [-1,0,1]:
            for a_y in [-1,0,1]:
                temp_str = f'[{a_x}, {a_y}]'
                if temp_str == action:
                    final_a_x = a_x
                    final_a_y = a_y

        prev_a_x = self.a_x
        prev_a_y = self.a_y

        accel_fail = np.random.uniform()
        if accel_fail >= 0.2:
            final_a_x = 0
            final_a_y = 0

        self.a_x = final_a_x
        self.a_y = final_a_y

        self.state_arr_x.append(self.environment.cur_x)
        self.state_arr_y.append(self.environment.cur_y)

        self.action_arr_x.append(self.a_x)
        self.action_arr_y.append(self.a_y)

        self.environment.prev_x = self.environment.cur_x
        self.environment.prev_y= self.environment.cur_y


    def initializeTables(self):
        '''
        Initialize Q, N, and policy tables
        '''
        self.q_table = self.createQTable()
        self.n_table = self.createNTable()
        self.policy_table = self.createPolicyTable()

    def recordReward(self, reward):
        # Store reward value
        self.reward_arr.append(reward)

    def getAcceleration(self):
        # Return acceleration
        return self.a_x, self.a_y

    def createQTable(self):
        '''
        Create table for storing Q Values
        '''
        track_file = self.environment.track_file

        row_ind_arr = []
        col_ind_arr = []
        char_arr = []

        with open(track_file) as f:
            # Get the number of rows/columns in the track
            row_col_count = f.readline().strip('\n').split(',')

            num_rows = int(row_col_count[0])
            num_cols = int(row_col_count[1])

            self.max_rows = num_rows
            self.max_cols = num_cols

            # Initialize row index
            row_ind = 0

            for line in f.readlines():
                line = line.strip('\n')

                # Initialize column index
                col_ind = 0
                for c in line:
                    row_ind_arr.append(row_ind)
                    col_ind_arr.append(col_ind)
                    char_arr.append(c)
                    col_ind +=1
                row_ind+=1

        a_x_neg_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_pos_arr = np.zeros(num_rows*num_cols)

        a_x_zero_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_pos_arr = np.zeros(num_rows*num_cols)

        a_x_pos_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_pos_arr = np.zeros(num_rows*num_cols)


        q_table = {'row_index': row_ind_arr,
                            'column_index': col_ind_arr,
                            'char_arr': char_arr,
                            '[-1, -1]': a_x_neg_y_neg_arr,
                            '[-1, 0]': a_x_neg_y_zero_arr,
                            '[-1, 1]': a_x_neg_y_pos_arr,

                            '[0, -1]': a_x_zero_y_neg_arr,
                            '[0, 0]': a_x_zero_y_zero_arr,
                            '[0, 1]': a_x_zero_y_pos_arr,

                            '[1, -1]': a_x_pos_y_neg_arr,
                            '[1, 0]': a_x_pos_y_zero_arr,
                            '[1, 1]': a_x_pos_y_pos_arr}

        q_table = pd.DataFrame(q_table)

    def getQTable(self):
        return self.q_table

    def getNTable(self):
        return self.n_table

    def createPolicyTable(self):
        track_file = self.environment.track_file

        row_ind_arr = []
        col_ind_arr = []
        char_arr = []

        with open(track_file) as f:
            # Get the number of rows/columns in the track
            row_col_count = f.readline().strip('\n').split(',')

            num_rows = int(row_col_count[0])
            num_cols = int(row_col_count[1])

            self.max_rows = num_rows
            self.max_cols = num_cols

            # Initialize row index
            row_ind = 0

            for line in f.readlines():
                line = line.strip('\n')

                # Initialize column index
                col_ind = 0
                for c in line:
                    row_ind_arr.append(row_ind)
                    col_ind_arr.append(col_ind)
                    char_arr.append(c)
                    col_ind +=1
                row_ind+=1

        a_x_neg_y_neg_arr = np.full((num_rows*num_cols), 1/9)
        a_x_neg_y_zero_arr = np.full((num_rows*num_cols), 1/9)
        a_x_neg_y_pos_arr = np.full((num_rows*num_cols), 1/9)

        a_x_zero_y_neg_arr = np.full((num_rows*num_cols), 1/9)
        a_x_zero_y_zero_arr = np.full((num_rows*num_cols), 1/9)
        a_x_zero_y_pos_arr = np.full((num_rows*num_cols), 1/9)

        a_x_pos_y_neg_arr = np.full((num_rows*num_cols), 1/9)
        a_x_pos_y_zero_arr = np.full((num_rows*num_cols), 1/9)
        a_x_pos_y_pos_arr = np.full((num_rows*num_cols), 1/9)


        policy_table = {'row_index': row_ind_arr,
                            'column_index': col_ind_arr,
                            'char_arr': char_arr,
                            '[-1, -1]': a_x_neg_y_neg_arr,
                            '[-1, 0]': a_x_neg_y_zero_arr,
                            '[-1, 1]': a_x_neg_y_pos_arr,

                            '[0, -1]': a_x_zero_y_neg_arr,
                            '[0, 0]': a_x_zero_y_zero_arr,
                            '[0, 1]': a_x_zero_y_pos_arr,

                            '[1, -1]': a_x_pos_y_neg_arr,
                            '[1, 0]': a_x_pos_y_zero_arr,
                            '[1, 1]': a_x_pos_y_pos_arr}

        policy_table = pd.DataFrame(policy_table)

        return policy_table

    def createQTable(self):
        track_file = self.environment.track_file

        row_ind_arr = []
        col_ind_arr = []
        char_arr = []

        with open(track_file) as f:
            # Get the number of rows/columns in the track
            row_col_count = f.readline().strip('\n').split(',')

            num_rows = int(row_col_count[0])
            num_cols = int(row_col_count[1])

            self.max_rows = num_rows
            self.max_cols = num_cols

            # Initialize row index
            row_ind = 0

            for line in f.readlines():
                line = line.strip('\n')

                # Initialize column index
                col_ind = 0
                for c in line:
                    row_ind_arr.append(row_ind)
                    col_ind_arr.append(col_ind)
                    char_arr.append(c)
                    col_ind +=1
                row_ind+=1

        a_x_neg_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_pos_arr = np.zeros(num_rows*num_cols)

        a_x_zero_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_pos_arr = np.zeros(num_rows*num_cols)

        a_x_pos_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_pos_arr = np.zeros(num_rows*num_cols)


        q_table = {'row_index': row_ind_arr,
                            'column_index': col_ind_arr,
                            'char_arr': char_arr,
                            '[-1, -1]': a_x_neg_y_neg_arr,
                            '[-1, 0]': a_x_neg_y_zero_arr,
                            '[-1, 1]': a_x_neg_y_pos_arr,

                            '[0, -1]': a_x_zero_y_neg_arr,
                            '[0, 0]': a_x_zero_y_zero_arr,
                            '[0, 1]': a_x_zero_y_pos_arr,

                            '[1, -1]': a_x_pos_y_neg_arr,
                            '[1, 0]': a_x_pos_y_zero_arr,
                            '[1, 1]': a_x_pos_y_pos_arr}

        q_table = pd.DataFrame(q_table)

        return q_table

    def createNTable(self):
        track_file = self.environment.track_file

        row_ind_arr = []
        col_ind_arr = []
        char_arr = []

        with open(track_file) as f:
            # Get the number of rows/columns in the track
            row_col_count = f.readline().strip('\n').split(',')

            num_rows = int(row_col_count[0])
            num_cols = int(row_col_count[1])

            self.max_rows = num_rows
            self.max_cols = num_cols

            # Initialize row index
            row_ind = 0

            for line in f.readlines():
                line = line.strip('\n')

                # Initialize column index
                col_ind = 0
                for c in line:
                    row_ind_arr.append(row_ind)
                    col_ind_arr.append(col_ind)
                    char_arr.append(c)
                    col_ind +=1
                row_ind+=1

        a_x_neg_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_neg_y_pos_arr = np.zeros(num_rows*num_cols)

        a_x_zero_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_zero_y_pos_arr = np.zeros(num_rows*num_cols)

        a_x_pos_y_neg_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_zero_arr = np.zeros(num_rows*num_cols)
        a_x_pos_y_pos_arr = np.zeros(num_rows*num_cols)


        n_table = {'row_index': row_ind_arr,
                            'column_index': col_ind_arr,
                            'char_arr': char_arr,
                            '[-1, -1]': a_x_neg_y_neg_arr,
                            '[-1, 0]': a_x_neg_y_zero_arr,
                            '[-1, 1]': a_x_neg_y_pos_arr,

                            '[0, -1]': a_x_zero_y_neg_arr,
                            '[0, 0]': a_x_zero_y_zero_arr,
                            '[0, 1]': a_x_zero_y_pos_arr,

                            '[1, -1]': a_x_pos_y_neg_arr,
                            '[1, 0]': a_x_pos_y_zero_arr,
                            '[1, 1]': a_x_pos_y_pos_arr}

        n_table = pd.DataFrame(n_table)

        return n_table

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


