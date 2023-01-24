import os
import time
import sys
import pandas as pd
import numpy as np
import csv
import math as m

from modules.Environment import Environment


# Suppress warnings that are a result of current diasgreements between pandas/numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
