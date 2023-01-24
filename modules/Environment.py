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
