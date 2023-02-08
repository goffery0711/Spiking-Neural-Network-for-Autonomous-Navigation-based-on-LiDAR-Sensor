#!/usr/bin/env python

import numpy as np
import pandas as pd
import math
import os.path
import time
import matplotlib.pyplot as plt
import cv2
from collections import deque

class Radarpreprocess():
    def __init__(self):
        self.i = 10    # 0
        self.vld_data = np.zeros(13000, dtype=int)
        self.queue = deque()
        self.queue.append(self.vld_data)

    def vld_callback_new(self, msg):

        points = np.asarray(msg.points)
        grid_len_y = 0.1
        grid_len_x = 0.2
        self.y_grid = int(4 / grid_len_y)   # 20
        self.x_grid = int(4 / grid_len_x)   # 20
        self.grid = np.zeros((self.x_grid, self.y_grid, 2), dtype=float)  # (20,20,2)

        self.grid[:, :, 1] = -99

        for point in points:
            if -2 <= point.x < 2 and 0 <= point.y < 4:
                grid_pos_y = int(point.y / grid_len_y)
                grid_pos_x = int((point.x + 2) / grid_len_x)
                if self.grid[grid_pos_x, grid_pos_y, 0] > point.z:
                    self.grid[grid_pos_x, grid_pos_y, 0] = point.z
                if self.grid[grid_pos_x, grid_pos_y, 1] < point.z:
                    self.grid[grid_pos_x, grid_pos_y, 1] = point.z
        return


    def vld_callback_diff(self, msg):
        t1 = time.time()
        self.vld_data = pd.DataFrame()
        x_data = []
        y_data = []
        z_data = []

        points = np.asarray(msg.points)

        for point in points:
            x_data.append(point.x)
            y_data.append(point.y)
            z_data.append(point.z)

        df_x = pd.DataFrame(x_data, columns=['x'])
        df_y = pd.DataFrame(y_data, columns=['y'])
        df_z = pd.DataFrame(z_data, columns=['z'])
        df_grid_org = pd.DataFrame(np.ones(len(df_x)) * 9999, columns=["grid_index"])
        df_pcl_pre = pd.concat([df_x, df_y, df_z, df_grid_org], axis=1)
        td = time.time() - t1
        print('1:', td)

        df_pcl = df_pcl_pre.drop(df_pcl_pre[(df_pcl_pre['x'] < -1) | (df_pcl_pre['x'] > 1) | (df_pcl_pre['y'] < 0) | (
                df_pcl_pre['y'] > 4)].index)

        td = time.time() - t1
        print('2:', td)

        # print(df_pcl.head(10))
        x_og = df_pcl.iloc[:, 0].min()
        y_og = df_pcl.iloc[:, 1].min()

        grid_len_x = 0.2
        grid_len_y = 0.2  # 0.5

        x_grid = int(int(math.ceil(df_pcl.iloc[:, 0].max() - df_pcl.iloc[:, 0].min())) / grid_len_x)  # 2/0.2 = 10
        y_grid = int(int(math.ceil(df_pcl.iloc[:, 1].max() - df_pcl.iloc[:, 1].min())) / grid_len_y)  # 4/0.5 = 8
        grid_num = x_grid * y_grid  # 80

        td = time.time() - t1
        print('3:', td)

        for i in range(len(df_pcl.index)):  # 16646
            n = int((df_pcl.iloc[i, 0] - x_og) / grid_len_x)
            m = int((df_pcl.iloc[i, 1] - y_og) / grid_len_y)
            df_pcl.iloc[i, 3] = m * x_grid + n

        grids = df_pcl.groupby('grid_index')

        td = time.time() - t1
        print('4:', td)

        resolution = 1   # 2

        for name, group in grids:
            # for i in range(resolution):  # resolution = 5
            y_min = (name / x_grid) * grid_len_y   # 0.5
            y_max = (name / x_grid) * grid_len_y + grid_len_y   # 0.5+0.1

            df_buff = group[(group['y'] >= y_min) & (group['y'] < y_max)]
            df_diff = df_buff.diff()
            path_index = df_diff[(abs(df_diff['z']) >= 0.02)].index
            df_path_buff = group.loc[path_index]
            if not df_path_buff.empty:
                self.vld_data = pd.concat([self.vld_data, df_path_buff], axis=0)  # ignore_index=True


        td = time.time() - t1
        print('5:', td)

        # print(len(self.vld_data))
        return

    def vld_callback(self, msg):
        self.vld_data = np.array(())

        ########
        self.vld_data_x = np.array(())
        self.vld_data_y = np.array(())

        points = np.asarray(msg.points)

        for point in points:
            if -0.26 <= point.z <= -0.22:   # [-0.25,-0.23]
                self.vld_data = np.append(self.vld_data, point.x)
                self.vld_data = np.append(self.vld_data, point.y)
                ##############
                if -1 <= point.x < 1 and 0 <= point.y < 4:
                    self.vld_data_x = np.append(self.vld_data_x, point.x)
                    self.vld_data_y = np.append(self.vld_data_y, point.y)
        # print(self.vld_data.shape)
        self.queue.append(self.vld_data)
        self.queue.popleft()
        # ######Plot#######
        # self.plot_lidar(self.vld_data_x, self.vld_data_y)
        return

    def plot_lidar(self, x_points, y_points):
        side_range = (-1, 1)  # left-most to right-most
        fwd_range = (0, 4)  # back-most to forward-most
        res = 0.01

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (x_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (y_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor and ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img -= int(np.floor(fwd_range[0] / res))

        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = int((side_range[1] - side_range[0]) / res)
        y_max = int((fwd_range[1] - fwd_range[0]) / res)
        im = np.zeros([y_max, x_max], dtype=np.uint8)  # initialize black picture

        # y_img = y_max - y_img
        # FILL PIXEL VALUES IN IMAGE ARRAY
        im[y_img, x_img] = 255  # lidar points as white color
        im = np.rot90(im, 2)  # rotate 90
        ###########
        cv2.imshow("img", im)
        cv2.waitKey(1)


    def getNewState_new(self):
        new_state = np.zeros((4, 8), dtype=int)

        for i in range(self.y_grid):
            for j in range(self.x_grid):
                if self.grid[j, i, 0] != 0 and self.grid[j, i, 1] != -99 and (self.grid[j, i, 1] - self.grid[j, i, 0]) >= 0.01:
                    new_state[int(j / 5), int(i / 5)] += 1
        # print(new_state)
        return new_state

    def showradarplot_new(self):
        plt.figure(1)
        plt.clf()
        plt.ion()
        grid = self.grid
        # new_state = np.zeros((4, 4), dtype=int)
        for i in range(self.y_grid):
            for j in range(self.x_grid):
                if grid[j, i, 0] != 0 and grid[j, i, 1] != -99 and (grid[j, i, 1] - grid[j, i, 0]) >= 0.01:
                    plt.scatter(int(j / 5), int(i / 5))

        # for i in range(len(self.vld_data) // 2):  # len(self.dvs_data)//2=925
        #     if -1 <= self.vld_data[i * 2] <= 1 and 0 <= self.vld_data[i * 2 + 1] <= 4:
        #         state_x = round(2 * (self.vld_data[i * 2] + 1))  # x:[-1,1]--[0,4]
        #         state_y = round(self.vld_data[i * 2 + 1])  # y:[0,4]
        #         if state_x < 4 and state_y < 4:
        #             # idx = (int(state_x), int(state_y))
        #             # new_state[idx] += 1
        #             plt.scatter(int(state_x), int(state_y))
        plt.xlim([0, 3])
        plt.ylim([0, 3])
        plt.pause(0.001)
        plt.clf()
        plt.ioff()

    def showradarplot(self):
        plt.figure(2)
        plt.clf()
        plt.ion()
        try:
            for i in range(len(self.vld_data) // 2):  # len(self.dvs_data)//2=925
                if -1 <= self.vld_data[i * 2] <= 1 and 0 <= self.vld_data[i * 2 + 1] <= 4:
                    state_x = round(2 * (self.vld_data[i * 2] + 1))  # x:[-1,1]--[0,4]
                    state_y = round(self.vld_data[i * 2 + 1])  # y:[0,4]
                    if state_x < 4 and state_y < 4:

                        plt.scatter(int(state_x), int(state_y))
        except:
            pass
        plt.xlim([0, 3])
        plt.ylim([0, 3])
        plt.pause(0.001)
        plt.clf()
        plt.ioff()

    def getNewState_diff(self):
        new_state = np.zeros((4, 4), dtype=int)  # (4,4)
        for i in range(len(self.vld_data)):
            try:
                if -1 <= self.vld_data.iloc[i, 0] <= 1 and 0 <= self.vld_data.iloc[i, 1] <= 4:
                    state_x = int(2 * (self.vld_data.iloc[i, 0] + 1))  # x:[-1,1]--[0,4]
                    state_y = int(self.vld_data.iloc[i, 1])  # y:[0,4]
                    if state_x < 4 and state_y < 4:
                        idx = (int(state_x), int(state_y))
                        new_state[idx] += 1
            except:
                pass
        # print(new_state)
        return new_state

    def getNewState(self):

        # new_state = np.zeros((8, 4), dtype=int)  # (8,4)
        # for i in range(len(self.vld_data) // 2):  # len(self.dvs_data)//2=9179
        # 	try:
        # 		if self.vld_data[i * 2] >= 0:
        # 			state_x = 8 * self.vld_data[i * 2]  # y:[0,10]--[0,80]
        # 			state_y = 4 * self.vld_data[i * 2 + 1]  # x:[0,10]--[0,40]   Then:[80,40]
        # 			if state_y >= 0:
        # 				idx = (int(state_x // 10), int(state_y // 10))  # [80,40]--[8,4]
        # 				new_state[idx] += 1
        # 	except:
        # 		pass
        # return new_state

        # # original used
        # # print(len(self.vld_data))
        # new_state = np.zeros((4, 4), dtype=int)  # (4,4)
        # for i in range(len(self.vld_data) // 2):  # len(self.dvs_data)//2=925
        #     try:
        #         if -1 <= self.vld_data[i * 2] <= 1 and 0 <= self.vld_data[i * 2 + 1] <= 4:
        #             state_x = round(2 * (self.vld_data[i * 2] + 1))  # x:[-1,1]--[0,4]
        #             state_y = round(self.vld_data[i * 2 + 1])  # y:[0,4]
        #             if state_x < 4 and state_y < 4:
        #                 idx = (int(state_x), int(state_y))
        #                 new_state[idx] += 1
        #
        #     except:
        #         pass
        # # print(new_state)
        # return new_state

        # Not Working
        # # print(len(self.vld_data))
        # new_state = np.zeros((32, 16), dtype=int)
        # for i in range(len(self.vld_data) // 2):  # len(self.dvs_data)//2=925
        #     try:
        #         if -1 <= self.vld_data[i * 2] <= 1 and 0 <= self.vld_data[i * 2 + 1] <= 4:
        #             state_x = round(16 * (self.vld_data[i * 2] + 1))  # x:[-1,1]--[0,32]
        #             state_y = round(4 * self.vld_data[i * 2 + 1])  # y:[0,4]--[0,16]
        #             if state_x < 32 and state_y < 16:
        #                 idx = (int(state_x), int(state_y))
        #                 new_state[idx] += 1
        #
        #     except:
        #
        # # print(new_state)
        # state_list = new_state.flatten()
        # # print(state_list)
        # return state_list


        # Not Working
        # # Initialize array in new size
        # resize_factor = 4  # from parameter
        # new_state = np.zeros(40 * 20, dtype=int)
        #
        # for i in range(len(self.vld_data) // 2):  # For every event in lidar vld frame
        #     try:
        #         if -1 <= self.vld_data[i * 2] <= 1 and 0 <= self.vld_data[i * 2 + 1] <= 4:
        #             state_y = 20 * self.vld_data[i * 2 + 1]     # [0,4] --> [0,80]  --  [0,20]
        #             state_x = 80 * (self.vld_data[i * 2] + 1)       # [-1,1] --> [0,160]  --  [0,40]
        #             idx = (state_y // resize_factor) * 40 + state_x // resize_factor  # resize_factor = 4
        #             idx = int(idx)
        #             if idx >= 800:
        #                 idx = 799
        #             new_state[idx] += 1
        #     except:
        #         pass
        # # print(new_state[100:612])
        # return new_state[100:612], new_state  # Here only first part  # Original DVS(middle): (6*32 : -10*32)=(6*32 : 22*32)


       # Initialize array in new size
        resize_factor = 4  # from parameter
        new_state = np.zeros(40 * 20, dtype=int)
        s = self.queue[-1]
        # print(s.shape, 'a')
        for i in range(len(s) // 2):  # For every event in lidar vld frame
            if -1 <= s[i * 2] <= 1 and 0 <= s[i * 2 + 1] <= 4:
                state_x = 80 * (s[i * 2] + 1)  # [-1,1] --> [0,160]  --  [0,40]
                state_y = 20 * s[i * 2 + 1]  # [0,4] --> [0,80]  --  [0,20]
                idx = (state_y // resize_factor) * 40 + state_x // resize_factor  # resize_factor = 4
                idx = int(idx)
                if idx >= 800:
                    idx = 799
                new_state[idx] += 1
        # print(new_state[100:612])
        return new_state, new_state  # Here only first part  # Original DVS(middle): (6*32 : -10*32)=(6*32 : 22*32)



    def showradarplot_vld(self):
        plt.clf()
        plt.ion()
        s = self.vld_data
        for i in range(len(s) // 2):  # For every event in DVS frame
            if -1 <= s[i * 2] <= 1 and 0 <= s[i * 2 + 1] <= 4:
                x = (80 * (s[i * 2] + 1)) // 4
                y = (20 * s[i * 2 + 1]) // 4
                plt.scatter(x, y)
                # plt.scatter(s[i * 2], s[i * 2 + 1])
        # # plt.xlim([-5,5])
        # plt.xlim([-2.5, 2.5])  # just keep the same as ylim
        # plt.ylim([0, 5])
        plt.xlim([0, 40])
        plt.ylim([0, 20])
        plt.pause(0.1)
        plt.clf()
        plt.ioff()
