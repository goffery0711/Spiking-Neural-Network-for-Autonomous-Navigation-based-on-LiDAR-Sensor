#!/usr/bin/env python

import numpy as np
import pandas as pd
import math
import os.path
import time
import matplotlib.pyplot as plt

class Radarpreprocess():
    def __init__(self):
        self.i = 10    # 0

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
        ## t1 = time.time()
        ## PCL data

        ## path = "/Thesis/Training-NN/Controller/R-STDP/test.pcd"
        ## o3d.io.write_point_cloud(path, msg)
        ## pcd = o3d.io.read_point_cloud(path)
        ## value = 16
        ## pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, value)

        ## cloud = pcl.PointCloud()
        ## pointcloud = np.array(msg, dtype=np.float32)
        ## cloud.from_array(pointcloud)
        ## pcl.save(cloud, "cloud.pcd", format='pcd')

        ## self.vld_data = pcd_new.points
        ## points = np.asarray(self.vld_data)

        # x_data = []
        # y_data = []
        # z_data = []

        self.vld_data = np.array(())

        points = np.asarray(msg.points)

        for point in points:
            # if self.i % 10 == 0:
            #     x_data.append(point.x)
            #     y_data.append(point.y)
            #     z_data.append(point.z)
            if -0.25 <= point.z <= -0.23:
                self.vld_data = np.append(self.vld_data, point.x)
                self.vld_data = np.append(self.vld_data, point.y)

        # output_dir = './lidar'
        # if self.i % 10 == 0 and self.i <= 1200:
        #     df_x = pd.DataFrame(x_data)
        #     df_x.to_csv(os.path.join(output_dir, 'x_' + str(self.i) + '.csv'))
        #     df_y = pd.DataFrame(y_data)
        #     df_y.to_csv(os.path.join(output_dir, 'y_' + str(self.i) + '.csv'))
        #     df_z = pd.DataFrame(z_data)
        #     df_z.to_csv(os.path.join(output_dir, 'z_' + str(self.i) + '.csv'))
        # self.i = self.i+1

        ## plt.scatter(range(len(self.z_data)), self.z_data)
        ## plt.show()
        ## print(len(self.x_data))   # <class 'geometry_msgs.msg._Point32.Point32'>
        ## print(self.vld_data.shape)
        ## td = time.time() - t1
        ## print(td)

        return

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

        # print(len(self.vld_data))
        new_state = np.zeros((4, 4), dtype=int)  # (4,4)
        for i in range(len(self.vld_data) // 2):  # len(self.dvs_data)//2=925
            try:
                if -1 <= self.vld_data[i * 2] <= 1 and 0 <= self.vld_data[i * 2 + 1] <= 4:
                    state_x = round(2 * (self.vld_data[i * 2] + 1))  # x:[-1,1]--[0,4]
                    state_y = round(self.vld_data[i * 2 + 1])  # y:[0,4]
                    if state_x < 4 and state_y < 4:
                        idx = (int(state_x), int(state_y))
                        new_state[idx] += 1

            except:
                pass
        # print(new_state)
        return new_state
