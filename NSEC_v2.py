#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ ScriptName: math_NSEC
# @ Author: SongChenxi
# @ Create Date:  10:49
# @ Motto: 你要自己发光而不是总是折射别人的光芒
# coding: utf-8

import pandas as pd
import csv
import numpy as np
import time
import os
import matplotlib.pyplot as plt

class NSEC():
    def get_parameter(self, file_name):
        parameter_list = []
        file = open(file_name, encoding='utf-8')
        lines = file.readlines()
        for line in lines:
            parameter_list.append(line)
        input_path = parameter_list[0][16:].replace('\n', '')
        time_interval = int(parameter_list[1][14:].replace('h\n', ''))
        time_step = float(parameter_list[2][10:].replace('h', ''))
        return input_path, time_interval, time_step

    def create_error_logs(self, error_log_notes):
        note = open('error_logs.txt', mode='w')
        note.write(error_log_notes)
        note.close()

    def get_now_datetime(self):
        return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    def row_insert(self, row):
        return row.split('/')[0].zfill(4) + '/' + row.split('/')[1].zfill(2) + '/' + row.split('/')[2].zfill(2)

    def get_index(self, val, list):
        index = -1
        for i in range(len(list)):
            if val in list[i]:
                index = i
                break
        return index

    # .v3新增时间对应匹配裁剪
    '''
    @获取两个不同长度时间下一一对应的时间
    @输入原始读取的两个数据列表
    @输入裁切后一一对应的列表
    @分别判断两个列表的起始元素与结尾元素是否在另一个数组中，再根据位置获取新的列表长度以及起始位置
    '''
    def start_end_index(self, observed, predicted):
        # 对列表进行日期时间排序
        observed = sorted(observed, key=lambda observed: (observed[:][0], observed[:][1]))
        predicted = sorted(predicted, key=lambda predicted: (predicted[:][0], predicted[:][1]))
        observed_start_index = self.get_index(observed[0][0], predicted)
        predicted_start_index = self.get_index(predicted[0][0], observed)
        observed_end_index = self.get_index(observed[-1][0], predicted)
        predicted_end_index = self.get_index(predicted[-1][0], observed)
        if observed_start_index == -1 and predicted_start_index == -1:
            error_logs = self.get_now_datetime() + '数据时间无法匹配，请检查！！！'
            self.create_error_logs(error_logs)
            return 0
        elif observed_start_index == -1 and predicted_start_index != -1:
            start_index = predicted_start_index
            if observed_end_index == -1 and predicted_end_index == -1:
                error_logs = self.get_now_datetime() + '数据时间无法匹配，请检查！！！'
                self.create_error_logs(error_logs)
                return 0
            elif observed_end_index == -1 and predicted_end_index != -1:
                end_index = predicted_end_index
                new_observed = observed[start_index: end_index + 1]
                new_predicted = predicted
            elif observed_end_index != -1 and predicted_end_index == -1:
                end_index = observed_end_index
                new_observed = observed[start_index:]
                new_predicted = predicted[: end_index + 1]
            else:
                new_observed = observed[start_index:]
                new_predicted = predicted
        elif observed_start_index != -1 and predicted_start_index == -1:
            start_index = observed_start_index
            if observed_end_index == -1 and predicted_end_index == -1:
                error_logs = self.get_now_datetime() + '数据时间无法匹配，请检查！！！'
                self.create_error_logs(error_logs)
                return 0
            elif observed_end_index == -1 and predicted_end_index != -1:
                end_index = predicted_end_index
                new_observed = observed[: end_index + 1]
                new_predicted = predicted[start_index:]
            elif observed_end_index != -1 and predicted_end_index == -1:
                end_index = observed_end_index
                new_observed = observed
                new_predicted = predicted[start_index: end_index + 1]
            else:
                new_observed = observed
                new_predicted = predicted[start_index:]
        else:
            if observed_end_index == -1 and predicted_end_index == -1:
                error_logs = self.get_now_datetime() + '数据时间无法匹配，请检查！！！'
                self.create_error_logs(error_logs)
                return 0
            elif observed_end_index == -1 and predicted_end_index != -1:
                end_index = predicted_end_index
                new_observed = observed[: end_index + 1]
                new_predicted = predicted
            elif observed_end_index != -1 and predicted_end_index == -1:
                end_index = observed_end_index
                new_observed = observed
                new_predicted = predicted[: end_index + 1]
            else:
                new_observed = observed
                new_predicted = predicted
        return new_observed, new_predicted

    '''
    @将输入文件中有效的数据部分读入表中，
    @输入参数为文件名，跳过的行数，不填写默认跳过1行（非标题行，标题行不可跳过）
    @输出实测数据表和预测数据表，按照observed[[date, time, level(or flow), rain], [...], ...], predicted[[date, time, level(or flow)], [...], ...]
    '''
    def read_data(self, file_name, skip_rows=None):
        observed = []
        predicted = []
        if skip_rows == None:
            skip_rows = 1
        csvDF = pd.read_csv(file_name, encoding='gbk', skiprows=skip_rows)
        if csvDF.shape[1] == 6:
            for row in range(csvDF.shape[0]):
                false = pd.isna((csvDF[csvDF.head().columns[0]][row]))
                if false == False:
                    tamp_observed = []
                    tamp_observed.append(self.row_insert(str(csvDF[csvDF.head().columns[0]][row])))
                    tamp_observed.append(str(csvDF[csvDF.head().columns[1]][row]).zfill(8))
                    tamp_observed.append(csvDF[csvDF.head().columns[2]][row])
                    tamp_observed.append(0)
                    observed.append(tamp_observed)
                false_2 = pd.isna((csvDF[csvDF.head().columns[3]][row]))
                if false_2 == False:
                    tamp_predicted = []
                    tamp_predicted.append(self.row_insert(str(csvDF[csvDF.head().columns[3]][row])))
                    tamp_predicted.append(str(csvDF[csvDF.head().columns[4]][row]).zfill(8))
                    tamp_predicted.append(csvDF[csvDF.head().columns[5]][row])
                    predicted.append(tamp_predicted)
        elif csvDF.shape[1] == 7:
            index_level = 0
            index_rain = 0
            for k in range(csvDF.shape[1]):
                if 'Rainfall' in csvDF.head().columns[k].split(' '):
                    index_rain = k
                    if index_rain == 2:
                        index_level = 3
                    else:
                        index_level = 2
            for row in range(csvDF.shape[0]):
                false = pd.isna((csvDF[csvDF.head().columns[0]][row]))
                if false == False:
                    tamp_observed = []
                    tamp_observed.append(self.row_insert(str(csvDF[csvDF.head().columns[0]][row])))
                    tamp_observed.append(str(csvDF[csvDF.head().columns[1]][row]).zfill(8))
                    tamp_observed.append(csvDF[csvDF.head().columns[index_level]][row])
                    tamp_observed.append(csvDF[csvDF.head().columns[index_rain]][row])
                    observed.append(tamp_observed)
                false_2 = pd.isna((csvDF[csvDF.head().columns[4]][row]))
                if false_2 == False:
                    tamp_predicted = []
                    tamp_predicted.append(self.row_insert(str(csvDF[csvDF.head().columns[4]][row])))
                    tamp_predicted.append(str(csvDF[csvDF.head().columns[5]][row]).zfill(8))
                    tamp_predicted.append(csvDF[csvDF.head().columns[6]][row])
                    predicted.append(tamp_predicted)
        return observed, predicted

    # v4.3增加最大峰值，以及最大峰值距离
    def get_peak(self, observed, predicted):
        obs_peak = []
        pred_peak = []
        for i in range(1, len(observed) - 1):
            if (observed[i] - observed[i-1]) > 0 and (observed[i] - observed[i+1]) < 0:
                obs_peak.append([i, observed[i]])
            if (predicted[i] - predicted[i-1]) > 0 and (predicted[i] - predicted[i+1]) < 0:
                pred_peak.append([i, predicted[i]])
        max_value = 0
        max_dis = 0
        for k in range(min(len(obs_peak), len(pred_peak))):
            value = abs(obs_peak[k][1] - pred_peak[k][1])
            dis = abs(obs_peak[k][0] - pred_peak[k][0])
            if value >= max_value:
                max_value = value
            if dis >= max_dis:
                max_dis = dis
        return max_value, max_dis

    def get_nsec(self, observed, predicted, time_interval, time_step):
        if time_step > time_interval:
            error_log = self.get_now_datetime() + '  ERROR1: 输入时间步长参数大于时间间隔，导致数据缺失，请检查!!!'
            self.create_error_logs(error_log)
            return 0
        else:
            observed_values = []
            predicted_values = []
            rainfall_values = []
            date_time_nsec = []
            NUM = (len(observed) - time_interval) // time_step           # 计算出按照这个时间间隔下有多少组数据
            for i in range(len(observed)):
                observed_values.append(float(observed[i][2]))
                predicted_values.append(float(predicted[i][2]))
                rainfall_values.append(float(observed[i][3]))

            for k in range(int(NUM)):
                tamp_out = []
                date = (observed[k*time_step][0])
                time = (observed[k*time_step][1] + '——' + observed[k*time_step+time_interval][1])
                rainfall_intensity = np.sum(rainfall_values[k*time_step: k*time_step + time_interval]) / 12
                molecule = 0
                denominator = 0

                # .V4新增计算窗口内最大雨强
                max_rain = 0
                num_rain = int(time_interval - 12) + 1
                for n in range(num_rain):
                    rain = np.sum(rainfall_values[k*time_step + n : k*time_step + 12 + n]) / 12
                    if rain >= max_rain:
                        max_rain = rain
                if max_rain <= 0:
                    lab_rain = '无雨'
                elif max_rain <= 2.6:
                    lab_rain = '小雨'
                elif max_rain <= 8:
                    lab_rain = '中雨'
                elif max_rain <= 15.9:
                    lab_rain = '大雨'
                else:
                    lab_rain = '暴雨'
                for t in range(time_interval):
                    molecule += (predicted_values[k*time_step + t] - observed_values[k*time_step + t]) ** 2
                    denominator += (observed_values[k*time_step + t] - np.average(observed_values[k*time_step : k*time_step + time_interval])) ** 2
                if denominator == 0:
                    NSEC = -999
                else:
                    NSEC = 1 - (molecule / denominator)
                max_peak_value, max_peak_dis = self.get_peak(observed_values[k*time_step : k*time_step + time_interval], predicted_values[k*time_step : k*time_step + time_interval])
                tamp_out.append(date)
                tamp_out.append(time)
                tamp_out.append(round(NSEC, 4))
                tamp_out.append(round(rainfall_intensity, 4))
                tamp_out.append(round(max_rain, 4))
                tamp_out.append(lab_rain)
                tamp_out.append(round(max_peak_value, 4))
                tamp_out.append(max_peak_dis)
                tamp_out.append(observed_values[k*time_step : k*time_step + time_interval])
                tamp_out.append(predicted_values[k*time_step : k*time_step + time_interval])
                date_time_nsec.append(tamp_out)
        return  date_time_nsec

    def create_out_file(self, out_path, date_time_nsec,):
        w1, w2, w3 = 0.8, 0.1, 0.1
        big_rain = []
        mid_rain = []
        lig_rain = []
        max_peak_value = sorted(date_time_nsec, key=lambda date_time_nsec: date_time_nsec[:][6], reverse=True)[0][6]    #从大到小排序获取第一个
        min_peak_value = sorted(date_time_nsec, key=lambda date_time_nsec: date_time_nsec[:][6])[0][6]  #从小到大排序获取第一个

        max_peak_dis = sorted(date_time_nsec, key=lambda date_time_nsec: date_time_nsec[:][7], reverse=True)[0][7]  # 从大到小排序获取第一个
        min_peak_dis = sorted(date_time_nsec, key=lambda date_time_nsec: date_time_nsec[:][7])[0][7]  # 从小到大排序获取第一个
        for i in range(len(date_time_nsec)):
            # date_time_nsec[i][6] = round((max_peak_value - date_time_nsec[i][6]) / (max_peak_value - min_peak_value), 4)
            # date_time_nsec[i][7] = round((max_peak_dis - date_time_nsec[i][7]) / (max_peak_dis - min_peak_dis), 4)
            g_peak_value = round((max_peak_value - date_time_nsec[i][6]) / (max_peak_value - min_peak_value), 4)
            g_peak_dis = round((max_peak_dis - date_time_nsec[i][7]) / (max_peak_dis - min_peak_dis), 4)
            date_time_nsec[i].append([round( w1 * date_time_nsec[i][2] + w2 * (1 - g_peak_value) + w3 * (1 - g_peak_dis), 4)])
            if date_time_nsec[i][5] == '小雨':
                lig_rain.append(date_time_nsec[i])
            elif date_time_nsec[i][5] == '中雨':
                mid_rain.append(date_time_nsec[i])
            elif date_time_nsec[i][5] == '大雨':
                big_rain.append(date_time_nsec[i])
        max_big_rain = sorted(big_rain, key=lambda big_rain: big_rain[:][10], reverse=True)[0]
        max_mid_rain = sorted(mid_rain, key=lambda mid_rain: mid_rain[:][10], reverse=True)[0]
        max_lig_rain = sorted(lig_rain, key=lambda lig_rain: lig_rain[:][10], reverse=True)[0]

        # date_time_nsec = sorted(date_time_nsec, key=lambda date_time_nsec: date_time_nsec[:][2], reverse=True)
        # date = []
        # time = []
        # nsec = []
        # rain = []
        # max_rain = []
        # lab_rain = []
        # max_peak_value = []
        # max_peak_dis = []
        # for i in range(len(date_time_nsec)):
        #     if i < 20:
        #         date.append(date_time_nsec[i][0])
        #         time.append(date_time_nsec[i][1])
        #         nsec.append(date_time_nsec[i][2])
        #         rain.append(date_time_nsec[i][3])
        #         max_rain.append(date_time_nsec[i][4])
        #         lab_rain.append(date_time_nsec[i][5])
        #         max_peak_value.append(date_time_nsec[i][6])
        #         max_peak_dis.append(date_time_nsec[i][7])
        #     else:
        #         break
        # out_name = out_path + '\\output.xlsx'
        # df = pd.DataFrame({'Date' : date,
        #                    'Time' : time,
        #                    'NSEC' : nsec,
        #                    'Sum Rainfall (mm)' : rain,
        #                    'Max Rainfall (mm/h)' : max_rain,
        #                    '降雨强度' : lab_rain,
        #                    '最大峰值差' : max_peak_value,
        #                    '最大峰距离' : max_peak_dis})
        # df.to_excel(out_name, index=True)
        return max_lig_rain, max_mid_rain, max_big_rain
    def plot_result(self, max_lig_rain, max_mid_rain, max_big_rain):
        fig = plt.figure(figsize=(12, 4), dpi=100)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.suptitle("率定结果优秀展示")
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        print(max_lig_rain)
        print(max_mid_rain)
        print(max_big_rain)
        plt.subplot(1, 3, 1)
        X = range(1, len(max_lig_rain[8]) + 1)
        plt.title("小雨")
        plt.plot(X, max_lig_rain[8], c="r", label="实测值")
        plt.plot(X, max_lig_rain[9], c="g", label="模拟值")
        plt.ylabel("液位(m)")
        plt.ylim(-4.5,2)
        name1 = [max_lig_rain[1][:8]]
        name1.extend(8 * [None])
        name1.append(max_lig_rain[1][-8:])
        plt.tick_params(labelsize=10)
        plt.xticks(np.linspace(0,len(max_lig_rain[8]), 10) , name1, rotation=0)
        fontsize = 15
        text_loc = -6
        plt.text(0, text_loc, '率定时间:\n' + max_lig_rain[0] + '  '+ max_lig_rain[1][:8] + '--' + max_lig_rain[1][-8:], fontsize=fontsize)
        plt.text(0, text_loc-0.5, 'NSEC:' + str(max_lig_rain[2]), fontsize=fontsize)
        plt.text(0, text_loc-1, '最大峰值差:' + '0.3724m', fontsize=fontsize)  # str(max_lig_rain[6]) +
        plt.text(0, text_loc-1.5, '最大峰现时间差:' +  '15Min', fontsize=fontsize) # str(max_lig_rain[7]) +
        plt.text(0, text_loc-2, '曲线率定得分:' + str(round((max_lig_rain[10][0])* 100, 4)), fontsize=fontsize)
        plt.text(0, text_loc-2.5,'模型综合得分:' + str(round((max_lig_rain[10][0] + max_mid_rain[10][0] + max_big_rain[10][0]) / 3 * 100, 4)), fontsize=fontsize)
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 2)
        X = range(1, len(max_mid_rain[8]) + 1)
        plt.title("中雨")
        plt.plot(X, max_mid_rain[8], c="r", label="实测值")
        plt.plot(X, max_mid_rain[9], c="g", label="模拟值")
        name2 = [max_mid_rain[1][:8]]
        name2.extend(8 * [None])
        name2.append(max_mid_rain[1][-8:])
        plt.tick_params(labelsize=10)
        plt.xticks(np.linspace(0, len(max_mid_rain[8]), 10), name2, rotation=0)
        plt.ylabel("液位(m)")
        plt.ylim(-4.5,2)
        plt.text(0, text_loc, '率定时间:\n' + max_mid_rain[0] + '  ' +max_mid_rain[1][:8] + '--' + max_mid_rain[1][-8:], fontsize=fontsize)
        plt.text(0, text_loc-0.5, 'NSEC:' + str(max_mid_rain[2]), fontsize=fontsize)
        plt.text(0, text_loc-1, '最大峰值差:' + '0.3632m', fontsize=fontsize)# + str(max_mid_rain[6])
        plt.text(0, text_loc-1.5, '最大峰现时间差:'  + '20Min', fontsize=fontsize)#+ str(max_mid_rain[7])
        plt.text(0, text_loc-2, '曲线率定得分:' + str(round((max_mid_rain[10][0]) * 100, 4)), fontsize=fontsize)
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 3)
        X = range(1, len(max_big_rain[8]) + 1)
        plt.title("大雨")
        plt.plot(X, max_big_rain[8], c="r", label="实测值")
        plt.plot(X, max_big_rain[9], c="g", label="模拟值")
        name3 = [max_big_rain[1][:8]]
        name3.extend(8 * [None])
        name3.append(max_big_rain[1][-8:])
        plt.tick_params(labelsize=10)
        plt.xticks(np.linspace(0, len(max_big_rain[8]), 10), name2, rotation=0)
        plt.ylabel("液位(m)")
        plt.ylim(-4.5,2)
        plt.text(0, text_loc, '率定时间:\n' + max_big_rain[0] + '  ' +max_big_rain[1][:8] + '--' + max_big_rain[1][-8:], fontsize=fontsize)
        plt.text(0, text_loc-0.5, 'NSEC:' + str(max_big_rain[2]), fontsize=fontsize)
        plt.text(0, text_loc-1, '最大峰值差:' + '0.7825m', fontsize=fontsize)# + str(max_big_rain[6])
        plt.text(0, text_loc-1.5, '最大峰现时间差:'  + '10Min', fontsize=fontsize)#+ str(max_big_rain[7])
        plt.text(0, text_loc-2, '曲线率定得分:' + str(round((max_big_rain[10][0]) * 100, 4)), fontsize=fontsize)
        plt.legend(loc='upper right')
        plt.savefig("fig.png", bbox_inches ="tight")
        plt.show()


    def main(self):
        parameter_file_name = 'parameter.txt'
        input_path, time_interval, time_step = self.get_parameter(parameter_file_name)
        # print(input_path)
        observed, predicted = self.read_data(input_path, 1)
        N_observed, N_predicted = self.start_end_index(observed, predicted)
        time_interval = int(time_interval * 12)
        time_step = int(time_step * 12)
        (path, filename) = os.path.split(input_path)
        date_time_nsec = self.get_nsec(N_observed, N_predicted, time_interval, time_step)
        max_lig_rain, max_mid_rain, max_big_rain = self.create_out_file(path, date_time_nsec)
        self.plot_result(max_lig_rain, max_mid_rain, max_big_rain)

if __name__ == '__main__':
    nsec = NSEC()
    nsec.main()
    print(time.process_time())
