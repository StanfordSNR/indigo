#!/usr/bin/env python

# Copyright 2018 Huawei Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


"""
Created on Mon Jul 23 16:19:51 2018

@author: y84107158
"""
import os
import csv
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('path',help='log mir')
args = parser.parse_args()
dir = args.path

try:
	os.makedirs(dir+'/rtt_figure')
	os.makedirs(dir+'/rtt_data')
except:
	pass

def data_smooth(data,interval):
	smooth_data = []
	for i in range(len(data))[::interval]:
		smooth_data.append(sum(data[i:i+interval])/len(data[i:i+interval]))
	return smooth_data

sender_filelist = glob.glob(r''+dir+'/sender_rtt*')

for file_path in sender_filelist:

	df_sender = pd.read_csv(file_path)
	sender_packets = float(df_sender.columns[0])
	filename = os.path.basename(file_path)
	df_receiver = pd.read_csv(dir+'receiver'+filename[10:])
	receiver_packets = float(df_receiver.columns[0])
	loss_rate = (sender_packets-receiver_packets)/sender_packets

	df_sender = df_sender[11:]

	plt.plot(df_sender,color='b',linewidth=0.1)
	plt.title('rtt')
	plt.xlabel('time')
	plt.ylabel('value')
	plt.savefig(dir+'/rtt_figure/'+filename+'_rtt.png',dpi=1000)
	plt.close()

	rtt_mean = np.mean(np.array(df_sender)[1:])
	ptile = np.percentile(df_sender,95)
	d_loss = ["loss rate", loss_rate]
	d_mean = ["mean", rtt_mean]
	d_ptile = ["95 percentile",ptile]

	csvFile = open("./"+filename+"_rtt.csv", "w")
	writer = csv.writer(csvFile)

	writer.writerow(d_loss)
	writer.writerow(d_mean)
	writer.writerow(d_ptile)

	csvFile.close()


	'''
	df_sender = pd.read_csv(file_path)
	print(df)
	print('draw',file_path)
	print(df[0])
	for i in range(9):
		df = df.drop('1')
	print(df.mean())
	print(df.columns)
	'''
	'''
	sd_best_cwnd = data_smooth(df[' best_cwnd'],10)
	sd_indigo_cwnd = data_smooth(df[' indigo_cwnd'],10)
	plt.plot(sd_best_cwnd,color='b',linewidth=1,label='best cwnd')
	plt.plot(sd_indigo_cwnd,color='r',linewidth=1,label='indigo cwnd')
	plt.legend()
	plt.title('best cwnd vs indigo cwnd')
	plt.xlabel('time')
	plt.ylabel('value')
	plt.savefig(path+'/log_figure/'+file+'_best_vs_indigo.png',dpi=1000)
	plt.close()
	'''


'''
for path,dirlist,filelist in os.walk(dir):
	for file in filelist:
		if os.path.splitext(file)[1] == '.log':
			file_path = path+'/'+file
			df = pd.read_csv(file_path)
			print('draw',file_path)
			sd_best_cwnd = data_smooth(df[' best_cwnd'],10)
			sd_indigo_cwnd = data_smooth(df[' indigo_cwnd'],10)
			plt.plot(sd_best_cwnd,color='b',linewidth=1,label='best cwnd')
			plt.plot(sd_indigo_cwnd,color='r',linewidth=1,label='indigo cwnd')
			plt.legend()
			plt.title('best cwnd vs indigo cwnd')
			plt.xlabel('time')
			plt.ylabel('value')
			plt.savefig(path+'/log_figure/'+file+'_best_vs_indigo.png',dpi=1000)
			plt.close()


			df['total'] = df.ix[:,[' tg_traffic',' sender_traffic']].apply(lambda x:x.sum(), axis =1)
			sd_tg_traffic = data_smooth(df[' tg_traffic'],10)
			sd_sender_traffic = data_smooth(df[' sender_traffic'],10)
			sd_total = data_smooth(df['total'],10)
			plt.plot(sd_tg_traffic,color='b',linewidth=1,label='tg traffic')
			plt.plot(sd_sender_traffic,color='r',linewidth=1,label='sender traffic')
			plt.plot(sd_total,color='green',linewidth=1,label='total traffic')
			plt.legend()
			plt.title('tg traffic vs sender traffic')
			plt.xlabel('time')
			plt.ylabel('value')
			plt.savefig(path+'/log_figure/'+file+'_tg_vs_sender.png.png',dpi=1000)
			plt.close()


			df_tg_traffic = df[' tg_traffic']
			sd_queueing_factor = data_smooth(df[' queueing_factor'],10)
			plt.plot(sd_queueing_factor,color='b',linewidth=1,label='tg traffic')
			plt.title('queueing_factor')
			plt.xlabel('time')
			plt.ylabel('value')
			plt.savefig(path+'/log_figure/'+file+'_queueing_factor.png',dpi=1000)
			plt.close()
'''
