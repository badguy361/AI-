from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import scipy.signal as ss

file_list=os.listdir("D:/AI_sleep_project/data/physionet.org/files/training/")

sample_rate=200
amp_threshold=0.5
index_threshold=75
window_step=1000

# for i in range(6,16):
file_ID=file_list[6]
file_path=f"D:/AI_sleep_project/data/physionet.org/files/training/{file_ID}/{file_ID}.mat"
records=loadmat(file_path)
ecg=records["val"][12,:]
scaler = MinMaxScaler().fit(ecg.reshape(-1,1))
ecg_scaled=scaler.transform(ecg.reshape(-1,1))
t1=int(0.2*sample_rate)
ecg_noise=ss.medfilt(ecg_scaled.flatten(),t1+1)
ecg_baseline=ecg_scaled.flatten()-ecg_noise
b,a=ss.butter(8,0.1,"lowpass")
ecg_baseline_low=ss.filtfilt(b,a,ecg_baseline)
b,a=ss.butter(8,0.007,"highpass")
ecg_baseline_filtered=ss.filtfilt(b,a,ecg_baseline_low)


heart_rate_list=[]
for count in range(len(ecg_baseline_filtered)//window_step):

    part_ecg=ecg_baseline_filtered[window_step*count:window_step*(count+1)]
    scaler = MaxAbsScaler().fit(part_ecg.reshape(-1,1))
    part_ecg_scaled=scaler.transform(part_ecg.reshape(-1,1))
    part_ecg_scaled_abs=np.abs(part_ecg_scaled)

    heart_rate_candidated_index=np.where(part_ecg_scaled_abs>amp_threshold)[0]

    heart_beat=1
    for i in range(0,len(heart_rate_candidated_index)-1):
        if heart_rate_candidated_index[i+1]-heart_rate_candidated_index[i]>index_threshold:
            heart_beat+=1
    heart_rate=heart_beat*60/(window_step/sample_rate) #time/per minute

    print(f"{window_step*count,window_step*(count+1)} | heart beat:{heart_beat} | heart_rate:{heart_rate}")
    heart_rate_list.append([heart_rate])

    fig,ax=plt.subplots(3,2,figsize=(20,14))
    ax[0,0].plot(ecg[window_step*count:window_step*(count+1)])
    ax[1,0].plot(ecg_scaled[window_step*count:window_step*(count+1)])
    ax[2,0].plot(ecg_baseline[window_step*count:window_step*(count+1)])
    ax[0,1].plot(ecg_baseline_filtered[window_step*count:window_step*(count+1)])
    ax[1,1].plot(part_ecg_scaled)
    ax[2,1].plot(part_ecg_scaled_abs)
    ax[0,0].set_title("raw data")
    ax[1,0].set_title("min max normalization",fontsize=15)
    ax[2,0].set_title("baseline correction",fontsize=15)
    ax[0,1].set_title("filtered data, frequency: 0.007 ~ 0.1",fontsize=15)
    ax[1,1].set_title(f"max absolute normalize",fontsize=15)
    ax[2,1].set_title(f"absolute, heart beats:{heart_beat}, heart rate: {heart_rate}",fontsize=15)
    ax[2,1].axhline(amp_threshold,c="red")
    if heart_rate> 100:
        fig.savefig(f"../part_ECG/heart rate bigger than 120/{window_step*count} to {window_step*(count+1)}.png",dpi=300)
        # plt.show(block=False)
    elif heart_rate<30:
        fig.savefig(f"../part_ECG/heart rate smaller than 30/{window_step*count} to {window_step*(count+1)}.png",dpi=300)
    else:
        fig.savefig(f"../part_ECG/preprocess ecg/{window_step*count} to {window_step*(count+1)}.png",dpi=300)    


