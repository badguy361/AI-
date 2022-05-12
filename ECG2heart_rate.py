from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import scipy.signal as ss
import warnings
import h5py
import json
warnings.filterwarnings("ignore")

file_list=os.listdir("D:/AI_sleep_project/data/physionet.org/files/training/")

sample_rate=200
amp_threshold=0.5
index_threshold=75
window_step=200*60 #per minute
heart_rate_boundary=30
integrity_rate_dict={"file ID":[],"integrity rate":[]}
for file_count in range(len(file_list)):
# for file_count in range(3):
    file_ID=file_list[file_count]
    print(f"{file_ID}")
    file_path=f"D:/AI_sleep_project/data/physionet.org/files/training/{file_ID}/{file_ID}.mat"
    label_data_path=f"D:/AI_sleep_project/data/physionet.org/files/training/{file_ID}/{file_ID}-arousal.mat"
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
    #arousal data load
    ######################################TODO##################################
    label_data=h5py.File(label_data_path, 'r')
    arousals=label_data["data"]["arousals"][:]
    arousals_df=pd.DataFrame(arousals,columns=["label"])
    ############################################################################
    heart_rate_list_per_minute=[]
    arousals_value_count_list=[]
    for count in range(len(ecg_baseline_filtered)//window_step):
        part_ecg=ecg_baseline_filtered[window_step*count:window_step*(count+1)]
        scaler = MaxAbsScaler().fit(part_ecg.reshape(-1,1))
        part_ecg_scaled=scaler.transform(part_ecg.reshape(-1,1))
        part_ecg_scaled_abs=np.abs(part_ecg_scaled)

        heart_rate_candidated_index=np.where(part_ecg_scaled_abs>amp_threshold)[0]
        heart_rate_list=[]
        heart_time_index=[]
        heart_beat=1

        for i in range(0,len(heart_rate_candidated_index)-1):
            if heart_rate_candidated_index[i+1]-heart_rate_candidated_index[i]>index_threshold:
                heart_beat+=1
                heart_time_index.append(heart_rate_candidated_index[i])
        # heart_rate=heart_beat*60/(window_step/sample_rate) #time/per minute
        for j in range(0,len(heart_time_index)-1):
            heart_beat_interval=heart_time_index[j+1]-heart_time_index[j]
            heart_rate=(1/(heart_beat_interval/sample_rate))*60
            heart_rate_list.append(heart_rate)
        if np.median(heart_rate_list)>heart_rate_boundary:
            heart_rate_median=np.median(heart_rate_list)
            heart_rate_list_per_minute.append(heart_rate_median)
        else:
            # heart_rate_list_per_minute.append(heart_rate_median)
            heart_rate_list_per_minute.append(np.nan)
        #arousal label value count
        arousals_value_count=arousals_df["label"][window_step*count:window_step*(count+1)].value_counts().to_dict()
        arousals_value_count_list.append(arousals_value_count)


    heart_rate_df=pd.DataFrame(heart_rate_list_per_minute,columns=["heart rate"])
    arousals_value_count_df=pd.DataFrame(arousals_value_count_list)
    joint_df=pd.merge(heart_rate_df,arousals_value_count_df,left_index=True, right_index=True)
    joint_df.dropna(subset="heart rate",inplace=True)
    joint_df.reset_index(drop=True,inplace=True)
    integrity_rate_dict["file ID"].append(file_ID)
    integrity_rate_dict["integrity rate"].append(len(joint_df)/len(heart_rate_df))
    print(100*np.round(len(joint_df)/len(heart_rate_df),3))
    joint_df.to_csv(f"../heart_rate_tb(drop unreasonable)/{file_ID}_heart_rate.csv")
with open('../heart_rate_tb(drop unreasonable)/integrity_rate.json', 'w') as fp:
    json.dump(integrity_rate_dict, fp)

    # fig,ax=plt.subplots(3,2,figsize=(20,14))
    # ax[0,0].plot(ecg[window_step*count:window_step*(count+1)])
    # ax[1,0].plot(ecg_scaled[window_step*count:window_step*(count+1)])
    # ax[2,0].plot(ecg_baseline[window_step*count:window_step*(count+1)])
    # ax[0,1].plot(ecg_baseline_filtered[window_step*count:window_step*(count+1)])
    # ax[1,1].plot(part_ecg_scaled)
    # ax[2,1].plot(part_ecg_scaled_abs)
    # ax[0,0].set_title("raw data")
    # ax[1,0].set_title("min max normalization",fontsize=15)
    # ax[2,0].set_title("baseline correction",fontsize=15)
    # ax[0,1].set_title("filtered data, frequency: 0.007 ~ 0.1",fontsize=15)
    # ax[1,1].set_title(f"max absolute normalize",fontsize=15)
    # ax[2,1].set_title(f"absolute, heart beats:{heart_beat}, heart rate: {heart_rate}",fontsize=15)
    # ax[2,1].axhline(amp_threshold,c="red")
    # fig.savefig(f"../part_ECG/preprocess ecg 1 min/{window_step*count} to {window_step*(count+1)}.png",dpi=300)  
    # if heart_rate> 100:
    # fig.savefig(f"../part_ECG/heart rate bigger than 120/{window_step*count} to {window_step*(count+1)}.png",dpi=300)
    # # plt.show(block=False)
    # elif heart_rate<30:
    # fig.savefig(f"../part_ECG/heart rate smaller than 30/{window_step*count} to {window_step*(count+1)}.png",dpi=300)
    # else:
    # fig.savefig(f"../part_ECG/preprocess ecg/{window_step*count} to {window_step*(count+1)}.png",dpi=300)    

