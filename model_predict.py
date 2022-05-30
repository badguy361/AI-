import torch
import numpy as np
from model import Net
from torch import argmax
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
import datetime

def vibration_preprocess(file_path): #1. unix time to date 2. sqrt(x**2+y**2+z**2) 
    data=pd.read_csv(f"{file_path}",header=None,\
                    sep=";",names=["UNIX time (ms)","X","Y","Z"])

    counts_df=data["UNIX time (ms)"].value_counts().\
        rename_axis('UNIX time (ms)').reset_index(name='counts')
    multiple_counts_mask=(counts_df["counts"]>1)

    #相同UNIX Time table
    multiple_counts_df=counts_df[multiple_counts_mask]

    result={"UNIX time (ms)":[],"X":[],"Y":[],"Z":[]}

    #取平均
    for unix_time in tqdm(multiple_counts_df["UNIX time (ms)"]):
        mask=(data["UNIX time (ms)"]==unix_time)
        result["UNIX time (ms)"].append(unix_time)
        result["X"].append(np.mean(data[mask]["X"]))
        result["Y"].append(np.mean(data[mask]["Y"]))
        result["Z"].append(np.mean(data[mask]["Z"]))    

    #平均table和總table取差集
    counts_mean_df=pd.DataFrame(result)
    final_data=data.append(counts_mean_df)
    final_data.drop_duplicates(subset=['UNIX time (ms)'],keep=False,inplace=True)
    #差集table和平均table結合
    final_data=final_data.append(counts_mean_df)
    final_data.sort_values(by=['UNIX time (ms)'],inplace=True)
    final_data.reset_index(drop=True,inplace=True)

    final_data.insert(1,"date time",0)
    final_data["root(square(XYZ))"]=np.sqrt(final_data["X"]**2+final_data["Y"]**2+final_data["Z"]**2)
    #UNIX time to Date time
    for i in tqdm(range(len(final_data))):
        final_data.loc[i,"date time"]=dt.fromtimestamp(final_data["UNIX time (ms)"][i]/1000,\
                                        datetime.timezone(datetime.timedelta(hours=8))) #UTC+8
    return final_data

def predict(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(n_feature=3, n_output=2)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    temp_list = []
    temp_list_2 = []
    for i in range(len(df)):
        temp = str(df['date time'][i])
        sec = ((temp.split(':')[2]).split('.')[0])[0:2]
        mini = temp.split(':')[1]
        hour = (temp.split(':')[0]).split(' ')[-1]
        time_str = hour  + mini + sec
        time_str2 = hour + ':' + mini + ':' + sec
        temp_list.append([float(time_str),float(df['X'][i]),float(df['Y'][i]),float(df['Z'][i])])
        temp_list_2.append(time_str2)
    temp_kill = 0
    save_list = []
    save_list_2 = []
    for k in range(len(temp_list)):
        if temp_list[k][0] != temp_kill:
            temp_kill = temp_list[k][0]
            save_list.append(temp_list[k])
            save_list_2.append(temp_list_2[k])
        temp_kill = temp_list[k][0]
    save_list = np.array(save_list)
    test = save_list[:,1:]
    time_record = save_list_2
    x_np = torch.from_numpy(test)
    x_np = x_np.reshape(len(x_np),3)
    x_np = x_np.to(device=device, dtype=torch.float32)
    pre_y=net(x_np)
    pre_y = pre_y.cpu().detach().numpy()
    predict_output = list(pre_y.argmax(1))
    Total_Sleep_Time = f'{len(predict_output)//3600}:{(len(predict_output)-((len(predict_output)//3600)*3600))//60}:{(len(predict_output)%3600)%60}'
    Total_Light_Sleep_Time = f'{(predict_output.count(0))//3600}:{((predict_output.count(0))-(((predict_output.count(0))//3600)*3600))//60}:{((predict_output.count(0))%3600)%60}'
    Total_Deep_Sleep_Time = f'{(predict_output.count(1))//3600}:{((predict_output.count(1))-(((predict_output.count(1))//3600)*3600))//60}:{((predict_output.count(1))%3600)%60}'
    Total_Deep_Sleep_probability = predict_output.count(1)/len(predict_output)*100
    Total_Light_Sleep_probability = predict_output.count(0)/len(predict_output)*100
    output = {'Predict':predict_output,'Time Record(s)':time_record,'Total Sleep Time':Total_Sleep_Time,'Total Light Sleep Time':Total_Light_Sleep_Time,'Total Deep Sleep Time':Total_Deep_Sleep_Time,'Deep Sleep Probability(%)':Total_Deep_Sleep_probability,'Light Sleep Probability(%)':Total_Light_Sleep_probability}
    return output

file_path = 'C:/Users/niko/Desktop/EarthScience/Senior_2/AI_proj/vibration_train/0525data.csv'
final_data=vibration_preprocess(file_path)

predict_output = predict(final_data)
df = pd.DataFrame(predict_output)
df.to_csv('output.csv')
