import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def model_ouput_plot(input_path,output_path,second_interval=60*30):
    
    data=pd.read_csv(f"{input_path}",index_col=[0])
    color={"light sleep":"#fc9d65","deep sleep":"#de5b0c"} #orange
    predict_light=(data["Predict"]==0)
    predict_deep=(data["Predict"]==1)

    sleep_dict={"time":[],"predict":[],"color":[]}
    for i in range(len(data)//second_interval):
        start_index=second_interval*i
        end_index=second_interval*(i+1)
        predict_light=(data["Predict"][start_index:end_index]==0)
        predict_deep=(data["Predict"][start_index:end_index]==1)

        sleep_dict["time"].append(data["Time Record(s)"][start_index][:-3])
        sleep_dict["predict"].append(1)

        # print(f"light:{len(data[start_index:end_index][predict_light]['Predict'])}")
        # print(f"deep:{len(data[start_index:end_index][predict_deep]['Predict'])}")
        # print("====================================")
        if (len(data[start_index:end_index][predict_deep]["Predict"])>=len(data[start_index:end_index][predict_light]["Predict"])):

            sleep_dict["color"].append(color["deep sleep"])
        else:
            sleep_dict["color"].append(color["light sleep"])

    sleep_tb=pd.DataFrame(sleep_dict)

    bg_color="#202035"
    label_color="white"
    fig,ax=plt.subplots(figsize=(17,5))
    lightsleep_patch = mpatches.Patch(color=color["light sleep"], label='light sleep')
    deepsleep_patch = mpatches.Patch(color=color["deep sleep"], label='deep sleep')
    ax.bar(sleep_tb["time"],sleep_tb["predict"],color=sleep_dict["color"])
    ax.set_xlabel("Time",fontsize=15,color=label_color)
    ax.set_title("Sleep Quality",fontsize=18,color=label_color)
    ax.legend(loc="upper right",handles=[lightsleep_patch,deepsleep_patch],\
            bbox_to_anchor=(0.96,1.15),fontsize=12)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axes.yaxis.set_visible(False)
    ax.tick_params(axis='x', colors=label_color)
    plt.xticks(fontsize=13)
    ax.spines[:].set_visible(False)
    fig.savefig(f"{output_path}/Sleep Quality.png",dpi=300)


model_ouput_plot(input_path="../model output.csv",output_path="..")
