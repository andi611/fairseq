import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
legend_fontsize = 17
linewidth = 2.0


model_name1 = "HuBERT"
model_name2 = "HuBERT 1/2"
filename = "valley-curve-hubert"



# hubert
roof = 0
# x = ["5.9M", "9.9M", "14.1M", "20.0M", "40.0M", "94.7M", "316.6M"]
x = ["30%", "50%", "70%", "100%", "200%", "473%", "1582%"]

Name_y1 = "ASR (WER)"
y1lim_diff = 0.5
y1 = np.array([20.0415, 17.2303, 16.0796, 14.5637, 14.7596, 15.1818, 20.0015])
data1 = pd.DataFrame({"x": x, Name_y1: y1, "Model": [model_name1] * len(x)})

subfig = sns.lineplot(
    data=data1,
    x="x",
    y=Name_y1,
    ax=axes[0],
    linewidth=linewidth,
    style="Model",
    color="tomato",
    markers=True,
    markersize=9,
)
subfig.set_title(None)
subfig.legend([Name_y1], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.68))
subfig.set(xlabel="Change in HuBERT model size")
subfig.set(ylim=(y1.min()-y1lim_diff, y1.max()+y1lim_diff+roof))


Name_y0 = "ASV (EER)"
y0lim_diff = 0.15
y0 = np.array([8.7487, 7.6299, 7.5504, 7.147, 6.6013, 6.8346, 10.4])
data0 = pd.DataFrame({"x": x, Name_y0: y0, "Model": [model_name2] * len(x)})

subfig0 = subfig.twinx()
sns.lineplot(
    data=data0,
    x="x",
    y=Name_y0,
    ax=subfig0,
    linewidth=linewidth,
    style="Model",
    color="red",
    markers=True,
    markersize=9,
)
subfig0.spines.right.set_position(("axes", 1.13)) # move the 2nd y-axis to the right
# subfig0.set(yticklabels=[], ylabel=None) # hide the 2nd y-axis
subfig0.set_title(None)
subfig0.legend([Name_y0], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.78))
subfig0.grid(False)
subfig0.set(ylim=(y0.min()-y0lim_diff, y0.max()+y0lim_diff))


Name_y2 = "Train Loss"
y2lim_diff = 0.08
y2 = np.array([4.971, 4.683, 4.563, 4.467, 4.402, 4.462, 4.881])
data2 = pd.DataFrame({"x": x, Name_y2: y2, "Model": [model_name1] * len(x)})

subfig2 = subfig.twinx()
sns.lineplot(
    data=data2,
    x="x",
    y=Name_y2,
    ax=subfig2,
    linewidth=linewidth,
    style="Model",
    color="green",
    markers=True,
    markersize=9,
)
# subfig2.spines.right.set_position(("axes", 1.15)) # move the 2nd y-axis to the right
subfig2.set(yticklabels=[], ylabel=None) # hide the 2nd y-axis
subfig2.set_title(None)
subfig2.legend([Name_y2], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.98))
subfig2.grid(False)
# subfig2.set(ylim=(y2.min()-y2lim_diff, y2.max()+y2lim_diff))


Name_y3 = "Valid Loss"
Yaxis_Name_for_2n3 = "Loss"
y3lim_diff = 0.08
y3 = np.array([4.64, 4.3558, 4.2742, 4.1788, 4.1478, 4.1978, 4.6688])
data3 = pd.DataFrame({"x": x, Yaxis_Name_for_2n3: y3, "Model": [model_name1] * len(x)})

subfig3 = subfig.twinx()
sns.lineplot(
    data=data3,
    x="x",
    y=Yaxis_Name_for_2n3,
    ax=subfig3,
    linewidth=linewidth,
    style="Model",
    color="royalblue",
    markers=True,
    markersize=9,
)
subfig3.legend([Name_y3], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.88))
subfig3.grid(False)

# set 2 and 3 to the same ylim
subfig2.set(ylim=(min(y2.min()-y2lim_diff, y3.min()-y3lim_diff), max(y2.max()+y2lim_diff, y3.max()+y3lim_diff)))
subfig3.set(ylim=(min(y2.min()-y2lim_diff, y3.min()-y3lim_diff), max(y2.max()+y2lim_diff, y3.max()+y3lim_diff)))


# HuBERT 1/2
roof = 0
x = ["30%", "50%", "70%", "100%", "200%", "473%", "1582%"]

Name_y1 = "ASR (WER)"
y1lim_diff = 0.5
y1 = np.array([20.9012, 18.3810, 17.4738, 16.4581, 16.6996, 18.1128, 23.7448])
data1 = pd.DataFrame({"x": x, Name_y1: y1, "Model": [model_name2] * len(x)})

subfig = sns.lineplot(
    data=data1,
    x="x",
    y=Name_y1,
    ax=axes[1],
    linewidth=linewidth,
    style="Model",
    color="tomato",
    markers=True,
    markersize=9,
)
subfig.set_title(None)
subfig.legend([Name_y1], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.68))
subfig.set(xlabel="Change in HuBERT model size, 1/2 budget")
subfig.set(ylim=(y1.min()-y1lim_diff, y1.max()+y1lim_diff+roof))


Name_y0 = "ASV (EER)"
y0lim_diff = 0.15
y0 = np.array([9.030, 8.091, 7.291, 7.603, 7.359, 8.685, 12.688])
data0 = pd.DataFrame({"x": x, Name_y0: y0, "Model": [model_name2] * len(x)})

subfig0 = subfig.twinx()
sns.lineplot(
    data=data0,
    x="x",
    y=Name_y0,
    ax=subfig0,
    linewidth=linewidth,
    style="Model",
    color="red",
    markers=True,
    markersize=9,
)
subfig0.spines.right.set_position(("axes", 1.13)) # move the 2nd y-axis to the right
# subfig0.set(yticklabels=[], ylabel=None) # hide the 2nd y-axis
subfig0.set_title(None)
subfig0.legend([Name_y0], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.78))
subfig0.grid(False)
subfig0.set(ylim=(y0.min()-y0lim_diff, y0.max()+y0lim_diff))


Name_y2 = "Train Loss"
y2lim_diff = 0.08
y2 = np.array([5.035, 4.775, 4.683, 4.631, 4.593, 4.694, 5.376])
data2 = pd.DataFrame({"x": x, Name_y2: y2, "Model": [model_name2] * len(x)})

subfig2 = subfig.twinx()
sns.lineplot(
    data=data2,
    x="x",
    y=Name_y2,
    ax=subfig2,
    linewidth=linewidth,
    style="Model",
    color="green",
    markers=True,
    markersize=9,
)
# subfig2.spines.right.set_position(("axes", 1.15)) # move the 2nd y-axis to the right
subfig2.set(yticklabels=[], ylabel=None) # hide the 2nd y-axis
subfig2.set_title(None)
subfig2.legend([Name_y2], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.98))
subfig2.grid(False)
# subfig2.set(ylim=(y2.min()-y2lim_diff, y2.max()+y2lim_diff))


Name_y3 = "Valid Loss"
Yaxis_Name_for_2n3 = "Loss"
y3lim_diff = 0.08
y3 = np.array([4.713, 4.473, 4.395, 4.339, 4.309, 4.490, 5.062])
data3 = pd.DataFrame({"x": x, Yaxis_Name_for_2n3: y3, "Model": [model_name2] * len(x)})

subfig3 = subfig.twinx()
sns.lineplot(
    data=data3,
    x="x",
    y=Yaxis_Name_for_2n3,
    ax=subfig3,
    linewidth=linewidth,
    style="Model",
    color="royalblue",
    markers=True,
    markersize=9,
)
subfig3.legend([Name_y3], loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.88))
subfig3.grid(False)

# set 2 and 3 to the same ylim
subfig2.set(ylim=(min(y2.min()-y2lim_diff, y3.min()-y3lim_diff), max(y2.max()+y2lim_diff, y3.max()+y3lim_diff)))
subfig3.set(ylim=(min(y2.min()-y2lim_diff, y3.min()-y3lim_diff), max(y2.max()+y2lim_diff, y3.max()+y3lim_diff)))



fig.tight_layout()
plt.savefig(f"{filename}.jpeg", format="jpeg", bbox_inches="tight")
plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")