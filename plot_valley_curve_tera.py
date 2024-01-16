import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
legend_fontsize = 17
linewidth = 2.0


model_name1 = "TERA"
model_name2 = "TERA 1/2"
filename = "valley-curve-TERA"



# TERA
roof = 0
x = ["30%", "50%", "70%", "100%", "200%", "476%", "1590%"]

Name_y1 = "ASR (WER)"
y1lim_diff = 0.5
y1 = np.array([21.0001, 18.0843, 17.1694, 16.2565, 13.9703, 17.2874, 21.9035])
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
subfig.set(xlabel="Change in TERA model size")
subfig.set(ylim=(y1.min()-y1lim_diff, y1.max()+y1lim_diff+roof))


Name_y0 = "ASV (EER)"
y0lim_diff = 0.15
y0 = np.array([10.509, 8.515, 8.218, 7.614, 6.766, 7.556, 10.705])
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
y2 = np.array([5.463, 5.299, 5.224, 5.178, 5.001, 5.185, 5.627])
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
y3 = np.array([5.135, 4.987, 4.910, 4.855, 4.693, 4.920, 5.446])
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


# TERA 1/2
roof = 0
x = ["30%", "50%", "70%", "100%", "200%", "476%", "1590%"]

Name_y1 = "ASR (WER)"
y1lim_diff = 0.5
y1 = np.array([22.4855, 19.7714, 19.1989, 17.9055, 18.1927, 19.7428, 24.456])
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
subfig.set(xlabel="Change in TERA model size, 1/2 budget")
subfig.set(ylim=(y1.min()-y1lim_diff, y1.max()+y1lim_diff+roof))


Name_y0 = "ASV (EER)"
y0lim_diff = 0.15
y0 = np.array([10.424, 9.422, 8.266, 8.083, 8.112, 9.470, 12.121])
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
y2 = np.array([5.597, 5.467, 5.448, 5.284, 5.265, 5.432, 5.888])
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
y3 = np.array([5.303, 5.191, 5.158, 4.999, 5.025, 5.277, 5.863])
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