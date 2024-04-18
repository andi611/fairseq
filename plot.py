import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def asr_exp():
    N = 2
    type_names = ["Wav2vec 2.0"] + ["HuBERT"]
    d = {"Model Size": ["Small"]*N,
        "WER": [19.37] + \
               [19.37],
        "Model" : copy.deepcopy(type_names)}
    
    d["Model Size"] += ["Base"]*N
    d["WER"] += [6.43] + \
                [6.42]
    d["Model"] += copy.deepcopy(type_names)
    
    d["Model Size"] += ["Large"]*N
    d["WER"] += [3.75] + \
                [3.62]
    d["Model"] += copy.deepcopy(type_names)

    df = pd.DataFrame(data=d)
    return df

if __name__ == "__main__":
    asr_data = asr_exp()
    asr_data["Task"] = "a) ASR"
    b_data = copy.deepcopy(asr_data)
    b_data["Task"] = "b) ASR"
    c_data = copy.deepcopy(asr_data)
    c_data["Task"] = "c) ASR"
    data = pd.concat([asr_data, b_data, c_data])

    plot_name = "model_size_exp"
    plt.figure(figsize=(20, 6), dpi=400)
    # sns.set_context("paper", font_scale=2.8)
    sns.set(style="whitegrid", color_codes=True, font_scale=2.4) # rc={"axes.grid":True}, 
    g = sns.catplot(x="Model Size", y="WER", hue="Model", col="Task", data=data, 
            ci="sd", legend=False, sharey=False, height=8, aspect=0.9, 
            kind="point", capsize=.2, palette="tab10",
            order=["Small", "Base", "Large"],
            hue_order=reversed(["Wav2vec 2.0", "HuBERT"]))
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Model Size", "WER")
    axes = g.axes
    axes[0,0].set_ylim(0,20)
    axes[0,1].set_ylim(-5,85)
    axes[0,2].set_ylim(-5,95)

    plt.legend(loc="lower right", fontsize="x-small", labelspacing=0.1, bbox_to_anchor=(1.01, 0.02))
    # plt.legend(loc="lower right", fontsize="medium", bbox_to_anchor=(1.01, 0.15), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f".plot_{plot_name}.jpg", dpi=400, pad_inches=0, bbox_inches="tight")