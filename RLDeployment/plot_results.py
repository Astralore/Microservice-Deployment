# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_all():
    if not os.path.exists("experiment_summary.csv"):
        print("无数据")
        return

    df = pd.read_csv("experiment_summary.csv")
    df_avg = df.groupby("Mode")[['Total_Energy_kJ', 'Avg_Reward', 'Makespan_sec']].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 能耗
    sns.barplot(data=df_avg, x="Mode", y="Total_Energy_kJ", ax=axes[0], palette="Reds")
    axes[0].set_title("Total Energy (kJ) ↓")
    for c in axes[0].containers: axes[0].bar_label(c, fmt='%.1f')

    # 奖励
    sns.barplot(data=df_avg, x="Mode", y="Avg_Reward", ax=axes[1], palette="Blues")
    axes[1].set_title("Avg Reward ↑")
    for c in axes[1].containers: axes[1].bar_label(c, fmt='%.1f')

    # 时间
    sns.barplot(data=df_avg, x="Mode", y="Makespan_sec", ax=axes[2], palette="Purples")
    axes[2].set_title("Makespan (s) ↓")
    for c in axes[2].containers: axes[2].bar_label(c, fmt='%.1f')

    plt.tight_layout()
    plt.savefig("result_summary.png")
    print("图表已保存: result_summary.png")

if __name__ == "__main__":
    plot_all()