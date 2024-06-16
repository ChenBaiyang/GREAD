import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

def plot_para():
    score_t =[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              [0.814,0.815,0.846,0.872,0.873,0.876,0.882,0.884,0.882],
              [0.420,0.425,0.452,0.493,0.514,0.529,0.533,0.536,0.536],]
    score_p =[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              [0.840,0.856,0.865,0.869,0.872,0.872,0.876,0.877,0.865],
              [0.486,0.497,0.507,0.519,0.524,0.530,0.536,0.535,0.525],]

    t = np.array(score_t)
    p = np.array(score_p)


    # Extracting data
    x_ticks = p[0]
    auc_scores = p[1]
    ap_scores = p[2]

    # Plotting
    plt.figure(figsize=(3.6, 3.6))
    plt.plot(x_ticks, auc_scores, marker='o', color='blue', label='AUC Score')
    plt.plot(x_ticks, ap_scores, marker='s', color='red', label='AP Score')
    # plt.title('Experimental Results')
    plt.xlabel('Percentile $p$', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.xticks(x_ticks)
    plt.xlim(0.08,0.92)
    plt.ylim(0.3,1)
    plt.legend(loc='lower right')
    # plt.grid(True)
    plt.margins(0, 0)
    plt.savefig('figs/para_p.pdf', dpi=600, pad_inches=0.02, bbox_inches='tight')
    plt.show()

    # Extracting data
    x_ticks = t[0]
    auc_scores = t[1]
    ap_scores = t[2]

    # Plotting
    plt.figure(figsize=(3.6, 3.6))
    plt.plot(x_ticks, auc_scores, marker='o', color='blue', label='AUC Score')
    plt.plot(x_ticks, ap_scores, marker='s', color='red', label='AP Score')
    # plt.title('Experimental Results')
    plt.xlabel('Threshold $\\tau$', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.xticks(x_ticks)
    plt.xlim(0.08,0.92)
    plt.ylim(0.3,1)
    plt.legend(loc='lower right')
    # plt.grid(True)
    plt.margins(0, 0)
    plt.savefig('figs/para_tau.pdf', dpi=600, pad_inches=0.02, bbox_inches='tight')
    plt.show()


def plot_candy_bar():
    x_ticks = ['IForest', 'SOD', 'LODA', 'DSVDD', 'ECOD', 'LUNAR', 'DIF', 'WFRDA', 'GREAD']
    labels = ['Categorical', 'Mixed', 'Numerical']

    AUC =[  [0.847,0.664,0.434,0.739,0.927,0.754,0.727,0.936,0.968],
            [0.851,0.772,0.577,0.716,0.843,0.839,0.756,0.853,0.882],
            [0.871,0.758,0.706,0.704,0.840,0.818,0.831,0.842,0.908]]

    AP =[   [0.483,0.355,0.102,0.268,0.620,0.380,0.268,0.826,0.824],
            [0.401,0.189,0.107,0.211,0.398,0.333,0.249,0.408,0.443],
            [0.495,0.306,0.340,0.292,0.457,0.357,0.433,0.415,0.606]]

    data = np.array(AP)

    # Define colors
    colors = ["grey", "darkorange", "brown", "green", "blue", "orange", "gold", "purple", "red"]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(9, 5))

    for i, label in enumerate(labels):
        ax = axs[i]
        ax.set_title(label, y=-0.13, fontsize=15)
        # ax.set_ylabel('AUC Scores', fontsize=14)

        for j in range(len(x_ticks)):
            ax.scatter(data[i][j], j, color=colors[j], s=150)  # Points
            ax.scatter(data[i][j], j, color=colors[j], s=110, marker="o", edgecolor='white',
                       linewidth=1)  # Outer circle
            ax.hlines(y=j, xmin=0, xmax=data[i][j], color=colors[j], linewidth=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 8.4)
        ax.set_yticks(range(len(x_ticks)))
        ax.set_yticklabels(x_ticks)
        #
        # # Add data labels
        # for j in range(len(x_ticks)):
        #     ax.text(data[i][j] - 0.02, j, f"{data[i][j]:.3f}", ha="right", va="center")

    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.savefig('figs/DataType_ap.pdf', dpi=600, pad_inches=0.02, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_para()
    plot_candy_bar()