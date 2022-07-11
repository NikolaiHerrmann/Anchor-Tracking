
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from brokenaxes import brokenaxes


FOR_REPORT = True
IS_APPROACH_1 = True


def get_max(std, mean):
    arr = []
    for i in range(len(std)):
        if std[i] + mean[i] > 1:
            arr.append(1 - mean[i])
        else:
            arr.append(std[i])
    return arr


# TODO read from files instead of hard-coding data like this
if IS_APPROACH_1:
    bayes_means = [0.9638957056449459, 0.8394952415180171, 0.8709492541707397]
    bayes_stds = [0.03347520042152208,
                  0.08906703297755617, 0.05378238684167435]

    knn_means = [0.9233383340967395, 0.7295975825079679, 0.8253797775822207]
    knn_stds = [0.0758034037463589, 0.08130397449116229, 0.07597677468172007]

    lr_means = [0.9617934191313409, 0.8406434987836782, 0.8500433064032523]
    lr_stds = [0.04412823395589815, 0.10397418891195506, 0.04231060746481784]

    dt_means = [0.9741438755684332, 0.8364755357622359, 0.8826594273468382]
    dt_stds = [0.028203942392481404, 0.08820772717688384, 0.05846056295431464]
else:
    bayes_means = [0.9909741674447557, 0.8686961965182707, 0.9467267618084829]
    bayes_stds = [0.017423635234707028,
                  0.06251998857721153, 0.04939381553222016]

    knn_means = [0.9352056795554238, 0.7304815425933159, 0.9118693509347856]
    knn_stds = [0.05869820977452408, 0.10877580949306631, 0.08390391235210742]

    lr_means = [0.9920634920634921, 0.8777935402082371, 0.9599126327989146]
    lr_stds = [0.01774657124999835, 0.06095699560190976, 0.04710962201740975]

    dt_means = [0.9721922896606441, 0.8906921126026329, 0.9370542759531547]
    dt_stds = [0.04299116166529955, 0.03184441244563159, 0.055101972208687774]

labels = ["Moving", "Stacking", "Disappearance"]
x = np.arange(len(labels))
width = 0.17
matplotlib.rcParams.update({'font.size': 11}) # not sure whether this even has any effect
approach_num = "1" if IS_APPROACH_1 else "2"

fig, ax = plt.subplots()
bax = brokenaxes(ylims=((0, 0.01), (0.61, 1.01)), hspace=0.02)

bax.errorbar(x - width * 1.5, bayes_means, yerr=(bayes_stds,
             get_max(bayes_stds, bayes_means)), ls='none', capsize=5, alpha=0.4)
bax.errorbar(x - width / 2, knn_means, yerr=(knn_stds, get_max(knn_stds,
             knn_means)), ls='none', alpha=0.4, capsize=5, color="purple")
bax.errorbar(x + width / 2, lr_means, yerr=(lr_stds, get_max(lr_stds,
             lr_means)), ls='none', capsize=5, color="g", alpha=0.4)
bax.errorbar(x + width * 1.5, dt_means, yerr=(dt_stds, get_max(dt_stds,
             dt_means)), ls='none', capsize=5, color="maroon", alpha=0.4)

bax.bar(x - width * 1.5, bayes_means, width,
        label='Bayes', color='lightskyblue')
bax.bar(x - width / 2, knn_means, width, label='KNN', color='plum')
bax.bar(x + width / 2, lr_means, width, label='Log Reg', color='lightgreen')
bax.bar(x + width * 1.5, dt_means, width, label='D-Tree', color='indianred')

bax.set_ylabel('Average F1 Score', labelpad=40, fontsize=13)
bax.set_xlabel('Testing Scenario', labelpad=30, fontsize=13)
if FOR_REPORT:
    bax.set_title("Approach " + approach_num +
                  " Classifier Performance Comparison", pad=20, fontsize=15)
bax.legend(ncol=2, loc='upper center')

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_axis_off()
bax.axs[1].set_xticks(x)
bax.axs[1].set_xticklabels(labels)

fig.set_size_inches(8, 6)

name = "a" + approach_num

if FOR_REPORT:
    plt.savefig(name + ".pdf")
else:
    plt.savefig(name + ".png", dpi=1000)

plt.show()
