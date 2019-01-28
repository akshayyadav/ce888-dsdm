import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def boostrap(sample, sample_size, iterations, ci=95):
    data = np.random.choice(a=sample, size=(iterations, sample_size))
    data_mean = np.mean(data)
    sorted_iteration_means = np.sort(np.mean(data, axis=1))
    alpha = ci/100
    lower_percentile = ((1.0-alpha)/2.0) * 100
    upper_percentile = (alpha+((1.0-alpha)/2.0)) * 100
    percentile_values = np.percentile(sorted_iteration_means, [lower_percentile, upper_percentile])
    # import pdb; pdb.set_trace()
    return data_mean, percentile_values[0], percentile_values[1]


if __name__ == "__main__":
    df = pd.read_csv('./salaries.csv')

    data = df.values.T[1]
    boots = []
    for i in range(100, 100000, 1000):
        boot = boostrap(data, data.shape[0], i)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

        df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
        sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

        sns_plot.axes[0, 0].set_ylim(0,)
        sns_plot.axes[0, 0].set_xlim(0, 100000)

        sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
        sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


        #print ("Mean: %f")%(np.mean(data))
        #print ("Var: %f")%(np.var(data))
