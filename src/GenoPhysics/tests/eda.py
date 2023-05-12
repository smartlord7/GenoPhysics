import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from util.stats import *


def main():
    PATH_DATA1 = '../../../out/bests_ge.txt'
    PATH_DATA2 = '../../../out/bests_gp.txt'
    matplotlib.use('TkAgg')
    NS = 0.05

    data = get_data(PATH_DATA1)
    data2 = get_data(PATH_DATA2)
    shp = data.shape
    n_runs = shp[0]
    n_gen = shp[1]
    data = data[:, n_gen - 1] # get the best ones which correspond to the last generation since elite is use
    describe_data(data)
    histogram(data, 'Histogram - Best Over %d Runs' % n_runs, 'Fitness', 'Frequency', bins=10)
    data = stats.zscore(data)
    histogram(data, 'Histogram - Best Over %d Runs' % n_runs, 'Fitness', 'Frequency', bins=10)
    norm_ks_p_value = test_normal_ks(data).pvalue
    print('Kolmogorov-Smirnov p-value: %.9f' % norm_ks_p_value)

    if NS < norm_ks_p_value:
        print('Data is normal at NS %.2f via KS', NS)

        _, cst_var_levene_p_value = levene(data)

        print('Levene p-value: %.9f' % cst_var_levene_p_value)

        if NS < cst_var_levene_p_value:
            print('Data has constant variance at NS %.2f via L', NS)
        else:
            print('Data has not constant variance at NS %.2f via L', NS)

    else:
        print('Data is not normal at NS %.2f via KS', NS)

        norm_sh_p_value = test_normal_sw(data).pvalue

        print('Shapiro-Wilk p-value: %.9f' % norm_sh_p_value)

        if NS < norm_ks_p_value:
            print('Data is normal at NS %.2f via SW', NS)
        else:
            print('Data is not normal at NS %.2f via SW', NS)


if __name__ == '__main__':
    main()
