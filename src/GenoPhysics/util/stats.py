"""
------------GenoPhysics: Kepler's Third Law of Planetary Motion------------
 University of Coimbra
 Masters in Intelligent Systems
 Evolutionary Computation
 1st year, 2nd semester
 Authors:
 Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
 Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
 Credits to:
 Ernesto Costa
 João Macedo
 Coimbra, 12th May 2023
 ---------------------------------------------------------------------------
"""

"""
stat_2016_alunos.py
Descriptive and inferential statistics in Python.
Use numpy, scipy and matplotlib.
"""
__author__ = 'Ernesto Costa'
__date__ = 'March 2015, March 2016'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# obtain data
def get_data(filename):
    """
       Loads data from a file using NumPy's `loadtxt` function.

       Parameters:
       -----------
       filename : str
           Name of the file to load the data from.

       Returns:
       --------
       numpy.ndarray
           The loaded data as a NumPy array.
    """
    data = np.loadtxt(filename)
    return data

def get_data_many(filename):
    """
       Loads data from a file using NumPy's `loadtxt` function and returns the transpose.

       Parameters:
       -----------
       filename : str
           Name of the file to load the data from.

       Returns:
       --------
       numpy.ndarray
           The transpose of the loaded data as a NumPy array.
    """
    data_raw = np.loadtxt(filename)
    data = data_raw.transpose()
    return data

def get_data_many_header(filename):
    """
       Loads data from a file using NumPy's `loadtxt` function and skips the first row, then returns the transpose.

       Parameters:
       -----------
       filename : str
           Name of the file to load the data from.

       Returns:
       --------
       numpy.ndarray
           The transpose of the loaded data as a NumPy array, with the first row skipped.
    """
    data = np.loadtxt(filename,skiprows=1)
    data = data.transpose()
    return data

# describing data

def describe_data(data):
    """
       Calculates various descriptive statistics for the given dataset.

       Parameters:
       -----------
       data : numpy.ndarray
           The dataset to describe, as a NumPy array of values.

       Returns:
       --------
       tuple of floats
           A tuple containing the following descriptive statistics:
           (minimum value, maximum value, mean, median, mode, variance, standard deviation,
           skewness, kurtosis, 25th percentile, 50th percentile, 75th percentile)
    """
    min_ = np.amin(data)
    max_ = np.amax(data)
    mean_ = np.mean(data)
    median_ = np.median(data)
    mode_ = st.mode(data)
    std_ = np.std(data)
    var_ = np.var(data)
    skew_ = st.skew(data)
    kurtosis_ = st.kurtosis(data)
    q_25, q_50, q_75 = np.percentile(data, [25,50,75])
    basic = 'Min: %s\nMax: %s\nMean: %s\nMedian: %s\nMode: %s\nVar: %s\nStd: %s'
    other = '\nSkew: %s\nKurtosis: %s\nQ25: %s\nQ50: %s\nQ75: %s'
    all_ = basic + other
    print(all_ % (min_,max_,mean_,median_,mode_,var_,std_,skew_,kurtosis_,q_25,q_50,q_75))
    return (min_,max_,mean_,median_,mode_,var_,std_,skew_,kurtosis_,q_25,q_50,q_75)

# visualizing data
def histogram(data,title,xlabel,ylabel,bins=25):
    """
      Function to plot a histogram.

      Parameters:
      -----------
      data: array-like
          Input data to plot the histogram from.
      title: str
          Title of the histogram plot.
      xlabel: str
          Label for the x-axis.
      ylabel: str
          Label for the y-axis.
      bins: int, optional (default=25)
          Number of bins to use in the histogram.

      Returns:
      --------
      None
    """
    plt.hist(data,bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def histogram_norm(data,title,xlabel,ylabel,bins=20):
    """
       Function to plot a normalized histogram and a fitted normal distribution curve.

       Parameters:
       -----------
       data: array-like
           Input data to plot the histogram and the fitted normal distribution curve from.
       title: str
           Title of the histogram plot.
       xlabel: str
           Label for the x-axis.
       ylabel: str
           Label for the y-axis.
       bins: int, optional (default=20)
           Number of bins to use in the histogram.

       Returns:
       --------
       None
    """
    plt.hist(data,normed=1,bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    min_,max_,mean_,median_,mode_,var_,std_,*X = describe_data(data)
    x = np.linspace(min_,max_,1000)
    pdf = st.norm.pdf(x,mean_,std_)
    plt.plot(x,pdf,'r')    
    plt.show()

def box_plot(data, labels):
    """
       Function to plot a box plot.

       Parameters:
       -----------
       data: array-like
           Input data to plot the box plot from.
       labels: array-like
           Labels for the x-axis.

       Returns:
       --------
       None
    """
    plt.boxplot(data,labels=labels)
    plt.show()


# Parametric??
def test_normal_ks(data):
    """
     Kolgomorov-Smirnov test for normality.

     Parameters:
     -----------
     data : array-like
         Input data for the test.

     Returns:
     --------
     tuple
         A tuple containing the test statistic and p-value.
     """
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.kstest(norm_data,'norm')

def test_normal_sw(data):
    """
    Shapiro-Wilk test for normality.

    Parameters:
    -----------
    data : array-like
        Input data for the test.

    Returns:
    --------
    tuple
        A tuple containing the test statistic and p-value.
    """
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.shapiro(norm_data)

def levene(data):
    """
    Levene's test for equality of variances.

    Parameters:
    -----------
    data : tuple
        Input data for the test. Should be in the form (data1, data2, ...).

    Returns:
    --------
    tuple
        A tuple containing the test statistic and p-value.
    """
    W,pval = st.levene(*data)
    return(W,pval)

# hypothesis testing
# Parametric
def t_test_ind(data1,data2, eq_var=True):
    """
    Two-sample independent t-test for parametric data.

    Parameters:
    -----------
    data1 : array-like
        Input data for the first sample.
    data2 : array-like
        Input data for the second sample.
    eq_var : bool, optional
        Whether or not to assume equal variances for the two samples. Default is True.

    Returns:
    --------
    tuple
        A tuple containing the test statistic and p-value.
    """
    t,pval = st.ttest_ind(data1,data2, equal_var=eq_var)
    return (t,pval)

def t_test_dep(data1,data2):
    """
    Two-sample dependent t-test for parametric data.

    Parameters:
    -----------
    data1 : array-like
        Input data for the first sample.
    data2 : array-like
        Input data for the second sample.

    Returns:
    --------
    tuple
        A tuple containing the test statistic and p-value.
    """
    t,pval = st.ttest_rel(data1,data2)
    return (t,pval)

def one_way_ind_anova(data):
    """
    Perform a one-way analysis of variance (ANOVA) test with independent samples.

    Parameters:
    -----------
    data: list of arrays
        A list containing the arrays of each group to be compared.

    Returns:
    --------
    tuple
        A tuple containing the calculated F-statistic and p-value of the test.
    """
    F,pval = st.f_oneway(*data)
    return (F,pval)


# Non Parametric
def mann_whitney(data1,data2):
    """
    Perform a Mann-Whitney U test with two independent samples.

    Parameters:
    -----------
    data1: array-like
        The first set of samples.
    data2: array-like
        The second set of samples.

    Returns:
    --------
    tuple
        A tuple containing the calculated U-statistic and p-value of the test.
    """
    return st.mannwhitneyu(data1, data2)

def wilcoxon(data1,data2):
    """
    Perform a Wilcoxon signed-rank test with two dependent samples.

    Parameters:
    -----------
    data1: array-like
        The first set of samples.
    data2: array-like
        The second set of samples.

    Returns:
    --------
    tuple
        A tuple containing the calculated test statistic and p-value of the test.
    """
    return st.wilcoxon(data1,data2)

def kruskal_wallis(data):
    """
    Perform a Kruskal-Wallis H test with independent samples.

    Parameters:
    -----------
    data: list of arrays
        A list containing the arrays of each group to be compared.

    Returns:
    --------
    tuple
        A tuple containing the calculated H-statistic and p-value of the test.
    """
    H,pval = st.kruskal(*data)
    return (H,pval)

def friedman_chi(data):
    """
    Perform a Friedman test with dependent samples.

    Parameters:
    -----------
    data: array-like
        A 2D array containing the data for all groups.

    Returns:
    --------
    tuple
        A tuple containing the calculated test statistic and p-value of the test.
    """
    F,pval = st.friedmanchisquare(*data)
    return (F,pval)    
    
# Effect size
def effect_size_t(stat,df):
    """
        Function to calculate Cohen's d effect size for a t-test.

        Parameters:
        -----------
        stat: float
            The t-statistic from the t-test.
        df: int
            Degrees of freedom from the t-test.

        Returns:
        --------
        float
            Cohen's d effect size.
    """
    r = np.sqrt(stat**2/(stat**2 + df))
    return r

def effect_size_mw(stat,n1,n2):
    """
    Function to calculate the effect size for a Mann-Whitney U test.

    Parameters:
    -----------
    stat: float
        The U-statistic from the Mann-Whitney U test.
    n1: int
        The size of the first sample.
    n2: int
        The size of the second sample.

    Returns:
    --------
    float
        Effect size.
    """
    n_ob = n1 + n2 
    mean = n1*n2/2
    std = np.sqrt(n1*n2*(n1+n2+1)/12)
    z_score = (stat - mean)/std
    print(z_score)
    return z_score/np.sqrt(n_ob)

def effect_size_wx(stat,n, n_ob):
    """
    Function to calculate the effect size for a Wilcoxon signed-rank test.

    Parameters:
    -----------
    stat: float
        The test statistic from the Wilcoxon signed-rank test.
    n: int
        The size of the effective sample (zero differences are excluded!).
    n_ob: int
        The total number of observations, which is the size of sample 1 plus
        the size of sample 2.

    Returns:
    --------
    float
        Effect size.
    """
    mean = n*(n+1)/4
    std = np.sqrt(n*(n+1)*(2*n+1)/24)
    z_score = (stat - mean)/std
    return z_score/np.sqrt(n_ob)


def main_1():
    """
       Function to run example analyses on a single group of data.

       This function retrieves pulse rate data, describes it, tests for normality
       using the Kolmogorov-Smirnov test, and displays a histogram.

       Parameters:
       -----------
       None

       Returns:
       --------
       None
    """
    # Pulse Rate example (one group)
    pr = get_data(prefix+'pulse_rate.txt')
    describe_data(pr)
    print(test_normal_ks(pr))    
    histogram(pr, 'Histogram','Pulse Rate', 'Frequency')
    
def main_11():
    """
       Function to create a box plot of pulse rate data.

       This function retrieves pulse rate data and creates a box plot.

       Parameters:
       -----------
       None

       Returns:
       --------
       None
    """
    # Pulse Rate example (one group)
    pr = get_data(prefix+'pulse_rate.txt')
    box_plot(pr,['PR'])

     
    
def main_111():
    """
        Function to run example analyses on a single group of data, with a
        normalized histogram.

        This function retrieves pulse rate data, describes it, tests for normality
        using the Kolmogorov-Smirnov test, and displays a normalized histogram.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
    """
    # Pulse Rate example (one group)
    pr = get_data(prefix+'pulse_rate.txt')
    describe_data(pr)
    print(test_normal_ks(pr))    
    histogram_norm(pr, 'Histogram','Pulse Rate', 'Frequency')  
    
def main_2():
    """
        Retrieves data about animals from a file and returns it.

        Returns:
        --------
        numpy.ndarray
            An array containing data about animals.
    """
    # Animals (four groups)
    data = get_data_many_header(prefix+'animals.txt')
    return data

    
def main_333():
    """
        Retrieves data about spiders from a file, performs statistical analysis and creates visualizations.

        Returns:
        --------
        None
    """
    # Spider example (2 groups)
    sp = get_data_many(prefix+'spider.txt')
    fake_sp = sp[0,:]
    real_sp = sp[1,:]
    box_plot([fake_sp,real_sp],['Fake','Real'])
    print('Picture')
    print(test_normal_ks(fake_sp))    
    histogram_norm(fake_sp, 'Histogram: Picture','Anxiety', 'Frequency') 
    print('Real')
    print(test_normal_ks(real_sp))    
    histogram_norm(fake_sp, 'Histogram: Real','Anxiety', 'Frequency')     
    print('Variance')
    print(levene([fake_sp , real_sp]))
    
    
def main_3():
    """
        Retrieves data about spiders from a file, performs a t-test and creates a box plot.

        Returns:
        --------
        None
    """
    # Spider example (2 groups)
    sp = get_data_many(prefix+'spider.txt')
    fake_sp = sp[0,:]
    real_sp = sp[1,:]
    t,pval = t_test_dep(fake_sp,real_sp)
    print('t: %s   p_value: %s' % (t,pval))
    r = effect_size_t(t,len(fake_sp)-1)
    print('Effect size:  %s' % r)
    
    min_ = min([np.amin(sp[i,:]) for i in range(len(sp))])
    max_ = max([np.amax(sp[i,:]) for i in range(len(sp))])    
    plt.axis(ymin=min_ - 20, ymax=max_ + 20)
    plt.title('Anxiety to spiders')
    plt.ylabel('Level')
    
    box_plot([fake_sp,real_sp],['Fake','Real'])





def main_311():
    """
        Function to conduct hypothesis testing and print results using the Mann-Whitney U test and the independent t-test.
        This function specifically applies these tests to spider-related anxiety levels, comparing two groups of individuals
        (fake_sp and real_sp).

        Parameters:
        -----------
        None

        Returns:
        --------
        None
    """
    # Spider example (2 groups)
    sp = get_data_many(prefix+'spider.txt')
    fake_sp = sp[0,:]
    real_sp = sp[1,:]
    u, p = mann_whitney(fake_sp,real_sp)
    print('u= %f   p = %s' % (u, p))
    t,p = t_test_ind(fake_sp,real_sp)
    print('t= %f   p = %s' % (u, p))

def main_31111():
    """
       Function to conduct hypothesis testing and print results using the Wilcoxon signed-rank test and effect size calculation.
       This function specifically applies these tests to spider-related anxiety levels, comparing two groups of individuals
       (fake_sp and real_sp), and plots a histogram of the anxiety levels.

       Parameters:
       -----------
       None

       Returns:
       --------
       None
    """
    # Spider example (2 groups)
    sp = get_data_many(prefix+'spider.txt')
    fake_sp = sp[0,:]
    real_sp = sp[1,:]
    T,pval = wilcoxon(fake_sp,real_sp)
    print('T: %s   p_value: %s' % (t,pval))
    r = effect_size_wx(T,18, 24)
    print('Effect size:  %s' % r)
   
    histogram(fake_sp, 'Anxiety to spiders','Level','Value' )     
    
if __name__ == '__main__':
    # The following variable should be edited by you
    prefix='../data/'
    #main_1()
    #main_11()
    #main_111()
    print(main_2())
    #main_3()
    #main_31()
    #main_311()
    #main_333()
    #main_31111()
       
    
    """
    # Spider example (2 groups)
    sp = get_data_many(prefix+'spider.txt')
    fake_sp = sp[0,:]
    real_sp = sp[1,:]
    t,pval = t_test_dep(fake_sp,real_sp)
    print(t,pval)
    r = effect_size_t(t,len(fake_sp)-1)
    print(r)
    #min_ = min([np.amin(sp[i,:]) for i in range(len(sp))])
    #max_ = max([np.amax(sp[i,:]) for i in range(len(sp))])    
    #plt.axis(ymin=min_ - 20, ymax=max_ + 20)new_arr
    #plt.title('Anxiety to spiders')
    #plt.ylabel('Level')
    #box_plot(pr,['PR'])
    #box_plot([fake_sp,real_sp],['Fake','Real'])
    #plt.title('Ansiety: fake spider')
    #histogram(fake_sp)
    #spy = get_data_two(prefix+'spider.txt')
    #print(mann_whitney(spy[0,:],spy[1,:]))
    #print(t_test_ind(fake_sp,real_sp))
    #print(t_test_dep(fake_sp,real_sp))    
    

    
    # Sphere example (3 groups)
    #sphere = get_data_many(prefix+'sphere_3.txt')
    #print(kruskal_wallis(sphere))
    #print(friedman_chi(sphere))
    #print(levene(sphere))    
    #print(one_way_ind_anova(sphere))

    # Effect size
    print(effect_size_t(t,len(fake_sp)-1))
    print(effect_size_mw(35.5,10,10))
    print(effect_size_wx(0,8,20))
    """