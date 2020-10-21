# https://stackoverflow.com/questions/8409095/set-markers-for-individual-points-on-a-line-in-matplotlib
# https://jeremy9959.net/Blog/bayesian-online-changepoint-fixed/

# General libs
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
import random
import time
import pandas as pd
from tabulate import tabulate


# Changepoint libs
import bayesian_changepoint_detection.bayesian_changepoint_detection.online_changepoint_detection as oncd
# from bocd.bocd.bocd import *
# from bocd.bocd.distribution import *
from bocd.bocd.hazard import *
import changefinder
import ruptures as rpt


# NUM_POINTS = [25,50,100,500,1000,1250]
# CHANGEPOINT_LOCATIONS = [6,16,25,125,250,1000]
CHANGEPOINT_LOCATIONS = [25, 50, 75, 125, 250, 375,
                         0, 100, 0, 500] # cases where there are no changepoints
NUM_POINTS = [100, 100, 100, 500, 500, 500,
              100, 100, 500, 500] # cases where there are no changepoints
t1 = 10
t2 = 100
lambda1 = 1
lambda2 = 10
RANDOM_SEED = 2222
PLOT_BOCPD = False
PLOT_CHANGEFINDER = False

def plot_data(data):
    x = np.arange(0, data['series'].shape[0])
    y = data['series']

    # plotting
    plt.title("Time Series, n={}; changepoint location at: {}; lambdas={}; ts={}".
              format(data['n'], data['c'], repr(data['lambdas']), repr(data['ts'])))
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.axvline(x=data['c'], linewidth=2, color='blue')
    plt.plot(x, y, color="red")
    plt.show()


def plot_multiple_data(data_list):
    fig, axs = plt.subplots(len(data_list))
    plt.subplots_adjust(hspace = 1)

    for i in range(len(data_list)):
        data = data_list[i]

        x = np.arange(0, data['series'].shape[0])
        y = data['series']

        # plotting
        plt.title("Time Series")
        axs[i].set_title("n={}; CPL: {}; lambdas={}; ts={}".
                  format(data['n'], data['c'], repr(data['lambdas']), repr(data['ts'])))
        # plt.ylabel("Y axis")
        axs[i].axvline(x=data['c'], linewidth=1, color='blue')
        if 'detected_changepoints' in data.keys():
            for changepoint_model in data['detected_changepoints']:
                # for j in range(len(data['detected_changepoints'][changepoint_model])):
                #     axs[i].axvline(x=data['detected_changepoints'][j], linewidth=1, color='red')
                scatter_marker_loc = data['detected_changepoints'][changepoint_model]
        # axs[i].scatter(x = scatter_marker_loc, y=0.5,
        #             marker=ALGORITHMS_PLOT_PARAMS[changepoint_model]['marker'],
        #             color=ALGORITHMS_PLOT_PARAMS[changepoint_model]['color'])
        axs[i].plot(x, y, color="black")
        # axs[i].scatter(x = scatter_marker_loc, y=0.5,
        #             marker=ALGORITHMS_PLOT_PARAMS[changepoint_model]['marker'],
        #             color=ALGORITHMS_PLOT_PARAMS[changepoint_model]['color'])
        plt.show()


def plot_multiple_data_by_algorithm(data_list, changepoint_algorithm):

    fig, axs = plt.subplots(len(data_list))
    fig.set_size_inches(8.25,11.75)

    plt.subplots_adjust(hspace = 1)
    plt.xlabel("Changepoint Algorithm: {}".format(changepoint_algorithm))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font = {'weight' : 'normal', 'size'   : 12}

    # plt.figure(figsize=(4, 7))

    for i in range(len(data_list)):
        data = data_list[i]

        x = np.arange(0, data['series'].shape[0])
        y = data['series']

        # plotting

        # axs[i].set_title("n={}; CPL: {}; lambdas={}; ts={}".
        #           format(data['n'], data['c'], repr(data['lambdas']), repr(data['ts'])))

        axs[i].set_title("Real Changepoint at: {} | Detected Changepoints at: {}".
                  format(data['c'], repr(data['detected_changepoints'][changepoint_algorithm])))

        # plt.ylabel("Y axis")
        axs[i].axvline(x=data['c'], linewidth=1, color='blue')
        if 'detected_changepoints' in data.keys():
            for j in range(len(data['detected_changepoints'][changepoint_algorithm])):
                axs[i].axvline(x=data['detected_changepoints'][changepoint_algorithm][j], linewidth=1, color='red')


        axs[i].plot(x, y, color="black")


        plt.savefig('figures/{}.png'.format(changepoint_algorithm),dpi=100)
        plt.show()



def generate_truncated_exp(n, c, ts = (t1,t2), lambdas = (lambda1, lambda2)):
    """
    https://stackoverflow.com/questions/40143718/python-scipy-truncated-exponential-distribution

    :param n: length of time series
    :param c: changepoint location
    :param t1: parameter of truncated exponential
    :param t2:
    :param lambdas: rate para
    :return:
    """
    lower, upper, scale = ts[0], ts[1], lambdas[0]
    X1 = ss.truncexpon(b=(upper - lower) / scale, loc=lower, scale=scale)
    lower, upper, scale = ts[0], ts[1], lambdas[1]
    d1 = X1.rvs(c)
    X2 = ss.truncexpon(b=(upper - lower) / scale, loc=lower, scale=scale)
    d2 = X2.rvs(n-c)
    series = np.concatenate((d1,d2))
    data = defaultdict()
    data['series'] = series
    data['c'] = c
    data['n'] = n
    data['ts'] = ts
    data['lambdas'] = lambdas
    data['distribution'] = ss.truncexpon
    data['detected_changepoints'] = dict()
    return data


def F_exp(x, lambda_):
    return 1 - np.exp(-lambda_ * x)


def F_exp_inverse(x, lambda_):
    return (-1) * (1 / lambda_) * np.log(1 - x)


def generated_truncated_exp_inverse_uniform_sampling(n, c, t1, t2, lambda1, lambda2):
    """
    https://stackoverflow.com/questions/40143718/python-scipy-truncated-exponential-distribution

    :param n: length of time series
    :param c: changepoint location
    :param t1: parameter of truncated exponential
    :param t2:
    :param lambdas: rate para
    :return:
    """

    # Generate a uniform random variable U ~ Uniform([F_exp(a),F_exp(b)]
    u1 = np.random.uniform(F_exp(t1,1/lambda2),F_exp(t2,1/lambda2), size = c)
    # Now, take F^(-1)(u)
    X1 = F_exp_inverse(u1,lambda2)

    u2 = np.random.uniform(F_exp(t1, 1/lambda1), F_exp(t2, 1/lambda1), size=n-c)
    X2 = F_exp_inverse(u2,lambda1)

    series = np.concatenate((X1,X2))
    data = defaultdict()
    data['series'] = series
    data['c'] = c
    data['n'] = n
    data['ts'] = [t1,t2]
    data['lambdas'] = [lambda1, lambda2]
    data['distribution'] = ss.truncexpon
    data['detected_changepoints'] = dict()
    return data


def L1TF(signal):
    import numpy as np
    import cvxpy as cp
    import scipy as scipy
    import cvxopt as cvxopt

    # Load time series data: S&P 500 price log.
    # y = np.loadtxt(open('data/snp500.txt', 'rb'), delimiter=",", skiprows=1)
    y = signal
    n = y.size

    # Form second difference matrix.
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)

    # Set regularization parameter.
    vlambda = 50

    # Solve l1 trend filtering problem.
    x = cp.Variable(shape=n)

    obj = cp.Minimize(0.5 * cp.sum_squares(y - x) + vlambda * cp.norm(D@x, 1) )
    prob = cp.Problem(obj)

    # ECOS and SCS solvers fail to converge before
    # the iteration limit. Use CVXOPT instead.
    prob.solve(solver=cp.CVXOPT, verbose=True)
    print('Solver status: {}'.format(prob.status))

    # Check for error.
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")

    print("optimal objective value: {}".format(obj.value))

    import matplotlib.pyplot as plt

    # Show plots inline in ipython.
    # %matplotlib inline

    # Plot properties.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font = {'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)

    # Plot estimated trend with original signal.
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(1,n+1), y, 'k:', linewidth=1.0)
    plt.plot(np.arange(1,n+1), np.array(x.value), 'b-', linewidth=2.0)
    plt.xlabel('date')
    plt.ylabel('log price')


if __name__ == "__main__":
    np.random.seed(seed=RANDOM_SEED)


    data_list = list()

    benchmark_list = list()
    # and each element of the list contains: "AlgorithmName", "NumPoints", "ChangepointLocation", "ExecutionTime"

    for i in range(len(NUM_POINTS)):
        # tmp_data = generate_truncated_exp(NUM_POINTS[i], CHANGEPOINT_LOCATIONS[i])
        tmp_data = generated_truncated_exp_inverse_uniform_sampling(NUM_POINTS[i],
                                                                    CHANGEPOINT_LOCATIONS[i],
                                                                    t1,t2,lambda1,lambda2)
        data_list.append(tmp_data)

    plot_multiple_data(data_list)

    for i in range(len(data_list)):
        signal = data_list[i]['series']

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # #    RUPTURES ALGORITHMS  # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        n = signal.shape[0]
        dim = 1
        sigma = 1
        model = 'rbf'

        RUPTURES_ALGORITHMS = {
            'Pelt': {
                'function': rpt.Pelt,
                'model_params': {
                    'model': 'rbf'
                },
                'predict_params': {
                    'pen': np.log(n) * dim * sigma ** 2
                }
            },
            'DynamicProgramming': {
                'function': rpt.Dynp,
                'model_params': {
                    'model': 'rbf'
                },
                'predict_params': {
                    'n_bkps': 1
                }
            },
            'BinarySegmentation': {
                'function': rpt.Binseg,
                'model_params': {
                    'model': 'rbf'
                },
                'predict_params': {
                    'pen': np.log(n) * dim * sigma ** 2
                }
            },
            'BottomUpSegmentation': {
                'function': rpt.BottomUp,
                'model_params': {
                    'model': 'rbf'
                },
                'predict_params': {
                    'pen': np.log(n) * dim * sigma ** 2
                }
            },
            'WindowBased': {
                'function': rpt.Window,
                'model_params': {
                    'model': 'rbf',
                    'width': 10,
                },
                'predict_params': {
                    'pen': np.log(n) * dim * sigma ** 2
                }
            },
        }

        ALGORITHMS_PLOT_PARAMS = {
            'Pelt': {
                'color': 'b',
                'marker': 'D'
            },
            'DynamicProgramming': {
                'color': 'b',
                'marker': '*'
            },
            'BinarySegmentation': {
                'color': 'b',
                'marker': 's'
            },
            'BottomUpSegmentation': {
                'color': 'b',
                'marker': 'o'
            },
            'WindowBased': {
                'color': 'b',
                'marker': '+'
            }
        }

        for a in RUPTURES_ALGORITHMS.keys():
            start_time = time.time()
            algo = RUPTURES_ALGORITHMS[a]['function'](**RUPTURES_ALGORITHMS[a]['model_params']).fit(signal)
            my_bkps = algo.predict(**RUPTURES_ALGORITHMS[a]['predict_params'])
            end_time = time.time() - start_time
            benchmark_list.append({"AlgorithmName": a,
                                   "TimeSeriesID": i,
                                   "NumPoints": NUM_POINTS[i],
                                   "ChangepointLocation": CHANGEPOINT_LOCATIONS[i],
                                   "ExecutionTime":end_time})
            print(a)
            print(my_bkps)
            data_list[i]['detected_changepoints'].update({a: my_bkps})

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # BAYESIAN ONLINE CHANGEPOINT DETECTION # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        start_time = time.time()
        R, maxes = oncd.online_changepoint_detection(signal,
                                                     hazard_func=partial(oncd.constant_hazard, 500),
                                                     observation_likelihood=oncd.StudentT(0.1, .01, 1, 0))
        end_time = time.time() - start_time
        Nw = 10
        all_cp_probs = R[Nw, Nw:-1]
        idx_changes = np.where(np.diff(maxes) < 0)[0]
        # R[m,n] R he matrix R is such that R[m,n] is the probability at time n that the run length is m.
        # The array maxes is the array of most likely run length at each time.
        Num_CP = 2
        top_n_changepoints = list(all_cp_probs.argsort()[-Num_CP:][::-1])

        data_list[i]['detected_changepoints'].update({'BayesianOnlineChangePointDetection': top_n_changepoints})
        benchmark_list.append({"AlgorithmName": "BayesianOnlineChangePointDetection",
                               "TimeSeriesID": i,
                               "NumPoints": NUM_POINTS[i],
                               "ChangepointLocation": CHANGEPOINT_LOCATIONS[i],
                               "ExecutionTime": end_time})

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # CHANGEFINDER  # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        smooth = int(0.05 * signal.shape[0])
        if smooth <= 3:
            smooth = 3

        start_time = time.time()
        cf = changefinder.ChangeFinder(r=0.01, order=1, smooth=smooth)

        ret = []
        for j in signal:
            score = cf.update(j)
            ret.append(score)

        end_time = time.time() - start_time

        Num_CP = 2
        top_n_changepoints = list(np.array(ret).argsort()[-Num_CP:][::-1])
        data_list[i]['detected_changepoints'].update({'ChangeFinder': top_n_changepoints})

        benchmark_list.append({"AlgorithmName": "ChangeFinder",
                               "TimeSeriesID": i,
                               "NumPoints": NUM_POINTS[i],
                               "ChangepointLocation": CHANGEPOINT_LOCATIONS[i],
                               "ExecutionTime": end_time})

        if PLOT_CHANGEFINDER:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(ret)
            ax2 = ax.twinx()
            ax2.plot(signal, 'r')
            plt.show()


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # PYRAMID RECURRENT NEURAL NETWORKS CPD # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO

        # # # # # # # # ACCURACY METRICS # # # # # # # #  # # # # # #

        # First of all, remove the detected changepoint which occurs either at the start of the series or at the end

    for i in range(len(data_list)):
        for algo in data_list[i]['detected_changepoints'].keys():
            copy_algo_cp = list()
            for k in data_list[i]['detected_changepoints'][algo]:
                if k != len(data_list[i]['series']) and k!=0:
                    # data_list[i]['detected_changepoints'][algo].remove(j)
                    copy_algo_cp.append(k)
            data_list[i]['detected_changepoints'][algo] = copy_algo_cp

    # plot multiple by algorithm
    LIST_OF_ALGORITHMS = ['Pelt', 'DynamicProgramming', 'BinarySegmentation', 'BottomUpSegmentation', 'WindowBased', 'BayesianOnlineChangePointDetection', 'ChangeFinder']
    for algorithm in LIST_OF_ALGORITHMS:
        plot_multiple_data_by_algorithm(data_list, algorithm)

    df_benchmark = pd.DataFrame(benchmark_list)
    pdtabulate = lambda df: tabulate(df, headers='keys', tablefmt='latex')
    print(pdtabulate(df_benchmark.loc[:35]))
    print(pdtabulate(df_benchmark.loc[36:]))

    # AlgorithmName #Paper #Code
    ref_list = [
        ['Pelt', "REFERENCE", "CODE"],
         ['DynamicProgramming', "REFERENCE", "CODE"],
         ['BinarySegmentation', "REFERENCE", "CODE"],
         ['BottomUpSegmentation', "REFERENCE", "CODE"],
         ['WindowBased', "REFERENCE", "CODE"],
         ['BayesianOnlineChangePointDetection', "REFERENCE", "CODE"],
         ['ChangeFinder', "REFERENCE", "CODE"]
    ]
    ref_df = pd.DataFrame(ref_list,columns=['AlgorithmName','Paper','Code'])
    print(pdtabulate(ref_df))





