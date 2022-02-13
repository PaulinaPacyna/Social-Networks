import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Snars:
    def __init__(self, Graph: nx.Graph):
        self.G = Graph
        self.degrees = dict(self.G.degree)
        self.degree_list = list(self.degrees.values())

    def plot_log_log_hist(self):
        plt.hist(self.degree_list, bins=np.exp(np.linspace(np.log(min(self.degree_list)),
                                                                         np.log(max(self.degree_list)), 20)))
        plt.yscale('log')
        plt.xscale('log')

    def survival(self, x: float) -> float:
        cdf = len(list(filter(lambda v: v < x, self.degree_list))) / len(self.degree_list)
        # nr of observations less than x/nr of observations
        return 1 - cdf

    def plot_survival(self):
        x_to_plot = np.unique(self.degree_list)
        y_to_plot = [self.survival(x) for x in x_to_plot]
        plt.plot(x_to_plot, y_to_plot)

    def est_alpha_lin_regression(self):
        bins = np.histogram(self.degree_list, bins=np.exp(np.linspace(np.log(min(self.degree_list)),
                                                                         np.log(max(self.degree_list)), 20)))
        log_y = bins[0].reshape(-1, 1)
        # middles of the intervals
        log_x = np.array([(a + b) / 2 for a, b in zip(bins[1][:-1], bins[1][1:], )]).reshape(-1, 1)
        model = LinearRegression(fit_intercept=True).fit(log_x, log_y)
        return model.coef_, model.intercept_

    def mle(self):
        x_min = min(self.degree_list)
        return 1 + len(self.degree_list) / (np.log(x_min) + sum(np.log(self.degree_list)))
