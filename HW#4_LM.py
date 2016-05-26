# ! /usr/bin/env python

import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt
import functools


def LM(x0, f, df, ddf, max_iter=256, 
	epsilon=1e-6, 
	show_interval=None):
    x = x0
    dim = x.shape[0]
    mu = 1
    f_s_k, q_k = 0, 0

    for _i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)
        if np.linalg.norm(dfx) < epsilon:
            break
        while True:
            try:
                M = ddfx + mu * np.identity(dim)
                L = np.linalg.cholesky(M)
                break
            except np.linalg.LinAlgError:
                mu *= 4
        s = np.linalg.solve(M, -dfx)
        new_f_s_k = f(x + s)
        new_q_k = fx + dfx.transpose().dot(s) + s.transpose().dot(ddfx).dot(s) / 2
        r_k = (new_f_s_k - f_s_k) / (new_q_k - q_k)
        f_s_k, q_k = new_f_s_k, new_q_k
        if r_k < 0.25:
            mu *= 4
        elif r_k > 0.75:
            mu /= 2
        if r_k > 0:
            x += s

        if show_interval and _i % show_interval == 0:
            print(
                'Iter %d: mu=%e, error=%e' %
                (_i, mu, np.linalg.norm(dfx)))
            pass

    return x


def Test_plot(domain, func_base, func_fit, data_set, legend_title=''):
    fig = plt.figure()
    base_range = func_base(domain)
    fit_range = func_fit(domain)
    plt.plot(domain, base_range, label='Ground Truth')
    plt.plot(domain, fit_range, label='Prediction')
    data_x, data_y = data_set
    plt.plot(
        data_x, data_y,
        marker='o', linestyle='None', label='Samples')
    plt.legend(title=legend_title)

    margin = 0.5
    plt.xlim(np.amin(domain) - margin, np.amax(domain) + margin)
    plt.ylim(np.amin(base_range) - margin, np.amax(fit_range) + margin)
    fig.show()
    pass


def Test():
    np.random.seed(0xdeadbeef)
	#Initialization for parameters
    dim = 4
    nsample = 256
    N = 64
    gauss_scale = 0.16
	#Generate ground truth
    func_base = np.sin
    domain_l, domain_r = 0 * np.pi, 2 * np.pi
    domain = np.linspace(domain_l, domain_r, num=nsample)
	
    data_x_gen = lambda N: np.linspace(domain_l, domain_r, num=N)
    data_y_gen = lambda data_x: [func_base(x) for x in data_x]
    data_y_noise_gen = lambda data_x, scale: [
        func_base(x) + np.random.normal(scale=scale) for x in data_x]
    data_x = data_x_gen(N)
    data_y_noise = data_y_noise_gen(data_x, gauss_scale)
    data_set = data_x, data_y_noise

    X = np.array([[i ** j for j in range(dim)] for i in data_x])
    y0 = data_y_noise
	
    k = 0.3
    f = lambda x: np.linalg.norm(X.dot(x) - y0) ** 2 + k / 2 * np.linalg.norm(x) ** 2
    df = lambda x: 2 * X.transpose().dot(X.dot(x) - y0) + k * x
    ddf = lambda x: 2 * X.transpose().dot(X) + k * np.identity(X.shape[1])

	#Perform LM algorithm
    A = LM(np.zeros(dim), f, df, ddf, show_interval=1)

    func_fit = np.polynomial.Polynomial(A)
    msg = 'Polynomial fitting using LM algorithm'
    Test_plot(domain, func_base, func_fit, data_set, msg)
    plt.show()
    pass


def main():
    Test()
    pass


if __name__ == '__main__':
    main()

