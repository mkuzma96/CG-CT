# Sensitivity analysis for unobserved confounding
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from main import *
from cv_hyper_search import *


# Sensitivity analysis implementation (from Jesson et al. 2022) with normal outcome distribution

def get_mu(w, gamma_term, y_means, y_samples):
    I1 = torch.mean(w * (y_samples - y_means), dim=1, keepdim=True)
    I2 = torch.mean(w, dim=1, keepdim=True)
    return y_means + (I1 / (gamma_term + I2))


def get_mu_plus(y_H, gamma_term, y_means, ysamples):
    diff = ysamples - y_H
    w = torch.heaviside(diff, torch.zeros_like(diff))
    return get_mu(w, gamma_term, y_means, ysamples)


def get_mu_minus(y_H, gamma_term, y_means, ysamples):
    diff = y_H - ysamples
    w = torch.heaviside(diff, torch.zeros_like(diff))
    return get_mu(w, gamma_term, y_means, ysamples)


def compute_bounds(y_means, y_var, gammas):
    # Sample from normal distribution with mean y and variance eps_hat
    y_samples = np.random.normal(y_means, np.sqrt(y_var), size=(5000, y_means.shape[0]))
    y_samples = torch.from_numpy(y_samples).float()
    y_samples = torch.transpose(y_samples, 0, 1)
    y_means = torch.from_numpy(np.expand_dims(y_means, axis=1)).float()

    # Compute bounds
    gamma_term = gammas.repeat(y_means.shape[0], 1)
    gamma_term = 1 / ((gamma_term * gamma_term) - 1)

    # Initial values
    mu_plus = torch.full_like(gamma_term, -1000)
    mu_minus = torch.full_like(gamma_term, 1000)
    y_plus = torch.zeros_like(gamma_term)
    y_minus = torch.zeros_like(gamma_term)

    # Start grid search
    for y_H in torch.unbind(y_samples, dim=1):
        y_H = torch.unsqueeze(y_H, dim=1)
        kappa_plus = get_mu_plus(y_H, gamma_term, y_means, y_samples)
        kappa_minus = get_mu_minus(y_H, gamma_term, y_means, y_samples)
        idx_plus = kappa_plus > mu_plus
        mu_plus[idx_plus] = kappa_plus[idx_plus]
        y_plus[idx_plus] = y_H.repeat((1, gamma_term.size(1)))[idx_plus]
        idx_minus = kappa_minus < mu_minus
        mu_minus[idx_minus] = kappa_minus[idx_minus]
        y_minus[idx_minus] = y_H.repeat((1, gamma_term.size(1)))[idx_minus]

    Q_plus = torch.zeros_like(gamma_term)
    Q_minus = torch.zeros_like(gamma_term)
    for i in range(gamma_term.size(1)):
        Q_plus[:, i:i + 1] = get_mu_plus(y_plus[:, i:i + 1], gamma_term, y_means, y_samples)[:, i:i + 1]
        Q_minus[:, i:i + 1] = get_mu_minus(y_minus[:, i:i + 1], gamma_term, y_means, y_samples)[:, i:i + 1]
    return [Q_plus.detach().numpy(), Q_minus.detach().numpy()]




def main():
    #Load trained model parameters
    GPS_mods = np.load('plot_data/tr_var.npz')
    b_coef = GPS_mods['b_coef']
    sig_hat = GPS_mods['sig_hat']
    a_coef = GPS_mods['a_coef']
    df_baes = GPS_mods['df_bae']


    # Real world - train
    df_train = pd.read_csv('data/HIV_data.csv')
    year = 2016
    df_train = df_train[df_train['year'] == year]
    df_train['hiv_cases'] = df_train['hiv_rate']*df_train['population'] # In thousands
    df_train.head(5)

    # Estimate residual variance on training data
    # Predict factual outcomes
    y_hat = np.empty((df_train.shape[0], 1))
    for i, country in enumerate(df_train['country'].unique()):
        country_ind = np.where(df_train['country'] == country)[0]
        a_obs = df_train[df_train['country']==country]['hiv_aid']
        a_obs = a_obs.iloc[0]
        y_hat[i, 0] = GPS_pred(np.concatenate(([[a_obs]], df_baes[country_ind, 2:, 0]), axis=1), b_coef[:, 0:(0 + 1)],
                 sig_hat[:, 0:(0 + 1)], a_coef[:, 0:(0 + 1)])

    #Factual outcome
    y = np.expand_dims(df_train['hiv_reduction'].to_numpy(), axis=1) * 100
    residuals = y / 100 - y_hat
    eps_hat = np.sum(residuals**2) / (df_train.shape[0])

    # Real world - test
    df = pd.read_csv('data/HIV_data.csv')
    year = 2017
    df = df[df['year'] == year]
    df['hiv_cases'] = df['hiv_rate']*df['population'] # In thousands
    df.head(5)
    a_sig = np.std(df['hiv_aid'])
    # Plot treatment response curves for all countries

    countries = ['South Africa', 'Mozambique', 'Burundi', 'Congo, Dem. Rep.', 'India', 'Indonesia']
    gammas = torch.tensor([1.25, 1.5])

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=18)
    color_cycle = plt.rcParams['axes.prop_cycle']()
    bound_colors = ["black", "grey"]
    bound_labels = [r"$\Gamma = 1.25$", r"$\Gamma = 1.5$"]

    # Country 1
    country = countries[0]
    a_obs = df[df['country']==country]['hiv_aid']
    a_obs = a_obs.iloc[0]
    if a_obs - a_sig > 0:
        a = np.arange(a_obs-a_sig, a_obs+a_sig, 3)
    else:
        a = np.arange(0, a_obs+a_sig, 3)
    y = np.empty(len(a))
    for i in range(len(a)):
        country_ind = np.where(df['country'] == country)[0]
        y[i] = GPS_pred(np.concatenate(([[a[i]]], df_baes[country_ind, 2:, 0]), axis=1) , b_coef[:,0:1], sig_hat[:,0:1], a_coef[:,0:1]) * 100

    # Bounds
    [Q_plus, Q_minus] = compute_bounds(y, eps_hat * 100 * 100, gammas)
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Plot bounds as dotted lines
        axs[0, 0].plot(a, Q_plus[:, i], color=bound_colors[i], linestyle='dashed', label=bound_labels[i])
        axs[0, 0].plot(a, Q_minus[:, i], color=bound_colors[i], linestyle='dashed')
    axs[0, 0].plot(a, y, label=country, **next(color_cycle))
    y_min = Q_minus[:, -1].min() - 3
    y_max = Q_plus[:, -1].max() + 3
    axs[0, 0].vlines(x=df.loc[df['country']==country, 'hiv_aid'], ymin=y_min, ymax=y_max,
                    colors='black', linestyle='dashed')
    axs[0, 0].set_ylim([y_min, y_max])

    # Country 2 ------------------------------------------
    country = countries[1]
    a_obs = df[df['country'] == country]['hiv_aid']
    a_obs = a_obs.iloc[0]
    if a_obs - a_sig > 0:
        a = np.arange(a_obs - a_sig, a_obs + a_sig, 3)
    else:
        a = np.arange(0, a_obs + a_sig, 3)
    y = np.empty(len(a))
    for i in range(len(a)):
        country_ind = np.where(df['country'] == country)[0]
        y[i] = GPS_pred(np.concatenate(([[a[i]]], df_baes[country_ind, 2:, 0]), axis=1) , b_coef[:,0:1], sig_hat[:,0:1], a_coef[:,0:1]) * 100

    # Bounds
    [Q_plus, Q_minus] = compute_bounds(y, eps_hat*10000, gammas)
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Plot bounds as dotted lines
        axs[0, 1].plot(a, Q_plus[:, i], color=bound_colors[i], linestyle='dashed')
        axs[0, 1].plot(a, Q_minus[:, i], color=bound_colors[i], linestyle='dashed')
    axs[0, 1].plot(a, y, label=country, **next(color_cycle))
    y_min = Q_minus[:, -1].min() - 3
    y_max = Q_plus[:, -1].max() + 3
    axs[0, 1].vlines(x=df.loc[df['country'] == country, 'hiv_aid'], ymin=y_min, ymax=y_max,
                     colors='black', linestyle='dashed')
    axs[0, 1].set_ylim([y_min, y_max])

    # Country 3 --------------------------------------------
    country = countries[2]
    a_obs = df[df['country'] == country]['hiv_aid']
    a_obs = a_obs.iloc[0]
    if a_obs - a_sig > 0:
        a = np.arange(a_obs - a_sig, a_obs + a_sig, 3)
    else:
        a = np.arange(0, a_obs + a_sig, 3)
    y = np.empty(len(a))
    for i in range(len(a)):
        country_ind = np.where(df['country'] == country)[0]
        y[i] = GPS_pred(np.concatenate(([[a[i]]], df_baes[country_ind, 2:, 0]), axis=1) , b_coef[:,0:1], sig_hat[:,0:1], a_coef[:,0:1]) * 100

    # Bounds
    [Q_plus, Q_minus] = compute_bounds(y, eps_hat * 10000, gammas)
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Plot bounds as dotted lines
        axs[0, 2].plot(a, Q_plus[:, i], color=bound_colors[i], linestyle='dashed')
        axs[0, 2].plot(a, Q_minus[:, i], color=bound_colors[i], linestyle='dashed')
    axs[0, 2].plot(a, y, label=country, **next(color_cycle))
    y_min = Q_minus[:, -1].min() - 3
    y_max = Q_plus[:, -1].max() + 3
    axs[0, 2].vlines(x=df.loc[df['country'] == country, 'hiv_aid'], ymin=y_min, ymax=y_max,
                     colors='black', linestyle='dashed')
    axs[0, 2].set_ylim([y_min, y_max])

    # Country 4 ------------------------------------------
    country = countries[3]
    a_obs = df[df['country'] == country]['hiv_aid']
    a_obs = a_obs.iloc[0]
    if a_obs - a_sig > 0:
        a = np.arange(a_obs - a_sig, a_obs + a_sig, 3)
    else:
        a = np.arange(0, a_obs + a_sig, 3)
    y = np.empty(len(a))
    for i in range(len(a)):
        country_ind = np.where(df['country'] == country)[0]
        y[i] = GPS_pred(np.concatenate(([[a[i]]], df_baes[country_ind, 2:, 0]), axis=1) , b_coef[:,0:1], sig_hat[:,0:1], a_coef[:,0:1]) * 100

    # Bounds
    [Q_plus, Q_minus] = compute_bounds(y, eps_hat*10000, gammas)
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Plot bounds as dotted lines
        axs[1, 0].plot(a, Q_plus[:, i], color=bound_colors[i], linestyle='dashed')
        axs[1, 0].plot(a, Q_minus[:, i], color=bound_colors[i], linestyle='dashed')
    axs[1, 0].plot(a, y, label=country, **next(color_cycle))
    y_min = Q_minus[:, -1].min() - 3
    y_max = Q_plus[:, -1].max() + 3
    axs[1, 0].vlines(x=df.loc[df['country'] == country, 'hiv_aid'], ymin=y_min, ymax=y_max,
                     colors='black', linestyle='dashed')
    axs[1, 0].set_ylim([y_min, y_max])

    # Country 5 ------------------------------------------
    country = countries[4]
    a_obs = df[df['country'] == country]['hiv_aid']
    a_obs = a_obs.iloc[0]
    if a_obs - a_sig > 0:
        a = np.arange(a_obs - a_sig, a_obs + a_sig, 3)
    else:
        a = np.arange(0, a_obs + a_sig, 3)
    y = np.empty(len(a))
    for i in range(len(a)):
        country_ind = np.where(df['country'] == country)[0]
        y[i] = GPS_pred(np.concatenate(([[a[i]]], df_baes[country_ind, 2:, 0]), axis=1) , b_coef[:,0:1], sig_hat[:,0:1], a_coef[:,0:1]) * 100

    # Bounds
    [Q_plus, Q_minus] = compute_bounds(y, eps_hat*10000, gammas)
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Plot bounds as dotted lines
        axs[1, 1].plot(a, Q_plus[:, i], color=bound_colors[i], linestyle='dashed')
        axs[1, 1].plot(a, Q_minus[:, i], color=bound_colors[i], linestyle='dashed')
    axs[1, 1].plot(a, y, label=country, **next(color_cycle))
    y_min = Q_minus[:, -1].min() - 3
    y_max = Q_plus[:, -1].max() + 3
    axs[1, 1].vlines(x=df.loc[df['country'] == country, 'hiv_aid'], ymin=y_min, ymax=y_max,
                     colors='black', linestyle='dashed')
    axs[1, 1].set_ylim([y_min, y_max])

    # Country 6 ------------------------------------------
    country = countries[5]
    a_obs = df[df['country'] == country]['hiv_aid']
    a_obs = a_obs.iloc[0]
    if a_obs - a_sig > 0:
        a = np.arange(a_obs - a_sig, a_obs + a_sig, 3)
    else:
        a = np.arange(0, a_obs + a_sig, 3)
    y = np.empty(len(a))
    for i in range(len(a)):
        country_ind = np.where(df['country'] == country)[0]
        y[i] = GPS_pred(np.concatenate(([[a[i]]], df_baes[country_ind, 2:, 0]), axis=1) , b_coef[:,0:1], sig_hat[:,0:1], a_coef[:,0:1]) * 100

    # Bounds
    [Q_plus, Q_minus] = compute_bounds(y, eps_hat*10000, gammas)
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Plot bounds as dotted lines
        axs[1, 2].plot(a, Q_plus[:, i], color=bound_colors[i], linestyle='dashed')
        axs[1, 2].plot(a, Q_minus[:, i], color=bound_colors[i], linestyle='dashed')
    axs[1, 2].plot(a, y, label=country, **next(color_cycle))
    y_min = Q_minus[:, -1].min() - 3
    y_max = Q_plus[:, -1].max() + 3
    axs[1, 2].vlines(x=df.loc[df['country'] == country, 'hiv_aid'], ymin=y_min, ymax=y_max,
                     colors='black', linestyle='dashed')
    axs[1, 2].set_ylim([y_min, y_max])


    # Create figure
    fig.text(0.5, 0.04, 'Development aid (in USD millions)', ha='center', va='center', size=18)
    fig.text(0.06, 0.5, "Relative reduction in HIV infection rate (in %)", ha='center', va='center',
             rotation='vertical', size=18)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(0.92, 0.65), loc='upper left')
    plt.savefig("causal_sensitivity.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()




