import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def tetramer_model(x, cb, mb, ct, mt, at, pKD1, P, dH1, dS1, dS2_factor, dS3_factor, L, logalpha, pKi, beta, loggamma):
    # Unit conversions
    T = x + 273.15
    KD1 = 10 ** (-pKD1)
    Ki = 10 ** (-pKi)
    alpha = 10 ** (logalpha)
    gamma = 10 ** (loggamma)

    bottom = cb + x * mb
    top = ct + x * mt + at * x**2

    dS2f = dS1 - dS2_factor
    dS3f = dS2f - dS3_factor

    KD2 = KD1 * alpha
    KD1_obs = KD1 * (1 + L / Ki) / (1 + L / (beta * beta * Ki))
    KD2_obs = KD2 * ((1 + L / (beta * Ki)) / (1 + L / (alpha * gamma * Ki)))
    Median_KD = np.sqrt(KD1_obs * KD2_obs)

    FL = (((2 * P + KD1_obs) - (((-2 * P - KD1_obs) ** 2) - 4 * P**2) ** 0.5) / 2) / P
    FT = 1 - (
        (((2 * P + KD2_obs) - (((-2 * P - KD2_obs) ** 2) - 4 * P**2) ** 0.5) / 2) / P
    )

    Theta_Dimer = ((1 - FL) + (1 - FT)) / (((1 - FT) / FT) + ((1 - FL) / FL))
    Conc_dept_affinity = KD1_obs + (Median_KD - KD1_obs) * (P / (Median_KD + P))

    NM = (
        (2 * P + Conc_dept_affinity)
        - (((-2 * P - Conc_dept_affinity) ** 2) - 4 * P**2) ** 0.5
    ) / 4

    Tetramer = (
        (2 * NM + KD2_obs) - (((-2 * NM - KD2_obs) ** 2) - 4 * NM**2) ** 0.5
    ) / 4
    Theta_tetramer = Tetramer / (P * 0.25)
    Theta_monomer = 1 - Theta_Dimer - Theta_tetramer

    dS1_obs = dS1 - (8.31 * np.log(1 + L / Ki))
    dS2_obs = dS2f - (8.31 * np.log(1 + (L / (beta * Ki))))
    dS3_obs = dS3f - (8.31 * np.log(1 + (L / (gamma * Ki))))

    dG_monomer = dH1 - T * dS1_obs
    dG_dimer = dH1 - T * dS2_obs
    dG_tetramer = dH1 - T * dS3_obs

    Ku_monomer = np.exp(dG_monomer / (8.31 * T))
    Ku_dimer = np.exp(dG_dimer / (8.31 * T))
    Ku_tetramer = np.exp(dG_tetramer / (8.31 * T))

    Y = (
        bottom
        + (Theta_monomer * ((top - bottom) / (1 + Ku_monomer)))
        + (Theta_Dimer * ((top - bottom) / (1 + Ku_dimer)))
        + (Theta_tetramer * ((top - bottom) / (1 + Ku_tetramer)))
    )
    return Y


def tetramer_linear(params, args):
    x = args[0]
    L = args[2]
    P = args[3]

    [pKD1, dH1, dS1, dS2_factor, dS3_factor, logalpha, pKi, beta, loggamma, cb, mb, ct, mt, at] = params

    T = x + 273.15
    KD1 = 10 ** (-pKD1)
    Ki = 10 ** (-pKi)
    alpha = 10 ** (logalpha)
    gamma = 10 ** (loggamma)

    bottom = cb + x * mb
    top = ct + x * mt + at * x**2

    dS2f = dS1 - dS2_factor
    dS3f = dS2f - dS3_factor

    KD2 = KD1 * alpha
    KD1_obs = KD1 * (1 + L / Ki) / (1 + L / (beta * beta * Ki))
    KD2_obs = KD2 * ((1 + L / (beta * Ki)) / (1 + L / (alpha * gamma * Ki)))
    Median_KD = np.sqrt(KD1_obs * KD2_obs)

    FL = (((2 * P + KD1_obs) - (((-2 * P - KD1_obs) ** 2) - 4 * P**2) ** 0.5) / 2) / P
    FT = 1 - (
        (((2 * P + KD2_obs) - (((-2 * P - KD2_obs) ** 2) - 4 * P**2) ** 0.5) / 2) / P
    )

    Theta_Dimer = ((1 - FL) + (1 - FT)) / (((1 - FT) / FT) + ((1 - FL) / FL))
    Conc_dept_affinity = KD1_obs + (Median_KD - KD1_obs) * (P / (Median_KD + P))

    NM = (
        (2 * P + Conc_dept_affinity)
        - (((-2 * P - Conc_dept_affinity) ** 2) - 4 * P**2) ** 0.5
    ) / 4

    Tetramer = (
        (2 * NM + KD2_obs) - (((-2 * NM - KD2_obs) ** 2) - 4 * NM**2) ** 0.5
    ) / 4
    Theta_tetramer = Tetramer / (P * 0.25)
    Theta_monomer = 1 - Theta_Dimer - Theta_tetramer

    dS1_obs = dS1 - (8.31 * np.log(1 + L / Ki))
    dS2_obs = dS2f - (8.31 * np.log(1 + (L / (beta * Ki))))
    dS3_obs = dS3f - (8.31 * np.log(1 + (L / (gamma * Ki))))

    dG_monomer = dH1 - T * dS1_obs
    dG_dimer = dH1 - T * dS2_obs
    dG_tetramer = dH1 - T * dS3_obs

    Ku_monomer = np.exp(dG_monomer / (8.31 * T))
    Ku_dimer = np.exp(dG_dimer / (8.31 * T))
    Ku_tetramer = np.exp(dG_tetramer / (8.31 * T))

    Y = (
        bottom
        + (Theta_monomer * ((top - bottom) / (1 + Ku_monomer)))
        + (Theta_Dimer * ((top - bottom) / (1 + Ku_dimer)))
        + (Theta_tetramer * ((top - bottom) / (1 + Ku_tetramer)))
    )

    return Y


def tetramer_linear_global(params, *args):
    global_params = params[0:9]
    total_cost = []

    param_sets = {}

    if len(args) == 4:
        t = [0]
    else:
        t = traces

    for i, trace in enumerate(t):
        # Deconstruct the parameters from the long list of params and args
        lower_param = 5 * i + 9
        upper_param = 5 * (i + 1) + 9
        lower_constant = 4 * i
        upper_constant = 4 * (i + 1)
        param_sets[trace] = {
            "local_params": params[lower_param:upper_param],
            "constants": args[lower_constant:upper_constant],
        }
    for trace in param_sets:
        all_params = [*global_params, *param_sets[trace]["local_params"]]
        result = tetramer_linear(all_params, args=param_sets[trace]["constants"])
        y_experimental = param_sets[trace]["constants"][1]
        cost = result - y_experimental
        total_cost = [*total_cost, *cost]

    return total_cost


traces = [1.25e-05, 0.0001, 3.13e-06, 0.0, 0.0002, 5e-05, 2.5e-05, 6.25]
