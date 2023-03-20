import datetime
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Utilities for running the PSO fits

- Section 1: Extracting and defining the core datasets and initial bounds for the PSO fits
- Section 2: Reformatting the data between runs (e.g. Output from PSO fit -> input to LGD fit)
- Section 3: Plotting, exporting and storing data from runs.
- Section 4: General helper functions used by other utils
"""

"""
Section 1: Experimental data and bounds.

Extract and define the core datasets and initial bounds for the PSO fits
"""

# Read the experimental dataset into a dataframe
experimental_values = pd.read_csv("./data/experimental_data.tsv", delimiter="\t")

# Parameter bounds choice: These starting bounds are broad and based on physically realistic space
# (e.g. positive melting enthalpies, binding constants > 0).

# These are the shared parameters, used in the local trace fit and the initial zero trace fit
local_trace_parameters = {
    "cb": {"l": 5000, "u": 20000},
    "mb": {"l": -1000, "u": 1000},
    "ct": {"l": 10000, "u": 80000},
    "mt": {"l": -100, "u": 100},
    "at": {"l": -5, "u": 0},
    "pKi": {"l": 1, "u": 7, "scope": "global"},
    "beta": {"l": -2, "u": 2, "scope": "global"},
    "loggamma": {"l": -30, "u": 30, "scope": "global"},
}

# The initial zero trace fit uses the same starting parameters, plus the global variables.
initial_parameters = {
    **local_trace_parameters,
    "pKD1": {"l": 4, "u": 8, "scope": "global"},
    "dH1": {"l": 1e5, "u": 6e5, "scope": "global"},
    "dS1": {"l": 500, "u": 2000, "scope": "global"},
    "dS2_factor": {"l": 0, "u": 70, "scope": "global"},
    "dS3_factor": {"l": 0, "u": 70, "scope": "global"},
    "logalpha": {"l": -4, "u": 2, "scope": "global"},
}

# Protein concentration
global_constants = {"P": 3e-6}

list_of_params = [
    "*pKD1",
    "*dH1",
    "*dS1",
    "*dS2_factor",
    "*dS3_factor",
    "*logalpha",
    "*pKi",
    "*beta",
    "*loggamma",
    "cb",
    "mb",
    "ct",
    "mt",
    "at",
]

"""
Section 2: Reformatting data between runs

- Zero trace PSO output -> Zero trace LGD input (get_linear_starting_point_from_pso_results)
- Zero trace LGD output -> Linear individual run PSO input
    - list of values to trace based Dict result (pivot_zero_trace_results)
    - get the starting points for an individual trace PSO fit (get_local_trace_configuration)
- All prior fits output -> Global PSO fit input
    - get constants: Take the optimized local params from each individual trace and set as constants for the global PSO fit (get_global_traces_constants_config)
    - get bounds of varied param (get_global_bounds_from_min_max)
- All prior fits output -> Global LGD fit input
    - get the starting points for linear gradient descent fit (get_points_for_global_lgd_fit)
    - format final results from global LGD step (get_final_results)
"""


def get_linear_starting_point_from_pso_results(
    starting_parameters: Dict,
    previous_results: Dict,
    x_data_points: np.ndarray,
    y_data_points: np.ndarray,
) -> tuple:
    """
    Generate the starting point inputs for the zero trace linear gradient descent fit

    :param starting_parameters: Initial bounds and parameters in PSO fit
    :param previous_results: Results from the PSO fit
    :param x_data_points: raw x data for trace
    :param y_data_points: raw y data for trace
    :return: The input parameters to the LGD fit
        - The initial guess to begin the LGD fit on (the final values of the PSO fit)
        - The arguments to the fit - x, y, L, P
        - The lower bounds for the variables
        - The upper bounds for the variables
    """
    zero_param_guess = []
    lower = []
    upper = []

    for var in list_of_params:
        zero_param_guess.append(previous_results[0][var])

    for var in sanitize_param_names(list_of_params):
        lower.append(starting_parameters[var]["l"])
        upper.append(starting_parameters[var]["u"])

    # This is for the zero trace only
    L = 0.0
    P = global_constants["P"]

    args = [x_data_points, y_data_points, L, P]

    return zero_param_guess, args, lower, upper


def pivot_zero_trace_results(results: List[int]) -> Dict:
    """
    Pivots results from a previous fit to the correct format to be passed in as constants for a subsequent fit
    :param results: Results from zero trace LGD step
    :return: Zero trace results
    """

    return {
        0.0: {
            "*pKD1": results[0],
            "*dH1": results[1],
            "*dS1": results[2],
            "*dS2_factor": results[3],
            "*dS3_factor": results[4],
            "*logalpha": results[5],
            "*pKi": results[6],
            "*beta": results[7],
            "*loggamma": results[8],
            "cb": results[9],
            "mb": results[10],
            "ct": results[11],
            "mt": results[12],
            "at": results[13],
        }
    }


def get_local_trace_configuration(zero_trace_results: Dict, conc: float) -> Dict:
    """
    Get the configuration for an individual trace PSO fit

    :param zero_trace_results: The formatted results of the zero trace LGD step (from pivot_zero_trace_results)
    :param conc: The concentration of the trace of interest
    :return: Return the formatted input parameters for an individual trace PSO fit.
    """
    trace_id = "{:.2e}".format(conc)
    x, y = get_x_y_values_for_concentration(conc)

    input_values = {
        trace_id: {
            "x": x,
            "y": y,
            "constants": {
                "L": conc,
                "pKD1": zero_trace_results[0]["*pKD1"],
                "dH1": zero_trace_results[0]["*dH1"],
                "dS1": zero_trace_results[0]["*dS1"],
                "dS2_factor": zero_trace_results[0]["*dS2_factor"],
                "dS3_factor": zero_trace_results[0]["*dS3_factor"],
                "logalpha": zero_trace_results[0]["*logalpha"],
            },
        }
    }

    return input_values


def get_global_traces_constants_config(
    local_results_pso: Dict, zero_trace_results_linear: Dict
) -> Dict:
    """
    Take the optimized local params from each individual trace and set as constants for the global PSO fit

    :param local_results_pso: Each local fit result
    :param zero_trace_results_linear: The zero trace LGD result
    :return: All the constants for the global PSO fit
    """
    global_fit_traces = {}
    for concentration in -np.sort(-experimental_values["conc"].unique()):
        concentration = float(concentration)
        trace_id = "{:.2e}".format(concentration)
        x, y = get_x_y_values_for_concentration(concentration)

        #
        local_params = local_results_pso[concentration]
        global_fit_traces[trace_id] = {
            "x": x,
            "y": y,
            "constants": {
                "L": concentration,
                "cb": local_params["cb"],
                "mb": local_params["mb"],
                "ct": local_params["ct"],
                "mt": local_params["mt"],
                "at": local_params["at"],
                "pKD1": zero_trace_results_linear[0]["*pKD1"],
                "dH1": zero_trace_results_linear[0]["*dH1"],
                "dS1": zero_trace_results_linear[0]["*dS1"],
                "dS2_factor": zero_trace_results_linear[0]["*dS2_factor"],
                "dS3_factor": zero_trace_results_linear[0]["*dS3_factor"],
                "logalpha": zero_trace_results_linear[0]["*logalpha"],
            },
        }

    return global_fit_traces


def get_global_bounds_from_min_max(results: Dict) -> Dict:
    """
    Get the bounds of the global fit parameters from the maximum and minimum values of the previous results

    :param results:  Dict of results per trace
    :return: formatted Dict containing parameters for Global PSO fit.
    """
    parameters = {}
    for param in ["pKi", "beta", "loggamma"]:
        parameters[param] = {
            "l": find_bounds_from_max_and_min_values(f"*{param}", results, "min"),
            "u": find_bounds_from_max_and_min_values(f"*{param}", results, "max"),
            "scope": "global",
        }

    return parameters


def get_points_for_global_lgd_fit(
    pso_based_traces: Dict, starting_parameters: Dict
) -> tuple:
    """
    Returns the starting guess for the final global linear gradient descent fit.
    The parameters are taken from the appropriate global/local fits from previous steps

    :param pso_based_traces: All the individual PSO traces (FIT 3)
    :param starting_parameters: Starting parameters of the original fit
    :return: Parameters ready for the LGD fit
    """
    local_params = []
    args = []
    full_list_params = []

    # These values are global and the same in every trace so we can take them from any trace
    for var in [
        "pKD1",
        "dH1",
        "dS1",
        "dS2_factor",
        "dS3_factor",
        "logalpha",
        "*pKi",
        "*beta",
        "*loggamma",
    ]:
        full_list_params.append(pso_based_traces[0.0001][var])

    for trace in pso_based_traces:
        trace_local_params = [pso_based_traces[trace][var] for var in ["cb", "mb", "ct", "mt", "at"]]
        local_params = [*local_params, *trace_local_params]

        x, y = get_x_y_values_for_concentration(trace)
        L = pso_based_traces[trace]["L"]
        P = global_constants["P"]
        args = [*args, x, y, L, P]

    full_list_params = [*full_list_params, *local_params]

    lower = []
    upper = []
    for var in [
        "pKD1",
        "dH1",
        "dS1",
        "dS2_factor",
        "dS3_factor",
        "logalpha",
        "pKi",
        "beta",
        "loggamma",
    ]:
        lower.append(starting_parameters[var]["l"])
        upper.append(starting_parameters[var]["u"])

    locals_low = []
    locals_up = []

    for var in ["cb", "mb", "ct", "mt", "at"]:
        locals_low.append(starting_parameters[var]["l"])
        locals_up.append(starting_parameters[var]["u"])

    # We pass in the same bounds for the local variables (ct, cb etc) for all traces
    # They are the same for all 8 traces
    lower = [*lower, *locals_low * 8]
    upper = [*upper, *locals_up * 8]

    return full_list_params, args, lower, upper


def get_final_results(
    global_vars: np.ndarray, traces: Dict, lgd_result: np.ndarray, args: List
) -> tuple:
    """
    Format the final results. Helps to determine which param is which result from the scipy List output.

    :param global_vars: Global variable results
    :param traces: Traces to iterate through
    :param lgd_result: Result of the LGD fit
    :param args: Args from the previous fit to determine constants

    :return: Formatted dict of final results, by global variables and per trace local variables
    e.g.
    {
    "global_parameters": {
        "0": {
            "*pKD1": 5.104076379393138,
            "*dH1": 307452.3415254857,
            "*dS1": 947.6358757635616,
            ...rest of global params
        }
    },
    "local_parameters": {
        "0.0002": {
            "cb": 17207.02994685999,
            "mb": -111.33510317958078,
            "ct": 36687.81545603862,
            "mt": 99.98018813298287,
            "at": -3.674301667007981
        },
        "0.0001": {
          ...
        }
        ... rest of traces
    }
    """

    final_results = {
        "global_parameters": {
            0: {
                "*pKD1": global_vars[0],
                "*dH1": global_vars[1],
                "*dS1": global_vars[2],
                "*dS2_factor": global_vars[3],
                "*dS3_factor": global_vars[4],
                "*logalpha": global_vars[5],
                "*pKi": global_vars[6],
                "*beta": global_vars[7],
                "*loggamma": global_vars[8],
            },
        },
        "local_parameters": {},
    }

    param_sets = {}
    for i, trace in enumerate(traces):
        # Deconstruct the parameters from the long list of params and args
        lower_param = 5 * i + 9
        upper_param = 5 * (i + 1) + 9
        lower_constant = 4 * i
        upper_constant = 4 * (i + 1)
        param_sets[trace] = {
            "local_params": lgd_result[lower_param:upper_param],
            "constants": args[lower_constant:upper_constant],
        }

        final_results["local_parameters"][trace] = {
            "cb": lgd_result[lower_param],
            "mb": lgd_result[lower_param + 1],
            "ct": lgd_result[lower_param + 2],
            "mt": lgd_result[lower_param + 3],
            "at": lgd_result[lower_param + 4],
        }

    return final_results, param_sets


"""
Section 3: plotting and storing data
"""


def plot_lgd_results(
    global_pso_results: Dict,
    global_variable_global_lgd_results: Dict,
    local_params_per_trace: Dict,
    run_time: str,
    model: Callable,
    run_number: int,
    save_fig: bool = False,
) -> str:
    """
    Plots the Global LGD fit, and stores the plots if requested
    :param global_pso_results:  Results of FIT 4: Global PSO fit
    :param global_variable_global_lgd_results: Global variable results of FIT 5: Global LGD fit
    :param local_params_per_trace: The local parameters from FIT 5, arranged by trace
    :param run_time: The time the run began, for cross referencing images against runs
    :param model: The model to be plotted against
    :param run_number: The number of the run, for cross referencing images against runs
    :param save_fig: Whether to store the image to disk
    :return: The name of the plot
    """
    for i, trace in enumerate(global_pso_results):
        colours = ["b", "g", "r", "c", "m", "royalblue", "limegreen", "orange"]
        trace_id = "{:.2e}".format(trace)
        x, y = get_x_y_values_for_concentration(trace)
        # Plot the raw data as dashed lines
        plt.plot(
            x, y, linestyle="dashed", linewidth=0.75, color=colours[i], label=trace_id
        )
        # Plot the fitted data using the model generated y values
        plt.plot(
            x,
            model(
                [
                    *global_variable_global_lgd_results,
                    *local_params_per_trace[trace]["local_params"],
                ],
                args=local_params_per_trace[trace]["constants"],
            ),
            linewidth=0.75,
            color=colours[i],
        )
    plt.title(f"Final fit after LGD refinement")
    plt.legend(loc="upper left")

    plot_name = f"Plots/{run_number} Global linear {run_time}.png"

    if save_fig:
        plt.savefig(f"Plots/{run_number} Global linear {run_time}.png")
    plt.show()

    return plot_name


def generate_results_to_store(
    run_number: int,
    zero_trace_results_pso: Dict,
    zero_trace_results_linear: Dict,
    individual_trace_results_pso: Dict,
    global_pso_results: Dict,
    global_lgd_final_results: Dict,
    global_pso_cost: float,
    global_lgd_cost: float,
) -> str:
    """
    Get all results from all runs to store to csv.
    :param run_number: The iteration of the run. Helpful if running multiple times so the CSV export can be tied to the plots by this number
    :param zero_trace_results_pso: Results of FIT 1: Zero trace PSO
    :param zero_trace_results_linear:  Results of FIT 2: Zero trace LGD
    :param individual_trace_results_pso:  Results of FIT 3: Individual trace PSO
    :param global_pso_results: Results of FIT 4: Global PSO fit
    :param global_lgd_final_results: Results of FIT 5: Global LGD fit
    :param global_pso_cost: The total cost of FIT 4: Global PSO fit
    :param global_lgd_cost: The total cost of FIT 5: Global LGD fit
    :return: path to the filename created
    """
    now = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now, "%d_%m_%Y %H:%M:%S")

    filename = f'CSV Exports/{run_number} {time_str} Cost={"{:.2e}".format(global_lgd_cost)}.csv'

    row_zero_pso = all_params(zero_trace_results_pso, "zero_trace_pso")
    row_zero_trace_linear = all_params(zero_trace_results_linear, "zero_trace_linear")
    row_global_pso = all_params(global_pso_results, "global_pso")
    row_global_linear = get_global_results_as_series(global_lgd_final_results["global_parameters"], "global_linear")
    local_results = get_local_results_as_series(global_lgd_final_results["local_parameters"], "global_linear")
    row_global_linear = row_global_linear.append(local_results)
    pso_cost_series = pd.Series(
        {"cost": global_pso_cost, "cost_e": "{:.2e}".format(global_pso_cost)},
        name="global_pso",
    )
    row_global_pso = row_global_pso.append(pso_cost_series)
    linear_cost_series = pd.Series(
        {"cost": global_lgd_cost, "cost_e": "{:.2e}".format(global_lgd_cost)},
        name="global_linear",
    )
    row_global_linear = row_global_linear.append(linear_cost_series)

    new_df = pd.DataFrame([row_global_pso, row_global_linear, row_zero_pso, row_zero_trace_linear])

    individual_pso = {}
    for c in individual_trace_results_pso:
        data_set = {"pso": {c: individual_trace_results_pso[c]}}
        individual_pso[c] = all_params(data_set["pso"], f"{c}_individual_pso", c)
        new_df = new_df.append(individual_pso[c])

    new_df.to_csv(filename)

    return filename


"""
Section 4: General utils for other functions
"""


def find_bounds_from_max_and_min_values(
    param: str, results: Dict, max_or_min: str
) -> float:
    """
    Generates the bounds for a fit based on the highest or lowest derived values of a previous run
    :param param: the named parameter, e.g. *dH1
    :param results: Dict of results per trace
    :param max_or_min: String 'max' or 'min', depending which value needs to be returned
    :return: The maximum or minimum value for the input parameter across all traces
    """

    values = []
    for p in results.values():
        values.append(p[param])

    if max_or_min == "min":
        return min(values)

    if max_or_min == "max":
        return max(values)


def get_x_y_values_for_concentration(c: float) -> tuple:
    """
    Get the x and y experimental values for a trace at a particular concentration.
    :param c: Concentration of the trace desired from the experimental results
    :return: Two numpy arrays for the x and y coordinates of the experimental results for that trace.
    """
    x = experimental_values[experimental_values["conc"] == c]["temp"].to_numpy()
    y = experimental_values[experimental_values["conc"] == c]["fluorescence"].to_numpy()

    return x, y


def sanitize_param_names(params: List[str]) -> List[str]:
    """
    :param params: List of the parameter names to sanitise as strings e.g. ['dH1', '*dS1']
    :return: A list of parameter names, with any *'s removed e.g. ['dH1', 'dS1']
    """
    return [x.replace("*", "") for x in params]


def get_global_results_as_series(
    trace_results: Dict, series_name: str, conc: float = 0.0
):
    """
    Return the global result set as a pandas series
    """

    def get_value(obj, key):
        if key in obj:
            return obj[key]
        elif key[1:] in obj:
            return obj[key[1:]]
        else:
            return None

    return pd.Series(
        {
            "*pKD1": get_value(trace_results[conc], "*pKD1"),
            "*dH1": get_value(trace_results[conc], "*dH1"),
            "*dS1": get_value(trace_results[conc], "*dS1"),
            "*dS2_factor": get_value(trace_results[conc], "*dS2_factor"),
            "*dS3_factor": get_value(trace_results[conc], "*dS3_factor"),
            "*logalpha": get_value(trace_results[conc], "*logalpha"),
            "*pKi": get_value(trace_results[conc], "*pKi"),
            "*beta": get_value(trace_results[conc], "*beta"),
            "*loggamma": get_value(trace_results[conc], "*loggamma"),
        },
        name=series_name,
    )


def get_local_results_as_series(trace_results, series_name):
    """
    Return the local result set as a pandas series
    """
    series = pd.Series({}, name=series_name)
    for conc in trace_results:
        indiv_series = pd.Series(
            {
                f"cb{conc}": trace_results[conc]["cb"],
                f"mb{conc}": trace_results[conc]["mb"],
                f"ct{conc}": trace_results[conc]["ct"],
                f"mt{conc}": trace_results[conc]["mt"],
                f"at{conc}": trace_results[conc]["at"],
            },
            name=series_name,
        )
        series = series.append(indiv_series)
    return series


def all_params(result_set: Dict, name: str, conc: float = 0.0):
    """
    Return all params in series
    """
    row = get_global_results_as_series(result_set, name, conc)
    local_results = get_local_results_as_series(result_set, name)
    row = row.append(local_results)
    return row
