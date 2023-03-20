import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from data.models import tetramer_linear, tetramer_linear_global, tetramer_model
from utils.pso_lib import PSOFit
from utils.utils import (
    experimental_values,
    generate_results_to_store,
    get_final_results,
    get_global_bounds_from_min_max,
    get_global_traces_constants_config,
    get_linear_starting_point_from_pso_results,
    get_local_trace_configuration,
    get_points_for_global_lgd_fit,
    get_x_y_values_for_concentration,
    global_constants,
    initial_parameters,
    local_trace_parameters,
    pivot_zero_trace_results,
    plot_lgd_results,
)


def run_pso(run_number: int, save_outputs: bool = False) -> int:
    """
    Runs the fitting model. There are 5 fitting steps in total:
    - FIT 1: PSO fit on zero (inhibitor concentration = 0.0) trace.
    - FIT 2: LGD (Linear gradient descent) is performed on the zero trace to refine fitting parameters
    - FIT 3: PSO fit on individual traces (i.e., each inhibitor concentration) to derive local parameters
    - FIT 4: PSO fit on global parameters for all traces
    - FIT 5: LGD refines all variables on all traces

    :param run_number: The number of the run. Used when storing output csv files and images. If not doing multiple
    runs, this function will only be run once.
    :param save_outputs: Boolean to determine whether to store the csv
    results and plots to disk.
    :return: int - overall cost of the final run
    """

    # Take the time the run begins to create the filepath.
    time = datetime.datetime.strftime(datetime.datetime.now(), "%d_%m_%Y %H:%M:%S")

    """
    1. FIT 1: PSO fit on 0.0 concentration trace.

    First fit zero trace (protein only, no inhibitor) to get initial values for the parameters.
    """
    x_zero_trace, y_zero_trace = get_x_y_values_for_concentration(0.0)
    zero_trace_input = {
        0.0: {
            "x": x_zero_trace,
            "y": y_zero_trace,
            "constants": {"L": 0.0},
        }
    }

    # For other applications, investigate adjusting the PSO factors (swarm_size, hyperparameters) for different results.
    fit_0 = PSOFit(
        model=tetramer_model,
        traces=zero_trace_input,
        parameters=initial_parameters,
        global_constants=global_constants,
        swarm_size=1000,
        iterations=400,
        hyperparameters={"c1": 4.05, "c2": 0.05, "w": 0.72984},
    )
    fit_0.fit()

    # Store the results from the fit to be used in the next fitting step.
    zero_trace_results = {0.0: fit_0[0].get_solution()}

    """
    FIT 2: Linear Gradient Descent (LGD) on zero trace

    Refine the zero trace PSO fit using least squares regression. (scipy optimize)
    The results of this fit will be used to set constants and bounds in step 3.
    """

    # Format the results of the FIT 1 PSO run to appropriate inputs for the LGD step.
    (zero_param_guess, zero_args, zero_lower, zero_upper) = get_linear_starting_point_from_pso_results(
        starting_parameters=initial_parameters,
        previous_results=zero_trace_results,
        x_data_points=x_zero_trace,
        y_data_points=y_zero_trace,
    )

    result_0_trace_lgd = opt.least_squares(
        fun=tetramer_linear_global,
        x0=zero_param_guess,
        args=zero_args,
        bounds=(zero_lower, zero_upper),
    )
    print("Linear fit zero trace cost", result_0_trace_lgd.cost)

    """
    3. FIT 3: PSO runs on all other traces.

    Run PSO on all other traces individually - the ligand concentration is varied in each case.
    The purpose of this step is to get a best-guess approach for each trace's local parameters.
    These are the linear/quadratic constants for each trace (ct, mt, cb, mb, at).

    The variables dependent on concentration (pKi, beta, loggamma) are also included as fitting parameters here.

    The variables that are not dependent on ligand concentration (pKD1, dH1, dS1, dS2_factor, dS3_factor, logalpha) 
    are held constant. The values for these constants are taken from the zero trace fits (FIT 2).
    """

    zero_trace_results_linear = pivot_zero_trace_results(result_0_trace_lgd.x)
    local_results_pso = {}

    # Run the PSO fit on each trace
    for concentration in -np.sort(-experimental_values["conc"].unique()):
        # Get the values to pass into the PSO fit
        local_trace = get_local_trace_configuration(zero_trace_results_linear, float(concentration))

        local_fit = PSOFit(
            model=tetramer_model,
            traces=local_trace,
            parameters=local_trace_parameters,
            global_constants=global_constants,
            swarm_size=500,
            iterations=300,
            hyperparameters={"c1": 4.05, "c2": 0.01, "w": 0.72984},
        )
        local_fit.fit()
        local_results_pso[local_fit[0].trace_constants["L"]] = local_fit[0].get_solution()

    """
    4. FIT 4: PSO global fit on ALL traces

    Run PSO on all traces together.
    Get a best guess for the global variables: 'pKi', 'beta', 'loggamma'
    Get the bounds from the maximum and minimum derived values of these variables from the results of the last step.
    """

    global_fit_traces = get_global_traces_constants_config(local_results_pso, zero_trace_results_linear)
    parameters_for_global_fit = get_global_bounds_from_min_max(local_results_pso)

    global_fit = PSOFit(
        model=tetramer_model,
        traces=global_fit_traces,
        parameters=parameters_for_global_fit,
        global_constants=global_constants,
        swarm_size=400,
        iterations=400,
        hyperparameters={"c1": 3.05, "c2": 0.01, "w": 0.72984},
    )

    global_fit.fit()
    global_fit.plot_fit(f"Plots/{run_number} Global PSO {time}", save_outputs)
    plt.show()

    """
    5. FIT 5: Optimize all variables on all traces using LGD.

    Take final Global PSO results and prepare as starting point guesses for the final Global LGD step
    """

    pso_based_traces = {}
    for trace in global_fit:
        pso_based_traces[trace.trace_constants["L"]] = {**trace.trace_constants,**trace.get_solution()}

    # Generate the configuration for the LGD fit
    _all_params, _args, _lower, _upper = get_points_for_global_lgd_fit(pso_based_traces, initial_parameters)

    result = opt.least_squares(
        tetramer_linear_global,
        _all_params,
        args=_args,
        bounds=(_lower, _upper),
    )
    global_lgd_results = result.x

    optimized_global_vars = global_lgd_results[0:9]

    # Get printable summary of params
    final_results, param_sets = get_final_results(optimized_global_vars, pso_based_traces, global_lgd_results, _args)

    # Plot results of LGD fit
    plot_lgd_results(
        global_pso_results=pso_based_traces,
        global_variable_global_lgd_results=optimized_global_vars,
        local_params_per_trace=param_sets,
        run_time=time,
        model=tetramer_linear,
        run_number=run_number,
        save_fig=False,
    )

    # Printing the results of each fit. To store in csv format, enable outputs below.
    print(json.dumps(final_results, indent=4))

    if save_outputs:
        generate_results_to_store(
            run_number=run_number,
            zero_trace_results_pso=zero_trace_results,
            zero_trace_results_linear=zero_trace_results_linear,
            individual_trace_results_pso=local_results_pso,
            global_pso_results=pso_based_traces,
            global_lgd_final_results=final_results,
            global_pso_cost=global_fit.cost,
            global_lgd_cost=result.cost,
        )

    return result.cost


if __name__ == "__main__":
    # To run multiple times to get a picture of performance/success rate, increase the number of runs.
    # By default this will only run the optimization once.
    NUMBER_OF_RUNS = 1

    # Set to True to save results csv export and plots to file
    SAVE_TO_DISK = False

    for i in range(NUMBER_OF_RUNS):
        run_pso(i, SAVE_TO_DISK)

    quit()
