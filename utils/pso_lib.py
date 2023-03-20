from numbers import Number
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyswarms.single import GlobalBestPSO


class PSOFit:
    """Perform PSO global fit of a provided model"""

    def __init__(
        self,
        model: callable,
        traces: dict,
        parameters: dict,
        x: np.ndarray = None,
        hyperparameters=None,
        end_hyperparameters=None,
        hyperparameter_options=None,
        swarm_size: int = 100,
        iterations: int = 100,
        global_constants: dict = None,
    ):

        # Set PSO (Pyswarms) customisations
        self.hyperparameters: dict = (
            hyperparameters
            if hyperparameters is not None
            else {"c1": 1.5, "c2": 1.5, "w": 0.5}
        )
        self.end_hyperparameters: dict = end_hyperparameters
        self.hyperparameter_options = hyperparameter_options
        self.swarm_size: int = swarm_size
        self.iterations: int = iterations

        self.shared_x: Optional[np.array] = x  # shared x data points, optional

        self.model: callable = model  # The fitting model, as a function.

        self.global_constants = (
            global_constants  # Constants applicable to all traces in the fit
        )

        # IDs of the local parameters
        self.local_parameters, self.global_parameters = self._populate_parameters(
            parameters
        )  # All parameters, including a local parameter for each trace.

        self.traces: List[Trace] = self._populate_traces(traces)
        self._experimental_data = (
            self._aggregate_experimental_data()
        )  # All raw points as one sequential array (in the order of self.traces)
        self.dimensions = (len(self.local_parameters) * len(self.traces)) + len(
            self.global_parameters
        )  # Total number of parameters

        self.all_x = self.get_all_x()
        self.stacked_x = self.get_stacked_x()
        self.reshaped_x = self.reshape_x()

        self.all_var_names = self._generate_all_var_names()

        self.bounds: Tuple[
            np.ndarray, np.ndarray
        ] = self._calculate_bounds()  # PSO bounds

        self.swarm_history = []  # Holds diagnostic info dfs for each iteration

        self.cost_history = []  # Pyswarms aggregate cost history object. (pyswarms.utils.plotters.plot_cost_history)
        self.cost = None
        self.solution = None

    def get_all_x(self) -> np.array:
        # return x points for all traces, stacked
        return np.hstack([trace.x for trace in self.traces])

    def get_stacked_x(self) -> np.array:
        # Return all x values, repeatedly stacked by swarm size
        return np.repeat(self.all_x, self.swarm_size)

    def reshape_x(self):
        # Rearrange and repeat x so that it can be broadcast across the swarm
        """
        x array in tabular representation:
        +--------------+------------+------------+-----+-------------------------+
        | x_position v | Particle_1 | Particle_2 | ... | Particle_1_{swarm_size} |
        +--------------+------------+------------+-----+-------------------------+
        | 1            |          5 |          5 | ... |                       5 |
        | 2            |         10 |         10 | ... |                      10 |
        | 3            |         20 |         20 | ... |                      20 |
        | ...          |        ... |        ... | ... |                     ... |
        | len_x        |        100 |        100 | ... |                     100 |
        +--------------+------------+------------+-----+-------------------------+
        """
        return np.reshape(self.stacked_x, (len(self.all_x), self.swarm_size))

    def _aggregate_experimental_data(self):
        all_data = np.empty((0, 1))
        for trace in self.traces:
            all_data = np.append(all_data, trace.y)
        return all_data

    def fit(self):
        # Call an instance of PSO
        optimizer = GlobalBestPSO(
            n_particles=self.swarm_size,
            dimensions=self.dimensions,
            options=self.hyperparameters,
            bounds=self.bounds,
            oh_strategy=self.hyperparameter_options,
        )

        # Perform optimization
        self.cost, self.solution = optimizer.optimize(
            self.opt_function, iters=self.iterations
        )
        self.cost_history = optimizer.cost_history

    def opt_function(self, p):
        model_data = self.model_staging(p, self.reshaped_x)
        real_data = self._experimental_data.reshape((len(self._experimental_data), 1))

        rss = np.sum(np.square(model_data - real_data), axis=0)

        # df to hold the Swarm characteristics for this iteration
        swarm_df = pd.DataFrame(p)
        swarm_df["rss"] = rss
        swarm_df["particle"] = swarm_df.index + 1
        swarm_df["iteration"] = len(self.swarm_history) + 1
        self.swarm_history.append(swarm_df)

        return rss

    def _generate_all_var_names(self):
        global_names = [f"*{x.name}" for x in self.global_parameters]

        local_names = []

        for trace in self.traces:
            trace_local = [f"{x.name} - {trace.id}" for x in self.local_parameters]
            local_names = local_names + trace_local

        return global_names + local_names

    @property
    def points_history(self):
        all_p_cost = pd.concat(self.swarm_history, ignore_index=True)

        names = self.all_var_names

        renamedict = dict(zip([i for i in range(len(names))], names))

        all_p_cost.rename(columns=renamedict, inplace=True)

        return all_p_cost

    def model_staging(self, X, x):
        """Stages the swarm data ready for the model. This is what handles the ability to use arrays and Numbers

        x = x-values, either shaped for the array or to be be shaped
        X = swarm array in format:

        +------------+---------+---------+-----+--------------------+
        | Particle v | Param_1 | Param_2 | ... | Param_{num_params} |
        +------------+---------+---------+-----+--------------------+
        | 1          |       6 |      12 | ... |                 25 |
        | 2          |       3 |       8 | ... |                 15 |
        | 3          |       8 |       4 | ... |                 21 |
        | ...        |     ... |     ... | ... |                ... |
        | swarm_size |       7 |       6 | ... |                 29 |
        +------------+---------+---------+-----+--------------------+
        """

        # If X requires re-shaping to match the swarm (e.g. is 0d array)
        if len(x.shape) == 1:
            x = np.reshape(x, (len(x), X.shape[0]))

        # set global parameters
        global_params = {
            parameter.name: X[:, i]
            for i, parameter in enumerate(self.global_parameters)
        }

        # Blank array to build up for each trace.
        swarm_iteration_array = np.ndarray((0, X.shape[0]))

        # Run the model for each trace
        for trace in self.traces:
            local_params = {
                parameter.name: X[:, i + trace.local_params_index[0]]
                for i, parameter in enumerate(self.local_parameters)
            }

            start, end = trace.positional_index
            x_trace = x[start:end]

            result = self.model(
                x_trace,
                **global_params,
                **local_params,
                **self.global_constants,
                **trace.trace_constants,
            )

            swarm_iteration_array = np.append(swarm_iteration_array, result, axis=0)

        # One long array of the results for each trace, sequentially
        return swarm_iteration_array

    def _populate_parameters(self, parameters):
        global_parameters = []
        local_parameters = []

        # switch to single iteration of params and move the local params (or the lookup index) to the traces
        for name, properties in parameters.items():
            kwargs = {
                "name": name,
                "upper_bound": properties["u"],
                "lower_bound": properties["l"],
            }

            if "rel" in properties:
                kwargs["relative_to"] = properties["rel"]

            if "scope" in properties:
                kwargs["scope"] = properties["scope"]
                global_parameters.append(Parameter(**kwargs))
            else:
                local_parameters.append(Parameter(**kwargs))

        return local_parameters, global_parameters

    def _populate_traces(self, traces):
        trace_objects = []
        existing_ids = []
        end_index = 0
        local_params_end_index = len(self.global_parameters)

        for trace in traces.items():
            id, properties = trace

            if id in existing_ids:
                raise ValueError(f"Non-unique trace id: {id}")
            else:
                if "x" not in properties.keys():
                    if self.shared_x is None:
                        raise ValueError(
                            f"Trace ({id}) has no x values, and no shared x is defined"
                        )
                    else:
                        properties["x"] = self.shared_x

                start_index = end_index
                end_index = start_index + len(properties["y"])

                local_params_start_index = local_params_end_index
                local_params_end_index = local_params_start_index + len(
                    self.local_parameters
                )

                trace_object = Trace(
                    parent_fit=self,
                    id=id,
                    positional_index=(start_index, end_index),
                    local_params_index=(
                        local_params_start_index,
                        local_params_end_index,
                    ),
                    **properties,
                )

                trace_objects.append(trace_object)
                existing_ids.append(id)

        return trace_objects

    def _calculate_bounds(self):
        bounds_dict = {"upper": [], "lower": []}

        def set_bounds(lower_bound, upper_bound):
            bounds_dict["upper"].append(upper_bound)
            bounds_dict["lower"].append(lower_bound)

        for global_parameter in self.global_parameters:
            set_bounds(global_parameter.lower_bound, global_parameter.upper_bound)

        for trace in self.traces:
            for local_parameter in self.local_parameters:
                if local_parameter.relative_to is None:
                    set_bounds(local_parameter.lower_bound, local_parameter.upper_bound)
                elif local_parameter.relative_to == "y_max":
                    set_bounds(
                        local_parameter.lower_bound + trace.y.max(),
                        local_parameter.upper_bound + trace.y.max(),
                    )
                elif local_parameter.relative_to == "y_min":
                    set_bounds(
                        local_parameter.lower_bound + trace.y.min(),
                        local_parameter.upper_bound + trace.y.min(),
                    )
                else:
                    raise ValueError(
                        f"Invalid relative to value: {local_parameter.relative_to}"
                    )

        return np.array(bounds_dict["lower"]), np.array(bounds_dict["upper"])

    def _get_all_parameters(self):
        all_params = self.global_parameters.copy()

        for i, trace in enumerate(self.traces):
            all_params = all_params + [f"{parameter}_{i}" for parameter in self.local_parameters]

        return all_params

    def _create_fit_plot(self, plot_name, save_fig=False):
        colours = ["b", "g", "r", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k"]

        # Plot raw data vs fitted curve
        for i, trace in enumerate(self.traces):
            plt.plot(
                trace.x,
                trace.y,
                linestyle="dashed",
                linewidth=0.5,
                color=colours[i],
                label=trace.id,
                alpha=0.5,
            )

        y_calced = self.model_staging(
            self.solution.reshape((1, len(self.solution))), self.all_x
        )

        for i, trace in enumerate(self.traces):
            # Determine the window of the complete data array which corresponds to this trace
            start, stop = trace.positional_index

            plt.plot(trace.x, y_calced[start:stop], linewidth=0.75, color=colours[i])

        plt.legend(loc="upper left")
        plt.title(f"Global PSO fit before LGD refinement.")
        if save_fig:
            plt.savefig(f"{plot_name}.png")

    def plot_fit(self, plot_name, save_plot=False):
        self._create_fit_plot(plot_name, save_plot)

    def __len__(self):
        return len(self.traces)

    def __iter__(self):
        for trace in self.traces:
            yield trace

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.traces[key]
        elif isinstance(key, str):
            for x in self.traces:
                if x.id == key:
                    return x
            else:
                raise KeyError(f"No trace with id '{key}'")


class Trace:
    """Trace object, holds trace-specific data"""

    def __init__(
        self,
        parent_fit: PSOFit,
        id: str,
        x: np.ndarray,
        y: np.ndarray,
        positional_index: Tuple[int, int],
        local_params_index: Tuple[int, int],
        constants: dict = None,
    ):
        self.parent_fit = parent_fit
        self.id: str = id  # A human readable identifier for the trace
        self.y: np.ndarray = y  # Y data points
        self.x: np.ndarray = x  # X data points
        self.positional_index = positional_index
        self.local_params_index = local_params_index
        self.trace_constants: dict = {} if dict is None else constants

    def get_solution(self):
        solution = np.append(
            self.parent_fit.solution[: len(self.parent_fit.global_parameters)],
            self.parent_fit.solution[
                self.local_params_index[0]: self.local_params_index[1]
            ].tolist(),
        )
        param_names = [f"*{x.name}" for x in self.parent_fit.global_parameters] + [
            f"{x.name}" for x in self.parent_fit.local_parameters
        ]
        return dict(zip(param_names, solution))


class Parameter:
    """A single parameter, used in the model"""

    def __init__(
        self,
        name: str,
        upper_bound: Number,
        lower_bound: Number,
        relative_to: str = None,
        scope=False,
    ):
        self.name: str = name  # Human readable variable identifier. Must match the variable name in the model
        self.is_global: bool = scope  # Global variable if True, else Local

        self.relative_to: Optional[
            str
        ] = relative_to  # If the bounds are calculated based on a trace-specific property (Can expand to global later)

        if upper_bound < lower_bound:
            raise ValueError(
                f"upper bound for {name} must be larger than lower bound (u={upper_bound}, l={lower_bound})"
            )

        self.upper_bound: Number = upper_bound
        self.lower_bound: Number = lower_bound
