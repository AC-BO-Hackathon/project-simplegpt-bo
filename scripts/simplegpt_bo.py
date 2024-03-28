import pandas as pd
import numpy as np


def show_image(image_path):
    from PIL import Image
    import matplotlib.pyplot as plt

    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def generate_adjusted_sample_data(num_samples=50):
    np.random.seed(0)
    # Generate random indexes for temperatures and pressures
    # within specified ranges and step sizes

    # Possible temperature values
    possible_temperatures = np.arange(20, 101, 10)
    # Possible pressure values
    possible_pressures = np.arange(1, 10.1, 0.5)

    temperatures = np.random.choice(possible_temperatures, num_samples)
    pressures = np.random.choice(possible_pressures, num_samples)

    chemicals = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    yield_ = np.random.uniform(low=10, high=100, size=num_samples)
    contaminant = np.random.uniform(low=1, high=50, size=num_samples)

    # Adjusting the initial function to match the required stepwise selection
    data = pd.DataFrame({
        'Temperature': temperatures,
        'Pressure': pressures,
        'Chemical1_w%': chemicals[:, 0],
        'Chemical2_w%': chemicals[:, 1],
        'Chemical3_w%': chemicals[:, 2],
        'Yield': yield_,
        'Contaminant': contaminant
    })

    return data


def single_objective_transformation(
        df,
        optimization_specification,
        weights=None
):
    from sklearn.preprocessing import minmax_scale

    # Extract targets and their optimization directions
    targets = optimization_specification['targets']

    # If weights are not provided, assign equal weights
    if weights is None:
        n_targets = len(targets)
        weights = {target: 1.0 / n_targets for target in targets}

    # Normalize targets
    for target, direction in targets.items():
        if direction == 'maximize':
            # For maximization, a higher value is
            # better so we use the values as is
            df[target + '_norm'] = minmax_scale(df[target])
        else:
            # For minimization, a lower value is better so
            # we invert the normalized values
            df[target + '_norm'] = 1 - minmax_scale(df[target])

    # Calculate the combined objective
    df['combined_objective'] = sum(
        df[target + '_norm'] * weights[target] for target in targets
    )

    return df


def generate_pool_from_user_data_no_duplicates(
        user_data,
        explanatory_vars,
        max_pool_size=1000
):
    np.random.seed(0)  # For reproducibility
    variable_values = {}

    for var in explanatory_vars:
        if (
            user_data[var].dtype == 'float64' or
            user_data[var].dtype == 'int64'
        ):
            unique_vals = np.sort(user_data[var].unique())
            variable_values[var] = unique_vals
        else:
            variable_values[var] = user_data[var].unique()

    total_combinations = np.prod(
        [len(variable_values[var]) for var in explanatory_vars]
    )

    pool_set = set()
    while len(pool_set) < min(max_pool_size, total_combinations):
        combination = tuple(
            np.random.choice(variable_values[var]) for var in explanatory_vars
        )
        pool_set.add(combination)

    pool = list(pool_set)

    df_pool = pd.DataFrame(data=pool, columns=explanatory_vars)

    return total_combinations, df_pool


def run_bayesian_optimization(
        data,
        candidate_pool,
        target_column,
        acquisition_function='tpe',
        bandwidth=0.2,
        kappa=2.576):

    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        RBF,
        WhiteKernel
    )
    from sklearn.preprocessing import MinMaxScaler

    np.random.seed(0)
    feature_columns = candidate_pool.columns
    scaler = MinMaxScaler()
    observed_features = scaler.fit_transform(data[feature_columns].values)
    observed_target = data[target_column].values.flatten()
    candidate_features = scaler.transform(candidate_pool.values)

    # Define the kernel
    kernel = ConstantKernel() * RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(observed_features, observed_target)

    # Choose the acquisition function
    if acquisition_function == 'tpe':
        scores = perform_tpe(
            observed_features,
            observed_target,
            candidate_features,
            bandwidth=bandwidth
        )
    elif acquisition_function == 'ei':
        scores = perform_ei(
            gp,
            candidate_features
        )
    elif acquisition_function == 'pi':
        scores = perform_pi(
            gp,
            candidate_features
        )
    elif acquisition_function == 'ucb':
        scores = perform_ucb(
            gp,
            candidate_features,
            kappa=kappa
        )
    else:
        raise ValueError(
            "Unsupported acquisition function: " + acquisition_function
        )

    # Assign scores to all samples and return the DataFrame
    candidate_pool_with_scores = candidate_pool.assign(Scores=scores)
    return candidate_pool_with_scores


def perform_ei(gp, candidate_features, xi=0.01):
    from scipy.stats import norm
    mu, sigma = gp.predict(candidate_features, return_std=True)
    best_target = np.max(gp.y_train_)

    with np.errstate(divide='warn'):
        imp = mu - best_target - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei


def perform_pi(gp, candidate_features, xi=0.01):
    from scipy.stats import norm
    mu, sigma = gp.predict(candidate_features, return_std=True)
    best_target = np.max(gp.y_train_)

    with np.errstate(divide='warn'):
        Z = (mu - best_target - xi) / sigma
        pi = norm.cdf(Z)
        return pi


def perform_ucb(gp, candidate_features, kappa=2.576):
    mu, sigma = gp.predict(candidate_features, return_std=True)
    ucb = mu + kappa * sigma
    return ucb


def perform_tpe(
        observed_features,
        observed_target,
        candidate_features,
        bandwidth=0.2
):
    gamma = 0.9
    lower_size = int(len(observed_target) * gamma)

    lower_indices = np.argsort(observed_target)[:lower_size]
    upper_indices = np.argsort(observed_target)[lower_size:]

    lower_features = observed_features[
        lower_indices
    ].reshape(-1, observed_features.shape[1])
    upper_features = observed_features[
        upper_indices
    ].reshape(-1, observed_features.shape[1])

    kde_lower = fit_kde(lower_features, bandwidth=bandwidth)
    kde_upper = fit_kde(upper_features, bandwidth=bandwidth)

    lower_val = kde_lower.score_samples(candidate_features)
    upper_val = kde_upper.score_samples(candidate_features)

    score = upper_val - lower_val
    return score


def fit_kde(features, bandwidth=0.2):
    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(features)
    return kde


def get_top_k_samples(candidate_pool_with_scores, top_k=5):
    top_k_samples = candidate_pool_with_scores.sort_values(
        'Scores',
        ascending=False
    ).head(top_k)
    return top_k_samples
