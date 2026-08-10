from typing import Optional, Any
import logging
from pathlib import Path
import time
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt


GLOBAL_LOGGER = logging.getLogger("BHM")


def build_hierarchical_model(
    raw_auoc_values: np.ndarray,
    library_names: list[str],
    variable_names: list[str],
    reference_variable: Optional[str] = None,
) -> tuple[az.InferenceData, dict]:
    """
    Builds a hierarchical Bayesian model on AUC data across different libraries and additional variables, assuming
    that every variable was repeated on a fixed set of seed blocks for each library.

    The model is formulated as follows (i denoting the library index, j denoting the variable index, and k
    denoting the seed block index):

    y_ijk ~ Beta(auc_logits_ijk * phi, (1 - auc_logits_ijk) * phi)
    auc_logits_ijk = sigmoid(intercept + library_effect_i + seed_effect_k + beta_j + library_variable_effect_ij)

    Args:
        raw_auoc_values (np.ndarray): A 3D array of shape (num_libraries, num_variables, num_seeds) containing
                                      the AUC values.
        library_names (list[str]): A list of library names corresponding to the first dimension of raw_auoc_values.
        variable_names (list[str]): A list of variable names corresponding to the second dimension of
                                    raw_auoc_values.
        reference_variable (Optional[str]): The variable to be used as the reference in the model. If None,
                                            the first variable in the list will be used as the reference.

    Returns:
        az.InferenceData: The inference data containing the posterior samples and diagnostics.
        dict: Metadata containing information about the libraries, variables, and seed blocks.
    """
    log_file_handler = logging.FileHandler(Path.cwd() / "hierarchical_bayesian_model.log")
    GLOBAL_LOGGER.addHandler(log_file_handler)
    GLOBAL_LOGGER.info(f"Starting to build a hierarchical Bayesian model with {len(library_names)} libraries, "
                       f"{len(variable_names)} variables, and {raw_auoc_values.shape[2]} seed blocks.")

    auc_values = raw_auoc_values.clip(1E-6, 1 - 1E-6)
    num_libraries, num_variables, num_seeds = auc_values.shape

    # Reorder variables to have the reference variable first
    reference = variable_names[0] if reference_variable is None else reference_variable
    reference_idx = variable_names.index(reference)
    variable_order = np.array([reference_idx] + [i for i in range(len(variable_names)) if i != reference_idx])
    variables = [variable_names[i] for i in variable_order]
    auc_values = auc_values[:, variable_order, :]

    # Reshape the AUC values to a 1D array and create index arrays for libraries, variables, and seeds
    auc_values = auc_values.reshape(-1)
    library_idx = np.repeat(np.arange(num_libraries), num_variables * num_seeds)
    variable_idx = np.tile(np.repeat(np.arange(num_variables), num_seeds), num_libraries)
    seed_idx = np.tile(np.arange(num_seeds), num_libraries * num_variables)
    seed_block_idx = library_idx * num_seeds + seed_idx
    num_seed_blocks = num_libraries * num_seeds

    # Define the hierarchical Bayesian model
    coords = {
        "observation": np.arange(len(auc_values)),
        "library": library_names,
        "variable": variables,
        "non_reference_variables": variables[1:],
        "seed": np.arange(num_seeds),
    }

    GLOBAL_LOGGER.info("Successfully processed the data for building a hierarchical Bayesian model.")

    with pm.Model(coords=coords) as model:

        intercept = pm.Normal("intercept", mu=0, sigma=2)

        # Effect relative to the reference variable
        beta_free = pm.Normal("beta_free", mu=0, sigma=1, dims="non_reference_variables")
        beta = pm.Deterministic("beta", pt.concatenate([pt.zeros(1), beta_free]), dims="variable")

        # Library-specifc effects
        sigma_library = pm.HalfNormal("sigma_library", sigma=0.7)
        z_library_raw = pm.Normal("z_library_raw", mu=0, sigma=1, dims="library")
        z_library = z_library_raw - pt.mean(z_library_raw)  # Centering library effects
        library_effect = pm.Deterministic("library_effect", sigma_library * z_library, dims="library")

        # Seed block effects
        sigma_seed = pm.HalfNormal("sigma_seed", sigma=0.5)
        z_seed_raw = pm.Normal("z_seed_raw", mu=0, sigma=1, dims=("library", "seed"))
        z_seed = z_seed_raw - pt.mean(z_seed_raw, axis=1, keepdims=True)  # Centering seed  effects within each library
        seed_effect = pm.Deterministic("seed_effect", sigma_seed * z_seed, dims=("library", "seed"))

        # Representation effects varying by library
        sigma_library_variable = pm.HalfNormal("sigma_library_variable", sigma=0.5)
        z_library_variable_raw = pm.Normal("z_library_variable_raw", mu=0, sigma=1, dims=("library", "non_reference_variables"))
        z_library_variable = z_library_variable_raw - pt.mean(z_library_variable_raw, axis=0, keepdims=True)  # Centering variable effects within each library
        library_variable_free = pm.Deterministic("library_variable_free", sigma_library_variable * z_library_variable, dims=("library", "non_reference_variables"))
        library_variable_effect = pm.Deterministic("library_variable_effect", pt.concatenate([pt.zeros((num_libraries, 1)), library_variable_free], axis=1), dims=("library", "variable"))

        # Full model
        eta = pm.Deterministic("eta", intercept + library_effect[library_idx] + seed_effect[library_idx, seed_idx] + beta[variable_idx] + library_variable_effect[library_idx, variable_idx], dims="observation")
        expected_auc = pm.Deterministic("expected_auc", pm.math.sigmoid(eta), dims="observation")
        phi = pm.LogNormal("phi", mu=np.log(30), sigma=1.0)
        pm.Beta("observed_aucs", alpha=expected_auc * phi, beta=(1 - expected_auc) * phi, observed=auc_values, dims="observation")

        # Fit the model using MCMC sampling
        start_time = time.time()
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            init="jitter+adapt_diag",
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )
        GLOBAL_LOGGER.info(f"Model fitting completed in {time.time() - start_time:.1f} seconds.")


        # Compute diagnostics and log them
        var_names = [variable.name for variable in model.free_RVs]
        summary = az.summary(trace, var_names=var_names, kind="diagnostics")
        rhat = summary["r_hat"].dropna()
        ess_bulk = summary["ess_bulk"].dropna()
        ess_tail = summary["ess_tail"].dropna()
        min_ess = 100 * trace.posterior.sizes["chain"]
        divergences = np.asarray(trace.sample_stats["diverging"])
        # tree_depth_hit_fraction = (np.asarray(trace.sample_stats["tree_depth"]) >= 10).mean()
        if "reached_max_treedepth" in trace.sample_stats:
            tree_depth_hit_fraction = (np.asarray(trace.sample_stats["reached_max_treedepth"], dtype=bool)).mean()
        else:
            tree_depth_hit_fraction = None
        mean_acceptance = np.asarray(trace.sample_stats["acceptance_rate"]).mean(axis=1)
        mean_step_size = np.asarray(trace.sample_stats["step_size"]).mean(axis=1)
        bfmi = np.asarray(az.bfmi(trace))
        full_summary = az.summary(trace, var_names=var_names, kind="all")
        relative_mcse = (full_summary["mcse_mean"] / full_summary["sd"].replace(0, np.nan)).dropna()

        sampling_ok = (
            rhat.max() <= 1.01
            and ess_bulk.min() >= min_ess
            and ess_tail.min() >= min_ess
            and relative_mcse.max() <= 0.05
            and divergences.sum() == 0
            and bfmi.min() >= 0.3
            and (tree_depth_hit_fraction <= 0.01 or tree_depth_hit_fraction is None)
        )

        GLOBAL_LOGGER.info(
            f"\nMCMC diagnostics:"
            f"\n  Sampling OK:                 {sampling_ok}"
            f"\n  Largest R-hat:               {rhat.max():.6f} (Index {rhat.idxmax()})"
            f"\n  Smallest bulk ESS:           {ess_bulk.min():.0f} ({ess_bulk.idxmin()})"
            f"\n  Smallest tail ESS:           {ess_tail.min():.0f} ({ess_tail.idxmin()})"
            f"\n  Largest relative MCSE:       {relative_mcse.max():.4f} ({relative_mcse.idxmax()})"
            f"\n  Divergences:                 {int(divergences.sum())}"
            f"\n  BFMI by chain:               {np.round(bfmi, 3).tolist()}"
            f"\n  Maximum-tree-depth hits:     {100 * tree_depth_hit_fraction:.1f}%"
            f"\n  Mean acceptance by chain:    {np.round(mean_acceptance, 3).tolist()}"
            f"\n  Mean step size by chain:     {np.round(mean_step_size, 3).tolist()}"
        )

        # Sample from the posterior predictive distribution
        start_time = time.time()
        pm.sample_posterior_predictive(trace, var_names=["observed_aucs"], random_seed=42, extend_inferencedata=True)
        GLOBAL_LOGGER.info(f"Posterior predictive sampling completed in {time.time() - start_time:.1f} seconds.")

        observed = auc_values.reshape(num_libraries, num_variables, num_seeds)
        predicted = np.asarray(trace.posterior_predictive["observed_aucs"]).reshape(-1, num_libraries, num_variables, num_seeds)
        ae = np.abs(observed.mean(axis=2) - predicted.mean(axis=(0, 3)))
        asd = np.abs(predicted.std(axis=3).mean(axis=0) - observed.std(axis=2))

        GLOBAL_LOGGER.info(
            f"\nPosterior predictive diagnostics:"
            f"\n  Mean absolute error (mean AUC): {ae.mean():.6f} (Max: {ae.max():.6f}, Min: {ae.min():.6f})"
            f"\n  Mean absolute error (std AUC):  {asd.mean():.6f} (Max: {asd.max():.6f}, Min: {asd.min():.6f})"
        )

        # Metadata for the trace
        metadata = {
            "auc_values": auc_values,
            "libraries": library_names,
            "num_libraries": num_libraries,
            "library_idx": library_idx,
            "variables": variables,
            "num_variables": num_variables,
            "variable_idx": variable_idx,
            "seed_idx": seed_idx,
            "num_seeds": num_seeds,
            "seed_block_idx": seed_block_idx,
            "num_seed_blocks": num_seed_blocks,
            "block_library_idx": np.repeat(np.arange(num_libraries), num_seeds),  # used
            "reference_variable": reference,
            "variable_order": variable_order
        }

        return trace, metadata


def _stack_posterior(trace: az.InferenceData, var_name: str, *dims) -> np.ndarray:
    """
    Stacks the posterior samples of a given variable from the trace into a 2D array.

    Args:
        trace (az.InferenceData): The inference data containing the posterior samples.
        var_name (str): The name of the variable to extract from the trace.
        *dims: Optional dimensions to stack along. If not provided, all dimensions will be stacked.

    Returns:
        np.ndarray: A 2D array where each row corresponds to a single sample and each column corresponds to a dimension
                    of the variable.
    """
    posterior = trace.posterior[var_name]
    stacked = posterior.stack(sample=("chain", "draw")).transpose("sample", *dims)
    return np.asarray(stacked)


def _extract_expected_auc_samples(trace: az.InferenceData, metadata: dict) -> np.ndarray:
    """
    Extracts the AUC samples from the posterior predictive distribution in the trace.

    Args:
        trace (az.InferenceData): The inference data containing the posterior samples.
        metadata (dict): Metadata containing information about the libraries, variables, and seed blocks.

    Returns:
        np.ndarray: A 3D array of shape (num_samples, num_libraries, num_variables) containing the AUC samples.
    """
    num_libraries, num_variables, num_seeds = metadata["num_libraries"], metadata["num_variables"], metadata["num_seeds"]
    samples = _stack_posterior(trace, "expected_auc", "observation")  # shape: (num_samples, num_observations)
    return samples.reshape(-1, num_libraries, num_variables, num_seeds).mean(axis=3)  # shape: (num_samples, num_libraries, num_variables)


def _extract_hdi(samples: np.ndarray, probability: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the highest density interval (HDI) for the given samples.

    Args:
        samples (np.ndarray): A 2D array of shape (num_samples, num_variables) containing the samples for which to
                              compute the HDI.
        probability (float): The probability mass to include in the HDI.

    Returns:
        tuple[np.ndarray, np.ndarray]: The lower and upper bounds of the HDI.
    """
    samples = np.sort(samples, axis=0)

    num_samples, num_variables = samples.shape
    interval_size = int(np.floor(probability * num_samples))
    intervals = samples[interval_size:, :] - samples[:num_samples - interval_size, :]

    lower_idx = np.argmin(intervals, axis=0)
    columns = np.arange(num_variables)

    return samples[lower_idx, columns], samples[lower_idx + interval_size, columns]


def extract_absolute_variable_effects(trace: az.InferenceData, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Extract library-averaged expected performance for every variable on the original AUOC scale.

    Expected AUOC is first averaged over paired seed blocks within each library and then averaged equally over the
    analyzed libraries. The result answers: "What AUOC is expected for each variable on average across these
    libraries?" Because this is an outcome-scale summary, its magnitude can depend on the baseline AUOC values of the
    libraries and should not be used as the primary standardized effect for comparisons across different AUOC metrics.

    Args:
        trace: Fitted hierarchical-model trace.
        metadata: Metadata returned by :func:`build_hierarchical_model`.
        hdi_probability: Probability mass of the reported highest-density intervals.

    Returns:
        Dictionary with:

        - ``all``: posterior samples, shape ``(num_variables, num_samples)
        - ``mean``: posterior mean expected AUOC for each variable;
        - ``hdi_lower`` and ``hdi_upper``: HDI limits;
        - ``prob_best``: posterior probability that each variable has the highest library-averaged expected AUOC.
    """
    expected_auc = _extract_expected_auc_samples(trace, metadata)  # shape: (num_samples, num_libraries, num_variables)
    averaged_samples = expected_auc.mean(axis=1)  # shape: (num_samples, num_variables)
    hdi_lower, hdi_upper = _extract_hdi(averaged_samples)

    best_idx = np.argmax(averaged_samples, axis=1)
    prob_best = np.mean(best_idx[:, None] == np.arange(averaged_samples.shape[1])[None, :], axis=0)

    return {
        "all": averaged_samples.T,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "prob_best": prob_best
    }


def extract_relative_variable_effects(trace: az.InferenceData, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Extract library-averaged variable effects relative to the reference variable.

    Relative effects are reported on the model's logit scale. For variable ``j``, the library-averaged effect is the
    posterior coefficient ``beta_j`` because the library-specific variable deviations are centered across libraries.
    This contrast is independent of the library baseline and is therefore the preferred summary for comparing effect
    magnitudes across libraries or across different normalized AUOC metrics.

    A logit effect of zero indicates no difference from the reference. Positive values favor the variable; negative
    values favor the reference. ``exp(logit_effect)`` is the ratio of the transformed expected-AUOC odds
    ``mu / (1 - mu)`` relative to the reference and equals one under no difference.

    Args:
        trace: Fitted hierarchical-model trace.
        metadata: Metadata returned by :func:`build_hierarchical_model`.
        hdi_probability: Probability mass of the reported highest-density intervals.

    Returns:
        Dictionary with:

        - ``all``: logit-effect samples, shape ``(num_variables, num_samples)``;
        - ``mean``: posterior mean logit effect;
        - ``hdi_lower`` and ``hdi_upper``: HDI limits on the logit scale;
        - ``prob_above_reference``: posterior probability that the logit effect is greater than zero;
        - ``prob_best``: posterior probability that the variable has the largest library-averaged logit effect;
        - ``effect_ratio_all``: exponentiated effect samples;
        - ``effect_ratio_mean``: posterior mean exponentiated effect;
        - ``effect_ratio_hdi_lower`` and ``effect_ratio_hdi_upper``: HDI limits of the exponentiated effect.

        For the reference variable, the logit effect is zero and the effect ratio is one by construction. Its
        ``prob_above_reference`` is returned as ``NaN`` because a variable is not compared with itself.
    """
    logit_samples = _stack_posterior(trace, "beta", "variable")  # shape: (num_samples, num_variables)
    hdi_lower, hdi_upper = _extract_hdi(logit_samples)

    probability_above_reference = np.mean(logit_samples > 0, axis=0)
    probability_above_reference[0] = np.nan  # Reference variable is not compared with itself

    best_idx = np.argmax(logit_samples, axis=1)
    prob_best = np.mean(best_idx[:, None] == np.arange(logit_samples.shape[1])[None, :], axis=0)

    effect_ratio_samples = np.exp(logit_samples)
    ratio_hdi_lower, ratio_hdi_upper = _extract_hdi(effect_ratio_samples)

    return {
        "all": logit_samples.T,
        "mean": logit_samples.mean(axis=0),
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "prob_above_reference": probability_above_reference,
        "prob_best": prob_best,
        "effect_ratio_all": effect_ratio_samples.T,
        "effect_ratio_mean": effect_ratio_samples.mean(axis=0),
        "effect_ratio_hdi_lower": ratio_hdi_lower,
        "effect_ratio_hdi_upper": ratio_hdi_upper
    }


def extract_absolute_variable_effects_per_library(trace: az.InferenceData, metadata: dict[str, Any]) -> np.ndarray:
    """
    Extract per-library expected performance for every variable on the original AUOC scale.

    Values are averaged over paired seed blocks within each library but not across libraries. This output is suitable
    for plotting library-resolved expected AUOC values and for showing practical performance on the original metric.

    Args:
        trace: Fitted hierarchical-model trace.
        metadata: Metadata returned by :func:`build_hierarchical_model`.

    Returns:
        Posterior samples with shape ``(num_libraries, num_variables, num_samples)``.
    """
    expected_auc = _extract_expected_auc_samples(trace, metadata)  # shape: (num_samples, num_libraries, num_variables)
    return expected_auc.transpose(1, 2, 0)


def extract_absolute_differences_per_library(trace: az.InferenceData, metadata: dict[str, Any]) -> np.ndarray:
    """
    Extract per-library differences in expected performance for every variable relative to the reference variable.

    Values are averaged over paired seed blocks within each library but not across libraries. This output is suitable
    for plotting library-resolved expected AUOC differences and for showing practical performance on the original metric.

    Args:
        trace: Fitted hierarchical-model trace.
        metadata: Metadata returned by :func:`build_hierarchical_model`.

    Returns:
        Posterior samples with shape ``(num_libraries, num_variables, num_samples)``.
    """
    expected_auc = _extract_expected_auc_samples(trace, metadata)  # shape: (num_samples, num_libraries, num_variables)
    reference_auc = expected_auc[:, :, 0][:, :, None]  # shape: (num_samples, num_libraries, 1)
    differences = expected_auc - reference_auc  # shape: (num_samples, num_libraries, num_variables)
    return differences.transpose(1, 2, 0)


def extract_relative_variable_effects_per_library(trace: az.InferenceData, metadata: dict[str, Any]) -> np.ndarray:
    """
    Extracts the relative performance heatmap from the hierarchical Bayesian model trace.

    Returns a 3D array of shape (num_libraries, num_variables, num_samples) containing the relative AUC for each
    library and variable, computed as the difference between the AUC of each variable and the AUC of the
    reference variable.
    """
    beta = _stack_posterior(trace, "beta", "variable")  # shape: (num_samples, num_variables)
    library_deviations = _stack_posterior(trace, "library_variable_effect", "library", "variable")  # shape: (num_samples, num_libraries, num_variables)
    relative_effects = beta[:, None, :] + library_deviations  # shape: (num_samples, num_libraries, num_variables)
    return relative_effects.transpose(1, 2, 0)



