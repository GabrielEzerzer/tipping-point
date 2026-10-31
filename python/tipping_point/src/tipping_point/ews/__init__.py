"""Early-warning signals, classic and latent, and the scoring protocol both domains share (#26).

indicators  rolling variance, autocorrelation, skewness, mean and slope; Gaussian detrending
trend       Kendall's tau against time; the pipeline that turns a series into its score s(t)
surrogates  phase-randomised, AR(1)-fitted and block-bootstrap resamples
scoring     AUC, surrogate p-values, alarm thresholds, lead times, hierarchical bootstrap
fss         scaled temperature, power-law fits at Tc, pass / fail / inconclusive verdicts
margins     paired margins of a trained checkpoint over its untrained twin, across model seeds
ising       adapter over the Ising simulator's runs, their latents tables and the tau_c record
prereg      pre-registration files: load, validate, freeze, stamp results, log trials

Every compute function takes (batch, time) numpy arrays with every parameter in samples; the
adapters convert tau_c and eps_L into samples. Run `python -m tipping_point.ews --help`.
"""
