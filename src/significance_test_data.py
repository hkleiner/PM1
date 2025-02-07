import scipy.stats as stats
import numpy as np

# Data in percentages
us_labour_statistics = np.array([36.67, 48.49, 52.50, 80.00, 56.75, 0.24, 19.20, 0.04])
ilostat_usa = np.array([42.57, 53.98, 55.40, 72.11, 57.56, 20.01, 14.72, 19.16])

# Initialize lists to store results_tests
anova_results = []

# Perform one-way ANOVA for each skill level
for us, ilo, uk_val in zip(us_labour_statistics, ilostat_usa, uk):
    f_stat, p_val = stats.f_oneway([us], [ilo], [uk_val])
    anova_results.append((f_stat, p_val))

# Display results_tests
for i, (f_stat, p_val) in enumerate(anova_results, start=1):
    print(f"ISCO-08 Level {i}: F-statistic = {f_stat:.2f}, p-value = {p_val:.4f}")

# Initialize lists to store results_tests
t_statistics = []
p_values = []

# Perform paired t-tests for each skill level
for us, ilo in zip(us_labour_statistics, ilostat_usa):
    t_stat, p_val = stats.ttest_1samp([us, ilo], 0)
    t_statistics.append(t_stat)
    p_values.append(p_val)

# Display results_tests
for i, (t_stat, p_val) in enumerate(zip(t_statistics, p_values), start=1):
    print(f"ISCO-08 Level {i}: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")
