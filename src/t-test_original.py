import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

"""
Replicating the original statistical test (t-test) performed by Cheong et al. 2024
@Investigating Gender and Racial Biases in DALL-E Mini Images
"""

data_original = {
# selected occupations:
    # removed occupations that were categorized as indistinct (from our coding)
    # removed occupations from which form an archetype or superset of several occupations (such as “civil servant” or “business person”)
    # removed occupations that could not be located in the labor statistics (such as “lexicographer”)
    "Occupation": [
        "Company Director",
        "Executive",
        "Manager",
        "Accountant",
        "Actor",
        "Architect",
        "Attorney",
        "Author",
        "Biologist",
        "Businessperson",  # kept however
        "Chiropractor",
        "Comedian",
        "Dietitian",
        "Doctor",
        "Engineer",
        "Journalist",
        "Judge",
        "Juggler",
        "Lawyer",
        "Lecturer",
        "Magician",
        "Newscaster",
        "Newsreader",
        "Nurse",
        "Optician",
        "Painter",
        "Pastor",
        "Physician",
        "Poet",
        "Psychologist",
        "Rapper",
        "Singer",
        "Software Engineer",
        "Solicitor",
        "Spokesperson",
        "Teacher",
        "Television presenter",
        "Writer",
        "Assistant",
        "Chef",
        "Interior Designer",
        "Personal trainer",
        "Photographer",
        "Pilot",
        "Tennis player",
        "Yoga Teacher",
        "Clerk",
        "Library Assistant",
        "Receptionist",
        "Secretary",
        "Telephone operator",
        "Telephonist",
        "Cook",
        "Flight Attendant",
        "Hairdresser",
        "Makeup Artist",
        "Police Officer",
        "Prison Officer",
        "Salesperson",
        "Waiter",
        "Farmer",
        "Baker",
        "Builder",
        "Butcher",
        "Electrician",
        "Plumber",
        "Miner"
    ],
    "DALL-E Mini P(women)": [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.22,
        0.00,
        0.00,
        0.00,
        0.00,
        1.00,
        0.00,
        0.00,
        0.40,
        0.00,
        0.00,
        0.00,
        0.09,
        0.00,
        1.00,
        1.00,
        1.00,
        0.56,
        0.00,
        0.00,
        0.00,
        0.00,
        0.65,
        0.00,
        1.00,
        0.00,
        0.00,
        1.00,
        0.78,
        0.70,
        0.25,
        0.89,
        0.00,
        0.18,
        0.63,
        0.10,
        0.00,
        0.11,
        1.00,
        0.22,
        1.00,
        1.00,
        1.00,
        0.86,
        0.50,
        0.00,
        1.00,
        1.00,
        1.00,
        0.00,
        0.00,
        1.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.17,
        0.00,
        0.00,
        0.00
    ],
    "Labor Statistics P(women)": [
        0.29,
        0.29,
        0.52,
        0.59,
        0.48,
        0.30,
        0.39,
        0.57,
        0.58,
        0.55,
        0.26,
        0.49,
        0.88,
        0.44,
        0.16,
        0.48,
        0.56,
        0.49,
        0.39,
        0.48,
        0.49,
        0.48,
        0.48,
        0.88,
        0.76,
        0.11,
        0.19,
        0.44,
        0.57,
        0.75,
        0.26,
        0.26,
        0.22,
        0.53,
        0.67,
        0.73,
        0.49,
        0.57,
        0.72,
        0.27,
        0.89,
        0.63,
        0.48,
        0.09,
        0.49,
        0.63,
        0.72,
        0.79,
        0.90,
        0.95,
        0.72,
        0.72,
        0.38,
        0.70,
        0.93,
        0.93,
        0.13,
        0.30,
        0.49,
        0.68,
        0.24,
        0.64,
        0.04,
        0.25,
        0.02,
        0.01,
        0.04
    ]
}

data_accumulated = {
            "ISCO-8": [
                "Occupation (ISCO-08): 1. Managers",
                "Occupation (ISCO-08): 2. Professionals",
                "Occupation (ISCO-08): 3. Technicians and associate professionals",
                "Occupation (ISCO-08): 4. Clerical support workers",
                "Occupation (ISCO-08): 5. Service and sales workers",
                "Occupation (ISCO-08): 7. Craft and related trades workers"
            ],
            "DALL-E Mini P(women)": [0.000, 0.276, 0.364, 0.763, 0.500, 0.034],
            f"Labor Statistics P(women)": [0.367, 0.485, 0.525, 0.800, 0.568, 0.192]
        }

df_original = pd.DataFrame(data_original)
df_accumulated = pd.DataFrame(data_accumulated)

# t-test

# t-test between Group 0/DALL-E Mini P(women) and Group 1/Labor Statistics P(women)
t_stat_original, p_value_original = ttest_ind(df_original["DALL-E Mini P(women)"], df_original["Labor Statistics P(women)"], equal_var=False)
t_stat_acc, p_value_acc = ttest_ind(df_accumulated["DALL-E Mini P(women)"], df_accumulated["Labor Statistics P(women)"], equal_var=False)

print("t-test results_tests:")
print("Original test based on occupations: ", t_stat_original, p_value_original)  # -2.8925851765896047 0.004631151886616149
print("Test based on skill-levels ISCO-08: ", t_stat_acc, p_value_acc, "\n")  # -2.8925851765896047 0.004631151886616149

"""
The Wilcoxon Signed-Rank Test is better than the t-test for your mapped data because:

No Assumption of Normality:
The t-test assumes that the differences between paired samples are normally distributed. In your case, the aggregation of 67 categories into 6 likely introduces non-normality, making the t-test less reliable.
The Wilcoxon test is non-parametric and doesn't require normality, so it works better for data with unknown or skewed distributions.

Handles Ordinal or Ranked Data:
By mapping the data into fewer categories, the granularity is reduced, and the data may no longer behave like continuous values. The Wilcoxon test uses ranks of differences rather than raw values, making it robust to such changes.

Sensitivity to Directionality:
The Wilcoxon test is sensitive to the direction of differences (whether one dataset is consistently higher or lower). In your context, this is crucial for detecting consistent biases introduced by the model, even if the magnitude of differences varies.

Robust to Outliers:
Aggregated data might amplify the impact of outliers, which can skew t-test results_tests. The rank-based Wilcoxon test minimizes this effect.
In summary, the Wilcoxon Signed-Rank Test is more suitable for your mapped data because it avoids strict parametric assumptions and is robust to aggregation effects, making it better equipped to detect differences in your transformed datasets.
"""

# Wilcoxon Signed-Rank Test

dall_e_mini_women_acc = np.array(data_accumulated["DALL-E Mini P(women)"])
labor_stats_women_acc = np.array(data_accumulated["Labor Statistics P(women)"])
dall_e_mini_women_ori = np.array(data_original["DALL-E Mini P(women)"])
labor_stats_women_ori = np.array(data_original["Labor Statistics P(women)"])

statistic_acc, p_value_acc = wilcoxon(dall_e_mini_women_acc, labor_stats_women_acc)
statistic_ori, p_value_ori = wilcoxon(dall_e_mini_women_ori, labor_stats_women_ori)

print("Wilcoxon Signed-Rank Test results_tests:")
print("Original test based on occupations: ", statistic_ori, p_value_ori)
print("Test based on skill-levels ISCO-08: ", statistic_acc, p_value_acc)
