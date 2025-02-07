import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

"""
WILCOXON SIGNED-RANK TEST
Analysis of DALL-E Mini generated images gender distribution against ILOSTAT distributions of 74 countries
DALL-E Mini Image data taken from 'Investigating Gender and Racial Biases in DALL-E Mini Images'
"""


def visualize(data, country):
    plt.figure(figsize=(10, 5))
    # Bar Chart: Compare Distributions
    plt.bar(data["ISCO-8"], data["DALL-E Mini P(women)"], color='skyblue', label="DALL-E Mini")
    plt.bar(data["ISCO-8"], data[f"ILOSTAT P(women)-{country}"], color='orange', alpha=0.7,
            label=f"ILOSTAT ({country})")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("P(Women)")
    plt.title(f"Comparison of P(Women): DALL-E Mini vs. {country}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Line Plot: Highlight Trends
    plt.figure(figsize=(10, 5))
    plt.plot(data["ISCO-8"], data["DALL-E Mini P(women)"], marker='o', label="DALL-E Mini", linestyle='--',
             color='blue')
    plt.plot(data["ISCO-8"], data[f"ILOSTAT P(women)-{country}"], marker='o', label=f"ILOSTAT ({country})",
             linestyle='-', color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("P(Women)")
    plt.title(f"Trends in P(Women): DALL-E Mini vs. {country}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bar Chart: Differences
    plt.figure(figsize=(10, 5))
    differences = np.array(data[f"ILOSTAT P(women)-{country}"]) - np.array(data["DALL-E Mini P(women)"])
    plt.bar(data["ISCO-8"], differences, color='lightgreen', edgecolor='black')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Difference (ILOSTAT - DALL-E Mini)")
    plt.title(f"Differences in P(Women): {country}")
    plt.tight_layout()
    plt.show()


bias, no_bias = list(), list()
results = list()

# get all countries and their according P(women)-distribution
path = "./data/Analysis-ILOSTAT - P(women) skill-level.csv"
out_path = "results_tests/t-test_results.csv"

# path = str(input("Path to .csv-file: "))
# out_path = str(input("Path to save results_tests as .csv-file: "))

ilostat_data_raw = pd.read_csv(path, sep=";")
ilostat_data = dict()

for continent, country, *sls in zip(
        ilostat_data_raw["Continent"],
        ilostat_data_raw["Country"],
        ilostat_data_raw["ISCO-08-1"],
        ilostat_data_raw["ISCO-08-2"],
        ilostat_data_raw["ISCO-08-3"],
        ilostat_data_raw["ISCO-08-4"],
        ilostat_data_raw["ISCO-08-5"],
        ilostat_data_raw["ISCO-08-7"],
):
    ilostat_data[country] = [list(float(num.replace(',', '.')) for num in sls), continent]

# iterating over all 74 countries in ILOSTAT data
for country, info in ilostat_data.items():
    p_women, continent = info
    if country != "Global":
        data = {
            "ISCO-8": [
                "Occupation (ISCO-08): 1. Managers",
                "Occupation (ISCO-08): 2. Professionals",
                "Occupation (ISCO-08): 3. Technicians and associate professionals",
                "Occupation (ISCO-08): 4. Clerical support workers",
                "Occupation (ISCO-08): 5. Service and sales workers",
                "Occupation (ISCO-08): 7. Craft and related trades workers"
            ],
            "DALL-E Mini P(women)": ilostat_data["Global"][0],
            f"ILOSTAT P(women)-{country}": p_women
        }

        df = pd.DataFrame(data)

        # t-test between Group 0/DALL-E Mini P(women) and Group 1/Labor Statistics P(women)
        t_stat, p_value = wilcoxon(np.array(df["DALL-E Mini P(women)"]),
                                   np.array(df[f"ILOSTAT P(women)-{country}"]))

        # Interpretation
        alpha = 0.05
        significant = p_value < alpha

        if significant:
            bias.append(country)

            # visualize(data, country)
        else:
            no_bias.append(country)

        results.append({
            "Continent": continent,
            "Country": country,
            "T-Statistic": t_stat,
            "P-Value": p_value,
            "Significant": significant
        })

        # print(f"Country: {country}  t-value: {t_stat}   p-value: {p_value}")

print("\n--- Results ---")
results_df = pd.DataFrame(results)
results_df.to_csv(out_path, index=False)

print(f"Countries with significant difference between distributions: {bias}")
print(f"Countries with no significant difference between distributions: {no_bias}")

