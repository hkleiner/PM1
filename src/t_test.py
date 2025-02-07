import pandas as pd
from scipy.stats import ttest_ind

"""
T-TEST
Analysis of DALL-E Mini generated images gender distribution against ILOSTAT distributions of 74 countries
DALL-E Mini Image data taken from 'Investigating Gender and Racial Biases in DALL-E Mini Images'
"""
# get all countries and their according P(women)-distribution
path = "./data/Analysis-ILOSTAT - P(women) skill-level.csv"
out_path = "results_tests/t-test_results.csv"

# path = str(input("Path to .csv-file: "))
# out_path = str(input("Path to save results_tests as .csv-file: "))

ilostat_data_raw = pd.read_csv(path, sep=";")
ilostat_data = dict()

results = list()

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
for country, (p_women, continent) in ilostat_data.items():
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
        t_stat, p_value = ttest_ind(df["DALL-E Mini P(women)"], df[f"ILOSTAT P(women)-{country}"], equal_var=False)

        # Interpretation
        alpha = 0.05
        significant = p_value < alpha

        results.append({
            "Continent": continent,
            "Country": country,
            "T-Statistic": t_stat,
            "P-Value": p_value,
            "Significant": significant
        })

        print(f"Country: {country}  t-value: {t_stat}   p-value: {p_value}")

print("\n--- Results ---")
results_df = pd.DataFrame(results)
results_df.to_csv(out_path, index=False)
