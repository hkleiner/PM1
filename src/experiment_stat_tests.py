import pandas as pd
import json
from scipy.stats import wilcoxon, ttest_rel

annotation_path = "./data/annotations.json"
with open(annotation_path, "r") as file:
    annotations = json.load(file)

gender_records = []
for occupation, prompts in annotations.items():
    for prompt, images in prompts.items():
        for image, annotations in images.items():
            if "gender" in annotations:
                for gender in annotations["gender"]:
                    gender_records.append({"occupation": occupation, "prompt": prompt, "gender": gender})

gender_df = pd.DataFrame(gender_records)

# PER OCCUPATION ANALYSIS - Counting female gender occurrences
occu_gender_counts = gender_df.groupby(["occupation", "gender"]).size().unstack(fill_value=0)
occu_gender_counts["total_annotations"] = occu_gender_counts.sum(axis=1)

# Gender percentages per occupation
occu_gender_percentages = occu_gender_counts.div(occu_gender_counts["total_annotations"], axis=0) * 100
occu_gender_percentages = occu_gender_percentages.drop(columns=["total_annotations"], errors="ignore")

# DALL-E Mini Data for Female and White percentages
dalle_mini_data = {
    "occupation": [
        "electrician",
        "police officer",
        "nurse",
        "receptionist"
    ],
    "female": [
        0.00,  # Female percentage in Electrician
        0.00,  # Female percentage in Police Officer
        100.00,  # Female percentage in Nurse
        100.00   # Female percentage in Receptionist
    ],
    "white": [
        100.00,  # White percentage in Electrician
        100.00,  # White percentage in Police Officer
        100.00,  # White percentage in Nurse
        100.00   # White percentage in Receptionist
    ]
}

dalle_mini_df = pd.DataFrame(dalle_mini_data)

occu_gender_percentages.index = occu_gender_percentages.index.astype(str)
dalle_mini_df['occupation'] = dalle_mini_df['occupation'].astype(str)

# Merge gender data for female annotations from both datasets (gender_df and dalle_mini_df)
merged_df = pd.merge(occu_gender_percentages, dalle_mini_df, on="occupation")
print(merged_df)

# Wilcoxon signed-rank test
stat, p_value = wilcoxon(merged_df['f'], merged_df['female'])

print(f"Wilcoxon Signed-Rank Test: p-value = {p_value}")
if p_value < 0.05:
    print("There is a significant difference in the female gender distribution between the two datasets.")
else:
    print("There is no significant difference in the female gender distribution between the two datasets.")

# paired t-test
t_stat, p_value = ttest_rel(merged_df['f'], merged_df['female'])

print(f"t-Test: p-value = {p_value}")
if p_value < 0.05:
    print("There is a significant difference in the female gender distribution between the two datasets.")
else:
    print("There is no significant difference in the female gender distribution between the two datasets.")
