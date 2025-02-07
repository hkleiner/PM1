import pandas as pd
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 8 prompts
# 8 occupations
# 9 images per prompt
# 8 x 9 = 72 images per occupation
# 72 x 8 = 576 images in total
# however, depending on the number of annotated persons within one picture, we have different numbers of annotations for
# each prompt


def write_latex_table(distribution, name, index=True):
    """
    Writes a LaTeX table from a given DataFrame.

    This function takes a pandas DataFrame, formats it into LaTeX code, and writes
    the LaTeX code to a specified file.

    Parameters:
    distribution (pd.DataFrame): The DataFrame to be converted into a LaTeX table.
    name (str): The name of the output file where the LaTeX code will be saved.
    index (bool): A boolean indicating whether to include the DataFrame's index
                  in the LaTeX table.

    Returns:
    None
    """
    # LaTeX-Tabelle formatieren
    latex_table = distribution.to_latex(float_format="%.2f", index=index)

    with open(name, 'w') as f:
        f.write(latex_table)

    print("File saved", file=sys.stderr)


prompt_template = json.load(open("./data/prompt_templates.json"))

annotation_path = "./data/annotations.json"
with open(annotation_path, "r") as file:
    annotations = json.load(file)

gender_records = []
race_records = []

for occupation, prompts in annotations.items():
    for prompt, images in prompts.items():
        for image, annotations in images.items():
            if "gender" in annotations:
                for gender in annotations["gender"]:
                    gender_records.append({"occupation": occupation, "prompt": prompt, "gender": gender})
            if "race" in annotations:
                for race in annotations["race"]:
                    race_records.append({"occupation": occupation, "prompt": prompt, "race": race})

# create separate DataFrames for gender and race
gender_df = pd.DataFrame(gender_records)
race_df = pd.DataFrame(race_records)

# PER OCCUPATION ANALYSIS
# count occurrences per occupation
occu_gender_counts = gender_df.groupby(["occupation", "gender"]).size().unstack(fill_value=0)
occu_race_counts = race_df.groupby(["occupation", "race"]).size().unstack(fill_value=0)

# total annotations per occupation
occu_gender_counts["total_annotations"] = occu_gender_counts.sum(axis=1)
occu_race_counts["total_annotations"] = occu_race_counts.sum(axis=1)

occu_gender_percentages = occu_gender_counts.div(occu_gender_counts["total_annotations"], axis=0) * 100
occu_race_percentages = occu_race_counts.div(occu_race_counts["total_annotations"], axis=0) * 100

occu_gender_percentages = occu_gender_percentages.drop(columns=["total_annotations"], errors="ignore")
occu_race_percentages = occu_race_percentages.drop(columns=["total_annotations"], errors="ignore")

#write_latex_table(occu_gender_counts, "./results_experiments/occu_gender_counts.txt", True)
#write_latex_table(occu_gender_counts, "./results_experiments/occu_race_counts.txt", True)
#write_latex_table(occu_gender_percentages, "./results_experiments/occu_gender_percentages.txt", True)
#write_latex_table(occu_gender_percentages, "./results_experiments/occu_race_percentages.txt", True)

print("Gender Counts:\n", occu_gender_percentages)
print("Race Counts:\n", occu_race_percentages)

# PER PROMPT ANALYSIS
prompt_gender_counts = gender_df.groupby(["occupation", "prompt", "gender"]).size().unstack(fill_value=0)
prompt_race_counts = race_df.groupby(["occupation", "prompt", "race"]).size().unstack(fill_value=0)

prompt_gender_counts["total_annotations"] = prompt_gender_counts.sum(axis=1)
prompt_race_counts["total_annotations"] = prompt_race_counts.sum(axis=1)

prompt_gender_percentages = prompt_gender_counts.div(prompt_gender_counts["total_annotations"], axis=0) * 100
prompt_race_percentages = prompt_race_counts.div(prompt_race_counts["total_annotations"], axis=0) * 100

prompt_gender_percentages = prompt_gender_percentages.drop(columns=["total_annotations"], errors="ignore")
prompt_race_percentages = prompt_race_percentages.drop(columns=["total_annotations"], errors="ignore")

#write_latex_table(prompt_gender_counts, "./results_experiments/prompt_gender_counts.txt", True)
#write_latex_table(prompt_gender_counts, "./results_experiments/prompt_race_counts.txt", True)
#write_latex_table(prompt_gender_percentages, "./results_experiments/prompt_gender_percentages.txt", True)
#write_latex_table(prompt_gender_percentages, "./results_experiments/prompt_race_percentages.txt", True)

# Print results_tests
print("Gender Percentages Per Prompt:\n", prompt_gender_percentages)
print("Race Counts Per Prompt:\n", prompt_race_percentages)

# PLOTTING
sns.set(style="whitegrid")
occupations = prompt_gender_percentages.index.get_level_values("occupation").unique()

for occupation in occupations:
    subset = prompt_gender_percentages.loc[occupation]
    plt.figure(figsize=(8, 5))

    for gender in subset.columns:
        sns.lineplot(
            x=subset.index.astype(str),
            y=subset[gender],
            label=gender,
            linestyle='--',
            alpha=0.8
        )

    # prompt_labels = [prompt_template[str(idx)].replace("[occupation (pl)]", occupation) for idx in subset.index]

    plt.title(f"Gender Distribution Trend for Occupation: {occupation}")
    plt.xlabel("Prompt No.")
    plt.ylabel("Percentage")
    # plt.xticks(range(len(subset)), labels=prompt_labels, rotation=45, ha="right")  # mapped labels
    plt.xticks(range(len(subset)))
    plt.legend(
        title="Annotated Gender",
        fontsize='small',
        markerscale=0.8
    )
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f"./results_experiments/figures/gender_trends_{occupation}.png", dpi=700)

sns.set(style="whitegrid")
occupations = prompt_race_percentages.index.get_level_values("occupation").unique()

for occupation in occupations:
    subset = prompt_race_percentages.loc[occupation]
    plt.figure(figsize=(8, 5))

    for race in subset.columns:
        sns.lineplot(
            x=subset.index.astype(str),
            y=subset[race],
            label=race,
            linestyle='--',
            alpha=0.8
        )

    # prompt_labels = [prompt_template[str(idx)].replace("[occupation (pl)]", occupation) for idx in subset.index]

    plt.title(f"Race Distribution Trend for Occupation: {occupation}")
    plt.xlabel("Prompt No.")
    plt.ylabel("Percentage")
    # plt.xticks(range(len(subset)), labels=prompt_labels, rotation=45, ha="right")  # mapped labels
    plt.xticks(range(len(subset)))
    plt.legend(
        title="Annotated Race",
        fontsize='small',
        markerscale=0.8
    )
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f"./results_experiments/figures/race_trends_{occupation}.png", dpi=700)
    plt.close()
