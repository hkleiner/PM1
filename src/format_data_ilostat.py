import pandas as pd

path_all = '/Users/hermine/Library/Mobile Documents/com~apple~CloudDocs/LMU/Master_HF-CL_NF-INF/Semester3/Profilierungsmodul_1/Exam/DALL-E_Mini_Images/Report/data/Statistiken/ilostat_EMP_TEMP_SEX_AGE_OCU_NB_A-filtered-2025-01-16-all.csv'
path_fem = '/Users/hermine/Library/Mobile Documents/com~apple~CloudDocs/LMU/Master_HF-CL_NF-INF/Semester3/Profilierungsmodul_1/Exam/DALL-E_Mini_Images/Report/data/Statistiken/ilostat_EMP_TEMP_SEX_AGE_OCU_NB_A-filtered-2025-01-16-female.csv'

ilostat_all = pd.read_csv(path_all)
ilostat_fem = pd.read_csv(path_fem)

ilostat_all["obs_value"] = ilostat_all["obs_value"].astype(float) * 1000
ilostat_fem["obs_value"] = ilostat_fem["obs_value"].astype(float) * 1000

ilostat_all.to_csv(path_all, index=False)
ilostat_fem.to_csv(path_fem, index=False)

