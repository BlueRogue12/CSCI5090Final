import pandas as pd
from datetime import datetime

# Load CSVs
patients = pd.read_csv("../data/patients.csv")
conditions = pd.read_csv("../data/conditions.csv")
medications = pd.read_csv("../data/medications.csv")
observations = pd.read_csv("../data/observations.csv", header=None)
encounters = pd.read_csv("../data/encounters.csv")
allergies = pd.read_csv("../data/allergies.csv")
procedures = pd.read_csv("../data/procedures.csv")

# Set column names for observations
observations.columns = [
    'DATE', 'PATIENT', 'ENCOUNTER', 'CATEGORY', 'CODE',
    'DESCRIPTION', 'VALUE', 'UNITS', 'TYPE'
]

# Step 1: Patient demographics and age
today = pd.to_datetime("2025-01-01")
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
patients['age'] = (today - patients['BIRTHDATE']).dt.days // 365
patients_filtered = patients[['Id', 'age', 'RACE', 'ETHNICITY', 'GENDER']].rename(columns={'Id': 'PATIENT'})

# Step 2: Conditions
conditions_filtered = conditions[['PATIENT', 'DESCRIPTION']].rename(columns={'DESCRIPTION': 'CONDITION'})
conditions_grouped = conditions_filtered.groupby('PATIENT').agg({'CONDITION': lambda x: list(set(x))}).reset_index()

# Step 3: Medications - best_medication and med_reason
meds_filtered = medications[['PATIENT', 'DESCRIPTION', 'REASONDESCRIPTION']].dropna()
first_meds = meds_filtered.groupby('PATIENT').first().reset_index()
first_meds.rename(columns={'DESCRIPTION': 'BEST_MEDICATION', 'REASONDESCRIPTION': 'MED_REASON'}, inplace=True)

# Step 4: All medications (optional, grouped)
all_meds_grouped = medications[['PATIENT', 'DESCRIPTION']].rename(columns={'DESCRIPTION': 'MEDICATION'})
all_meds_grouped = all_meds_grouped.groupby('PATIENT').agg({'MEDICATION': lambda x: list(set(x))}).reset_index()

# Step 5: Smoking status
smoking = observations[
    observations['DESCRIPTION'].str.contains("smoking status", case=False, na=False)
]
smoking_status = smoking.groupby('PATIENT').agg({'VALUE': 'last'}).reset_index()
smoking_status.rename(columns={'VALUE': 'SMOKING_STATUS'}, inplace=True)

# Step 6: Presenting problem from encounters
encounter_reason = encounters[['PATIENT', 'REASONDESCRIPTION']].dropna()
presenting_problem = encounter_reason.groupby('PATIENT').agg({'REASONDESCRIPTION': 'first'}).reset_index()
presenting_problem.rename(columns={'REASONDESCRIPTION': 'PRESENTING_PROBLEM'}, inplace=True)

# Step 7: Allergies
allergies_filtered = allergies[['PATIENT', 'DESCRIPTION']].rename(columns={'DESCRIPTION': 'ALLERGY'})
allergies_grouped = allergies_filtered.groupby('PATIENT').agg({'ALLERGY': lambda x: list(set(x))}).reset_index()

# Step 8: Pregnancy detection from procedures
pregnancy_procs = procedures[
    procedures['DESCRIPTION'].str.contains("pregnan|obstetric|labor|delivery|prenatal", case=False, na=False) |
    procedures['REASONDESCRIPTION'].str.contains("pregnan|obstetric|labor|delivery|prenatal", case=False, na=False)
]
pregnancy_flags = pregnancy_procs[['PATIENT']].drop_duplicates()
pregnancy_flags['PREGNANT'] = 'yes'

pregnancy_all = pd.DataFrame(patients_filtered['PATIENT'].unique(), columns=['PATIENT'])
pregnancy_all = pregnancy_all.merge(pregnancy_flags, on='PATIENT', how='left')
pregnancy_all['PREGNANT'] = pregnancy_all['PREGNANT'].fillna('no')

# Step 9: Merge everything
merged = patients_filtered.merge(conditions_grouped, on='PATIENT', how='left')
merged = merged.merge(all_meds_grouped, on='PATIENT', how='left')
merged = merged.merge(first_meds, on='PATIENT', how='left')
merged = merged.merge(smoking_status, on='PATIENT', how='left')
merged = merged.merge(presenting_problem, on='PATIENT', how='left')
merged = merged.merge(allergies_grouped, on='PATIENT', how='left')
merged = merged.merge(pregnancy_all, on='PATIENT', how='left')
merged = merged.dropna(subset=['BEST_MEDICATION', 'PRESENTING_PROBLEM'])

# Final preview or export step
merged.to_csv("../data/cleaned_dataset.csv", index=False)
# print(f"Total patients merged: {len(merged)}")
# print(f"Patients with a best medication assigned: {merged['BEST_MEDICATION'].notnull().sum()}")