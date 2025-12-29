from pathlib import Path
import pandas as pd

# Class definitions

CLASS_MAP = {
    0: "No cardiac disease",
    1: "Ischemic heart disease",
    2: "Arrhythmias and conduction disorders",
    3: "Valvular and pump failure disorders",
    4: "Inflammatory and structural heart disease"
}

# ICD → class mapping (prefix-based)
ICD_TO_CLASS = {
    1: ['I20', 'I21', 'I22', 'I24', 'I25'],          # Ischemic heart disease
    2: ['I44', 'I45', 'I46', 'I47', 'I48', 'I49'],  # Arrhythmias & conduction
    3: ['I34', 'I35', 'I36', 'I42', 'I50'],         # Valvular & pump failure
    4: ['I30', 'I31', 'I33', 'I40']                 # Inflammatory & structural
}

# Priority used when multiple classes exist for a patient
CLASS_PRIORITY = [1, 2, 3, 4]

# Paths

script_dir = Path(__file__).resolve().parent
FEATURE_PATH = script_dir.parent.parent / "Tasks" / "1" / "Features"

# Data loading

def load_patient_profiles():
    return pd.read_csv(FEATURE_PATH / "patient_profile_clustering.csv")

def load_diagnoses_data():
    return pd.read_csv(FEATURE_PATH / "patient_profiles_core.csv")

# ICD → class helpers

def icd_to_class(icd_code):
    if pd.isna(icd_code):
        return None

    icd_str = str(icd_code).strip()

    for class_id, prefixes in ICD_TO_CLASS.items():
        if any(icd_str.startswith(p) for p in prefixes):
            return class_id

    return None

def resolve_patient_class(icd_codes):
    found_classes = set()

    for code in icd_codes:
        cls = icd_to_class(code)
        if cls is not None:
            found_classes.add(cls)

    if not found_classes:
        return 0  # No cardiac disease

    for cls in CLASS_PRIORITY:
        if cls in found_classes:
            return cls

# Label extraction

def extract_multiclass_labels(patients: pd.DataFrame) -> pd.DataFrame:
    diagnoses = load_diagnoses_data()

    patient_labels = (
        diagnoses
        .groupby('subject_id')['icd_code']
        .apply(resolve_patient_class)
        .reset_index(name='cardiac_class')
    )

    patients_with_labels = patients[['subject_id']].merge(
        patient_labels,
        on='subject_id',
        how='left'
    )

    patients_with_labels['cardiac_class'] = (
        patients_with_labels['cardiac_class']
        .fillna(0)
        .astype(int)
    )

    return patients_with_labels

# Dataset creation

def create_multiclass_labeled_dataset():
    patients = load_patient_profiles()
    labels = extract_multiclass_labels(patients)

    dataset = patients.merge(labels, on='subject_id', how='left')
    dataset['cardiac_class'] = dataset['cardiac_class'].fillna(0).astype(int)

    print("\n" + "=" * 50)
    print("MULTICLASS DATASET SUMMARY")
    print("=" * 50)
    print(dataset['cardiac_class'].value_counts().sort_index())
    print("\nClass definitions:")
    for k, v in CLASS_MAP.items():
        print(f"{k}: {v}")
    print("=" * 50 + "\n")

    return dataset

# Main

if __name__ == "__main__":
    dataset = create_multiclass_labeled_dataset()

    OUTPUT_PATH = script_dir.parent.parent / "Tasks" / "4" / "data"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_NAME = "patient_profiles_with_multiclass_labels.csv"
    dataset.to_csv(OUTPUT_PATH / OUTPUT_NAME, index=False)

    print(f"Dataset saved to: {OUTPUT_PATH / OUTPUT_NAME}")