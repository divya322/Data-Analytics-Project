from pathlib import Path
import pandas as pd


# ICD codes for ischemic diseases
ischemic_codes = ['I20', 'I21', 'I22', 'I24', 'I25']
script_dir = Path(__file__).resolve().parent  
FEATURE_PATH = script_dir.parent.parent / "Tasks" /   "1" / "Features"

def load_patient_profiles():
    """
    Load the patient profiles dataset
    """
    PATIENT_PROFILES_NAME = "patient_profile_clustering.csv"
    
    patients = pd.read_csv(FEATURE_PATH / PATIENT_PROFILES_NAME)
    return patients


def load_diagnoses_data():
    """
    Load the dataset containing patient ICD codes
    """
    
    DIAGNOSES_NAME = "patient_profiles_core.csv" 
    
    diagnoses = pd.read_csv(FEATURE_PATH / DIAGNOSES_NAME)
    return diagnoses


def has_ischemic_code(icd_code):
    """
    Check if an ICD code corresponds to ischemic disease
    
    Parameters:
    -----------
    icd_code : str or float
        ICD code to check
    
    Returns:
    --------
    bool : True if it's an ischemic code, False otherwise
    """
    if pd.isna(icd_code):
        return False
    
    icd_str = str(icd_code).strip()
    
    # Check if the code starts with any ischemic code
    return any(icd_str.startswith(code) for code in ischemic_codes)


def extract_labels(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Extract binary labels for ischemic disease for each patient
    
    Parameters:
    -----------
    patients : pd.DataFrame
        DataFrame with patient profiles (must contain 'subject_id')
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns ['subject_id', 'ischemic_disease']
                   where ischemic_disease is 0 (healthy) or 1 (ischemic)
    """
    # Load diagnosis data
    diagnoses = load_diagnoses_data()
    
    # Create binary label by aggregating per patient
    patient_labels = diagnoses.groupby('subject_id')['icd_code'].apply(
        lambda codes: int(any(has_ischemic_code(code) for code in codes))
    ).reset_index()
    
    # Rename column
    patient_labels.columns = ['subject_id', 'ischemic_disease']
    
    # Merge with patient profiles to ensure all patients have a label
    patients_with_labels = patients[['subject_id']].merge(
        patient_labels,
        on='subject_id',
        how='left'
    )
    
    # Fill NaN with 0 (patients without ischemic diagnosis)
    patients_with_labels['ischemic_disease'] = patients_with_labels['ischemic_disease'].fillna(0).astype(int)
    
    return patients_with_labels


def create_labeled_dataset():
    """
    Create the complete dataset with patient profiles and labels
    
    Returns:
    --------
    pd.DataFrame : Complete dataset with all features and 'ischemic_disease' label
    """
    # Load patient profiles
    patients = load_patient_profiles()
    
    # Extract labels
    labels = extract_labels(patients)
    
    # Merge to create final dataset
    labeled_dataset = patients.merge(labels, on='subject_id', how='left')
    
    # Verify there are no NaN in labels
    labeled_dataset['ischemic_disease'] = labeled_dataset['ischemic_disease'].fillna(0).astype(int)
    
    # Statistics
    print(f"\n{'='*50}")
    print(f"DATASET CREATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total patients: {len(labeled_dataset)}")
    print(f"\nIschemic disease distribution:")
    print(labeled_dataset['ischemic_disease'].value_counts().sort_index())
    print(f"\nPercentage ischemic: {labeled_dataset['ischemic_disease'].mean()*100:.2f}%")
    print(f"{'='*50}\n")
    
    return labeled_dataset


# Usage
if __name__ == "__main__":
    # Create dataset with labels
    dataset = create_labeled_dataset()
    
    # Save the result
    script_dir = Path(__file__).resolve().parent
    OUTPUT_PATH = script_dir.parent.parent / "Tasks" / "4" / "data"
    OUTPUT_NAME = "patient_profiles_with_labels.csv"
    
    dataset.to_csv(OUTPUT_PATH / OUTPUT_NAME, index=False)
    print(f"Dataset saved to: {OUTPUT_PATH / OUTPUT_NAME}")
    
    # Show some examples
    print("\nExamples of ischemic patients:")
    print(dataset[dataset['ischemic_disease'] == 1][['subject_id', 'ischemic_disease']].head())
    
    print("\nExamples of non-ischemic patients:")
    print(dataset[dataset['ischemic_disease'] == 0][['subject_id', 'ischemic_disease']].head())
