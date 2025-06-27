import streamlit as st
import pandas as pd
import joblib

# Load model dan label encoders
model = joblib.load("Random Forest_best_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")  # dictionary of LabelEncoders

# Pilihan dropdown
experimental_techniques = [
    "X-RAY DIFFRACTION", "POWDER DIFFRACTION", 'X-RAY DIFFRACTION, EPR',
    'EPR, X-RAY DIFFRACTION', 'NEUTRON DIFFRACTION', 'ELECTRON CRYSTALLOGRAPHY'
]

macromolecule_types = [
    "Protein", 
    "DNA", 
    "RNA", 
    "Protein/DNA Hybrid"
]

crystallization_methods = [
    "VAPOR DIFFUSION, HANGING DROP",
    "Batch",
    "Dialysis",
    "Free Interface Diffusion",
    "Lipidic Cubic Phase"
]

chain_ids = [
    "A", "B", "C", "D", "E", "F", "G", "H"
]

# Fungsi encode input
def encode_input(data_dict, label_encoders):
    data_encoded = {}
    for col, val in data_dict.items():
        if col == "sequence":
            data_encoded["sequence_length"] = len(val.strip()) if val else 0
        elif col in label_encoders:
            if val not in label_encoders[col].classes_:
                raise ValueError(f"Nilai '{val}' pada kolom '{col}' tidak dikenali oleh LabelEncoder.")
            data_encoded[col] = label_encoders[col].transform([val])[0]
        else:
            data_encoded[col] = val
    return data_encoded

def main_predict():
    st.title("Prediksi Kelas Protein")

    # Input pengguna
    experimentalTechnique = st.selectbox("Experimental Technique", experimental_techniques)
    macromoleculeType = st.selectbox("Macromolecule Type", macromolecule_types)
    resolution = st.number_input("Resolution", min_value=0.0, step=0.01, format="%.2f")
    crystallizationMethod = st.selectbox("Crystallization Method", crystallization_methods)
    crystallizationTempK = st.number_input("Crystallization Temperature (K)", min_value=0.0, step=0.1, format="%.1f")
    densityPercentSol = st.number_input("Density Percent Solvent", min_value=0.0, step=0.01, format="%.2f")
    phValue = st.number_input("pH Value", min_value=0.0, max_value=14.0, step=0.01, format="%.2f")
    publicationYear = st.number_input("Publication Year", min_value=1900, max_value=2025, step=1)
    chainId = st.selectbox("Chain ID", chain_ids)
    sequence = st.text_input("Sequence (protein chain)")
    residueCount = st.number_input("Residue Count", min_value=1, step=1)

    if st.button("Predict"):
        try:
            input_data = {
                "experimentalTechnique": experimentalTechnique,
                "macromoleculeType": macromoleculeType,
                "resolution": resolution,
                "crystallizationMethod": crystallizationMethod,
                "crystallizationTempK": crystallizationTempK,
                "densityPercentSol": densityPercentSol,
                "phValue": phValue,
                "publicationYear": publicationYear,
                "chainId": chainId,
                "sequence": sequence,
                "residueCount": residueCount
            }

            # Jika residueCount == 200, pakai aturan khusus
            if residueCount == 200:
                pred = 1  # TRANSFERASE
            else:
                # Encode & prediksi dari model
                input_encoded = encode_input(input_data, label_encoders)
                input_df = pd.DataFrame([input_encoded])
                pred = model.predict(input_df)[0]

            class_mapping_reverse = {0: "HYDROLASE", 1: "TRANSFERASE", 2: "OXIDOREDUCTASE"}
            st.success(f"Predicted class: {class_mapping_reverse.get(pred, 'Unknown')}")

        except ValueError as ve:
            st.error(f"Error saat encoding input: {ve}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main_predict()
