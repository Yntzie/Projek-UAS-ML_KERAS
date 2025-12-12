import streamlit as st
import pandas as pd
import pickle
import os

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Prediksi Performance Index (Ridge)",
    page_icon="🎓",
    layout="centered"
)

# =============================================================================
# LOAD MODEL RIDGE
# =============================================================================
# Pastikan path folder 'model/' sudah benar dan file ada di dalamnya
FILE_MODEL_RIDGE = 'model/BestModel_Ridge_KERAS.pkl'

@st.cache_resource
def load_model(filename):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error memuat model {filename}: {e}")
        return None

model_ridge = load_model(FILE_MODEL_RIDGE)

# =============================================================================
# HEADER
# =============================================================================
st.title("🎓 Prediksi Performance Index Siswa")
st.write(f"""
Aplikasi ini menggunakan model **Ridge Regression** untuk memprediksi nilai performa siswa.
""")
st.divider()

# =============================================================================
# FORM INPUT FITUR
# =============================================================================
st.subheader("Masukkan Data Siswa")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "Jam Belajar (Hours Studied)", 
        min_value=0, max_value=24, value=5, step=1
    )
    
    previous_scores = st.number_input(
        "Nilai Sebelumnya (Previous Scores)", 
        min_value=0, max_value=100, value=75, step=1
    )
    
    extracurricular = st.selectbox(
        "Ekstrakurikuler",
        options=["Yes", "No"]
    )

with col2:
    sleep_hours = st.number_input(
        "Jam Tidur (Sleep Hours)", 
        min_value=0, max_value=24, value=7, step=1
    )
    
    sample_papers = st.number_input(
        "Latihan Soal (Sample Papers)", 
        min_value=0, max_value=100, value=2, step=1
    )

# =============================================================================
# PRE-PROCESSING
# =============================================================================
# Konversi input sesuai format training (Yes=1, No=0)
extracurricular_value = 1 if extracurricular == "Yes" else 0

input_data = {
    'Hours Studied': [hours_studied],
    'Previous Scores': [previous_scores],
    'Extracurricular Activities': [extracurricular_value],
    'Sleep Hours': [sleep_hours],
    'Sample Question Papers Practiced': [sample_papers]
}

input_df = pd.DataFrame(input_data)

# =============================================================================
# PREDIKSI DAN HASIL
# =============================================================================
st.markdown("---")

if st.button("Prediksi", type="primary"):
    # Cek ketersediaan model
    if model_ridge is None:
        st.error(f"File model '{FILE_MODEL_RIDGE}' tidak ditemukan. Pastikan file .pkl ada di folder 'model/'.")
    else:
        try:
            # Prediksi menggunakan Ridge
            prediction = model_ridge.predict(input_df)
            hasil_prediksi = prediction[0] if hasattr(prediction, '__iter__') else prediction

            # Tampilkan Hasil
            st.success("Hasil Prediksi")
            st.metric(
                label="Performance Index", 
                value=f"{hasil_prediksi:.2f}",
                delta=None
            )
            
            # Tambahan Info Visual
            if hasil_prediksi >= 80:
                st.info("Kategori: Sangat Baik 🌟")
            elif hasil_prediksi >= 60:
                st.info("Kategori: Baik 👍")
            else:
                st.warning("Kategori: Perlu Peningkatan ⚠️")

        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.write("Tips: Pastikan nama dan urutan kolom input sama persis dengan saat model dilatih.")