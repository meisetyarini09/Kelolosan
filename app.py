import streamlit as st
import pandas as pd
import pickle  # atau joblib, tergantung model kamu

# Muat model
with open('model_nb.pkl', 'rb') as file:  # Ganti nama file sesuai model kamu
    nb = pickle.load(file)

st.title("Prediksi Kelolosan Ujian")

st.markdown("### Masukkan data di bawah ini:")

# Form input pengguna
sleep_hours = st.number_input("Sleep Hours (dalam jam):", min_value=0.0, max_value=24.0, step=0.5)
motivation_level = st.selectbox("Motivation Level", options=[0, 1, 2], format_func=lambda x: ['Low', 'Medium', 'High'][x])
teacher_quality = st.selectbox("Teacher Quality", options=[0, 1, 2], format_func=lambda x: ['Low', 'Medium', 'High'][x])

# Tombol prediksi
if st.button("Prediksi"):
    # Buat DataFrame dari input pengguna
    new_data = pd.DataFrame([[sleep_hours, motivation_level, teacher_quality]],
                            columns=['Sleep_Hours', 'Motivation_Level', 'Teacher_Quality'])

    # Lakukan prediksi
    try:
        pred_code = nb.predict(new_data)[0]
        label_map = {1: "Lolos", 0: "Tidak lolos"}
        pred_label = label_map.get(pred_code, "Tidak diketahui")

        st.success("Hasil Prediksi:")
        st.write(f"**Sleep Hours:** {sleep_hours}")
        st.write(f"**Motivation Level:** {['Low', 'Medium', 'High'][motivation_level]}")
        st.write(f"**Teacher Quality:** {['Low', 'Medium', 'High'][teacher_quality]}")
        st.write(f"### 🎓 Prediksi kategori kelolosan ujian adalah: **{pred_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
