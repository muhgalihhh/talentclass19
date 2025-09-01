import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Sidebar untuk memilih tugas
tugas = st.sidebar.selectbox(
    "Pilih Tugas",
    ("Tugas 1", "Tugas 2 (coming soon)")
)

if tugas == "Tugas 1":
    st.title('Tugas 1: Energy Data Analyst - Monitoring and Optimization of Renewables')

    # Load dataset
    df = pd.read_csv('dataset_tugas_1_energy_data_analyst.csv')
    st.header('Bagian 1 – Statistik Deskriptif')

    # Statistik deskriptif
    stats = {}
    for col in ['Irradiance_kWh_m2', 'Temperature_C', 'Humidity_%', 'Solar_Output_MWh']:
        stats[col] = {
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std Dev': df[col].std()
        }
    st.write(pd.DataFrame(stats).T)

    # Histogram
    st.subheader('Histogram Irradiance')
    fig, ax = plt.subplots()
    ax.hist(df['Irradiance_kWh_m2'].dropna(), bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel('Irradiance (kWh/m²)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Boxplot Solar Output
    st.subheader('Boxplot Solar Output (MWh)')
    fig2, ax2 = plt.subplots()
    ax2.boxplot(df['Solar_Output_MWh'].dropna())
    ax2.set_ylabel('Solar Output (MWh)')
    st.pyplot(fig2)

    st.header('Bagian 2 – Data Cleaning & Outlier')
    # Missing values
    total_missing = df.isnull().sum()
    st.write('Missing values per column:')
    st.write(total_missing)

    # Strategi penanganan missing values
    st.subheader('Data setelah menghapus baris dengan missing values')
    df_dropna = df.dropna()
    st.write(df_dropna)

    st.subheader('Data setelah mengisi missing values dengan mean')
    df_fillna = df.copy()
    for col in ['Irradiance_kWh_m2', 'Temperature_C', 'Humidity_%', 'Solar_Output_MWh']:
        df_fillna[col] = df_fillna[col].fillna(df_fillna[col].mean())
    st.write(df_fillna)

    # Outlier detection
    st.subheader('Deteksi Outlier Solar Output (Boxplot)')
    fig3, ax3 = plt.subplots()
    ax3.boxplot(df_fillna['Solar_Output_MWh'])
    ax3.set_ylabel('Solar Output (MWh)')
    st.pyplot(fig3)

    st.header('Bagian 3 – Hubungan Antar Variabel')
    # Korelasi
    corr = df_fillna[['Irradiance_kWh_m2', 'Temperature_C', 'Humidity_%', 'Solar_Output_MWh']].corr()
    st.write('Correlation Matrix:')
    st.write(corr)

    st.subheader('Heatmap Korelasi')
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

    # Interpretasi korelasi
    target_corr = corr['Solar_Output_MWh'].drop('Solar_Output_MWh')
    strongest = target_corr.abs().idxmax()
    st.write(f"Variabel dengan korelasi terkuat terhadap Solar Output: {strongest} (korelasi = {target_corr[strongest]:.2f})")

    st.info('Dokumentasi otomatis: seluruh proses, analisis, dan visualisasi ditampilkan di aplikasi ini.')

elif tugas == "Tugas 2 (coming soon)":
    st.title('Tugas 2')
    st.info('Dokumentasi Tugas 2 akan segera tersedia.')
