import matplotlib
matplotlib.use('Agg')  # Set backend for Streamlit Cloud compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def run():
    # Set page config for better UI
    st.set_page_config(page_title="Energy Data Analysis", layout="wide")
    
    # Main title with emoji and styling
    st.title("🔋 Energy Data Analyst: Monitoring and Optimization of Renewables")
    st.markdown("---")
    
    # Load data with progress indicator
    with st.spinner("📂 Loading dataset..."):
        df = pd.read_csv('Tugas_1/dataset_tugas_1_energy_data_analyst.csv')
    
    # Show dataset overview
    st.success(f"✅ Dataset loaded successfully! Total records: {len(df)}")
    
    # Dataset preview
    with st.expander("📊 Dataset Preview"):
        st.write("**First 5 rows of the dataset:**")
        st.dataframe(df.head())
        st.write("**Dataset Info:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    st.markdown("---")
    
    # ============ BAGIAN 1: STATISTIK DESKRIPTIF ============
    st.header("📈 Bagian 1 – Statistik Deskriptif")
    st.markdown("*Analisis statistik dasar untuk memahami karakteristik data*")
    
    # Calculate descriptive statistics
    numeric_cols = ['Irradiance_kWh_m2', 'Temperature_C', 'Humidity_%', 'Solar_Output_MWh']
    
    st.subheader("📊 Statistik Deskriptif Lengkap")
    stats_df = df[numeric_cols].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)
    
    # Custom statistics table
    st.subheader("🎯 Statistik Kunci (Mean, Median, Standar Deviasi)")
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'Mean': round(df[col].mean(), 2),
            'Median': round(df[col].median(), 2),
            'Std Dev': round(df[col].std(), 2)
        }
    
    stats_display = pd.DataFrame(stats).T
    st.dataframe(stats_display, use_container_width=True)
    
    # Insights from descriptive stats
    st.info("💡 **Insight:** Statistik di atas menunjukkan distribusi dan variabilitas setiap variabel energi.")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Histogram Irradiance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['Irradiance_kWh_m2'].dropna(), bins=15, color='gold', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Irradiance (kWh/m²)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Solar Irradiance')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.caption("Distribusi irradiasi matahari menunjukkan pola energi yang tersedia")
    
    with col2:
        st.subheader("📦 Boxplot Solar Output")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        box_plot = ax2.boxplot(df['Solar_Output_MWh'].dropna(), patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('Solar Output (MWh)')
        ax2.set_title('Solar Output Distribution & Outliers')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        st.caption("Boxplot membantu mengidentifikasi outlier dalam produksi energi")
    
    st.markdown("---")
    
    # ============ BAGIAN 2: DATA CLEANING & OUTLIER ============
    st.header("🧹 Bagian 2 – Data Cleaning & Outlier Detection")
    st.markdown("*Membersihkan data dan menangani nilai yang hilang serta outlier*")
    
    # Missing values analysis
    st.subheader("🔍 Analisis Missing Values")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage (%)': missing_percentage.round(2)
    })
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Missing Values Summary:**")
        st.dataframe(missing_df)
    
    with col2:
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(8, 6))
        missing_percentage[missing_percentage > 0].plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Missing Values Percentage by Column')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Columns')
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    if missing_data.sum() > 0:
        st.warning(f"⚠️ Found {missing_data.sum()} missing values in total")
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Data cleaning strategies
    st.subheader("🛠️ Strategi Penanganan Missing Values")
    
    tab1, tab2 = st.tabs(["📋 Hapus Baris", "🔢 Isi dengan Mean"])
    
    with tab1:
        st.write("**Strategi 1: Menghapus baris dengan missing values**")
        df_dropna = df.dropna()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Original", len(df))
        with col2:
            st.metric("Setelah Drop NA", len(df_dropna))
        
        if len(df_dropna) < len(df):
            st.info(f"📉 {len(df) - len(df_dropna)} baris dihapus ({((len(df) - len(df_dropna))/len(df)*100):.1f}%)")
        
        with st.expander("Preview Data Cleaned (Drop NA)"):
            st.dataframe(df_dropna.head())
    
    with tab2:
        st.write("**Strategi 2: Mengisi missing values dengan mean**")
        df_fillna = df.copy()
        for col in numeric_cols:
            if df_fillna[col].isnull().sum() > 0:
                mean_val = df_fillna[col].mean()
                df_fillna[col] = df_fillna[col].fillna(mean_val)
                st.info(f"🔢 {col}: Filled {df[col].isnull().sum()} missing values with mean = {mean_val:.2f}")
        
        with st.expander("Preview Data Cleaned (Fill with Mean)"):
            st.dataframe(df_fillna.head())
    
    # Choose strategy for further analysis
    st.subheader("🎯 Pilihan Strategi untuk Analisis Lanjutan")
    strategy = st.radio(
        "Pilih strategi data cleaning:",
        ["Fill with Mean", "Drop Missing Rows"],
        index=0
    )
    
    if strategy == "Fill with Mean":
        df_clean = df_fillna.copy()
        st.success("✅ Menggunakan data dengan missing values diisi mean")
    else:
        df_clean = df_dropna.copy()
        st.success("✅ Menggunakan data dengan missing rows dihapus")
    
    # Outlier detection
    st.subheader("🎯 Deteksi Outlier")
    
    col1, col2 = st.columns(2)
    with col1:
        # Boxplot for outlier detection
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        df_clean[numeric_cols].boxplot(ax=ax3)
        ax3.set_title('Outlier Detection - All Variables')
        ax3.set_ylabel('Values')
        plt.xticks(rotation=45)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
    
    with col2:
        # Statistical outlier detection
        st.write("**Outlier Statistics (IQR Method):**")
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outlier_summary.append({
                'Variable': col,
                'Outliers Count': len(outliers),
                'Percentage': f"{(len(outliers)/len(df_clean)*100):.1f}%"
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True)
    
    st.markdown("---")
    
    # ============ BAGIAN 3: HUBUNGAN ANTAR VARIABEL ============
    st.header("🔗 Bagian 3 – Hubungan Antar Variabel")
    st.markdown("*Analisis korelasi untuk memahami hubungan antar variabel energi*")
    
    # Correlation analysis
    st.subheader("📊 Correlation Matrix")
    corr = df_clean[numeric_cols].corr()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Correlation Matrix Values:**")
        st.dataframe(corr.round(3), use_container_width=True)
    
    with col2:
        # Correlation heatmap
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, 
                    square=True, fmt='.3f', cbar_kws={"shrink": .8},
                    mask=mask, ax=ax4)
        ax4.set_title('Correlation Heatmap - Energy Variables')
        st.pyplot(fig4)
    
    # Key insights from correlation
    st.subheader("🎯 Key Insights dari Analisis Korelasi")
    target_corr = corr['Solar_Output_MWh'].drop('Solar_Output_MWh')
    strongest_var = target_corr.abs().idxmax()
    strongest_val = target_corr[strongest_var]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Strongest Correlation", strongest_var.replace('_', ' '), f"{strongest_val:.3f}")
    with col2:
        second_strongest = target_corr.abs().nlargest(2).index[1]
        st.metric("Second Strongest", second_strongest.replace('_', ' '), f"{target_corr[second_strongest]:.3f}")
    with col3:
        avg_corr = target_corr.abs().mean()
        st.metric("Average |Correlation|", f"{avg_corr:.3f}")
    
    # Interpretation
    st.subheader("🧠 Interpretasi Hasil")
    interpretation_text = f"""
    **Analisis Korelasi Solar Output:**
    
    1. **Variabel dengan pengaruh terkuat:** `{strongest_var.replace('_', ' ')}` (korelasi = {strongest_val:.3f})
    2. **Interpretasi korelasi:**
       - Korelasi > 0.7: Hubungan kuat positif
       - Korelasi 0.3-0.7: Hubungan sedang
       - Korelasi < 0.3: Hubungan lemah
    
    3. **Insight bisnis:** 
       {'Irradiance memiliki hubungan yang sangat kuat dengan output solar' if 'Irradiance' in strongest_var else 
        'Temperature memiliki pengaruh signifikan terhadap efisiensi panel solar' if 'Temperature' in strongest_var else
        'Humidity berpengaruh terhadap kondisi operasional panel solar' if 'Humidity' in strongest_var else
        'Perlu analisis lebih lanjut untuk memahami faktor dominan'}
    """
    
    st.info(interpretation_text)
    
    # Summary and recommendations
    st.markdown("---")
    st.header("📋 Summary & Recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Hasil Analisis")
        st.write(f"""
        - **Dataset:** {len(df)} records, {len(df.columns)} variables
        - **Missing Values:** {missing_data.sum()} total
        - **Outliers:** Detected in multiple variables
        - **Key Correlation:** {strongest_var.replace('_', ' ')} → Solar Output ({strongest_val:.3f})
        """)
    
    with col2:
        st.subheader("🎯 Rekomendasi")
        st.write("""
        - Monitor variabel dengan korelasi tinggi
        - Implementasi strategi penanganan outlier
        - Optimalisasi berdasarkan faktor dominan
        - Pengumpulan data berkualitas tinggi
        """)
    
    st.success("🎉 Analisis data energi telah selesai! Semua tahap telah dilakukan sesuai dengan metodologi data science yang tepat.")
    
    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        csv_clean = df_clean.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv_clean,
            file_name="cleaned_energy_data.csv",
            mime="text/csv"
        )
    with col2:
        corr_csv = corr.to_csv()
        st.download_button(
            label="📊 Download Correlation Matrix",
            data=corr_csv,
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )
