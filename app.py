import streamlit as st

from Tugas_1 import tugas1_ui
from Tugas_2 import tugas2_ui

def show_home_dashboard():
    """Halaman home dashboard dengan identitas dan informasi kursus"""
    # Set page config
    st.set_page_config(
        page_title="Energy Data Analyst Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header dengan styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #2E8B57; font-size: 3rem; margin-bottom: 0.5rem;">
            🌱 Energy Data Analyst Dashboard
        </h1>
        <h2 style="color: #4682B4; font-size: 1.5rem; margin-bottom: 2rem;">
            Monitoring and Optimization of Renewables
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Informasi identitas dalam card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    ">
        <h2 style="margin-bottom: 1rem; font-size: 2.5rem;">👨‍💻 Muhamad Galih</h2>
        <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Peserta Kursus Talent Class Kemnaker</p>
        <p style="font-size: 1rem; opacity: 0.9;">Energy Data Analyst: Monitoring and Optimization of Renewables</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Informasi kursus
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Tentang Kursus
        
        **Program:** Talent Class Kemnaker  
        **Judul:** Energy Data Analyst: Monitoring and Optimization of Renewables  
        **Fokus:** Analisis data energi terbarukan untuk optimasi dan monitoring  
        
        **Kompetensi yang Dipelajari:**
        - 📊 Analisis statistik deskriptif
        - 🧹 Data cleaning dan preprocessing
        - 🔍 Deteksi dan penanganan outlier
        - 📈 Analisis korelasi antar variabel
        - 📋 Visualisasi data energi
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Tujuan Aplikasi
        
        Aplikasi ini dibuat untuk mendokumentasikan semua tugas dan pembelajaran 
        dari kursus Energy Data Analyst. Setiap tugas mencakup:
        
        - ✅ Implementasi teknik analisis data
        - 📊 Visualisasi hasil analisis
        - 📝 Dokumentasi proses dan temuan
        - 💾 Export hasil untuk referensi
        
        **Status Tugas:**
        - ✅ Tugas 1: Energy Data Analysis
        - ✅ Tugas 2: Basic Forecasting
        """)
    
    # Statistik aplikasi
    st.markdown("---")
    st.markdown("### 📊 Ringkasan Aplikasi")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Tugas", "2", "2 Selesai")
    with col2:
        st.metric("📊 Dataset", "2", "Energy + Forecast")
    with col3:
        st.metric("🔧 Tools", "Python", "ML + Statistics")
    with col4:
        st.metric("📈 Status", "Complete", "All Done")

# Sidebar navigation
st.sidebar.title('🧭 Navigasi')
st.sidebar.markdown("---")

# Navigation buttons
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home Dashboard", "📊 Tugas 1: Energy Data Analysis", "📈 Tugas 2: Forecasting"],
    index=0
)

# Main content based on selection
if page == "🏠 Home Dashboard":
    show_home_dashboard()
elif page == "📊 Tugas 1: Energy Data Analysis":
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tugas 1: Analisis data energi terbarukan dengan fokus pada statistik deskriptif, data cleaning, dan analisis korelasi.")
    tugas1_ui.run()
elif page == "📈 Tugas 2: Forecasting":
    st.sidebar.markdown("---")
    st.sidebar.info("� Tugas 2: Forecasting solar energy output menggunakan machine learning dan time series analysis.")
    tugas2_ui.run()
