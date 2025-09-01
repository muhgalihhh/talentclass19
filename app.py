import streamlit as st

from Tugas_1 import tugas1_ui

# Jika ada tugas lain, import juga modulnya
# from Tugas_2 import tugas2_ui

st.sidebar.title('Menu Tugas')
tugas = st.sidebar.selectbox(
    'Pilih Tugas',
    ['Tugas 1', 'Tugas 2 (coming soon)']
)

if tugas == 'Tugas 1':
    tugas1_ui.run()
elif tugas == 'Tugas 2 (coming soon)':
    st.title('Tugas 2')
    st.info('Dokumentasi Tugas 2 akan segera tersedia.')
yes
