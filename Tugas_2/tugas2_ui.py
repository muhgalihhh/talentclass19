import matplotlib

matplotlib.use('Agg')
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def _load_data():
    df = pd.read_csv('Tugas_2/forecasting_dataset.csv')
    # Normalise column names (remove units for easier referencing)
    rename_map = {
        'Irradiance (kWh/m²)': 'Irradiance',
        'Temperature (°C)': 'Temperature',
        'Humidity (%)': 'Humidity',
        'Solar Output (MWh)': 'Solar_Output'
    }
    df = df.rename(columns=rename_map)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_Number'] = (df['Date'] - df['Date'].min()).dt.days + 1
    return df


def _simple_linear_regression(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = model.score(x.reshape(-1, 1), y)
    coef = model.coef_[0]
    intercept = model.intercept_
    return model, y_pred, r2, coef, intercept


def _multivariate_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    return model, y_pred, r2


def _evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # Hitung MSE lalu akar kuadrat manual agar kompatibel dengan versi scikit-learn lama
    # (beberapa versi tidak mendukung argumen squared=False)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Hindari pembagian dengan nol
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    return mae, rmse, mape


def run():
    st.title('📈 Tugas 2: Basic Forecasting – Solar Energy')
    st.markdown('**Linear Regression, Time Feature Engineering, and Multivariate Modeling**')
    st.markdown('---')

    with st.spinner('📂 Loading dataset...'):
        df = _load_data()
    st.success(f"Dataset loaded: {df.shape[0]} rows")

    with st.expander('👀 Dataset Preview'):
        st.dataframe(df.head())
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric('Rows', df.shape[0])
        with col2: st.metric('Columns', df.shape[1])
        with col3: st.metric('Date Range', f"{df['Date'].min().date()} → {df['Date'].max().date()}")
        with col4: st.metric('Missing Values', int(df.isna().sum().sum()))

    st.markdown('---')
    # ================= BAGIAN 1 =================
    st.header('🔹 Bagian 1 – Regresi Linear Sederhana (Irradiance → Solar Output)')
    X_irrad = df['Irradiance'].values
    y = df['Solar_Output'].values
    model1, y_pred1, r2_1, coef1, intercept1 = _simple_linear_regression(X_irrad, y)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(X_irrad, y, color='gold', alpha=0.7, label='Actual')
        x_line = np.linspace(X_irrad.min(), X_irrad.max(), 100)
        y_line = model1.predict(x_line.reshape(-1,1))
        ax.plot(x_line, y_line, color='red', label='Trendline')
        ax.set_xlabel('Irradiance (kWh/m²)')
        ax.set_ylabel('Solar Output (MWh)')
        ax.set_title('Linear Regression: Irradiance vs Solar Output')
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader('Persamaan Regresi')
        st.write(f"Y = {intercept1:.4f} + {coef1:.4f} * Irradiance")
        st.metric('R²', f"{r2_1:.4f}")
        st.caption('Menunjukkan seberapa besar variasi Solar Output dijelaskan oleh Irradiance.')

    # Tabel Actual vs Prediksi untuk Bagian 1
    with st.expander('📋 Tabel Actual vs Predicted – Model Irradiance'):
        mae1, rmse1, mape1 = _evaluate(y, y_pred1)
        part1_df = pd.DataFrame({
            'Date': df['Date'],
            'Irradiance': df['Irradiance'],
            'Actual_Solar_Output': y,
            'Predicted_Solar_Output': y_pred1,
            'Error': y - y_pred1
        })
        st.dataframe(part1_df.head(30), use_container_width=True)
        m1c1, m1c2, m1c3 = st.columns(3)
        with m1c1: st.metric('MAE', f"{mae1:.4f}")
        with m1c2: st.metric('RMSE', f"{rmse1:.4f}")
        with m1c3: st.metric('MAPE (%)', f"{mape1:.2f}")
        # Download button
        buf1 = io.StringIO()
        part1_df.to_csv(buf1, index=False)
        st.download_button(
            label='⬇️ Download Tabel (CSV)',
            data=buf1.getvalue(),
            file_name='bagian1_actual_vs_predicted.csv',
            mime='text/csv'
        )

    st.markdown('---')
    # ================= BAGIAN 2 =================
    st.header('🔹 Bagian 2 – Menambahkan Fitur Waktu (Day Number)')
    X_day = df['Day_Number'].values
    model2, y_pred2, r2_2, coef2, intercept2 = _simple_linear_regression(X_day, y)

    col1, col2 = st.columns([2,1])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(7,5))
        ax2.scatter(X_day, y, color='skyblue', alpha=0.7, label='Actual')
        x_line2 = np.linspace(X_day.min(), X_day.max(), 100)
        y_line2 = model2.predict(x_line2.reshape(-1,1))
        ax2.plot(x_line2, y_line2, color='darkblue', label='Trendline')
        ax2.set_xlabel('Day Number')
        ax2.set_ylabel('Solar Output (MWh)')
        ax2.set_title('Linear Regression: Day Number vs Solar Output')
        ax2.grid(alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
    with col2:
        st.subheader('Persamaan Regresi Waktu')
        st.write(f"Y = {intercept2:.4f} + {coef2:.4f} * Day_Number")
        st.metric('R² (Time)', f"{r2_2:.4f}")
        diff = r2_2 - r2_1
        st.caption(f"Perbandingan dengan model Irradiance: ΔR² = {diff:.4f}")

    st.info('Interpretasi: Jika R² meningkat signifikan, waktu (trend) berkontribusi pada variasi Solar Output.')

    st.markdown('---')
    # ================= BAGIAN 3 =================
    st.header('🔹 Bagian 3 – Regresi Multivariat (Irradiance, Temperature, Humidity)')
    feature_cols = ['Irradiance', 'Temperature', 'Humidity']
    X_multi = df[feature_cols].values
    model3, y_pred3, r2_3 = _multivariate_regression(X_multi, y)
    # statsmodels untuk p-value & adjusted R2
    X_sm = sm.add_constant(df[feature_cols])
    ols_model = sm.OLS(y, X_sm).fit()
    adj_r2_3 = ols_model.rsquared_adj

    st.subheader('Koefisien & Signifikansi')
    coef_df = pd.DataFrame({
        'Feature': ['Intercept'] + feature_cols,
        'Coefficient': ols_model.params.values,
        'p-value': ols_model.pvalues.values
    })
    # Tambahkan interpretasi signifikansi
    def signif(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        if p < 0.1: return '.'
        return ''
    coef_df['Signif'] = coef_df['p-value'].apply(signif)
    st.dataframe(coef_df.style.format({'Coefficient': '{:.5f}', 'p-value': '{:.4f}'}), use_container_width=True)
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric('R² (Multivariate)', f"{r2_3:.4f}")
    with mcol2:
        st.metric('Adjusted R²', f"{adj_r2_3:.4f}")
    st.caption('Signif legend: *** <0.001, ** <0.01, * <0.05, . <0.1')

    # Feature importance (absolute coef scale normalized)
    imp = np.abs(model3.coef_)
    imp_norm = imp / imp.sum()
    fig_imp, ax_imp = plt.subplots(figsize=(5,4))
    sns.barplot(x=imp_norm, y=feature_cols, ax=ax_imp, palette='viridis')
    ax_imp.set_xlabel('Normalized Importance')
    ax_imp.set_title('Feature Importance (|Coefficient| normalized)')
    st.pyplot(fig_imp)

    st.caption('Catatan: Nilai koefisien belum distandardisasi; perbandingan magnitude dapat dipengaruhi oleh skala variabel.')

    with st.expander('🔍 Standardized Coefficients (Optional)'):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_multi)
        model_std, y_pred_std, _ = _multivariate_regression(X_scaled, y)
        std_coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Std_Coefficient': model_std.coef_
        }).sort_values('Std_Coefficient', key=lambda x: np.abs(x), ascending=False)
        st.dataframe(std_coef_df)
        st.caption('Koefisien setelah standardisasi dapat dibandingkan langsung antar fitur.')

    st.markdown('---')
    # ================= BAGIAN 4 =================
    st.header('🔹 Bagian 4 – Evaluasi Model')
    eval_rows = []
    for name, pred, r2 in [
        ('Irradiance → Output', y_pred1, r2_1),
        ('Day_Number → Output', y_pred2, r2_2),
        ('Multivariate', y_pred3, r2_3)
    ]:
        mae, rmse, mape = _evaluate(y, pred)
        eval_rows.append({'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape})
    eval_df = pd.DataFrame(eval_rows).round(4)
    st.subheader('📊 Ringkasan Metrik')
    st.dataframe(eval_df, use_container_width=True)

    best_model = eval_df.sort_values('RMSE').iloc[0]
    st.success(f"Model terbaik (berdasarkan RMSE terendah): {best_model['Model']} – RMSE={best_model['RMSE']:.4f}, MAPE={best_model['MAPE (%)']:.2f}%")

    with st.expander('📈 Plot Actual vs Predicted (Multivariate)'):
        figp, axp = plt.subplots(figsize=(7,5))
        axp.plot(df['Date'], y, label='Actual', marker='o')
        axp.plot(df['Date'], y_pred3, label='Predicted', marker='x')
        axp.set_xlabel('Date')
        axp.set_ylabel('Solar Output (MWh)')
        axp.set_title('Actual vs Predicted – Multivariate Model')
        axp.grid(alpha=0.3)
        axp.legend()
        st.pyplot(figp)

    with st.expander('📉 Error Distribution (Multivariate)'):
        errors = y - y_pred3
        figerr, axerr = plt.subplots(figsize=(6,4))
        sns.histplot(errors, bins=15, kde=True, ax=axerr, color='salmon')
        axerr.set_title('Residual Distribution')
        axerr.set_xlabel('Error (Actual - Predicted)')
        st.pyplot(figerr)

    # Interpretation
    st.markdown('---')
    st.header('🧠 Interpretasi & Insight')
    # Tentukan fitur paling berpengaruh (mengabaikan intercept)
    coef_only = coef_df[coef_df['Feature'] != 'Intercept']
    strongest_feature = coef_only.iloc[np.argmax(np.abs(coef_only['Coefficient']))]['Feature']
    interpretation = f"Fitur paling berpengaruh (berdasarkan |koefisien| OLS): {strongest_feature}. "\
                     f"Model multivariat meningkatkan R² menjadi {r2_3:.3f} dibanding model sederhana (Irradiance R²={r2_1:.3f}). "\
                     f"Hal ini menunjukkan bahwa memasukkan variabel lingkungan lain membantu menjelaskan variasi Solar Output."
    st.write(interpretation)

    st.info('Rekomendasi: Gunakan model multivariat untuk perencanaan produksi energi harian dan monitoring anomali.')

    # Downloads
    st.markdown('---')
    st.header('💾 Download Outputs')
    # Combined predictions table
    result_df = df[['Date','Irradiance','Temperature','Humidity','Solar_Output']].copy()
    result_df['Pred_Irradiance'] = y_pred1
    result_df['Pred_Day'] = y_pred2
    result_df['Pred_Multivariate'] = y_pred3
    result_df['Error_Multivariate'] = result_df['Solar_Output'] - result_df['Pred_Multivariate']

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label='📥 Download Prediction Results (CSV)',
        data=csv_buffer.getvalue(),
        file_name='tugas2_predictions.csv',
        mime='text/csv'
    )

    eval_buffer = io.StringIO()
    eval_df.to_csv(eval_buffer, index=False)
    st.download_button(
        label='📊 Download Evaluation Metrics (CSV)',
        data=eval_buffer.getvalue(),
        file_name='tugas2_evaluation.csv',
        mime='text/csv'
    )

    st.success('🎉 Forecasting Task Completed!')
