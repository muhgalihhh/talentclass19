import matplotlib

matplotlib.use('Agg')  # Set backend for Streamlit Cloud compatibility

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Set modern style for matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def run():
    # Main title with emoji and styling
    st.title("📈 Tugas 2: Basic Forecasting")
    st.markdown("**Praktik Forecasting Energi Surya: Linear & Multivariable Regression**")
    st.markdown("**Durasi:** ±1 jam | **Dataset:** forecasting_dataset.csv")
    st.markdown("---")
    
    # Tujuan pembelajaran
    with st.expander("🎯 Tujuan Pembelajaran"):
        st.markdown("""
        - ✅ Memahami penerapan regresi linear untuk forecasting
        - ✅ Menambahkan variabel waktu sebagai fitur prediktif  
        - ✅ Membangun model regresi multivariat
        - ✅ Mengevaluasi performa model dengan metrik statistik
        """)
    
    # Load data with progress indicator
    with st.spinner("📂 Loading forecasting dataset..."):
        df = pd.read_csv('Tugas_2/forecasting_dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        # Add Day Number as requested in instructions
        df['Day_Number'] = range(1, len(df) + 1)
    
    # Show dataset overview
    st.success(f"✅ Dataset loaded successfully! Total records: {len(df)}")
    
    # Dataset description
    with st.expander("📊 Deskripsi Dataset"):
        st.markdown("""
        | Kolom | Deskripsi |
        |-------|-----------|
        | Date | Tanggal pencatatan data (format harian) |
        | Irradiance (kWh/m²) | Intensitas radiasi matahari |
        | Temperature (°C) | Suhu lingkungan |
        | Humidity (%) | Kelembaban lingkungan |
        | Solar Output (MWh) | Output energi surya aktual |
        """)
        
        st.write("**Preview Dataset:**")
        display_df = df[['Date', 'Irradiance (kWh/m²)', 'Temperature (°C)', 'Humidity (%)', 'Solar Output (MWh)', 'Day_Number']].head(10)
        st.dataframe(display_df, width=800)
    
    st.markdown("---")
    
    # ============ BAGIAN 1: REGRESI LINEAR SEDERHANA ============
    st.header("📊 Bagian 1 - Regresi Linear Sederhana")
    st.markdown("**Tujuan:** Memprediksi Solar Output berdasarkan Irradiance")
    
    # Linear regression: Irradiance vs Solar Output
    X_irradiance = df[['Irradiance (kWh/m²)']]
    y = df['Solar Output (MWh)']
    
    model_irradiance = LinearRegression()
    model_irradiance.fit(X_irradiance, y)
    y_pred_irradiance = model_irradiance.predict(X_irradiance)
    
    # Calculate R²
    r2_irradiance = model_irradiance.score(X_irradiance, y)
    
    # Get regression equation
    coef_irradiance = model_irradiance.coef_[0]
    intercept_irradiance = model_irradiance.intercept_
    
    # Scatter plot with trendline (Enhanced visualization)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with gradient colors
    scatter = ax.scatter(df['Irradiance (kWh/m²)'], df['Solar Output (MWh)'], 
                        c=df['Temperature (°C)'], cmap='viridis', 
                        alpha=0.7, s=60, edgecolors='white', linewidth=0.5, 
                        label='Data Points')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20, fontsize=12)
    
    # Trendline with enhanced styling
    ax.plot(df['Irradiance (kWh/m²)'], y_pred_irradiance, 
            color='#FF6B6B', linewidth=3, linestyle='--', 
            label='Regression Line', alpha=0.9)
    
    # Enhanced styling
    ax.set_xlabel('Irradiance (kWh/m²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solar Output (MWh)', fontsize=14, fontweight='bold')
    ax.set_title('🌞 Scatter Plot: Irradiance vs Solar Output\n(Colored by Temperature)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(loc='upper left', framealpha=0.9, shadow=True)
    
    # Enhanced equation box
    equation_text = f'Regression Equation:\ny = {coef_irradiance:.4f}x + {intercept_irradiance:.4f}\nR² = {r2_irradiance:.4f}'
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                     edgecolor='navy', alpha=0.9, linewidth=2))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Persamaan Regresi")
        st.write(f"**y = {coef_irradiance:.4f}x + {intercept_irradiance:.4f}**")
        st.write(f"**R² = {r2_irradiance:.4f}**")
    
    with col2:
        st.subheader("💡 Interpretasi")
        strength = "Kuat" if r2_irradiance > 0.7 else "Sedang" if r2_irradiance > 0.5 else "Lemah"
        st.write(f"Kekuatan hubungan: **{strength}**")
        st.write(f"Setiap kenaikan 1 kWh/m² irradiance meningkatkan solar output sebesar {coef_irradiance:.4f} MWh")
    
    st.markdown("---")
    
    # ============ BAGIAN 2: TAMBAH FITUR WAKTU ============
    st.header("⏰ Bagian 2 - Tambah Fitur Waktu")
    st.markdown("**Tujuan:** Mengetahui apakah waktu punya pengaruh terhadap output")
    
    # Linear regression: Day Number vs Solar Output
    X_day = df[['Day_Number']]
    
    model_day = LinearRegression()
    model_day.fit(X_day, y)
    y_pred_day = model_day.predict(X_day)
    
    # Calculate R²
    r2_day = model_day.score(X_day, y)
    
    # Get regression equation
    coef_day = model_day.coef_[0]
    intercept_day = model_day.intercept_
    
    # Enhanced scatter plot for Day Number
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with seasonal colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(df)))
    scatter = ax.scatter(df['Day_Number'], df['Solar Output (MWh)'], 
                        c=colors, alpha=0.7, s=70, edgecolors='white', 
                        linewidth=0.5, label='Data Points')
    
    # Enhanced trendline
    ax.plot(df['Day_Number'], y_pred_day, 
            color='#FF4444', linewidth=3, linestyle='-', 
            label='Regression Line', alpha=0.9)
    
    # Add confidence interval shading
    residuals = y - y_pred_day
    std_residuals = np.std(residuals)
    upper_bound = y_pred_day + 1.96 * std_residuals
    lower_bound = y_pred_day - 1.96 * std_residuals
    
    ax.fill_between(df['Day_Number'], lower_bound, upper_bound, 
                   alpha=0.2, color='red', label='95% Confidence Interval')
    
    # Enhanced styling
    ax.set_xlabel('Day Number (Sequential Days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solar Output (MWh)', fontsize=14, fontweight='bold')
    ax.set_title('📅 Time Series: Day Number vs Solar Output\n(with Confidence Interval)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(loc='best', framealpha=0.9, shadow=True)
    
    # Enhanced equation box
    equation_text_day = f'Temporal Regression:\ny = {coef_day:.6f}x + {intercept_day:.4f}\nR² = {r2_day:.4f}'
    ax.text(0.02, 0.98, equation_text_day, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', 
                     edgecolor='darkgreen', alpha=0.9, linewidth=2))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Comparison of R² values with enhanced visualization
    st.subheader("📊 Perbandingan Nilai R²")
    
    # Create enhanced comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Bar chart comparison
    models = ['Irradiance\nvs Solar Output', 'Day Number\nvs Solar Output']
    r2_values = [r2_irradiance, r2_day]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(models, r2_values, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2, width=0.6)
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison\n(R² Scores)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(r2_values) * 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, r2_val in zip(bars, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Right: Donut chart showing model strength
    def get_strength_category(r2):
        if r2 > 0.7:
            return "Strong", "#2ECC71"
        elif r2 > 0.5:
            return "Moderate", "#F39C12"
        else:
            return "Weak", "#E74C3C"
    
    strength_irr, color_irr = get_strength_category(r2_irradiance)
    strength_day, color_day = get_strength_category(r2_day)
    
    # Create donut chart
    sizes = [r2_irradiance, r2_day]
    labels = [f'Irradiance\n({strength_irr})', f'Day Number\n({strength_day})']
    colors_donut = [color_irr, color_day]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_donut, 
                           autopct='%1.3f', startangle=90, 
                           wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title('Model Strength Distribution\n(R² Values)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display comparison table with enhanced styling
    comparison_df = pd.DataFrame({
        'Model': ['Irradiance vs Solar Output', 'Day Number vs Solar Output'],
        'R²': [r2_irradiance, r2_day],
        'Strength': [get_strength_category(r2_irradiance)[0], get_strength_category(r2_day)[0]],
        'Persamaan': [f'y = {coef_irradiance:.4f}x + {intercept_irradiance:.4f}',
                     f'y = {coef_day:.6f}x + {intercept_day:.4f}']
    })
    st.dataframe(comparison_df, width=900)
    
    # Interpretation
    if r2_irradiance > r2_day:
        st.info("💡 **Kesimpulan:** Irradiance memiliki hubungan yang lebih kuat dengan Solar Output dibanding Day Number")
    else:
        st.info("💡 **Kesimpulan:** Day Number memiliki hubungan yang lebih kuat dengan Solar Output dibanding Irradiance")
    
    st.markdown("---")
    
    # ============ BAGIAN 3: REGRESI MULTIVARIAT ============
    st.header("🔢 Bagian 3 - Regresi Multivariat")
    st.markdown("**Tujuan:** Meningkatkan akurasi dengan banyak variabel")
    
    # Multiple regression: Irradiance, Temperature, Humidity as predictors
    X_multi = df[['Irradiance (kWh/m²)', 'Temperature (°C)', 'Humidity (%)']]
    
    model_multi = LinearRegression()
    model_multi.fit(X_multi, y)
    y_pred_multi = model_multi.predict(X_multi)
    
    # Calculate metrics
    r2_multi = model_multi.score(X_multi, y)
    adjusted_r2 = 1 - (1 - r2_multi) * (len(y) - 1) / (len(y) - X_multi.shape[1] - 1)
    
    # Get coefficients
    coefficients = model_multi.coef_
    intercept_multi = model_multi.intercept_
    feature_names = X_multi.columns
    
    # Create results table
    results_df = pd.DataFrame({
        'Fitur': ['Intercept'] + list(feature_names),
        'Koefisien': [intercept_multi] + list(coefficients),
        'Interpretasi': [
            'Nilai dasar ketika semua fitur = 0',
            f'Pengaruh per 1 kWh/m² irradiance',
            f'Pengaruh per 1°C temperature',
            f'Pengaruh per 1% humidity'
        ]
    })
    
    st.subheader("📋 Hasil Regresi Multivariat")
    st.dataframe(results_df, width=700)
    
    # Display R² metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{r2_multi:.4f}")
    with col2:
        st.metric("Adjusted R²", f"{adjusted_r2:.4f}")
    with col3:
        st.metric("Fitur", len(feature_names))
    
    # Regression equation
    st.subheader("📈 Persamaan Regresi Multivariat")
    equation_parts = [f"{intercept_multi:.4f}"]
    for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
        sign = "+" if coef >= 0 else ""
        equation_parts.append(f"{sign}{coef:.4f}×{feature}")
    
    equation_multi = f"**Solar Output = {' '.join(equation_parts)}**"
    st.markdown(equation_multi)
    
    # Enhanced Actual vs Predicted plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with residual coloring
    residuals = y - y_pred_multi
    scatter = ax.scatter(y, y_pred_multi, c=residuals, cmap='RdYlBu_r', 
                        alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
    
    # Add colorbar for residuals
    cbar = plt.colorbar(scatter)
    cbar.set_label('Residuals (Actual - Predicted)', rotation=270, labelpad=20, fontsize=12)
    
    # Perfect prediction line
    perfect_line = [y.min(), y.max()]
    ax.plot(perfect_line, perfect_line, 'k--', linewidth=3, 
            label='Perfect Prediction', alpha=0.8)
    
    # Add confidence bands
    x_line = np.linspace(y.min(), y.max(), 100)
    rmse = np.sqrt(mean_squared_error(y, y_pred_multi))
    ax.fill_between(x_line, x_line - rmse, x_line + rmse, 
                   alpha=0.2, color='gray', label=f'±RMSE ({rmse:.3f})')
    
    # Enhanced styling
    ax.set_xlabel('Actual Solar Output (MWh)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Solar Output (MWh)', fontsize=14, fontweight='bold')
    ax.set_title('🎯 Model Performance: Actual vs Predicted\n(Multivariable Regression)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(loc='upper left', framealpha=0.9, shadow=True)
    
    # Enhanced stats box
    stats_text = f'Model Performance:\nR² = {r2_multi:.4f}\nRMSE = {rmse:.4f}\nMAE = {mean_absolute_error(y, y_pred_multi):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                     edgecolor='orange', alpha=0.9, linewidth=2))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # ============ BAGIAN 4: EVALUASI MODEL ============
    st.header("📊 Bagian 4 - Evaluasi Model")
    st.markdown("**Tujuan:** Mengukur keakuratan hasil prediksi model")
    
    # Calculate error metrics for all models
    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mae, rmse, mape
    
    # Calculate metrics for all models
    mae_irr, rmse_irr, mape_irr = calculate_metrics(y, y_pred_irradiance)
    mae_day, rmse_day, mape_day = calculate_metrics(y, y_pred_day)
    mae_multi, rmse_multi, mape_multi = calculate_metrics(y, y_pred_multi)
    
    # Create performance comparison table
    performance_df = pd.DataFrame({
        'Model': ['Irradiance Only', 'Day Number Only', 'Multivariable'],
        'R²': [r2_irradiance, r2_day, r2_multi],
        'MAE': [mae_irr, mae_day, mae_multi],
        'RMSE': [rmse_irr, rmse_day, rmse_multi],
        'MAPE (%)': [mape_irr, mape_day, mape_multi]
    })
    
    st.subheader("📈 Ringkasan Evaluasi Model")
    
    # Create enhanced performance visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: R² Comparison
    models = ['Irradiance\nOnly', 'Day Number\nOnly', 'Multivariable']
    r2_vals = [r2_irradiance, r2_day, r2_multi]
    colors_r2 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars1 = ax1.bar(models, r2_vals, color=colors_r2, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('R² Score Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, r2_val in zip(bars1, r2_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MAE Comparison
    mae_vals = [mae_irr, mae_day, mae_multi]
    bars2 = ax2.bar(models, mae_vals, color=colors_r2, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel('MAE (Mean Absolute Error)', fontweight='bold')
    ax2.set_title('MAE Comparison (Lower is Better)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mae_val in zip(bars2, mae_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mae_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: RMSE Comparison
    rmse_vals = [rmse_irr, rmse_day, rmse_multi]
    bars3 = ax3.bar(models, rmse_vals, color=colors_r2, alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_ylabel('RMSE (Root Mean Squared Error)', fontweight='bold')
    ax3.set_title('RMSE Comparison (Lower is Better)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, rmse_val in zip(bars3, rmse_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{rmse_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: MAPE Comparison
    mape_vals = [mape_irr, mape_day, mape_multi]
    bars4 = ax4.bar(models, mape_vals, color=colors_r2, alpha=0.8, edgecolor='white', linewidth=2)
    ax4.set_ylabel('MAPE (%)', fontweight='bold')
    ax4.set_title('MAPE Comparison (Lower is Better)', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, mape_val in zip(bars4, mape_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mape_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Enhanced performance table with color coding
    performance_df = pd.DataFrame({
        'Model': ['Irradiance Only', 'Day Number Only', 'Multivariable'],
        'R²': [r2_irradiance, r2_day, r2_multi],
        'MAE': [mae_irr, mae_day, mae_multi],
        'RMSE': [rmse_irr, rmse_day, rmse_multi],
        'MAPE (%)': [mape_irr, mape_day, mape_multi]
    })
    st.dataframe(performance_df.round(4), width=900)
    
    # Add prediction and error columns to dataset
    df_results = df.copy()
    df_results['Predicted_Irradiance'] = y_pred_irradiance
    df_results['Error_Irradiance'] = y_pred_irradiance - y
    df_results['Predicted_Multi'] = y_pred_multi
    df_results['Error_Multi'] = y_pred_multi - y
    
    # Show sample predictions with enhanced visualization
    with st.expander("📋 Sample Predictions dan Errors"):
        sample_cols = ['Date', 'Solar Output (MWh)', 'Predicted_Multi', 'Error_Multi']
        sample_data = df_results[sample_cols].head(10)
        st.dataframe(sample_data, width=700)
        
        # Add residual analysis plot
        st.markdown("**📊 Residual Analysis (First 20 observations):**")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residual plot
        sample_20 = df_results.head(20)
        x_range = range(len(sample_20))
        
        ax1.bar(x_range, sample_20['Error_Multi'], 
               color=['red' if x < 0 else 'green' for x in sample_20['Error_Multi']],
               alpha=0.7, edgecolor='white', linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax1.set_xlabel('Observation Number', fontweight='bold')
        ax1.set_ylabel('Residual (Error)', fontweight='bold')
        ax1.set_title('Residual Plot (First 20 Obs.)\nPositive = Overprediction, Negative = Underprediction', 
                     fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Actual vs Predicted for sample
        ax2.scatter(sample_20['Solar Output (MWh)'], sample_20['Predicted_Multi'], 
                   c=sample_20['Error_Multi'], cmap='RdYlGn_r', s=100, 
                   edgecolors='white', linewidth=1, alpha=0.8)
        
        # Perfect prediction line
        min_val = min(sample_20['Solar Output (MWh)'].min(), sample_20['Predicted_Multi'].min())
        max_val = max(sample_20['Solar Output (MWh)'].max(), sample_20['Predicted_Multi'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Actual Solar Output (MWh)', fontweight='bold')
        ax2.set_ylabel('Predicted Solar Output (MWh)', fontweight='bold')
        ax2.set_title('Actual vs Predicted (Sample)\nColor = Residual Magnitude', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Enhanced Feature importance visualization
    st.subheader("🔍 Analisis Fitur")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Horizontal bar chart
    abs_coef = np.abs(coefficients)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    y_pos = np.arange(len(feature_names))
    
    bars = ax1.barh(y_pos, abs_coef, color=colors, alpha=0.8, 
                    edgecolor='white', linewidth=2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names, fontsize=12)
    ax1.set_xlabel('|Coefficient| (Absolute Value)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance\n(Absolute Coefficients)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, coef) in enumerate(zip(bars, abs_coef)):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{coef:.4f}', ha='left', va='center', fontweight='bold')
    
    # Right plot: Pie chart showing relative importance
    sizes = abs_coef / abs_coef.sum() * 100
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=feature_names, 
                                      colors=colors, autopct='%1.1f%%',
                                      shadow=True, startangle=90,
                                      textprops={'fontsize': 11})
    
    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Relative Feature Importance\n(Percentage Distribution)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Final interpretation
    st.subheader("💡 Interpretasi Hasil")
    
    best_model = performance_df.loc[performance_df['R²'].idxmax(), 'Model']
    most_important_feature = feature_names[np.argmax(abs_coef)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Terbaik:**")
        st.success(f"🏆 {best_model}")
        st.write(f"R² = {performance_df.loc[performance_df['R²'].idxmax(), 'R²']:.4f}")
    
    with col2:
        st.markdown("**Fitur Paling Penting:**")
        st.info(f"📊 {most_important_feature}")
        st.write(f"Koefisien = {coefficients[np.argmax(abs_coef)]:.4f}")
    
    # Summary insights
    st.markdown("---")
    st.subheader("📝 Kesimpulan Analisis")
    
    insights = f"""
    **Hasil Utama:**
    - Model terbaik: **{best_model}** dengan R² = {performance_df.loc[performance_df['R²'].idxmax(), 'R²']:.4f}
    - Fitur paling berpengaruh: **{most_important_feature}**
    - RMSE model multivariat: **{rmse_multi:.4f} MWh**
    - MAPE model multivariat: **{mape_multi:.2f}%**
    
    **Rekomendasi:**
    - Fokus monitoring pada {most_important_feature.lower()} untuk prediksi yang lebih akurat
    - Model multivariat memberikan prediksi terbaik dengan mengkombinasikan semua fitur
    - Akurasi prediksi dapat ditingkatkan dengan data historis yang lebih panjang
    """
    
    st.markdown(insights)
    
    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    with col1:
        results_csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset dengan Prediksi",
            data=results_csv,
            file_name="forecasting_results.csv",
            mime="text/csv"
        )
    
    with col2:
        performance_csv = performance_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Performance Metrics",
            data=performance_csv,
            file_name="model_performance_comparison.csv",
            mime="text/csv"
        )
    
    st.success("🎉 Forecasting analysis selesai! Semua tahap regresi linear dan multivariat telah dilakukan sesuai instruksi.")
