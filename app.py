import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.linear_model import LinearRegression, HuberRegressor

# Page Configuration
st.set_page_config(page_title="SMU Admission Analysis", layout="wide")
st.title("🎓 Sangmyung University Admission Statistical Analysis")
st.markdown("### Interactive Dashboard & AI Ensemble Prediction")

# ---------------------------------------------------------
# 1. DATA LOADING & CLEANING FUNCTIONS
# ---------------------------------------------------------

@st.cache_data
def load_data():
    """
    Loads and processes both data files assuming they are saved as CSV UTF-8.
    """
    
    # --- Part A: Load the Main History File (2020-2023, 2025) ---
    try:
        df_main_raw = pd.read_csv("smu_admission_results.xlsx - 상명대입결.csv", header=None, encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading Main History file: {e}")
        return pd.DataFrame()

    # Map based on file inspection:
    years_map = {
        2025: (2, 8),
        2023: (8, 14),
        2022: (14, 20),
        2021: (20, 26),
        2020: (26, 32)
    }
    
    clean_rows = []

    if not df_main_raw.empty:
        start_row = 3 if len(df_main_raw) > 3 else 0
        
        for index, row in df_main_raw.iloc[start_row:].iterrows():
            if len(row) < 32:
                padding = pd.Series([np.nan] * (32 - len(row)))
                row = pd.concat([row, padding], ignore_index=True)

            dept_group = row[0]
            major = row[1]
            
            if pd.isna(major):
                continue
                
            for year, (start_col, end_col) in years_map.items():
                try:
                    year_data = row.iloc[start_col:end_col].values
                except:
                    continue
                
                if len(year_data) < 6 or all(pd.isna(x) for x in year_data):
                    continue
                    
                try:
                    clean_rows.append({
                        "Year": int(year),
                        "Department": dept_group,
                        "Major": major,
                        "Category": "Initial Accepted (최초합격자)",
                        "Max": year_data[0],
                        "Avg": year_data[1],
                        "Min": year_data[2]
                    })
                    clean_rows.append({
                        "Year": int(year),
                        "Department": dept_group,
                        "Major": major,
                        "Category": "Final Registered (최종등록자)",
                        "Max": year_data[3],
                        "Avg": year_data[4],
                        "Min": year_data[5]
                    })
                except IndexError:
                    continue

    df_history = pd.DataFrame(clean_rows)

    # --- Part B: Load the 2024 File ---
    try:
        df_2024_raw = pd.read_csv("smu_admission_results.xlsx - 24학년도 수시입결.csv", encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading 2024 file: {e}")
        return df_history

    rows_2024 = []
    current_dept = None
    
    if not df_2024_raw.empty:
        if len(df_2024_raw.columns) >= 3:
            for index, row in df_2024_raw.iterrows():
                raw_dept = row.iloc[0]
                raw_major = row.iloc[1]
                raw_grade = row.iloc[2]
                
                if pd.notna(raw_dept) and str(raw_dept).strip() != "모집단위":
                    current_dept = raw_dept
                    
                if pd.isna(raw_major) or str(raw_major).strip() in ["모집단위", "최종등록자 70% cut 성적"]:
                    continue
                    
                try:
                    grade_val = float(str(raw_grade).replace(',', ''))
                except:
                    grade_val = None

                if grade_val is not None:
                    rows_2024.append({
                        "Year": 2024,
                        "Department": current_dept,
                        "Major": raw_major,
                        "Category": "Final Registered (최종등록자)",
                        "Max": np.nan, 
                        "Avg": grade_val,
                        "Min": np.nan 
                    })

    df_2024 = pd.DataFrame(rows_2024)

    if df_history.empty and df_2024.empty:
        return pd.DataFrame(columns=['Year', 'Major', 'Category', 'Avg']) 

    final_df = pd.concat([df_history, df_2024], ignore_index=True)
    
    cols_to_numeric = ['Max', 'Avg', 'Min']
    for col in cols_to_numeric:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    return final_df

# ---------------------------------------------------------
# 2. ENSEMBLE PREDICTION ENGINE
# ---------------------------------------------------------
def run_ensemble_prediction(df_major):
    """
    Runs Weighted Linear Regression and Robust Huber Regression
    to predict the next year's grade.
    """
    # Filter for Final Registered data only, sorted by year
    data = df_major[df_major['Category'] == "Final Registered (최종등록자)"].dropna(subset=['Avg']).sort_values('Year')
    
    if len(data) < 3:
        return None, None # Not enough data points for regression

    X = data['Year'].values.reshape(-1, 1)
    y = data['Avg'].values
    
    next_year = [[2026]] # Predict for 2026
    
    # --- Model 1: Weighted Linear Regression ---
    # Weight formula: (Year - 2019)^2. 
    # This gives exponential importance to recent years (2025 weight=36 vs 2020 weight=1)
    weights = (data['Year'] - 2019) ** 2
    
    model_weighted = LinearRegression()
    model_weighted.fit(X, y, sample_weight=weights)
    pred_weighted = model_weighted.predict(next_year)[0]
    
    # --- Model 2: Robust Huber Regression ---
    # Robust against outliers (sudden spikes/drops in competition)
    # Epsilon 1.35 is standard for 95% efficiency on normal data
    model_robust = HuberRegressor(epsilon=1.35) 
    model_robust.fit(X, y)
    pred_robust = model_robust.predict(next_year)[0]
    
    # --- Ensemble: Simple Average ---
    final_pred = (pred_weighted + pred_robust) / 2
    
    return final_pred, data

# ---------------------------------------------------------
# 3. APP UI & LOGIC
# ---------------------------------------------------------

try:
    df = load_data()
    if df.empty:
        st.error("No valid data loaded. Please ensure files are saved as 'CSV UTF-8' in Excel.")
        st.stop()
    st.sidebar.success("Data Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.header("1. Select Major")
if 'Major' in df.columns:
    all_majors = sorted(df['Major'].dropna().unique().astype(str))
    selected_major = st.sidebar.selectbox("Major (전공)", all_majors)
else:
    st.error("Could not find 'Major' column.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("2. My Chances Calculator")
user_grade = st.sidebar.number_input("My Grade (나의 내신 등급)", min_value=1.0, max_value=9.0, value=2.5, step=0.1, format="%.2f")

# Filter data
major_data = df[df['Major'] == selected_major].sort_values(by="Year")

# --- Run Prediction ---
predicted_grade, regression_data = run_ensemble_prediction(major_data)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Grade Trends & 2026 Forecast")
    
    fig = go.Figure()

    # Define colors
    color_map = {
        "Initial Accepted (최초합격자)": "#3366CC", 
        "Final Registered (최종등록자)": "#DC3912" 
    }

    # Plot Historical Lines
    for category in major_data['Category'].unique():
        subset = major_data[major_data['Category'] == category]
        fig.add_trace(go.Scatter(
            x=subset['Year'], 
            y=subset['Avg'],
            mode='lines+markers+text',
            name=f"{category} (Avg)",
            line=dict(color=color_map.get(category, "gray"), width=3),
            marker=dict(size=8),
            text=subset['Avg'],
            textposition="top center"
        ))
        
        # Add Range (Min/Max) for Initial
        if category == "Initial Accepted (최초합격자)": 
             fig.add_trace(go.Scatter(x=subset['Year'], y=subset['Max'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
             fig.add_trace(go.Scatter(x=subset['Year'], y=subset['Min'], mode='none', fill='tonexty', fillcolor='rgba(51, 102, 204, 0.1)', showlegend=False, name='Range'))

    # --- Plot Prediction (If available) ---
    if predicted_grade:
        last_year_val = regression_data.iloc[-1]['Avg']
        last_year_x = regression_data.iloc[-1]['Year']
        
        # Draw dotted line connecting last real point to predicted point
        fig.add_trace(go.Scatter(
            x=[last_year_x, 2026],
            y=[last_year_val, predicted_grade],
            mode='lines+markers+text',
            name="2026 AI Prediction",
            line=dict(color="purple", width=3, dash='dot'),
            marker=dict(size=10, symbol='star'),
            text=[None, f"{predicted_grade:.2f}"],
            textposition="top center"
        ))

    # --- User Grade Line ---
    fig.add_hline(y=user_grade, line_dash="dash", line_color="green", annotation_text="My Grade", annotation_position="bottom right")

    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=2020, dtick=1, title="Year"),
        yaxis=dict(title="Grade (Lower is Better)", autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 Ensemble Analysis")
    
    if predicted_grade:
        st.info(f"**Predicted 2026 Cutoff:** {predicted_grade:.2f}")
        
        # Compare User Grade to PREDICTED grade (more advanced)
        diff = user_grade - predicted_grade
        
        st.markdown("Based on the **Ensemble Model** (Weighted + Robust Regression):")
        
        if diff < -0.5:
            st.success(f"**Very Safe**\n\nYour grade ({user_grade}) is well above the projected cutoff ({predicted_grade:.2f}).")
        elif diff <= 0:
            st.success(f"**Safe**\n\nYour grade ({user_grade}) meets the projected cutoff.")
        elif diff < 0.2:
            st.warning(f"**Competitive**\n\nYour grade is slightly below the projection. It will be tight.")
        else:
            st.error(f"**High Risk**\n\nYour grade is significantly lower than the projected cutoff.")

        with st.expander("View Logic Details"):
            st.markdown("""
            **Methodology:**
            1. **Weighted Regression:** Assigns higher importance to recent years (2024, 2025) to capture current momentum.
            2. **Robust Regression (Huber):** Minimizes the impact of outlier years (spikes/drops) to find the stable trend.
            3. **Ensemble:** Averages both models for the final prediction.
            """)
    else:
        st.warning("Insufficient data points to run regression models.")

# ---------------------------------------------------------
# 4. STATISTICAL SUMMARY
# ---------------------------------------------------------
st.divider()
st.subheader("📈 Statistical Insights")
c1, c2, c3 = st.columns(3)

final_data = major_data[major_data['Category'] == "Final Registered (최종등록자)"]
if not final_data.empty:
    c1.metric("Best Final Avg", f"{final_data['Avg'].min():.2f}")
    c2.metric("Lowest Final Avg", f"{final_data['Avg'].max():.2f}")
    c3.metric("5-Year Average", f"{final_data['Avg'].mean():.2f}")
else:
    st.info("Insufficient data.")

st.divider()
with st.expander("View Raw Data"):
    st.dataframe(df)