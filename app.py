import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import io

# --- 1. CONFIGURATION AND INITIAL SETUP ---
MODEL_PATH = 'models/pdm_rf_model.pkl'

# Cost constants (for calculating repair budget)
COST_LOW_RISK_REPAIR = 200  # Minor maintenance cost
COST_HIGH_RISK_REPAIR = 1500  # Major repair cost

# Thresholds for classification
RISK_THRESHOLD = 37  # Weekly score below which is considered high risk

# --- 2. MODEL LOADING AND ANALYSIS FUNCTIONS ---

@st.cache_resource
def load_model_assets():
    """Loads the trained model and features."""
    try:
        model_assets = joblib.load(MODEL_PATH)
        return model_assets
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please run 'python model_training.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.stop()

# NOTE: @st.cache_data removed to resolve non-hashable argument error
def analyze_data(df, model_assets):
    """
    Analyzes the DataFrame, makes predictions, calculates cost, and structures data.
    """
    model = model_assets['model']
    features = model_assets['features']
    
    # Check if required features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Error: Uploaded CSV is missing required features: {', '.join(missing_features)}. Please check your column names.")
        return None

    # Prepare features for prediction
    X = df[features]
    
    # 1. Predict Risk and Confidence (Probability)
    df['Predicted_Risk'] = model.predict(X) 
    
    # Get probability of being High Risk (Class 1)
    probabilities = model.predict_proba(X)
    df['Risk_Confidence'] = [p[1] for p in probabilities] # Probability of HIGH RISK (1)
    
    # 2. Define Risk Level based on predicted class
    df['Risk_Level'] = df['Predicted_Risk'].apply(
        lambda x: 'HIGH RISK (Action Needed)' if x == 1 else 'Low Compliance Risk'
    )
    
    # 3. Calculate Estimated Cost
    df['Estimated_Repair_Cost'] = df['Predicted_Risk'].apply(
        lambda x: COST_HIGH_RISK_REPAIR if x == 1 else COST_LOW_RISK_REPAIR
    )
    
    return df

# --- 3. DATA AGGREGATION AND PLOTTING FUNCTIONS ---

@st.cache_data
def aggregate_monthly_data(df):
    """Aggregates prediction results by machine and month."""
    
    # Ensure dates are valid
    df['record_date'] = pd.to_datetime(df['record_date'], errors='coerce')
    df.dropna(subset=['record_date'], inplace=True)
    
    # Create Year-Month column for grouping and charting
    df['YearMonth'] = df['record_date'].dt.to_period('M').astype(str)
    
    # Calculate Monthly Failure Probability (Avg Confidence of High Risk)
    monthly_prob = df.groupby('YearMonth')['Risk_Confidence'].mean().reset_index()
    monthly_prob.rename(columns={'Risk_Confidence': 'Avg_Failure_Prob'}, inplace=True)
    monthly_prob['Avg_Failure_Prob'] = monthly_prob['Avg_Failure_Prob'] * 100 # Convert to percentage

    # Calculate Total Estimated Cost (Sum of all predicted repair costs)
    monthly_cost = df.groupby('YearMonth')['Estimated_Repair_Cost'].sum().reset_index()
    monthly_cost.rename(columns={'Estimated_Repair_Cost': 'Total_Estimated_Cost'}, inplace=True)

    # Calculate Average Weekly Score (Compliance Metric)
    monthly_compliance = df.groupby('YearMonth')['weekly_score'].mean().reset_index()
    monthly_compliance.rename(columns={'weekly_score': 'Avg_Weekly_Score'}, inplace=True)

    return monthly_prob, monthly_cost, monthly_compliance

def create_monthly_probability_chart(df_prob):
    """Creates a professional monthly probability bar chart."""
    fig = px.bar(
        df_prob,
        x='YearMonth',
        y='Avg_Failure_Prob',
        color='Avg_Failure_Prob',
        color_continuous_scale=px.colors.sequential.Sunsetdark,
        title='Monthly Failure Prediction Probability (Fleet Average)',
        labels={'YearMonth': 'Month', 'Avg_Failure_Prob': 'Failure Probability (%)'},
        template='plotly_dark'
    )
    fig.update_layout(
        title_font_size=20,
        height=400,
        yaxis_range=[0, df_prob['Avg_Failure_Prob'].max() * 1.2 if not df_prob.empty else 100]
    )
    st.plotly_chart(fig, use_container_width=True)

def create_monthly_cost_chart(df_cost):
    """Creates a monthly estimated repair cost bar chart."""
    fig = px.bar(
        df_cost,
        x='YearMonth',
        y='Total_Estimated_Cost',
        title='Total Estimated Repair Cost by Month (Fleet Budget)',
        labels={'YearMonth': 'Month', 'Total_Estimated_Cost': 'Estimated Cost (USD)'},
        color_discrete_sequence=['#4f46e5'],
        template='plotly_dark'
    )
    fig.update_layout(
        title_font_size=20,
        height=400,
        yaxis_tickprefix='$'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_adherence_chart(df_compliance):
    """Creates a chart for average weekly compliance score."""
    fig = px.line(
        df_compliance,
        x='YearMonth',
        y='Avg_Weekly_Score',
        title='Average Weekly Compliance Score by Month',
        labels={'YearMonth': 'Month', 'Avg_Weekly_Score': 'Avg Weekly Score'},
        line_shape='spline',
        markers=True,
        color_discrete_sequence=['#10b981'],
        template='plotly_dark'
    )
    fig.add_hline(y=RISK_THRESHOLD, line_dash="dash", line_color="red", annotation_text="High Risk Threshold", annotation_position="bottom right")
    fig.update_layout(title_font_size=20, height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. MAIN STREAMLIT APPLICATION LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="PdM Analysis")
    st.markdown("""
        <style>
        .reportview-container .main {background-color: #0d1117;}
        h1 {color: #e5e7eb;}
        .stButton>button {border-radius: 20px;}
        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            background-color: #161b22;
            color: #e5e7eb;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- Sidebar for Upload ---
    with st.sidebar:
        st.header("Upload Maintenance Data (CSV)")
        st.info("Upload structured maintenance logs (CSV) for predictive classification and visualization.")
        
        uploaded_file = st.file_uploader("Upload structured maintenance logs CSV", type="csv")
        
        st.markdown("---")
        st.subheader("Model uses features:")
        st.code("weekly_score, monthly_score")
        
    # --- Main Dashboard ---
    st.title("⚙️ Proactive Maintenance Insight")

    # Load assets first
    model_assets = load_model_assets()

    if uploaded_file is not None:
        try:
            # Read and analyze data
            df_uploaded = pd.read_csv(uploaded_file)
            
            df_analyzed = analyze_data(df_uploaded.copy(), model_assets)
            
            if df_analyzed is None:
                return
            
            # --- Aggregation for Charts ---
            df_prob, df_cost, df_compliance = aggregate_monthly_data(df_analyzed)

            # --- Calculate Top-Level Metrics ---
            total_machines = df_analyzed['unit_id'].nunique()
            high_risk_count = df_analyzed[df_analyzed['Predicted_Risk'] == 1]['unit_id'].nunique()
            avg_compliance = df_analyzed['weekly_score'].mean()
            total_cost = df_analyzed['Estimated_Repair_Cost'].sum()
            
            # --- 1. Metric Row ---
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Machines Analyzed</h3>
                        <h2>{total_machines}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Machines Predicted High Risk</h3>
                        <h2>{high_risk_count}</h2>
                        <p style='color: #ef4444; font-size: 14px;'>↑ {high_risk_count / total_machines * 100:.1f}% of Fleet</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Average Weekly Compliance</h3>
                        <h2>{avg_compliance:.1f}</h2>
                        <p style='color: #10b981; font-size: 14px;'>Target: >{RISK_THRESHOLD}</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Estimated Repair Budget</h3>
                        <h2>${total_cost:,.0f}</h2>
                        <p style='color: #38bdf8; font-size: 14px;'>Budget for analyzed period</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.header("2. Monthly Analysis & Fleet Trends")
            
            # --- 2. Monthly Probability Bar Chart (Matching Reference Image Style) ---
            st.subheader("Monthly Failure Prediction Probability (Fleet Trend)")
            create_monthly_probability_chart(df_prob)

            # --- 3. Compliance and Cost Analysis ---
            colA, colB = st.columns(2)
            
            with colA:
                 st.subheader("Monthly Estimated Repair Cost (Fleet Trend)")
                 create_monthly_cost_chart(df_cost)

            with colB:
                st.subheader("Compliance Adherence Trend")
                create_adherence_chart(df_compliance)
                
            st.markdown("---")
            st.header("3. Analyzed Data Table")
            st.dataframe(df_analyzed, use_container_width=True)

            # --- Save for Future Learning ---
            st.markdown("---")
            st.subheader("Future Learning")
            csv_export = df_analyzed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Save Predictions for Future Learning (CSV)",
                data=csv_export,
                file_name=f'predictions_{uploaded_file.name}',
                mime='text/csv',
                help="Download this file to use the predictions (Risk Level, Cost) for retraining a new model later."
            )
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.stop()
            
    else:
        st.info("Awaiting CSV file upload to perform predictive maintenance analysis...")

if __name__ == '__main__':
    main()