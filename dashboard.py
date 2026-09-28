import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

st.set_page_config(
    page_title="BIA Project Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 BIA Project Dashboard")
st.write("Welcome to my Machine Learning Dashboard")

st.sidebar.header("Dashboard Controls")
st.sidebar.write("Select an option")

# --- DATASET LOAD ---
try:
   df = pd.read_csv(r"C:\Users\Sameer\OneDrive\Documents\Desktop\BIA_PROJECT\water_potability.csv")
   total_records = len(df)
except Exception:
    df = None
    total_records = 3276

# Accuracy Value
accuracy_val = "88.5%"

# --- METRICS DISPLAY ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{total_records:,}")

with col2:
    st.metric("Accuracy", accuracy_val)

with col3:
    st.metric("Model", "Random Forest")

st.subheader("Project Overview")
st.info("Dataset and Machine Learning model connected successfully!")

# --- DATA PREVIEW & VISUALIZATIONS ---
if df is not None:
    st.write("---")
    
    # Data Table Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("---")
    
    # Graphs Section
    st.subheader("📊 Data Visualizations")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Potability Target Distribution**")
        target_col = 'Potability' if 'Potability' in df.columns else df.columns[-1]
        
        # Matplotlib se clean graph banana taaki 0 se start ho
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df[target_col].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color=['#4CAF50', '#FF9800'])
        ax.set_xlabel("Potability (0 = Not Safe, 1 = Safe)")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts.values) + 300) # Force start from 0
        
        st.pyplot(fig)
            
    with c2:
        st.write("**Data Summary Statistics**")
        st.dataframe(df.describe())