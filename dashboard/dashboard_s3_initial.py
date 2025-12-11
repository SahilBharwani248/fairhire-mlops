"""
FairHire Dashboard (AWS S3 Version)
====================================
This version reads data from S3 instead of local files.

To use this:
1. Upload your data to S3
2. Update the BUCKET_NAME variable below
3. Run: streamlit run dashboard_s3.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import boto3
from io import StringIO, BytesIO

# =============================================================================
# CONFIGURATION - UPDATE THIS!
# =============================================================================

BUCKET_NAME = "fairhire-mlops-project"  # <-- Change this to your bucket name

# S3 paths (update if your folder structure is different)
S3_PATHS = {
    "model_a_shortlist": "models/model_A/modelA_shortlist_base.csv",
    "model_b_shortlist": "models/model_B/modelB_shortlist_base.csv",
    "model_a_metrics": "models/model_A/modelA_metrics.json",
    "model_b_metrics": "models/model_B/modelB_metrics.json",
    "model_b_fairness": "models/model_B/modelB_fairness_metrics.json",
    "model_a_demographics": "models/model_A/demographics_test.csv",
    "model_b_demographics": "models/model_B/demographics_test_modelB.csv",
}

# =============================================================================
# S3 CLIENT SETUP
# =============================================================================

# Create S3 client
# If running on EC2 with IAM role, credentials are automatic
# If running locally, make sure you've run 'aws configure'
s3_client = boto3.client('s3')

# =============================================================================
# S3 DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_csv_from_s3(s3_path):
    """
    Load a CSV file from S3 into a pandas DataFrame.
    
    How it works:
    1. s3_client.get_object() downloads the file from S3
    2. The file comes as bytes in response['Body']
    3. We read those bytes into pandas
    """
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_path)
        csv_content = response['Body'].read().decode('utf-8')
        return pd.read_csv(StringIO(csv_content))
    except Exception as e:
        st.error(f"Error loading {s3_path} from S3: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_json_from_s3(s3_path):
    """
    Load a JSON file from S3 into a Python dictionary.
    """
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_path)
        json_content = response['Body'].read().decode('utf-8')
        return json.loads(json_content)
    except Exception as e:
        st.error(f"Error loading {s3_path} from S3: {str(e)}")
        return {}


def load_shortlist(model_name):
    """Load candidate shortlist for selected model."""
    if model_name == "Model A":
        return load_csv_from_s3(S3_PATHS["model_a_shortlist"])
    else:
        return load_csv_from_s3(S3_PATHS["model_b_shortlist"])


def load_model_metrics(model_name):
    """Load performance metrics for selected model."""
    if model_name == "Model A":
        return load_json_from_s3(S3_PATHS["model_a_metrics"])
    else:
        return load_json_from_s3(S3_PATHS["model_b_metrics"])


def load_fairness_metrics():
    """Load fairness metrics (currently only Model B has these)."""
    return load_json_from_s3(S3_PATHS["model_b_fairness"])


def load_demographics(model_name):
    """Load demographic information for test candidates."""
    if model_name == "Model A":
        return load_csv_from_s3(S3_PATHS["model_a_demographics"])
    else:
        return load_csv_from_s3(S3_PATHS["model_b_demographics"])


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="FairHire - Fairness Auditing Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# =============================================================================
# HEADER
# =============================================================================

st.title("⚖️ FairHire: Fairness Auditing Dashboard")

st.markdown("""
**FairHire** is an MLOps-driven fairness auditing framework that evaluates whether 
hiring models produce biased outcomes across demographic groups.

Select a model below to view its predictions and fairness analysis.
""")

# Show S3 connection status
try:
    s3_client.head_bucket(Bucket=BUCKET_NAME)
    st.sidebar.success(f" Connected to S3: {BUCKET_NAME}")
except Exception as e:
    st.sidebar.error(f" Cannot connect to S3: {BUCKET_NAME}")
    st.sidebar.error(f"Error: {str(e)}")

# =============================================================================
# MODEL SELECTION
# =============================================================================

st.sidebar.header("🔧 Configuration")

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["Model A", "Model B"],
    help="Model A: Trained on original data (baseline). Model B: Trained on biased data."
)

st.sidebar.markdown("---")

st.sidebar.markdown(f"""
### Model Descriptions

**Model A (Current Model)**  
- RandomForest Classifier
- Trained on original resume features
- Demographics NOT used in training

**Model B (Historic Model)**  
- LightGBM Classifier  
- Trained on biased dataset
- Simulates historical bias in data
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Source:** `s3://{BUCKET_NAME}/`")

# =============================================================================
# LOAD DATA
# =============================================================================

shortlist = load_shortlist(selected_model)
metrics = load_model_metrics(selected_model)
demographics = load_demographics(selected_model)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Candidate Shortlist", 
    "📊 Performance Metrics", 
    "⚖️ Fairness Report",
    "🔄 Model Comparison"
])

# =============================================================================
# TAB 1: CANDIDATE SHORTLIST
# =============================================================================

with tab1:
    st.header(f"Candidate Shortlist - {selected_model}")
    
    st.markdown("""
    This table shows all candidates ranked by their predicted hire probability.
    Higher probability = model thinks candidate is more likely to be a good hire.
    """)
    
    if shortlist.empty:
        st.warning("Could not load shortlist data from S3.")
    else:
        # Determine column names based on model
        if selected_model == "Model A":
            prob_col = "ModelA_Hire_Prob" if "ModelA_Hire_Prob" in shortlist.columns else None
            pred_col = "ModelA_Pred_Label" if "ModelA_Pred_Label" in shortlist.columns else None
            rank_col = "ModelA_Rank" if "ModelA_Rank" in shortlist.columns else None
        else:
            prob_col = "ModelB_Hire_Prob" if "ModelB_Hire_Prob" in shortlist.columns else None
            pred_col = "ModelB_Pred_Label" if "ModelB_Pred_Label" in shortlist.columns else None
            rank_col = "ModelB_Rank" if "ModelB_Rank" in shortlist.columns else None
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            job_roles = ["All"] + sorted(shortlist["Job_Role"].unique().tolist())
            selected_role = st.selectbox("Filter by Job Role", job_roles)
        
        with col2:
            top_k = st.slider("Show Top K Candidates", min_value=10, max_value=200, value=50)
        
        with col3:
            pred_filter = st.selectbox("Filter by Prediction", ["All", "Hire", "Reject"])
        
        # Apply filters
        filtered_df = shortlist.copy()
        
        if selected_role != "All":
            filtered_df = filtered_df[filtered_df["Job_Role"] == selected_role]
        
        if pred_filter != "All" and pred_col:
            filtered_df = filtered_df[filtered_df[pred_col] == pred_filter]
        
        if rank_col and rank_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(rank_col).head(top_k)
        else:
            filtered_df = filtered_df.head(top_k)
        
        # Select columns to display
        display_cols = ["Skills", "Experience", "Education", "Job_Role", "AI_Score"]
        if prob_col and prob_col in filtered_df.columns:
            display_cols.append(prob_col)
        if pred_col and pred_col in filtered_df.columns:
            display_cols.append(pred_col)
        if rank_col and rank_col in filtered_df.columns:
            display_cols.append(rank_col)
        
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
        st.markdown(f"*Showing {len(filtered_df)} candidates*")


# =============================================================================
# TAB 2: PERFORMANCE METRICS
# =============================================================================

with tab2:
    st.header(f"Performance Metrics - {selected_model}")
    
    if not metrics:
        st.warning("Could not load metrics from S3.")
    else:
        st.markdown("These metrics measure how well the model predicts hiring decisions.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Accuracy",
                value=f"{metrics.get('accuracy', 0):.1%}",
                help="Percentage of correct predictions"
            )
        
        with col2:
            st.metric(
                label="Precision",
                value=f"{metrics.get('precision', 0):.1%}",
                help="Of predicted 'Hire', how many were correct?"
            )
        
        with col3:
            st.metric(
                label="Recall",
                value=f"{metrics.get('recall', 0):.1%}",
                help="Of actual good candidates, how many did we find?"
            )
        
        with col4:
            st.metric(
                label="F1 Score",
                value=f"{metrics.get('f1_score', 0):.1%}",
                help="Balance of precision and recall"
            )
        
        with col5:
            st.metric(
                label="AUC",
                value=f"{metrics.get('auc', 0):.3f}",
                help="Model's discrimination ability"
            )
        
        st.markdown("---")
        
        st.subheader("Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            - **Model Type:** {metrics.get('model_type', 'N/A')}
            - **Task Type:** {metrics.get('task_type', 'N/A')}
            - **Target Variable:** {metrics.get('target_variable', 'N/A')}
            - **Test Samples:** {metrics.get('test_samples', 'N/A')}
            """)
        
        with col2:
            class_dist = metrics.get('class_distribution', {})
            if class_dist:
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    title="Prediction Distribution",
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 3: FAIRNESS REPORT
# =============================================================================

with tab3:
    st.header("⚖️ Fairness Report")
    
    if selected_model == "Model B":
        fairness = load_fairness_metrics()
        
        if not fairness:
            st.warning("Could not load fairness metrics from S3.")
        else:
            st.markdown("""
            Fairness metrics measure whether the model treats different demographic groups equally.
            
            **Thresholds:**
            - Demographic Parity Gap > 0.1 = Potential bias
            - Selection Rate Ratio < 0.8 = Violates 4/5ths rule
            """)
            
            st.markdown("---")
            
            demo_tabs = st.tabs(["Gender", "Race", "Age Group", "Disability Status"])
            
            for i, (demo_attr, demo_tab) in enumerate(zip(
                ["Gender", "Race", "Age_Group", "Disability_Status"], 
                demo_tabs
            )):
                with demo_tab:
                    if demo_attr in fairness:
                        demo_metrics = fairness[demo_attr]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            dp_gap = demo_metrics.get('demographic_parity_max_gap', None)
                            if dp_gap is not None and not pd.isna(dp_gap):
                                status = "🔴" if dp_gap > 0.1 else "🟢"
                                st.metric(
                                    label=f"{status} Demographic Parity Gap",
                                    value=f"{dp_gap:.3f}"
                                )
                            else:
                                st.metric(label="Demographic Parity Gap", value="N/A")
                        
                        with col2:
                            sr_ratio = demo_metrics.get('topk_min_over_max', None)
                            if sr_ratio is not None and not pd.isna(sr_ratio):
                                status = "🔴" if sr_ratio < 0.8 else "🟢"
                                st.metric(
                                    label=f"{status} Selection Rate Ratio",
                                    value=f"{sr_ratio:.3f}"
                                )
                            else:
                                st.metric(label="Selection Rate Ratio", value="N/A")
                        
                        with col3:
                            eo_gap = demo_metrics.get('equal_opportunity_max_gap', None)
                            if eo_gap is not None and not pd.isna(eo_gap):
                                status = "🔴" if eo_gap > 0.1 else "🟢"
                                st.metric(
                                    label=f"{status} Equal Opportunity Gap",
                                    value=f"{eo_gap:.3f}"
                                )
                            else:
                                st.metric(label="Equal Opportunity Gap", value="N/A")
                    else:
                        st.info(f"No fairness metrics for {demo_attr}")
            
            st.markdown("---")
            
            # Summary chart
            st.subheader("Fairness Summary")
            
            summary_data = []
            for attr in ["Gender", "Race", "Disability_Status"]:
                if attr in fairness:
                    dp_gap = fairness[attr].get('demographic_parity_max_gap', None)
                    if dp_gap is not None and not pd.isna(dp_gap):
                        summary_data.append({
                            "Attribute": attr,
                            "Demographic Parity Gap": dp_gap,
                            "Status": "Needs Review" if dp_gap > 0.1 else "Acceptable"
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                fig = px.bar(
                    summary_df,
                    x="Attribute",
                    y="Demographic Parity Gap",
                    color="Status",
                    color_discrete_map={"Acceptable": "#2ecc71", "Needs Review": "#e74c3c"},
                    title="Demographic Parity Gap by Attribute"
                )
                fig.add_hline(y=0.1, line_dash="dash", line_color="red",
                             annotation_text="Threshold (0.1)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Fairness metrics only available for Model B. Select Model B from sidebar.")


# =============================================================================
# TAB 4: MODEL COMPARISON
# =============================================================================

with tab4:
    st.header("🔄 Model Comparison")
    
    metrics_a = load_model_metrics("Model A")
    metrics_b = load_model_metrics("Model B")
    
    if not metrics_a or not metrics_b:
        st.warning("Could not load metrics for comparison.")
    else:
        comparison_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
            "Model A": [
                metrics_a.get('accuracy', 0),
                metrics_a.get('precision', 0),
                metrics_a.get('recall', 0),
                metrics_a.get('f1_score', 0),
                metrics_a.get('auc', 0)
            ],
            "Model B": [
                metrics_b.get('accuracy', 0),
                metrics_b.get('precision', 0),
                metrics_b.get('recall', 0),
                metrics_b.get('f1_score', 0),
                metrics_b.get('auc', 0)
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df["Difference"] = comparison_df["Model B"] - comparison_df["Model A"]
        
        st.dataframe(comparison_df.style.format({
            "Model A": "{:.3f}",
            "Model B": "{:.3f}",
            "Difference": "{:+.3f}"
        }), use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Model A', x=comparison_data["Metric"], 
                            y=comparison_data["Model A"], marker_color='#3498db'))
        fig.add_trace(go.Bar(name='Model B', x=comparison_data["Metric"], 
                            y=comparison_data["Model B"], marker_color='#e74c3c'))
        fig.update_layout(title="Performance Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>FairHire - MLOps Fairness Auditing Framework</p>
    <p>University of Chicago | MLOps Final Project</p>
    <p>Data Source: AWS S3</p>
</div>
""", unsafe_allow_html=True)
