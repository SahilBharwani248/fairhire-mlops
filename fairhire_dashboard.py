"""
FairHire Dashboard - Production Ready Version
Integrates with AWS S3 and Airflow pipeline outputs
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import boto3
from io import StringIO
from datetime import datetime

# =============================================================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# =============================================================================

st.set_page_config(
    page_title="FairHire - ML Fairness Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET_NAME = "airflow-bucket-amu"

# Fairness metric definitions
FAIRNESS_DEFINITIONS = {
    "demographic_parity_max_gap": {
        "name": "Demographic Parity Gap",
        "description": "Measures the maximum difference in selection rates between demographic groups. A gap > 0.1 indicates potential bias.",
        "formula": "max(selection_rate_group_i) - min(selection_rate_group_j)",
        "threshold": 0.1,
        "ideal": "Close to 0 (no gap)"
    },
    "topk_min_over_max": {
        "name": "Selection Rate Ratio (4/5ths Rule)",
        "description": "Ratio of minimum to maximum selection rates across groups. Values < 0.8 violate the 4/5ths rule and indicate disparate impact.",
        "formula": "min(selection_rate) / max(selection_rate)",
        "threshold": 0.8,
        "ideal": "Close to 1.0 (equal rates)"
    },
    "equal_opportunity_max_gap": {
        "name": "Equal Opportunity Gap",
        "description": "Measures differences in True Positive Rate (TPR) across groups. High gaps mean qualified candidates from some groups are less likely to be selected.",
        "formula": "max(TPR_group_i) - min(TPR_group_j)",
        "threshold": 0.1,
        "ideal": "Close to 0 (equal TPR)"
    },
    "rank_ordering_max_gap": {
        "name": "Rank Ordering Gap",
        "description": "Measures differences in average ranking positions between demographic groups. Higher gaps indicate systematic ranking bias.",
        "formula": "max(avg_rank_group_i) - min(avg_rank_group_j)",
        "threshold": "Context-dependent",
        "ideal": "Low gap (similar rankings)"
    },
    "score_distribution_overlap": {
        "name": "Score Distribution Overlap",
        "description": "Measures how similar the score distributions are between groups (0-1 scale). Low overlap indicates systematic scoring differences.",
        "formula": "Histogram overlap metric",
        "threshold": "No strict threshold",
        "ideal": "Close to 1.0 (high overlap)"
    }
}

# =============================================================================
# S3 CLIENT
# =============================================================================

@st.cache_resource
def get_s3_client():
    return boto3.client('s3')

s3_client = get_s3_client()

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def list_manual_runs(model_name):
    """List all manual run folders in S3 for a specific model"""
    try:
        model_prefix = "modelA" if model_name == "Model A" else "modelB"
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f'models/{model_prefix}/manual_',
            Delimiter='/'
        )
        
        runs = []
        for prefix in response.get('CommonPrefixes', []):
            run_path = prefix['Prefix']
            run_name = run_path.split('/')[-2]
            runs.append(run_name)
        
        return sorted(runs, reverse=True)
    except Exception as e:
        st.error(f"Error listing runs: {str(e)}")
        return []

@st.cache_data(ttl=300)
def load_csv_from_s3(s3_path):
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_path)
        csv_content = response['Body'].read().decode('utf-8')
        return pd.read_csv(StringIO(csv_content))
    except Exception as e:
        st.warning(f"Could not load {s3_path}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_json_from_s3(s3_path):
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_path)
        json_content = response['Body'].read().decode('utf-8')
        return json.loads(json_content)
    except Exception as e:
        st.warning(f"Could not load {s3_path}: {str(e)}")
        return {}

def get_model_paths(model_name, run_id=None):
    """Get S3 paths for a specific model and run"""
    model_prefix = "modelA" if model_name == "Model A" else "modelB"
    
    # If no specific run selected, get the latest run
    if not run_id:
        runs = list_manual_runs(model_name)
        if runs:
            run_id = runs[0]  # Most recent
    
    if run_id:
        base_path = f"models/{model_prefix}/{run_id}/"
    else:
        # Fallback to base directory (shouldn't happen with manual runs)
        base_path = f"models/{model_prefix}/"
    
    return {
        "metrics": f"{base_path}{model_prefix}_metrics.json",
        "fairness": f"{base_path}{model_prefix}_fairness_metrics.json",
        "shortlist": f"{base_path}{model_prefix}_shortlist_base.csv",
        "base_path": base_path
    }

# =============================================================================
# HEADER
# =============================================================================

st.title("⚖️ FairHire: ML Fairness Auditing Dashboard")

st.markdown("""
This dashboard analyzes hiring model performance and fairness across demographic groups.
Select a model and job role to view detailed metrics and shortlisted candidates.
""")

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

st.sidebar.header("🔧 Configuration")

# S3 connection status
try:
    s3_client.head_bucket(Bucket=BUCKET_NAME)
    st.sidebar.success(f"✅ Connected to S3: {BUCKET_NAME}")
except Exception as e:
    st.sidebar.error(f"❌ S3 Connection Error")
    st.error(f"Cannot connect to S3 bucket '{BUCKET_NAME}': {str(e)}")
    st.info("💡 Make sure AWS credentials are configured correctly")
    st.stop()

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    ["Model A", "Model B"],
    help="Model A: RandomForest baseline | Model B: LightGBM on biased data"
)

# Run selection
st.sidebar.markdown("---")
st.sidebar.markdown("### Run Selection")

manual_runs = list_manual_runs(selected_model)

if manual_runs:
    # Show most recent run by default
    selected_run = st.sidebar.selectbox(
        "Select Run",
        manual_runs,
        help="Select a specific training run to analyze"
    )
    st.sidebar.info(f"📅 {len(manual_runs)} run(s) available")
else:
    st.sidebar.warning("⚠️ No runs found for this model")
    selected_run = None

# Get paths
model_paths = get_model_paths(selected_model, selected_run)

# Show current paths for debugging
with st.sidebar.expander("🔍 Debug Info"):
    st.code(f"""
Base Path: {model_paths['base_path']}
Metrics: {model_paths['metrics']}
Fairness: {model_paths['fairness']}
Shortlist: {model_paths['shortlist']}
    """)

# =============================================================================
# LOAD DATA
# =============================================================================

if not selected_run:
    st.error("No training runs found for the selected model. Please run the Airflow pipeline first.")
    st.stop()

with st.spinner('Loading data from S3...'):
    metrics = load_json_from_s3(model_paths["metrics"])
    fairness = load_json_from_s3(model_paths["fairness"])
    shortlist = load_csv_from_s3(model_paths["shortlist"])

# Job role selection
job_roles = ["All"]
if not shortlist.empty and "Job_Role" in shortlist.columns:
    unique_roles = shortlist["Job_Role"].unique().tolist()
    job_roles.extend(sorted([str(role) for role in unique_roles if pd.notna(role)]))

selected_job_role = st.sidebar.selectbox(
    "Filter by Job Role",
    job_roles,
    help="Filter analysis by specific job role"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Current Selection:**
- Model: {selected_model}
- Job Role: {selected_job_role}
- Run: {selected_run if selected_run else 'None'}
""")

# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Performance Analysis",
    "⚖️ Fairness Analysis", 
    "📋 Generate Shortlist"
])

# =============================================================================
# TAB 1: PERFORMANCE ANALYSIS
# =============================================================================

with tab1:
    st.header(f"📊 Performance Metrics - {selected_model}")
    
    if not metrics:
        st.error("Could not load performance metrics from S3")
        st.info(f"Looking for: {model_paths['metrics']}")
    else:
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.1%}")
        with col5:
            st.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
        
        st.markdown("---")
        
        # Additional details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.markdown(f"""
            - **Model Type:** {metrics.get('model_type', 'N/A')}
            - **Test Samples:** {metrics.get('test_samples', 'N/A'):,}
            - **Target Variable:** {metrics.get('target_variable', 'N/A')}
            - **Training Date:** {metrics.get('timestamp', 'N/A')}
            """)
        
        with col2:
            st.subheader("Prediction Distribution")
            class_dist = metrics.get('class_distribution', {})
            if class_dist:
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    title="Predictions by Class",
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No prediction distribution available")
        
        # Job role filtering impact
        if selected_job_role != "All" and not shortlist.empty:
            st.markdown("---")
            st.subheader(f"Metrics for Job Role: {selected_job_role}")
            
            role_data = shortlist[shortlist["Job_Role"] == selected_job_role]
            st.info(f"Found {len(role_data)} candidates for this role ({len(role_data)/len(shortlist)*100:.1f}% of total)")

# =============================================================================
# TAB 2: FAIRNESS ANALYSIS
# =============================================================================

with tab2:
    st.header("⚖️ Fairness Analysis")
    
    if not fairness:
        st.warning(f"No fairness metrics available for {selected_model}.")
        st.info(f"Looking for: {model_paths['fairness']}")
    else:
        # Show metric definitions
        with st.expander("📖 Understanding Fairness Metrics", expanded=False):
            for metric_key, metric_info in FAIRNESS_DEFINITIONS.items():
                st.markdown(f"### {metric_info['name']}")
                st.markdown(f"**Description:** {metric_info['description']}")
                st.markdown(f"**Formula:** `{metric_info['formula']}`")
                st.markdown(f"**Threshold:** {metric_info['threshold']}")
                st.markdown(f"**Ideal Value:** {metric_info['ideal']}")
                st.markdown("---")
        
        st.markdown("### Fairness Metrics by Demographic Group")
        st.markdown("**Thresholds:** 🔴 Bias detected | 🟢 Fair")
        
        # Analyze each demographic attribute
        for demo_attr in ["Gender", "Race", "Age_Group", "Disability_Status"]:
            if demo_attr in fairness:
                st.subheader(demo_attr.replace("_", " "))
                demo_metrics = fairness[demo_attr]
                
                # Create metrics grid
                cols = st.columns(4)
                
                # Demographic Parity Gap
                with cols[0]:
                    dp_gap = demo_metrics.get('demographic_parity_max_gap')
                    if dp_gap is not None and not pd.isna(dp_gap):
                        status = "🔴" if dp_gap > 0.1 else "🟢"
                        st.metric(
                            f"{status} Demographic Parity",
                            f"{dp_gap:.3f}",
                            help=FAIRNESS_DEFINITIONS['demographic_parity_max_gap']['description']
                        )
                    else:
                        st.metric("Demographic Parity", "N/A")
                
                # Selection Rate Ratio
                with cols[1]:
                    sr = demo_metrics.get('topk_min_over_max')
                    if sr is not None and not pd.isna(sr):
                        status = "🔴" if sr < 0.8 else "🟢"
                        st.metric(
                            f"{status} Selection Rate",
                            f"{sr:.3f}",
                            help=FAIRNESS_DEFINITIONS['topk_min_over_max']['description']
                        )
                    else:
                        st.metric("Selection Rate", "N/A")
                
                # Equal Opportunity Gap
                with cols[2]:
                    eo = demo_metrics.get('equal_opportunity_max_gap')
                    if eo is not None and not pd.isna(eo):
                        status = "🔴" if eo > 0.1 else "🟢"
                        st.metric(
                            f"{status} Equal Opportunity",
                            f"{eo:.3f}",
                            help=FAIRNESS_DEFINITIONS['equal_opportunity_max_gap']['description']
                        )
                    else:
                        st.metric("Equal Opportunity", "N/A")
                
                # Rank Ordering Gap
                with cols[3]:
                    ro = demo_metrics.get('rank_ordering_max_gap')
                    if ro is not None and not pd.isna(ro):
                        st.metric(
                            "Rank Ordering Gap",
                            f"{ro:.1f}",
                            help=FAIRNESS_DEFINITIONS['rank_ordering_max_gap']['description']
                        )
                    else:
                        st.metric("Rank Ordering Gap", "N/A")
                
                # Interpretation section with detailed explanations
                with st.expander("📊 What do these numbers mean?", expanded=False):
                    st.markdown("### Interpretation Guide")
                    
                    # Demographic Parity Explanation
                    if dp_gap is not None and not pd.isna(dp_gap):
                        dp_pct = dp_gap * 100
                        if dp_gap > 0.1:
                            st.error(f"""
                            **🔴 Demographic Parity Gap: {dp_pct:.1f} percentage points**
                            
                            This means the **hiring rate varies by up to {dp_pct:.1f}%** between different {demo_attr.lower()} groups.
                            
                            - A gap > 10% suggests **potential bias** in how frequently different groups are hired
                            - Example: If one group has 80% hire rate and another has 60%, the gap is 20 percentage points
                            - **Action needed:** Investigate why certain groups have significantly different hiring outcomes
                            """)
                        else:
                            st.success(f"""
                            **🟢 Demographic Parity Gap: {dp_pct:.1f} percentage points**
                            
                            The hiring rates are **fairly consistent** across {demo_attr.lower()} groups (within {dp_pct:.1f}%).
                            
                            - This suggests the model treats different groups relatively equally
                            - Small variations are expected and acceptable
                            """)
                    
                    # Selection Rate Explanation
                    if sr is not None and not pd.isna(sr):
                        if sr < 0.8:
                            st.error(f"""
                            **🔴 Selection Rate Ratio: {sr:.3f} (4/5ths Rule Violation)**
                            
                            The group with the **lowest top-50 representation** appears {sr:.1%} as often as the highest group.
                            
                            - The "4/5ths rule" requires this ratio to be ≥ 0.80 to avoid disparate impact
                            - Current ratio of {sr:.3f} means underrepresented groups appear in top rankings at **{(1-sr)*100:.1f}% lower rates**
                            - **Legal risk:** This could be evidence of systemic bias in candidate ranking
                            """)
                        else:
                            st.success(f"""
                            **🟢 Selection Rate Ratio: {sr:.3f} (4/5ths Rule: PASS)**
                            
                            All {demo_attr.lower()} groups are **adequately represented** in the top-50 ranked candidates.
                            
                            - The least represented group appears at {sr:.1%} the rate of the most represented
                            - This meets the 80% threshold (4/5ths rule) used in employment law
                            """)
                    
                    # Equal Opportunity Explanation
                    if eo is not None and not pd.isna(eo):
                        eo_pct = eo * 100
                        if eo > 0.1:
                            st.error(f"""
                            **🔴 Equal Opportunity Gap: {eo_pct:.1f} percentage points**
                            
                            Among **truly qualified candidates** (ground truth = Hire), the model's positive prediction rate varies by {eo_pct:.1f}% across groups.
                            
                            - This means qualified candidates from some groups are **{eo_pct:.1f}% less likely** to be identified as hireable
                            - **Impact:** Systematically missing qualified talent from underrepresented groups
                            - **Action:** Model may need retraining with bias mitigation techniques
                            """)
                        else:
                            st.success(f"""
                            **🟢 Equal Opportunity Gap: {eo_pct:.1f} percentage points**
                            
                            The model **identifies qualified candidates equally well** across {demo_attr.lower()} groups.
                            
                            - True Positive Rates (TPR) are consistent
                            - Qualified individuals have similar chances of being hired regardless of group
                            """)
                    
                    # Rank Ordering Explanation
                    if ro is not None and not pd.isna(ro):
                        if ro > 10:
                            st.warning(f"""
                            **⚠️ Rank Ordering Gap: {ro:.1f} positions**
                            
                            On average, candidates from different {demo_attr.lower()} groups appear **{ro:.1f} positions apart** in the ranking.
                            
                            - Higher-ranked groups appear {ro:.0f} slots earlier in the shortlist
                            - **Impact:** Some groups may consistently miss out on top positions even with similar qualifications
                            - Consider reviewing ranking algorithm for hidden biases
                            """)
                        else:
                            st.info(f"""
                            **✓ Rank Ordering Gap: {ro:.1f} positions**
                            
                            Average ranking positions are **relatively similar** across groups ({ro:.1f} positions difference).
                            
                            - Small variations in average rank are normal
                            - No strong evidence of systematic ranking bias
                            """)
                
                # Per-group details
                with st.expander(f"📋 Detailed breakdown by {demo_attr} group"):
                    per_group = demo_metrics.get('per_group_details', {})
                    if per_group:
                        df_per_group = pd.DataFrame(per_group).T
                        st.dataframe(df_per_group, use_container_width=True)
                    else:
                        st.info(f"""
                        **Per-group statistics not available in saved metrics.**
                        
                        The fairness analysis calculated aggregate metrics (gaps and ratios) but didn't save 
                        individual group statistics. To see detailed per-group breakdowns:
                        
                        1. Re-run the fairness analysis with `save_per_group_details=True`
                        2. Or check the Airflow logs for the full analysis output
                        3. Example stats: Hiring rate per group, Top-50 representation per group, etc.
                        """)
                
                st.markdown("---")
        
        # Overall fairness summary
        st.subheader("📊 Overall Fairness Summary")
        
        bias_count = 0
        total_metrics = 0
        
        for demo_attr in ["Gender", "Race", "Age_Group", "Disability_Status"]:
            if demo_attr in fairness:
                demo = fairness[demo_attr]
                
                dp_gap = demo.get('demographic_parity_max_gap')
                if dp_gap and not pd.isna(dp_gap):
                    total_metrics += 1
                    if dp_gap > 0.1:
                        bias_count += 1
                
                sr = demo.get('topk_min_over_max')
                if sr and not pd.isna(sr):
                    total_metrics += 1
                    if sr < 0.8:
                        bias_count += 1
                
                eo = demo.get('equal_opportunity_max_gap')
                if eo and not pd.isna(eo):
                    total_metrics += 1
                    if eo > 0.1:
                        bias_count += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Metrics", total_metrics)
        with col2:
            st.metric("Metrics Showing Bias", bias_count)
        with col3:
            fairness_score = ((total_metrics - bias_count) / total_metrics * 100) if total_metrics > 0 else 0
            st.metric("Fairness Score", f"{fairness_score:.1f}%")
        
        if bias_count > 0:
            st.error(f"⚠️ {bias_count} metric(s) indicate potential bias. Review recommended.")
        else:
            st.success("✅ All fairness metrics within acceptable thresholds!")

# =============================================================================
# TAB 3: GENERATE SHORTLIST
# =============================================================================

with tab3:
    st.header(f"📋 Candidate Shortlist - {selected_model}")
    
    if shortlist.empty:
        st.error("Could not load shortlist data from S3")
        st.info(f"Looking for: {model_paths['shortlist']}")
    else:
        # Determine column names based on model
        model_prefix = "ModelA" if selected_model == "Model A" else "ModelB"
        prob_col = f"{model_prefix}_Hire_Prob"
        pred_col = f"{model_prefix}_Pred_Label"
        rank_col = f"{model_prefix}_Rank"
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_filter = st.selectbox(
                "Filter by Prediction",
                ["All", "Hire", "Reject"]
            )
        
        with col2:
            top_k = st.slider(
                "Number of Candidates to Show",
                min_value=10,
                max_value=min(200, len(shortlist)),
                value=50
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                [rank_col, prob_col, "Experience", "Skills"] if rank_col in shortlist.columns else [prob_col, "Experience", "Skills"]
            )
        
        # Apply filters
        filtered_df = shortlist.copy()
        
        # Job role filter
        if selected_job_role != "All":
            filtered_df = filtered_df[filtered_df["Job_Role"] == selected_job_role]
        
        # Prediction filter
        if pred_filter != "All" and pred_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[pred_col] == pred_filter]
        
        # Sort
        if sort_by in filtered_df.columns:
            ascending = sort_by == rank_col
            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
        
        # Limit to top K
        filtered_df = filtered_df.head(top_k)
        
        # Select display columns (exclude demographics and AI_Score)
        display_cols = ["Skills", "Experience", "Education", "Job_Role"]
        
        # Add model-specific columns
        for col in [prob_col, pred_col, rank_col]:
            if col in filtered_df.columns:
                display_cols.append(col)
        
        # Ensure columns exist
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Display summary
        st.markdown(f"**Showing {len(filtered_df)} candidates** (filtered from {len(shortlist)} total)")
        
        # Display table
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = filtered_df[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Shortlist as CSV",
            data=csv,
            file_name=f"{model_prefix}_shortlist_{selected_job_role}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Statistics
        if pred_col in filtered_df.columns:
            st.markdown("---")
            st.subheader("Shortlist Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hire_count = (filtered_df[pred_col] == "Hire").sum()
                st.metric("Recommended for Hire", hire_count)
            
            with col2:
                if prob_col in filtered_df.columns:
                    avg_prob = filtered_df[prob_col].mean()
                    st.metric("Average Hire Probability", f"{avg_prob:.2%}")
            
            with col3:
                if "Experience" in filtered_df.columns:
                    avg_exp = filtered_df["Experience"].mean()
                    st.metric("Average Experience (years)", f"{avg_exp:.1f}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
FairHire - ML Fairness Auditing Dashboard | University of Chicago<br>
Connected to S3: {BUCKET_NAME} | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)
