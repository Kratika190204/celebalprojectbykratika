import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Import custom modules
from clv_model import EnhancedCLVPredictor
from utils import (
    save_feedback, save_feature_request, load_feedback, load_feature_requests,
    validate_customer_data, format_currency, format_percentage,
    generate_sample_data, export_data_to_csv, create_email_link
)
from feedback_manager import FeedbackManager
from documentation_handler import DocumentationHandler

# Set page configuration
st.set_page_config(
    page_title="CLV Model Interface",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    with open('styles.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css()
except FileNotFoundError:
    st.warning("CSS file not found. Using default styling.")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'manual_customers' not in st.session_state:
    st.session_state.manual_customers = pd.DataFrame()
if 'show_all_customers' not in st.session_state:
    st.session_state.show_all_customers = False
if 'trigger_prediction' not in st.session_state:
    st.session_state.trigger_prediction = False
if 'show_feedback_dashboard' not in st.session_state:
    st.session_state.show_feedback_dashboard = False
if 'show_feature_requests' not in st.session_state:
    st.session_state.show_feature_requests = False
if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False
if 'show_feature_request_form' not in st.session_state:
    st.session_state.show_feature_request_form = False


# Initialize components
clv_predictor = EnhancedCLVPredictor()
feedback_manager = FeedbackManager()
doc_handler = DocumentationHandler()

# Check if we have a saved model
MODEL_FILE = 'clv_model.joblib'

# Add a simple way to force retraining (for development/testing)
FORCE_RETRAIN = False  # Set to True if you want to retrain the model

# Only initialize model if not already fitted (avoid retraining on every load)
if not clv_predictor.is_fitted and not FORCE_RETRAIN:
    # Try to load existing model first
    if os.path.exists(MODEL_FILE):
        try:
            with st.spinner("Loading existing CLV model..."):
                clv_predictor.load_model(MODEL_FILE)
                st.success("✅ CLV Model loaded from saved file!")
        except Exception as e:
            st.warning(f"Could not load saved model: {str(e)}")
            FORCE_RETRAIN = True  # Fall back to training new model

# Train new model if needed
if not clv_predictor.is_fitted or FORCE_RETRAIN:
    with st.spinner("Training new CLV model (this may take a moment)..."):
        try:
            # Save sample data to temp file first
            sample_data = generate_sample_data()
            sample_data.to_csv('temp_sample_data.csv', index=False)
            
            # Run the pipeline
            clv_predictor.run_full_pipeline('temp_sample_data.csv')
            
            # Save the trained model
            clv_predictor.save_model(MODEL_FILE)
            
            # Clean up temp file only after successful initialization
            if os.path.exists('temp_sample_data.csv'):
                os.remove('temp_sample_data.csv')
                
            st.success("✅ CLV Model trained and saved successfully!")
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            # Clean up temp file even if there's an error
            if os.path.exists('temp_sample_data.csv'):
                os.remove('temp_sample_data.csv')
else:
    st.success("✅ CLV Model ready!")

def handle_manual_entry():
    """Handle manual customer data entry"""
    st.sidebar.write("### Enter customer data:")
    
    with st.sidebar.form("manual_entry"):
        customer_id = st.text_input("Customer ID", f"CUST_{len(st.session_state.manual_customers) + 1:04d}")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        total_purchases = st.number_input("Total Purchases", min_value=0, value=5)
        avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=50.0)
        days_since_first = st.number_input("Days Since First Purchase", min_value=1, value=180)
        days_since_last = st.number_input("Days Since Last Purchase", min_value=0, value=30)
        acquisition_channel = st.selectbox("Acquisition Channel", 
                                         ['Online', 'Store', 'Social Media', 'Referral'], 
                                         index=0)
        location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'], index=0)
        subscription_status = st.selectbox("Subscription Status", 
                                         ['Active', 'Inactive', 'None'], 
                                         index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            add_customer = st.form_submit_button("➕ Add Customer", type="secondary")
        with col2:
            add_and_predict = st.form_submit_button("🔮 Add & Predict", type="primary")
        
        if add_customer or add_and_predict:
            # Validation logic (same as original)
            validation_errors = []
            
            if not customer_id.strip():
                validation_errors.append("Customer ID cannot be empty!")
            else:
                if not st.session_state.manual_customers.empty and 'customer_id' in st.session_state.manual_customers.columns:
                    existing_ids = st.session_state.manual_customers['customer_id'].tolist()
                    if customer_id.strip() in existing_ids:
                        validation_errors.append(f"Customer ID '{customer_id}' already exists!")
            
            if age < 18 or age > 100:
                validation_errors.append("Age must be between 18 and 100!")
            if total_purchases < 0:
                validation_errors.append("Total purchases cannot be negative!")
            if avg_order_value <= 0:
                validation_errors.append("Average order value must be greater than 0!")
            if days_since_first < 1:
                validation_errors.append("Days since first purchase must be at least 1!")
            if days_since_last < 0:
                validation_errors.append("Days since last purchase cannot be negative!")
            if days_since_last > days_since_first:
                validation_errors.append("Days since last purchase cannot be greater than days since first purchase!")
            
            if validation_errors:
                for error in validation_errors:
                    st.sidebar.error(error)
            else:
                new_customer = pd.DataFrame({
                    'customer_id': [customer_id.strip()], 
                    'age': [age], 
                    'total_purchases': [total_purchases],
                    'avg_order_value': [avg_order_value], 
                    'days_since_first_purchase': [days_since_first],
                    'days_since_last_purchase': [days_since_last], 
                    'acquisition_channel': [acquisition_channel],
                    'location': [location], 
                    'subscription_status': [subscription_status]
                })
                
                if st.session_state.manual_customers.empty:
                    st.session_state.manual_customers = new_customer
                else:
                    st.session_state.manual_customers = pd.concat([st.session_state.manual_customers, new_customer], 
                                                                ignore_index=True)
                
                st.session_state.customer_data = st.session_state.manual_customers.copy()
                st.sidebar.success(f"✅ Customer {customer_id} added! Total: {len(st.session_state.manual_customers)}")
                
                if add_and_predict:
                    st.session_state.trigger_prediction = True
    
    return not st.session_state.manual_customers.empty

def display_manual_customers():
    """Display and manage manually entered customers"""
    if not st.session_state.manual_customers.empty:
        st.sidebar.write("### Added Customers")
        
        for idx, row in st.session_state.manual_customers.iterrows():
            with st.sidebar.expander(f"👤 {row['customer_id']}", expanded=False):
                st.write(f"*Age:* {row['age']}")
                st.write(f"*Purchases:* {row['total_purchases']}")
                st.write(f"*Avg Order:* ${row['avg_order_value']:.2f}")
                st.write(f"*Channel:* {row['acquisition_channel']}")
                
                if st.button(f"🗑 Remove", key=f"remove_{idx}"):
                    st.session_state.manual_customers = st.session_state.manual_customers.drop(idx).reset_index(drop=True)
                    if not st.session_state.manual_customers.empty:
                        st.session_state.customer_data = st.session_state.manual_customers.copy()
                    else:
                        st.session_state.customer_data = None
                    st.rerun()
        
        st.sidebar.write("### Bulk Actions")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📋 View All", type="secondary"):
                st.session_state.show_all_customers = True
        
        with col2:
            if st.button("🗑 Clear All", type="secondary"):
                st.session_state.manual_customers = pd.DataFrame()
                st.session_state.customer_data = None
                st.session_state.predictions = None
                st.sidebar.success("All customers cleared!")
                st.rerun()
        
        if st.sidebar.button("💾 Export Customer List", type="primary"):
            csv = st.session_state.manual_customers.to_csv(index=False)
            st.sidebar.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"manual_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="export_manual"
            )

def display_feature_request_form():
    """Display feature request form"""
    st.subheader("💡 Request a Feature")
    
    with st.form("feature_request_form"):
        st.write("*Help us improve the CLV Model Interface by suggesting new features:*")
        
        col1, col2 = st.columns(2)
        with col1:
            feature_title = st.text_input("Feature Title", placeholder="Brief title for your feature request")
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
        
        with col2:
            category = st.selectbox("Category", 
                                  ["UI/UX Improvement", "New Analysis Feature", "Data Export", 
                                   "Performance", "Integration", "Other"])
            user_email = st.text_input("Your Email (Optional)", placeholder="your.email@example.com")
        
        feature_description = st.text_area("Feature Description", 
                                         height=150,
                                         placeholder="Please describe the feature you'd like to see. Include:\n- What problem it would solve\n- How you envision it working\n- Any specific requirements")
        
        use_case = st.text_area("Use Case", 
                               height=100,
                               placeholder="Describe a specific scenario where this feature would be helpful")
        
        additional_notes = st.text_area("Additional Notes (Optional)", 
                                      height=80,
                                      placeholder="Any additional information, mockups, or references")
        
        submitted = st.form_submit_button("🚀 Submit Feature Request", type="primary")
        
        if submitted:
            if not feature_title.strip():
                st.error("Please provide a feature title!")
            elif not feature_description.strip():
                st.error("Please provide a feature description!")
            else:
                request_data = {
                    'feature_title': feature_title.strip(),
                    'feature_description': feature_description.strip(),
                    'use_case': use_case.strip() if use_case.strip() else "Not provided",
                    'additional_notes': additional_notes.strip() if additional_notes.strip() else "None",
                    'priority': priority,
                    'category': category,
                    'user_email': user_email.strip() if user_email.strip() else "Anonymous",
                    'status': 'Pending'
                }
                
                if save_feature_request(request_data):
                    st.success("🎉 Thank you! Your feature request has been submitted successfully.")
                    st.info("💡 Our development team will review your request and prioritize it accordingly.")
                else:
                    st.error("❌ Sorry, there was an error submitting your request. Please try again.")

def display_feedback_dashboard():
    """Display feedback dashboard for administrators"""
    st.write("DEBUG: display_feedback_dashboard function called")
    
    # Use the feedback manager's admin dashboard
    try:
        feedback_manager.display_admin_feedback_dashboard()
        st.write("DEBUG: feedback_manager.display_admin_feedback_dashboard() completed successfully")
    except Exception as e:
        st.error(f"DEBUG: Error in feedback dashboard: {str(e)}")
        st.write("DEBUG: Showing fallback feedback display")
        
        # Fallback: simple feedback display
        st.subheader("📊 Feedback Dashboard (Fallback)")
        
        # Load feedback directly
        feedbacks = load_feedback()
        
        if not feedbacks:
            st.info("No feedback received yet.")
            return
        
        st.write(f"DEBUG: Found {len(feedbacks)} feedback entries")
        
        # Show simple list
        for i, feedback in enumerate(feedbacks):
            st.write(f"**Feedback {i+1}:** {feedback.get('feedback_text', 'No text')}")
            st.write(f"**Type:** {feedback.get('feedback_type', 'Unknown')}")
            st.write(f"**Rating:** {feedback.get('rating', 'N/A')}/5")
            st.write(f"**Date:** {feedback.get('timestamp', 'Unknown')}")
            st.write("---")

def display_feature_requests_admin():
    """Display feature requests admin dashboard"""
    st.subheader("💡 Feature Requests Dashboard")
    
    # Load feature requests
    feature_requests = load_feature_requests()
    
    if not feature_requests:
        st.info("No feature requests received yet.")
        return
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Requests", len(feature_requests))
    
    with col2:
        pending_requests = len([r for r in feature_requests if r.get('status') == 'Pending'])
        st.metric("Pending", pending_requests)
    
    with col3:
        high_priority = len([r for r in feature_requests if r.get('priority') == 'High'])
        st.metric("High Priority", high_priority)
    
    # Display feature requests in a simple table
    st.subheader("📋 All Feature Requests")
    
    # Convert to DataFrame for display
    df = pd.DataFrame(feature_requests)
    
    # Format the data for display
    display_df = df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Reorder columns for better display
    display_df = display_df[['timestamp', 'feature_title', 'priority', 'category', 'status', 'user_email']]
    display_df.columns = ['Date', 'Title', 'Priority', 'Category', 'Status', 'Email']
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    # Show individual feature request details in expanders
    st.subheader("📝 Detailed Feature Requests")
    for i, request in enumerate(reversed(feature_requests)):
        priority_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(request.get('priority', 'Medium'), "⚪")
        status_color = {"Pending": "🟡", "Approved": "🟢", "Rejected": "🔴", "In Progress": "🔵", "Completed": "🟣"}.get(request.get('status', 'Pending'), "⚪")
        
        with st.expander(f"{priority_color} {status_color} Request #{request['id']} - {request.get('feature_title', 'Untitled')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Description:** {request.get('feature_description', 'No description')}")
                if request.get('use_case') and request.get('use_case') != "Not provided":
                    st.write(f"**Use Case:** {request.get('use_case')}")
                if request.get('additional_notes') and request.get('additional_notes') != "None":
                    st.write(f"**Additional Notes:** {request.get('additional_notes')}")
            
            with col2:
                st.write(f"**Priority:** {request.get('priority', 'Medium')}")
                st.write(f"**Category:** {request.get('category', 'Other')}")
                st.write(f"**Status:** {request.get('status', 'Pending')}")
                st.write(f"**Date:** {request.get('timestamp', 'Unknown')[:19]}")
                if request.get('user_email') and request.get('user_email') != "Anonymous":
                    st.write(f"**Email:** {request.get('user_email')}")

def main():
    # Check for admin access first
    admin_access = st.sidebar.checkbox("🔐 Admin Access", help="For developers only")
    
    if admin_access:
        admin_password = st.sidebar.text_input("Admin Password", type="password")
        if admin_password == "clv_admin_2024":  # Change this password
            st.sidebar.success("✅ Admin access granted")
            
            # Admin navigation
            admin_action = st.sidebar.selectbox("Admin Actions", 
                                              ["Main Interface", "View Feedback Dashboard", "View Feature Requests"])
            
            if admin_action == "View Feedback Dashboard":
                st.session_state.show_feedback_dashboard = True
            elif admin_action == "View Feature Requests":
                st.session_state.show_feature_requests = True
            else:
                st.session_state.show_feedback_dashboard = False
                st.session_state.show_feature_requests = False
        else:
            if admin_password:
                st.sidebar.error("❌ Invalid admin password")
    
    # Show admin dashboards if requested
    if st.session_state.show_feedback_dashboard:
        display_feedback_dashboard()
        return
    
    if st.session_state.show_feature_requests:
        display_feature_requests_admin()
        return
    
    # Show feedback form if requested
    if st.session_state.show_feedback_form:
        feedback_manager.display_feedback_form()
        if st.button("← Back to Main Interface", type="secondary"):
            st.session_state.show_feedback_form = False
            st.rerun()
        return
    
    # Show feature request form if requested
    if st.session_state.show_feature_request_form:
        display_feature_request_form()
        if st.button("← Back to Main Interface", type="secondary"):
            st.session_state.show_feature_request_form = False
            st.rerun()
        return
    
    # MAIN INTERFACE - Only show this when not in other modes
    # Header with text and instructions on the left, image on the right
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("""
        <h1 class="main-header">
            <span style="background: linear-gradient(90deg, #0a1f44, #4a6fa5);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;">
                Customer Lifetime Value (CLV) Predictor
            </span>
        </h1>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 Welcome to the CLV Predictor!
        This tool helps you predict Customer Lifetime Value (CLV) for your customers using advanced machine learning models.

        **Getting Started:**
        1. **Choose a data source** from the sidebar:
           - Upload your own CSV file with customer data
           - Generate sample data to explore the tool
           - Manually enter customer information
        2. **Configure prediction parameters** in the sidebar
        3. **Generate predictions** and explore the results

        **Required Data Fields:**
        - `customer_id`: Unique identifier for each customer
        - `age`: Customer age
        - `total_purchases`: Total number of purchases made
        - `avg_order_value`: Average order value in dollars
        - `days_since_first_purchase`: Days since first purchase
        - `days_since_last_purchase`: Days since last purchase
        - `acquisition_channel`: How the customer was acquired
        - `location`: Customer location (Urban, Suburban, Rural)
        - `subscription_status`: Current subscription status
        """)
    with col_right:
        st.image("img1.jpg", caption="CLV Model Overview", use_container_width=True)
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-header">📊 Model Configuration</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-header">Data Input</div>', unsafe_allow_html=True)
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Data", "Manual Entry"],
        label_visibility="collapsed"
    )
    
    customer_data = None
    
    # Data source handling (same as original logic)
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your customer data CSV file"
        )
        
        if uploaded_file is not None:
            try:
                customer_data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ File uploaded successfully! ({len(customer_data)} customers)")
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {str(e)}")
    
    elif data_source == "Use Sample Data":
        if st.sidebar.button("Generate Sample Data", type="primary"):
            customer_data = generate_sample_data()
            st.session_state.customer_data = customer_data
            st.sidebar.success(f"✅ Sample data generated! ({len(customer_data)} customers)")
        
        if st.session_state.customer_data is not None:
            customer_data = st.session_state.customer_data

    elif data_source == "Manual Entry":
        has_manual_data = handle_manual_entry()
        display_manual_customers()
        
        if has_manual_data:
            customer_data = st.session_state.manual_customers.copy()
        
        if st.session_state.show_all_customers:
            st.subheader("👥 All Manually Added Customers")
            if not st.session_state.manual_customers.empty:
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(st.session_state.manual_customers, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No customers added yet.")
            if st.button("❌ Close View", type="primary"):
                st.session_state.show_all_customers = False
                st.rerun()
    
    # Model Parameters
    st.sidebar.markdown('<div class="sidebar-header">🎛 Prediction Parameters</div>', unsafe_allow_html=True)
    
    time_horizon = st.sidebar.selectbox(
        "Time Horizon",
        [6, 12, 24, 36],
        index=1,
        help="Prediction period in months"
    )
    
    discount_rate = st.sidebar.slider(
        "Discount Rate",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Annual discount rate for future cash flows"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Minimum confidence level for predictions"
    )
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if customer_data is not None:
        # Display data preview and metrics (same as original)
        st.subheader("📋 Data Preview")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(customer_data.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality indicators
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{len(customer_data)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            missing_pct = (customer_data.isnull().sum().sum() / (len(customer_data) * len(customer_data.columns))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>Data Completeness</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{100-missing_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_purchases = customer_data['total_purchases'].mean() if 'total_purchases' in customer_data.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Purchases</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{avg_purchases:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            avg_order_value = customer_data['avg_order_value'].mean() if 'avg_order_value' in customer_data.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Order Value</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">${avg_order_value:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data validation
        validation_results = validate_customer_data(customer_data)
        
        if validation_results['errors']:
            st.error("❌ Data validation errors found:")
            for error in validation_results['errors']:
                st.error(f"• {error}")
        
        if validation_results['warnings']:
            st.warning("⚠️ Data validation warnings:")
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
        
        if not validation_results['errors']:
            st.success("✅ Data validation passed!")
            
            # Prediction button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔮 Generate CLV Predictions", type="primary", use_container_width=True):
                    st.session_state.trigger_prediction = True
            
            # Handle prediction trigger
            if st.session_state.trigger_prediction:
                with st.spinner("🔄 Generating CLV predictions..."):
                    try:
                        # Prepare data for prediction
                        X_pred = clv_predictor.prepare_features_for_prediction(customer_data)
                        
                        # Make predictions
                        predictions = clv_predictor.predict_clv(X_pred)
                        
                        # Create predictions DataFrame
                        predictions_df = customer_data.copy()
                        predictions_df['predicted_clv'] = predictions
                        
                        # Add additional metrics
                        predictions_df['customer_segment'] = clv_predictor.segment_customers(predictions)
                        predictions_df['churn_risk'] = 1 - (predictions_df['predicted_clv'] / predictions_df['predicted_clv'].max())
                        predictions_df['percentile_rank'] = predictions_df['predicted_clv'].rank(pct=True) * 100
                        
                        # Add confidence intervals (simplified)
                        std_dev = predictions_df['predicted_clv'].std()
                        predictions_df['confidence_lower'] = predictions_df['predicted_clv'] - (1.96 * std_dev)
                        predictions_df['confidence_upper'] = predictions_df['predicted_clv'] + (1.96 * std_dev)
                        
                        # Store predictions in session state
                        st.session_state.predictions = predictions_df
                        st.session_state.trigger_prediction = False
                        
                        st.success("✅ CLV predictions generated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating predictions: {str(e)}")
                        st.session_state.trigger_prediction = False
        
        # Display predictions if available
        if st.session_state.predictions is not None:
            predictions_df = st.session_state.predictions
            
            st.markdown('<div class="prediction-results">', unsafe_allow_html=True)
            st.subheader("📈 Prediction Results")
            
            # Summary metrics with responsive grid
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_clv = predictions_df['predicted_clv'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Average CLV</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">${avg_clv:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_clv = predictions_df['predicted_clv'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total CLV</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">${total_clv:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                high_value_customers = len(predictions_df[predictions_df['customer_segment'] == 'High Value'])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>High Value Customers</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{high_value_customers}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_churn_risk = predictions_df['churn_risk'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Churn Risk</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{avg_churn_risk:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.subheader("📊 Analytics Dashboard")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Get all visualizations
            visualizations = clv_predictor.create_visualizations(predictions_df)
            
            # Create tabs for different analysis categories
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🎯 Core Metrics", 
                "📈 Customer Analysis", 
                "💰 Revenue Insights", 
                "⚠️ Risk Analysis", 
                "📋 Detailed Reports",
                "🎯 Actionable Insights"
            ])
            
            with tab1:
                st.subheader("🎯 Core CLV Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(visualizations['clv_distribution'], use_container_width=True, config={'displayModeBar': False})
                    st.plotly_chart(visualizations['customer_segments'], use_container_width=True, config={'displayModeBar': False})
                
                with col2:
                    st.plotly_chart(visualizations['segment_boxplot'], use_container_width=True, config={'displayModeBar': False})
                    if visualizations['feature_importance']:
                        st.plotly_chart(visualizations['feature_importance'], use_container_width=True, config={'displayModeBar': False})
            
            with tab2:
                st.subheader("📈 Customer Behavior Analysis")
                
                # Row 1: Age and Purchase Analysis
                col1, col2 = st.columns(2)
                with col1:
                    if visualizations['age_vs_clv']:
                        st.plotly_chart(visualizations['age_vs_clv'], use_container_width=True, config={'displayModeBar': False})
                with col2:
                    if visualizations['purchases_vs_clv']:
                        st.plotly_chart(visualizations['purchases_vs_clv'], use_container_width=True, config={'displayModeBar': False})
                
                # Row 2: AOV and Channel Analysis
                col1, col2 = st.columns(2)
                with col1:
                    if visualizations['aov_vs_clv']:
                        st.plotly_chart(visualizations['aov_vs_clv'], use_container_width=True, config={'displayModeBar': False})
                with col2:
                    if visualizations['channel_segment']:
                        st.plotly_chart(visualizations['channel_segment'], use_container_width=True, config={'displayModeBar': False})
                
                # Row 3: Location and Subscription Analysis
                col1, col2 = st.columns(2)
                with col1:
                    if visualizations['location_clv']:
                        st.plotly_chart(visualizations['location_clv'], use_container_width=True, config={'displayModeBar': False})
                with col2:
                    if visualizations['subscription_clv']:
                        st.plotly_chart(visualizations['subscription_clv'], use_container_width=True, config={'displayModeBar': False})
            
            with tab3:
                st.subheader("💰 Revenue & Business Impact")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(visualizations['revenue_impact'], use_container_width=True, config={'displayModeBar': False})
                    
                    # Revenue breakdown table
                    st.subheader("💵 Revenue Breakdown by Segment")
                    revenue_df = visualizations['segment_revenue'].copy()
                    revenue_df['predicted_clv'] = revenue_df['predicted_clv'].apply(lambda x: f"${x:,.2f}")
                    revenue_df['percentage'] = revenue_df['percentage'].apply(lambda x: f"{x:.1f}%")
                    revenue_df.columns = ['Segment', 'Total CLV', 'Percentage']
                    st.dataframe(revenue_df, use_container_width=True)
                
                with col2:
                    st.plotly_chart(visualizations['percentile_distribution'], use_container_width=True, config={'displayModeBar': False})
                    
                    # Summary statistics
                    st.subheader("📊 Segment Statistics")
                    summary_df = visualizations['summary_stats'].copy()
                    # Format numeric columns
                    for col in summary_df.columns:
                        if 'predicted_clv' in col and col != 'customer_segment':
                            summary_df[col] = summary_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                        elif 'churn_risk' in col:
                            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    st.dataframe(summary_df, use_container_width=True)
            
            with tab4:
                st.subheader("⚠️ Risk & Churn Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(visualizations['churn_distribution'], use_container_width=True, config={'displayModeBar': False})
                    st.plotly_chart(visualizations['clv_vs_churn'], use_container_width=True, config={'displayModeBar': False})
                
                with col2:
                    st.plotly_chart(visualizations['risk_reward_matrix'], use_container_width=True, config={'displayModeBar': False})
                    
                    if visualizations['confidence_intervals']:
                        st.plotly_chart(visualizations['confidence_intervals'], use_container_width=True, config={'displayModeBar': False})
            
            with tab5:
                st.subheader("📋 Detailed Reports & Tables")
                
                # Top customers
                st.subheader("🏆 Top 10 Customers by CLV")
                st.dataframe(visualizations['top_customers'], use_container_width=True)
                
                # Risk assessment table
                st.subheader("⚠️ High-Risk Customers (Churn Risk > 50%)")
                high_risk_mask = predictions_df['churn_risk'] > 0.5
                high_risk_customers = predictions_df[high_risk_mask].sort_values('predicted_clv', ascending=False).head(10)[
                    ['customer_id', 'predicted_clv', 'customer_segment', 'churn_risk', 'percentile_rank']
                ].copy()
                if len(high_risk_customers) > 0:
                    high_risk_customers['predicted_clv'] = high_risk_customers['predicted_clv'].round(2)
                    high_risk_customers['churn_risk'] = (high_risk_customers['churn_risk'] * 100).round(1)
                    high_risk_customers['percentile_rank'] = high_risk_customers['percentile_rank'].round(1)
                    high_risk_customers.columns = ['Customer ID', 'Predicted CLV ($)', 'Segment', 'Churn Risk (%)', 'Percentile Rank (%)']
                    st.dataframe(high_risk_customers, use_container_width=True)
                else:
                    st.info("No high-risk customers found (all customers have churn risk ≤ 50%)")
                
                # Segment comparison table
                st.subheader("📊 Segment Comparison")
                segment_comparison = predictions_df.groupby('customer_segment').agg({
                    'predicted_clv': ['count', 'mean', 'median', 'sum'],
                    'churn_risk': 'mean',
                    'percentile_rank': 'mean'
                }).round(2)
                segment_comparison.columns = ['Customer Count', 'Avg CLV', 'Median CLV', 'Total CLV', 'Avg Churn Risk', 'Avg Percentile']
                segment_comparison = segment_comparison.reset_index()
                st.dataframe(segment_comparison, use_container_width=True)
            
            with tab6:
                st.subheader("🎯 Actionable Insights & Recommendations")
                
                # Calculate advanced metrics
                clv_metrics = clv_predictor.calculate_customer_lifetime_metrics(predictions_df)
                recommendations = clv_predictor.generate_customer_action_recommendations(predictions_df)
                
                # Key Metrics Overview
                st.subheader("📊 Key Business Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Customers", f"{clv_metrics['total_customers']:,}")
                    st.metric("Total CLV", f"${clv_metrics['total_clv']:,.2f}")
                
                with col2:
                    st.metric("Average CLV", f"${clv_metrics['avg_clv']:,.2f}")
                    st.metric("Median CLV", f"${clv_metrics['median_clv']:,.2f}")
                
                with col3:
                    st.metric("High Risk Customers", f"{clv_metrics['high_risk_count']:,}")
                    st.metric("High Risk %", f"{clv_metrics['high_risk_percentage']:.1f}%")
                
                with col4:
                    st.metric("Top 10% Revenue Share", f"{clv_metrics['top_10_percent_share']:.1f}%")
                    st.metric("High Risk CLV Value", f"${clv_metrics['high_risk_clv_value']:,.2f}")
                
                # Customer Tier Analysis
                st.subheader("🏆 Customer Value Tiers")
                tier_df = clv_metrics['tier_analysis'].copy()
                tier_df.columns = ['Tier', 'Customer Count', 'Total CLV', 'Average CLV', 'Avg Churn Risk']
                tier_df['Total CLV'] = tier_df['Total CLV'].apply(lambda x: f"${x:,.2f}")
                tier_df['Average CLV'] = tier_df['Average CLV'].apply(lambda x: f"${x:,.2f}")
                tier_df['Avg Churn Risk'] = tier_df['Avg Churn Risk'].apply(lambda x: f"{x:.1%}")
                st.dataframe(tier_df, use_container_width=True)
                
                # Actionable Recommendations
                st.subheader("🎯 Strategic Recommendations")
                
                for strategy, data in recommendations.items():
                    with st.expander(f"📋 {strategy.replace('_', ' ').title()}", expanded=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Customer Count", f"{data['count']:,}")
                            st.metric("Average CLV", f"${data['avg_clv']:,.2f}")
                        
                        with col2:
                            st.write("**Recommended Actions:**")
                            for action in data['actions']:
                                st.write(f"• {action}")
                
                # Revenue Concentration Analysis
                st.subheader("💰 Revenue Concentration Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Revenue Distribution:**")
                    st.write(f"• Top 10% of customers generate {clv_metrics['top_10_percent_share']:.1f}% of total revenue")
                    st.write(f"• High-risk customers represent {clv_metrics['high_risk_percentage']:.1f}% of customer base")
                    st.write(f"• High-risk customers hold ${clv_metrics['high_risk_clv_value']:,.2f} in potential revenue")
                
                with col2:
                    st.write("**Business Implications:**")
                    if clv_metrics['top_10_percent_share'] > 50:
                        st.warning("⚠️ High revenue concentration - focus on customer diversification")
                    else:
                        st.success("✅ Healthy revenue distribution across customer base")
                    
                    if clv_metrics['high_risk_percentage'] > 30:
                        st.error("🚨 High churn risk - immediate retention focus needed")
                    elif clv_metrics['high_risk_percentage'] > 15:
                        st.warning("⚠️ Moderate churn risk - proactive retention recommended")
                    else:
                        st.success("✅ Low churn risk - focus on growth and expansion")
                
                # CLV Percentile Analysis
                st.subheader("📈 CLV Percentile Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("25th Percentile", f"${clv_metrics['clv_25th_percentile']:,.2f}")
                with col2:
                    st.metric("75th Percentile", f"${clv_metrics['clv_75th_percentile']:,.2f}")
                with col3:
                    st.metric("90th Percentile", f"${clv_metrics['clv_90th_percentile']:,.2f}")
                with col4:
                    st.metric("95th Percentile", f"${clv_metrics['clv_95th_percentile']:,.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed results table with filters
            st.subheader("📋 Detailed Predictions")
            st.markdown('<div class="filter-controls">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                segment_filter = st.multiselect(
                    "Filter by Segment",
                    predictions_df['customer_segment'].unique(),
                    default=predictions_df['customer_segment'].unique()
                )
            
            with col2:
                min_clv = st.number_input("Minimum CLV", value=0.0)
            
            with col3:
                max_churn_risk = st.slider("Max Churn Risk", 0.0, 1.0, 1.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            filtered_df = predictions_df[
                (predictions_df['customer_segment'].isin(segment_filter)) &
                (predictions_df['predicted_clv'] >= min_clv) &
                (predictions_df['churn_risk'] <= max_churn_risk)
            ]
            
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_df.style.format({
                'predicted_clv': '${:,.2f}',
                'confidence_lower': '${:,.2f}',
                'confidence_upper': '${:,.2f}',
                'avg_order_value': '${:,.2f}',
                'churn_risk': '{:.2%}',
                'percentile_rank': '{:.1f}%'
            }), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Export functionality
            st.subheader("📥 Export Results")
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=f"clv_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='CLV_Predictions', index=False)
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"clv_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Email support section
            st.markdown("---")
            
            # Row 1: Contact Support and Request Feature (symmetrical)
            col1, col2 = st.columns(2)
            
            with col1:
                support_email = "kratikasoni73@gmail.com"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 15px;">Contact Support</h4>
                    <p style="color: white; margin-bottom: 15px;">Have questions or need assistance?</p>
                    <a href="mailto:{support_email}?subject=CLV Model Support Request&body=Hello,%0D%0A%0D%0AI need help with the CLV Model Interface.%0D%0A%0D%0APlease describe your issue:%0D%0A" 
                       style="display: inline-block; background: white; color: #667eea; padding: 12px 24px; 
                              border-radius: 25px; text-decoration: none; font-weight: bold; 
                              transition: all 0.3s ease;">
                        📧 Email Support: {support_email}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create a container for the styled box with the button inside
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; text-align: center; position: relative;">
                    <h4 style="color: white; margin-bottom: 15px;">Request Feature</h4>
                    <p style="color: white; margin-bottom: 15px;">Suggest new features for the CLV Model</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Position the button inside the box using negative margin
                st.markdown("""
                <div style="margin-top: -50px; margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                if st.button("💡 Request Feature", key="request_feature_btn", type="primary", use_container_width=True):
                    st.session_state.show_feature_request_form = True
                    st.session_state.show_feature_requests = False  # Ensure admin dashboard doesn't open
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Row 2: View Documentation and Provide Feedback (symmetrical)
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📚 View Documentation", type="secondary", use_container_width=True):
                    doc_handler.display_documentation()
            
            with col2:
                if st.button("💬 Provide Feedback", type="secondary", use_container_width=True):
                    st.session_state.show_feedback_form = True
            
            # Row 3: Version and License (centered)
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <p style="margin: 0; color: #666; font-size: 14px;">Version 1.0</p>
                <p style="margin: 0; color: #666; font-size: 14px;">© 2024 CLV Model Interface</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Please select a data source and load customer data to begin CLV analysis.")
    
    # Close main content container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with documentation and feedback
    st.markdown("---")
    
    # Row 1: Contact Support and Request Feature (symmetrical)
    col1, col2 = st.columns(2)
    
    with col1:
        support_email = "kratikasoni73@gmail.com"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: white; margin-bottom: 15px;">Contact Support</h4>
            <p style="color: white; margin-bottom: 15px;">Have questions or need assistance?</p>
            <a href="mailto:{support_email}?subject=CLV Model Support Request&body=Hello,%0D%0A%0D%0AI need help with the CLV Model Interface.%0D%0A%0D%0APlease describe your issue:%0D%0A" 
               style="display: inline-block; background: white; color: #667eea; padding: 12px 24px; 
                      border-radius: 25px; text-decoration: none; font-weight: bold; 
                      transition: all 0.3s ease;">
                📧 Email Support: {support_email}
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create a container for the styled box with the button inside
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; position: relative;">
            <h4 style="color: white; margin-bottom: 15px;">Request Feature</h4>
            <p style="color: white; margin-bottom: 15px;">Suggest new features for the CLV Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Position the button inside the box using negative margin
        st.markdown("""
        <div style="margin-top: -50px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        if st.button("💡 Request Feature", key="request_feature_btn_2", type="primary", use_container_width=True):
            st.session_state.show_feature_request_form = True
            st.session_state.show_feature_requests = False  # Ensure admin dashboard doesn't open
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 2: View Documentation and Provide Feedback (symmetrical)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 View Documentation", type="secondary", use_container_width=True):
            doc_handler.display_documentation()
    
    with col2:
        if st.button("💬 Provide Feedback", type="secondary", use_container_width=True):
            st.session_state.show_feedback_form = True
    
    # Row 3: Version and License (centered)
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p style="margin: 0; color: #666; font-size: 14px;">Version 1.0</p>
        <p style="margin: 0; color: #666; font-size: 14px;">© 2024 CLV Model Interface</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()