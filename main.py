import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import os
import sys
import traceback
from io import BytesIO
import base64
import hashlib

# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================

class AuthenticationSystem:
    def __init__(self):
        self.users = {
            'admin': {
                'password': self._hash_password('admin123'),
                'role': 'admin',
                'name': 'System Administrator'
            },
            'analyst': {
                'password': self._hash_password('analyst123'),
                'role': 'analyst',
                'name': 'Financial Analyst'
            },
            'operator': {
                'password': self._hash_password('operator123'),
                'role': 'operator',
                'name': 'Loan Operator'
            }
        }
    
    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_login(self, username, password):
        if username in self.users:
            hashed_password = self._hash_password(password)
            if self.users[username]['password'] == hashed_password:
                return self.users[username]
        return None
    
    def get_user_role(self, username):
        return self.users.get(username, {}).get('role', 'guest')

# =============================================================================
# INITIAL CONFIGURATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'applications' not in st.session_state:
        st.session_state.applications = []
    if 'kaggle_data' not in st.session_state:
        st.session_state.kaggle_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

# =============================================================================
# DATA AND MODEL LOADING
# =============================================================================

def load_kaggle_data():
    """Load and process the Kaggle dataset"""
    try:
        file_paths = [
            "rf_kaggle.csv",
            "./rf_kaggle.csv",
            "source/rf_kaggle.csv"
        ]
        
        for path in file_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.session_state.kaggle_data = df
                return df
        
        # Create demo data if file not found
        st.warning("⚠️ rf_kaggle.csv not found. Using demo data.")
        return create_demo_data()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_demo_data()

def create_demo_data():
    """Create demo dataset when real data is not available"""
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'loan_amount': np.random.normal(25000, 15000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'debt_to_income': np.random.normal(35, 15, n_samples),
        'risk_score': np.random.uniform(0, 1, n_samples),
        'status': np.random.choice(['approved', 'rejected', 'under_review'], n_samples, p=[0.6, 0.25, 0.15]),
        'application_date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'fraud_flag': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(demo_data)
    st.session_state.kaggle_data = df
    return df

def load_model():
    """Load the trained ML model"""
    try:
        model_paths = [
            "rf_kaggle.joblib",
            "./rf_kaggle.joblib",
            "source/rf_kaggle.joblib"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                return model
        
        st.warning("⚠️ Model file not found. Using demo mode.")
        return create_demo_model()
        
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return create_demo_model()

def create_demo_model():
    """Create a demo model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

# =============================================================================
# LOAN APPLICATION FORM SYSTEM
# =============================================================================

class LoanApplicationForm:
    def __init__(self):
        self.sections = {
            "personal_data": "📋 Personal Information",
            "financial_info": "💰 Financial Information", 
            "loan_details": "🏠 Loan Details",
            "review": "👁️ Review & Submit"
        }
        self.current_section = "personal_data"
        
    def render_progress_bar(self):
        sections = list(self.sections.keys())
        current_index = sections.index(self.current_section)
        progress = (current_index + 1) / len(sections)
        
        st.markdown(f"""
        <div style='margin: 20px 0;'>
            <div style='background: #e0e0e0; border-radius: 10px; height: 20px;'>
                <div style='background: linear-gradient(135deg, #1e3c72, #2a5298); width: {progress*100}%; 
                         border-radius: 10px; height: 20px; text-align: center; color: white; 
                         font-weight: bold; font-size: 0.8em; line-height: 20px;'>
                    {int(progress*100)}% Complete
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(sections))
        for i, section in enumerate(sections):
            with cols[i]:
                status = "✅" if i < current_index else "🔵" if i == current_index else "⚪"
                st.markdown(f"<div style='text-align: center; font-size: 0.8em;'>{status}<br>{self.sections[section]}</div>", 
                          unsafe_allow_html=True)

    def section_personal_data(self):
        st.subheader("📋 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name *", placeholder="John Smith")
            age = st.number_input("Age *", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            marital_status = st.selectbox("Marital Status *", ["Single", "Married", "Divorced", "Widowed"])
            
        with col2:
            tax_id = st.text_input("Tax ID *", placeholder="123456789")
            phone = st.text_input("Phone *", placeholder="+1 555 123 4567")
            email = st.text_input("Email *", placeholder="john.smith@email.com")
            nationality = st.selectbox("Nationality *", ["Citizen", "Permanent Resident", "Other"])
            
        if st.button("Continue to Financial Information →", type="primary"):
            if all([full_name, tax_id, phone, email]):
                st.session_state.personal_data = {
                    'full_name': full_name, 'age': age, 'gender': gender,
                    'marital_status': marital_status, 'tax_id': tax_id,
                    'phone': phone, 'email': email, 'nationality': nationality
                }
                self.current_section = "financial_info"
                st.rerun()
            else:
                st.error("Please fill all required fields (*)")

    def section_financial_info(self):
        st.subheader("💰 Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input("Monthly Net Income ($) *", min_value=0.0, value=3000.0, step=100.0)
            employment_type = st.selectbox("Employment Type *", ["Employed", "Self-Employed", "Unemployed", "Retired"])
            employment_duration = st.number_input("Current Employment Duration (months) *", min_value=0, value=24)
            
        with col2:
            monthly_expenses = st.number_input("Monthly Expenses ($) *", min_value=0.0, value=1500.0, step=50.0)
            credit_history = st.selectbox("Credit History *", ["Excellent", "Good", "Fair", "Poor"])
            savings = st.number_input("Savings & Investments ($)", min_value=0.0, value=10000.0, step=1000.0)
        
        financial_capacity = monthly_income - monthly_expenses
        st.info(f"💡 **Monthly Financial Capacity:** ${financial_capacity:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                self.current_section = "personal_data"
                st.rerun()
        with col2:
            if st.button("Continue to Loan Details →", type="primary"):
                st.session_state.financial_info = {
                    'monthly_income': monthly_income, 'employment_type': employment_type,
                    'employment_duration': employment_duration, 'monthly_expenses': monthly_expenses,
                    'credit_history': credit_history, 'savings': savings,
                    'financial_capacity': financial_capacity
                }
                self.current_section = "loan_details"
                st.rerun()

    def section_loan_details(self):
        st.subheader("🏠 Loan Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount ($) *", min_value=1000.0, value=25000.0, step=1000.0)
            loan_term = st.slider("Loan Term (months) *", min_value=12, max_value=360, value=120)
            loan_purpose = st.selectbox("Loan Purpose *", ["Home Purchase", "Debt Consolidation", "Education", "Medical", "Business", "Other"])
            
        with col2:
            interest_rate = st.slider("Annual Interest Rate (%)", min_value=1.0, max_value=15.0, value=5.0, step=0.1)
            monthly_payment = (loan_amount * (interest_rate/100/12)) / (1 - (1 + interest_rate/100/12)**(-loan_term))
            st.metric("Estimated Monthly Payment", f"${monthly_payment:,.2f}")
            
            income = st.session_state.get('financial_info', {}).get('monthly_income', 3000)
            if income > 0:
                debt_to_income = (monthly_payment / income) * 100
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                self.current_section = "financial_info"
                st.rerun()
        with col2:
            if st.button("Review Application →", type="primary"):
                st.session_state.loan_details = {
                    'loan_amount': loan_amount, 'loan_term': loan_term,
                    'loan_purpose': loan_purpose, 'interest_rate': interest_rate,
                    'monthly_payment': monthly_payment, 'debt_to_income': debt_to_income
                }
                self.current_section = "review"
                st.rerun()

    def section_review(self):
        st.subheader("👁️ Application Review")
        
        # Display application summary in table format
        self.display_application_summary()
        
        consent = st.checkbox("✅ I confirm all provided information is accurate and truthful *")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Loan Details"):
                self.current_section = "loan_details"
                st.rerun()
        with col2:
            if st.button("🚀 Submit Application", type="primary"):
                if consent:
                    return True
                else:
                    st.error("Please confirm the information accuracy")
        return False

    def display_application_summary(self):
        """Display application summary in professional table format"""
        personal = st.session_state.get('personal_data', {})
        financial = st.session_state.get('financial_info', {})
        loan = st.session_state.get('loan_details', {})
        
        # Personal Information Table
        st.subheader("👤 Personal Information")
        personal_df = pd.DataFrame([
            ["Full Name", personal.get('full_name', '')],
            ["Age", personal.get('age', '')],
            ["Gender", personal.get('gender', '')],
            ["Marital Status", personal.get('marital_status', '')],
            ["Tax ID", personal.get('tax_id', '')],
            ["Phone", personal.get('phone', '')],
            ["Email", personal.get('email', '')],
            ["Nationality", personal.get('nationality', '')]
        ], columns=["Field", "Value"])
        st.dataframe(personal_df, use_container_width=True, hide_index=True)
        
        # Financial Information Table
        st.subheader("💰 Financial Information")
        financial_df = pd.DataFrame([
            ["Monthly Income", f"${financial.get('monthly_income', 0):,.2f}"],
            ["Employment Type", financial.get('employment_type', '')],
            ["Employment Duration", f"{financial.get('employment_duration', 0)} months"],
            ["Monthly Expenses", f"${financial.get('monthly_expenses', 0):,.2f}"],
            ["Credit History", financial.get('credit_history', '')],
            ["Savings", f"${financial.get('savings', 0):,.2f}"],
            ["Financial Capacity", f"${financial.get('financial_capacity', 0):,.2f}"]
        ], columns=["Field", "Value"])
        st.dataframe(financial_df, use_container_width=True, hide_index=True)
        
        # Loan Details Table
        st.subheader("🏠 Loan Details")
        loan_df = pd.DataFrame([
            ["Loan Amount", f"${loan.get('loan_amount', 0):,.2f}"],
            ["Loan Term", f"{loan.get('loan_term', 0)} months"],
            ["Loan Purpose", loan.get('loan_purpose', '')],
            ["Interest Rate", f"{loan.get('interest_rate', 0)}%"],
            ["Monthly Payment", f"${loan.get('monthly_payment', 0):,.2f}"],
            ["Debt-to-Income Ratio", f"{loan.get('debt_to_income', 0):.1f}%"]
        ], columns=["Field", "Value"])
        st.dataframe(loan_df, use_container_width=True, hide_index=True)

    def get_application_data(self):
        """Combine all application data"""
        return {
            'personal_data': st.session_state.get('personal_data', {}),
            'financial_info': st.session_state.get('financial_info', {}),
            'loan_details': st.session_state.get('loan_details', {}),
            'application_id': f"APP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'submission_date': datetime.now(),
            'status': 'submitted',
            'assigned_to': st.session_state.current_user
        }

    def render_form(self):
        """Render complete form"""
        self.render_progress_bar()
        
        if self.current_section == "personal_data":
            self.section_personal_data()
        elif self.current_section == "financial_info":
            self.section_financial_info()
        elif self.current_section == "loan_details":
            self.section_loan_details()
        elif self.current_section == "review":
            return self.section_review()
        return False

# =============================================================================
# APPLICATION ANALYSIS SYSTEM
# =============================================================================

class ApplicationAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def extract_features(self, application_data):
        """Extract features from application form for model prediction"""
        personal = application_data.get('personal_data', {})
        financial = application_data.get('financial_info', {})
        loan = application_data.get('loan_details', {})
        
        features = {
            'age': personal.get('age', 30),
            'monthly_income': financial.get('monthly_income', 3000),
            'monthly_expenses': financial.get('monthly_expenses', 1500),
            'loan_amount': loan.get('loan_amount', 25000),
            'loan_term': loan.get('loan_term', 120),
            'savings': financial.get('savings', 10000),
            'employment_duration': financial.get('employment_duration', 24),
            'debt_to_income': loan.get('debt_to_income', 30)
        }
        return features
    
    def predict_risk(self, features):
        """Make prediction using ML model"""
        try:
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            probability = self.model.predict_proba(feature_array)[0, 1]
            return probability
        except Exception as e:
            # Fallback for demo
            base_risk = 0.3
            if features.get('debt_to_income', 0) > 40:
                base_risk += 0.3
            if features.get('savings', 0) < features.get('loan_amount', 0) * 0.1:
                base_risk += 0.2
            return min(base_risk + random.uniform(-0.1, 0.1), 0.95)
    
    def get_decision_explanation(self, probability, features):
        """Generate detailed decision explanation"""
        if probability < 0.3:
            return {
                'level': "✅ LOW RISK",
                'color': "#10B981",
                'message': "Application Approved - Low Risk Profile",
                'reasons': [
                    "Strong financial capacity demonstrated",
                    "Stable financial history",
                    "Loan amount appropriate for profile",
                    "Good debt-to-income ratio"
                ],
                'next_steps': [
                    "We will contact you within 24 hours for finalization",
                    "Please prepare supporting documentation",
                    "Funds disbursement in 3-5 business days"
                ],
                'status': 'approved'
            }
        elif probability < 0.7:
            return {
                'level': "⚠️ MODERATE RISK", 
                'color': "#F59E0B",
                'message': "Application Under Review - Additional Assessment Required",
                'reasons': [
                    "Recommended additional documentation review",
                    "Suggested verification of references",
                    "Possible need for additional guarantees",
                    "Moderate debt-to-income ratio"
                ],
                'next_steps': [
                    "Additional documentation required",
                    "We will contact you for clarifications",
                    "Review period: 2-3 business days"
                ],
                'status': 'under_review'
            }
        else:
            return {
                'level': "🚨 HIGH RISK",
                'color': "#DC2626", 
                'message': "High Risk Application - Recommended for Decline",
                'reasons': [
                    "Inconsistencies in provided data",
                    "Insufficient financial capacity",
                    "High risk of default",
                    "Concerns about repayment ability"
                ],
                'next_steps': [
                    "We recommend reviewing your financial situation",
                    "You may reapply after 6 months",
                    "Contact us for alternative solutions"
                ],
                'status': 'rejected'
            }

    def render_risk_visualization(self, probability):
        """Render visual risk assessment"""
        fig, ax = plt.subplots(figsize=(10, 2))
        
        ax.axvspan(0, 30, alpha=0.3, color='#10B981', label='Low Risk')
        ax.axvspan(30, 70, alpha=0.3, color='#F59E0B', label='Moderate Risk')
        ax.axvspan(70, 100, alpha=0.3, color='#DC2626', label='High Risk')
        
        risk_percent = probability * 100
        ax.axvline(x=risk_percent, color='black', linestyle='--', linewidth=2)
        ax.plot(risk_percent, 0, 'ko', markersize=10)
        
        ax.set_xlim(0, 100)
        ax.set_xlabel('Risk Probability (%)', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_dashboard():
    """Render the main dashboard with real data"""
    st.markdown("<h2>📊 Dashboard Overview</h2>", unsafe_allow_html=True)
    
    # Load data
    df = st.session_state.kaggle_data
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_apps = len(df)
        st.metric("Total Applications", f"{total_apps:,}")
    
    with col2:
        approval_rate = len(df[df['status'] == 'approved']) / len(df) * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    with col3:
        fraud_rate = df['fraud_flag'].mean() * 100
        st.metric("Fraud Detection Rate", f"{fraud_rate:.1f}%")
    
    with col4:
        avg_processing = 2.3  # This would come from real data
        st.metric("Avg Processing Time", f"{avg_processing} days")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Application Status Distribution")
        status_counts = df['status'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['risk_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Amount vs Risk Score")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df['loan_amount'], df['risk_score'], c=df['fraud_flag'], 
                           alpha=0.6, cmap='coolwarm')
        ax.set_xlabel('Loan Amount')
        ax.set_ylabel('Risk Score')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Monthly Applications")
        monthly_data = df.groupby(df['application_date'].dt.to_period('M')).size()
        fig, ax = plt.subplots(figsize=(8, 6))
        monthly_data.plot(kind='line', ax=ax, marker='o')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Applications')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Recent Applications Table
    st.subheader("Recent Applications")
    recent_apps = df.tail(10)[['application_date', 'age', 'income', 'loan_amount', 'risk_score', 'status']]
    st.dataframe(recent_apps, use_container_width=True)

# =============================================================================
# APPLICATION LOOKUP COMPONENT
# =============================================================================

def render_application_lookup():
    """Render application lookup interface"""
    st.markdown("<h2>🔍 Application Lookup</h2>", unsafe_allow_html=True)
    
    df = st.session_state.kaggle_data
    
    # Search filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("Search by ID or Name")
    
    with col2:
        status_filter = st.selectbox("Status Filter", ["All", "approved", "rejected", "under_review"])
    
    with col3:
        risk_threshold = st.slider("Risk Score Threshold", 0.0, 1.0, 0.5)
    
    # Filter data
    filtered_df = df.copy()
    
    if search_term:
        filtered_df = filtered_df[filtered_df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
    
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    
    filtered_df = filtered_df[filtered_df['risk_score'] >= risk_threshold]
    
    # Display results
    st.subheader(f"Found {len(filtered_df)} Applications")
    
    if len(filtered_df) > 0:
        # Select columns to display
        display_columns = ['application_date', 'age', 'income', 'loan_amount', 'credit_score', 'risk_score', 'status']
        display_df = filtered_df[display_columns].head(50)  # Limit display
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export to CSV"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="applications_export.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Show Statistics"):
                st.subheader("Dataset Statistics")
                st.write(filtered_df[['age', 'income', 'loan_amount', 'risk_score']].describe())

# =============================================================================
# ANALYTICS COMPONENT
# =============================================================================

def render_analytics():
    """Render advanced analytics dashboard"""
    st.markdown("<h2>📈 Analytics & Reports</h2>", unsafe_allow_html=True)
    
    df = st.session_state.kaggle_data
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            [df['application_date'].min(), df['application_date'].max()]
        )
    
    with col2:
        loan_range = st.slider(
            "Loan Amount Range",
            float(df['loan_amount'].min()),
            float(df['loan_amount'].max()),
            (float(df['loan_amount'].min()), float(df['loan_amount'].max()))
        )
    
    with col3:
        age_range = st.slider(
            "Age Range",
            int(df['age'].min()),
            int(df['age'].max()),
            (int(df['age'].min()), int(df['age'].max()))
        )
    
    # Filter data
    filtered_df = df[
        (df['application_date'] >= pd.to_datetime(date_range[0])) &
        (df['application_date'] <= pd.to_datetime(date_range[1])) &
        (df['loan_amount'] >= loan_range[0]) &
        (df['loan_amount'] <= loan_range[1]) &
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1])
    ]
    
    # Analytics Charts
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Demographic Insights", "Performance Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud Distribution by Loan Purpose")
            # This would be based on actual loan purpose data
            fraud_data = pd.DataFrame({
                'purpose': ['Home', 'Car', 'Business', 'Personal', 'Education'],
                'fraud_rate': [0.05, 0.12, 0.18, 0.08, 0.03]
            })
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(fraud_data['purpose'], fraud_data['fraud_rate'], color='lightcoral')
            ax.set_ylabel('Fraud Rate')
            ax.set_xlabel('Loan Purpose')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Risk Correlation Heatmap")
            numeric_cols = ['age', 'income', 'loan_amount', 'credit_score', 'risk_score']
            corr_matrix = filtered_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution by Status")
            fig, ax = plt.subplots(figsize=(8, 6))
            for status in filtered_df['status'].unique():
                status_data = filtered_df[filtered_df['status'] == status]
                ax.hist(status_data['age'], alpha=0.6, label=status, bins=15)
            ax.legend()
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Income vs Loan Amount")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(filtered_df['income'], filtered_df['loan_amount'], 
                               c=filtered_df['risk_score'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Income')
            ax.set_ylabel('Loan Amount')
            plt.colorbar(scatter, label='Risk Score')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Performance Metrics Over Time")
        
        # Monthly performance
        monthly_perf = filtered_df.groupby(filtered_df['application_date'].dt.to_period('M')).agg({
            'risk_score': 'mean',
            'fraud_flag': 'mean',
            'loan_amount': 'count'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(monthly_perf['application_date'].astype(str), monthly_perf['risk_score'], marker='o')
        ax1.set_ylabel('Average Risk Score')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(monthly_perf['application_date'].astype(str), monthly_perf['fraud_flag'] * 100)
        ax2.set_ylabel('Fraud Rate (%)')
        ax2.set_xlabel('Month')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

# =============================================================================
# LOGIN PAGE
# =============================================================================

def render_login_page(auth_system):
    """Render the login page"""
    st.markdown("""
        <div style='text-align: center; padding: 50px 0;'>
            <h1 style='color: #1e3c72; font-size: 2.5em;'>🏦 Loan Fraud Detection System</h1>
            <p style='color: #666; font-size: 1.2em;'>Secure Application Processing Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.subheader("🔐 User Login")
                
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                if st.form_submit_button("Login", type="primary"):
                    if username and password:
                        user_info = auth_system.verify_login(username, password)
                        if user_info:
                            st.session_state.authenticated = True
                            st.session_state.current_user = username
                            st.session_state.user_role = user_info['role']
                            st.success(f"Welcome, {user_info['name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.error("Please enter both username and password")
            
            # Demo credentials
            with st.expander("Demo Credentials"):
                st.write("""
                **Admin:** admin / admin123
                **Analyst:** analyst / analyst123  
                **Operator:** operator / operator123
                """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="Loan Fraud Detection System",
        layout="wide",
        page_icon="🛡️"
    )
    
    # Initialize authentication
    auth_system = AuthenticationSystem()
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        render_login_page(auth_system)
        return
    
    # Load data and model
    if st.session_state.kaggle_data is None:
        with st.spinner("Loading data..."):
            load_kaggle_data()
    
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
    
    # Main header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
                <h1 style='color: white; margin: 0; font-size: 2em;'>🏦 Loan Application Fraud Detection System</h1>
                <p style='color: #e0e0e0; margin: 5px 0 0 0; font-size: 1.1em;'>
                Intelligent Loan Application Assessment - Fraud Detection & Prevention
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; text-align: center;'>
                <p style='margin: 0; font-weight: bold;'>👤 {st.session_state.current_user}</p>
                <p style='margin: 0; color: #666;'>{st.session_state.user_role.title()}</p>
                <button onclick="window.location.href='?logout=true'" style='margin-top: 10px; padding: 5px 15px; background: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer;'>
                    Logout
                </button>
            </div>
        """, unsafe_allow_html=True)
        
        # Handle logout
        if st.button("Logout", key="logout_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🆕 New Application", 
        "🔍 Application Lookup", 
        "📈 Analytics & Reports"
    ])
    
    # TAB 1: DASHBOARD
    with tab1:
        render_dashboard()
    
    # TAB 2: NEW APPLICATION
    with tab2:
        st.markdown("<h2>📝 New Loan Application</h2>", unsafe_allow_html=True)
        
        if 'form' not in st.session_state:
            st.session_state.form = LoanApplicationForm()
        
        form = st.session_state.form
        
        with st.container():
            st.markdown("""
                <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1e3c72;'>
                    <h3 style='color: #1e3c72; margin: 0;'>Loan Application Form</h3>
                    <p style='color: #666; margin: 5px 0;'>Complete all sections to submit your application</p>
                </div>
            """, unsafe_allow_html=True)
            
            submission_ready = form.render_form()
            
            if submission_ready:
                st.success("✅ Application submitted successfully!")
                
                with st.spinner("🔍 Analyzing application..."):
                    application_data = form.get_application_data()
                    analyzer = ApplicationAnalyzer(st.session_state.model)
                    features = analyzer.extract_features(application_data)
                    probability = analyzer.predict_risk(features)
                    decision = analyzer.get_decision_explanation(probability, features)
                    
                    # Update application status
                    application_data['status'] = decision['status']
                    application_data['risk_score'] = probability
                    
                    # Store application
                    st.session_state.applications.append(application_data)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center;'>🎯 Analysis Results</h3>", unsafe_allow_html=True)
                    
                    # Result card
                    st.markdown(f"""
                        <div style='background: {decision['color']}; padding: 25px; border-radius: 15px; color: white; text-align: center;'>
                            <h2 style='margin: 0; font-size: 1.8em;'>{decision['level']}</h2>
                            <p style='font-size: 1.2em; margin: 10px 0;'>{decision['message']}</p>
                            <p style='font-size: 1.1em; margin: 0;'>Risk Probability: {probability*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk visualization
                    st.markdown("### 📊 Risk Assessment")
                    analyzer.render_risk_visualization(probability)
                    
                    # Detailed explanation
                    st.markdown("### 📋 Decision Rationale")
                    for i, reason in enumerate(decision['reasons'], 1):
                        st.markdown(f"{i}. {reason}")
                    
                    # Next steps
                    st.markdown("### 💡 Recommended Next Steps")
                    for i, step in enumerate(decision['next_steps'], 1):
                        st.markdown(f"{i}. {step}")
                    
                    # Professional summary export
                    st.markdown("### 📄 Application Summary")
                    summary_df = pd.DataFrame([
                        ["Application ID", application_data['application_id']],
                        ["Submission Date", application_data['submission_date'].strftime('%Y-%m-%d %H:%M')],
                        ["Status", decision['status'].title()],
                        ["Risk Score", f"{probability*100:.1f}%"],
                        ["Assigned To", st.session_state.current_user]
                    ], columns=["Field", "Value"])
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Export buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"application_{application_data['application_id']}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        if st.button("🆕 Submit New Application", type="primary"):
                            for key in ['form', 'personal_data', 'financial_info', 'loan_details']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
    
    # TAB 3: APPLICATION LOOKUP
    with tab3:
        render_application_lookup()
    
    # TAB 4: ANALYTICS
    with tab4:
        render_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p><strong>Loan Application Fraud Detection System</strong> • AI-Powered Risk Assessment</p>
            <p style='font-size: 0.8em;'>Secure • Fast • Accurate • Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
