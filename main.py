import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
import sys
import traceback

# =============================================================================
# INITIAL CONFIGURATION
# =============================================================================

try:
    st.set_page_config(
        page_title="Loan Application Fraud Detection System",
        layout="wide",
        page_icon="🛡️"
    )
    
    DEBUG_MODE = False

except Exception as e:
    st.error(f"Initial configuration error: {e}")

# =============================================================================
# DATA AND MODEL LOADING FUNCTIONS
# =============================================================================

def create_demo_model():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error creating demo model: {e}")
        return None

def load_model():
    try:
        model_paths = [
            "source/rf_kaggle.joblib",
            "./source/rf_kaggle.joblib", 
            "rf_kaggle.joblib"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                if DEBUG_MODE: 
                    st.sidebar.success(f"✅ Model loaded from: {path}")
                else:
                    st.sidebar.success("✅ Model loaded")
                return model
        
        st.sidebar.warning("⚠️ Model file not found. Using demo mode.")
        return create_demo_model()
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        return create_demo_model()

@st.cache_resource
def load_cached_model():
    return load_model()

# =============================================================================
# LOAN APPLICATION FORM SYSTEM
# =============================================================================

class LoanApplicationForm:
    def __init__(self):
        self.sections = {
            "personal_data": "📋 Personal Information",
            "financial_info": "💰 Financial Information", 
            "loan_details": "🏠 Loan Details",
            "guarantees": "🛡️ Guarantees & References"
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
        
        # Show steps
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
            nationality = st.selectbox("Nationality *", ["US Citizen", "Permanent Resident", "Other"])
            
        # Validation
        required_fields = [full_name, tax_id, phone, email]
        all_filled = all(required_fields)
        
        if st.button("Continue to Financial Information →", type="primary"):
            if all_filled:
                # Store data in session state
                st.session_state.personal_data = {
                    'full_name': full_name,
                    'age': age,
                    'gender': gender,
                    'marital_status': marital_status,
                    'tax_id': tax_id,
                    'phone': phone,
                    'email': email,
                    'nationality': nationality
                }
                self.current_section = "financial_info"
                st.rerun()
            else:
                st.error("Please fill all required fields (*)")

    def section_financial_info(self):
        st.subheader("💰 Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input("Monthly Net Income ($) *", 
                                           min_value=0.0, value=3000.0, step=100.0)
            employment_type = st.selectbox("Employment Type *", 
                                         ["Employed", "Self-Employed", "Unemployed", "Retired"])
            employment_duration = st.number_input("Current Employment Duration (months) *", 
                                                min_value=0, value=24)
            
        with col2:
            monthly_expenses = st.number_input("Monthly Expenses ($) *", 
                                             min_value=0.0, value=1500.0, step=50.0)
            credit_history = st.selectbox("Credit History *", 
                                        ["Excellent", "Good", "Fair", "Poor"])
            savings = st.number_input("Savings & Investments ($)", 
                                    min_value=0.0, value=10000.0, step=1000.0)
        
        # Financial capacity calculator
        financial_capacity = monthly_income - monthly_expenses
        st.info(f"💡 **Monthly Financial Capacity:** ${financial_capacity:,.2f}")
        
        if st.button("← Back", key="back_finance"):
            self.current_section = "personal_data"
            st.rerun()
            
        if st.button("Continue to Loan Details →", type="primary", key="next_finance"):
            # Store financial data
            st.session_state.financial_info = {
                'monthly_income': monthly_income,
                'employment_type': employment_type,
                'employment_duration': employment_duration,
                'monthly_expenses': monthly_expenses,
                'credit_history': credit_history,
                'savings': savings,
                'financial_capacity': financial_capacity
            }
            self.current_section = "loan_details"
            st.rerun()

    def section_loan_details(self):
        st.subheader("🏠 Loan Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount ($) *", 
                                       min_value=1000.0, value=25000.0, step=1000.0)
            loan_term = st.slider("Loan Term (months) *", 
                                min_value=12, max_value=360, value=120)
            loan_purpose = st.selectbox("Loan Purpose *", 
                                      ["Home Purchase", "Debt Consolidation", 
                                       "Education", "Medical", "Business", "Other"])
            
        with col2:
            # Automatic calculator
            if loan_amount > 0 and loan_term > 0:
                interest_rate = st.slider("Annual Interest Rate (%)", 
                                        min_value=1.0, max_value=15.0, value=5.0, step=0.1)
                monthly_payment = (loan_amount * (interest_rate/100/12)) / (1 - (1 + interest_rate/100/12)**(-loan_term))
                st.metric("Estimated Monthly Payment", f"${monthly_payment:,.2f}")
                
                # Affordability check
                income = st.session_state.get('financial_info', {}).get('monthly_income', 3000)
                if income > 0:
                    debt_to_income = (monthly_payment / income) * 100
                    st.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%")
        
        if st.button("← Back", key="back_loan"):
            self.current_section = "financial_info"
            st.rerun()
            
        if st.button("Continue to Guarantees →", type="primary", key="next_loan"):
            # Store loan data
            st.session_state.loan_details = {
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'loan_purpose': loan_purpose,
                'interest_rate': interest_rate,
                'monthly_payment': monthly_payment,
                'debt_to_income': debt_to_income
            }
            self.current_section = "guarantees"
            st.rerun()

    def section_guarantees(self):
        st.subheader("🛡️ Guarantees & References")
        
        col1, col2 = st.columns(2)
        
        with col1:
            guarantee_type = st.selectbox("Guarantee Type", 
                                        ["Mortgage", "Co-signer", "Collateral", "None"])
            guarantee_value = st.number_input("Guarantee Value ($)", 
                                           min_value=0.0, value=0.0, step=1000.0)
            bank_reference = st.text_area("Bank Reference", 
                                        placeholder="Bank name, account details, etc.")
            
        with col2:
            documents = st.file_uploader("Supporting Documents", 
                                       type=['pdf', 'jpg', 'png', 'docx'], 
                                       accept_multiple_files=True,
                                       help="Upload ID, proof of income, bank statements")
            additional_notes = st.text_area("Additional Notes",
                                          placeholder="Any additional relevant information...")
            
            # Consent checkbox
            consent = st.checkbox("✅ I confirm all provided information is accurate and truthful *")
        
        if st.button("← Back", key="back_guarantee"):
            self.current_section = "loan_details"
            st.rerun()
            
        if st.button("🔍 Submit Application for Analysis", type="primary", key="submit_app"):
            if consent:
                # Store guarantees data
                st.session_state.guarantees = {
                    'guarantee_type': guarantee_type,
                    'guarantee_value': guarantee_value,
                    'bank_reference': bank_reference,
                    'documents': len(documents) if documents else 0,
                    'additional_notes': additional_notes
                }
                return True
            else:
                st.error("Please confirm that all information is accurate and truthful")
        return False

    def get_application_data(self):
        """Combine all application data"""
        return {
            'personal_data': st.session_state.get('personal_data', {}),
            'financial_info': st.session_state.get('financial_info', {}),
            'loan_details': st.session_state.get('loan_details', {}),
            'guarantees': st.session_state.get('guarantees', {}),
            'application_id': f"APP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'submission_date': datetime.now()
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
        elif self.current_section == "guarantees":
            return self.section_guarantees()
        return False

# =============================================================================
# APPLICATION ANALYSIS AND DECISION SYSTEM
# =============================================================================

class ApplicationAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def extract_features(self, application_data):
        """Extract features from application form for model prediction"""
        personal = application_data.get('personal_data', {})
        financial = application_data.get('financial_info', {})
        loan = application_data.get('loan_details', {})
        
        # Feature mapping based on application data
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
            # Convert features to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            probability = self.model.predict_proba(feature_array)[0, 1]
            return probability
        except Exception as e:
            # Fallback for demo
            st.warning("Using demo risk analysis")
            # Simulate risk based on application data
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
                ]
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
                ]
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
                ]
            }

    def render_risk_visualization(self, probability, decision):
        """Render visual risk assessment"""
        # Risk meter
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Risk zones
        ax.axvspan(0, 30, alpha=0.3, color='#10B981', label='Low Risk')
        ax.axvspan(30, 70, alpha=0.3, color='#F59E0B', label='Moderate Risk')
        ax.axvspan(70, 100, alpha=0.3, color='#DC2626', label='High Risk')
        
        # Risk indicator
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
# MAIN INTERFACE
# =============================================================================

def main():
    # Main header
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 25px; border-radius: 15px; margin-bottom: 25px;'>
            <h1 style='color: white; text-align: center; margin: 0; font-size: 2.2em;'>🏦 Loan Application Fraud Detection System</h1>
            <p style='color: #e0e0e0; text-align: center; font-size: 1.1em; margin: 10px 0 0 0;'>
            Intelligent Loan Application Assessment - Fraud Detection & Prevention
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    clf = load_cached_model()
    analyzer = ApplicationAnalyzer(clf)
    
    # Tab system
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🆕 New Application", 
        "🔍 Application Lookup", 
        "📈 Analytics"
    ])
    
    # TAB 2: NEW APPLICATION (Main Feature)
    with tab2:
        st.markdown("<h2 style='font-size: 1.8em;'>📝 New Loan Application</h2>", unsafe_allow_html=True)
        
        # Initialize form
        if 'form' not in st.session_state:
            st.session_state.form = LoanApplicationForm()
        
        form = st.session_state.form
        
        # Form container
        with st.container():
            st.markdown("""
                <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1e3c72;'>
                    <h3 style='color: #1e3c72; margin: 0;'>Loan Application Form</h3>
                    <p style='color: #666; margin: 5px 0;'>Complete all sections to submit your application</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Render form
            submission_ready = form.render_form()
            
            # Process submission
            if submission_ready:
                st.success("✅ Application submitted successfully!")
                
                # Analyze application
                with st.spinner("🔍 Analyzing application..."):
                    # Get application data
                    application_data = form.get_application_data()
                    
                    # Extract features and predict
                    features = analyzer.extract_features(application_data)
                    probability = analyzer.predict_risk(features)
                    decision = analyzer.get_decision_explanation(probability, features)
                    
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
                    analyzer.render_risk_visualization(probability, decision)
                    
                    # Detailed explanation
                    st.markdown("### 📋 Decision Rationale")
                    for i, reason in enumerate(decision['reasons'], 1):
                        st.markdown(f"{i}. {reason}")
                    
                    # Next steps
                    st.markdown("### 💡 Recommended Next Steps")
                    for i, step in enumerate(decision['next_steps'], 1):
                        st.markdown(f"{i}. {step}")
                    
                    # Application summary
                    with st.expander("📄 Application Summary"):
                        st.json(application_data)
                    
                    # Reset form for new application
                    if st.button("📝 Submit New Application", type="primary"):
                        for key in ['form', 'personal_data', 'financial_info', 'loan_details', 'guarantees']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()

    # Simplified other tabs for example
    with tab1:
        st.markdown("<h2>📊 Dashboard Overview</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Applications", "1,247")
        with col2:
            st.metric("Approval Rate", "68%")
        with col3:
            st.metric("Avg Processing Time", "2.3 days")
        
    with tab3:
        st.markdown("<h2>🔍 Application Lookup</h2>", unsafe_allow_html=True)
        st.info("Search for existing applications by ID, name, or date range")
        
    with tab4:
        st.markdown("<h2>📈 Analytics & Reports</h2>", unsafe_allow_html=True)
        st.info("View detailed analytics and generate reports")

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
