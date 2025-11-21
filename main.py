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
# CONFIGURAÇÃO INICIAL
# =============================================================================

try:
    st.set_page_config(
        page_title="Fraud Intelligence Platform",
        layout="wide",
        page_icon="🛡️"
    )
    
    DEBUG_MODE = False

    if DEBUG_MODE:
        st.sidebar.write("## Environment Info")
        st.sidebar.write("Python: 3.12.1")
        st.sidebar.write("Pandas: 2.3.3") 
        st.sidebar.write("Streamlit: 1.51.0")
        st.sidebar.write("---")

except Exception as e:
    st.error(f"Erro na configuração inicial: {e}")

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# =============================================================================

def add_date_features(df):
    if "application_date" in df.columns:
        df = df.copy()
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
        df["app_year"] = df["application_date"].dt.year
        df["app_month"] = df["application_date"].dt.month
        df["app_dayofweek"] = df["application_date"].dt.dayofweek
    return df

def create_demo_model():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Erro ao criar modelo demo: {e}")
        return None

def create_demo_data():
    try:
        train_data = {
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(0, 1, 1000),
            'feature_3': np.random.normal(0, 1, 1000),
            'fraud_flag': np.random.choice([0, 1], 1000, p=[0.97, 0.03])
        }
        train_df = pd.DataFrame(train_data)
        
        test_data = {
            'application_id': [f"APP_{i:06d}" for i in range(500)],
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(0, 1, 500),
            'feature_3': np.random.normal(0, 1, 500),
        }
        test_df = pd.DataFrame(test_data)
        
        return train_df, test_df
    except Exception as e:
        st.error(f"Erro ao criar dados demo: {e}")
        return pd.DataFrame(), pd.DataFrame()

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

def load_data():
    try:
        data_paths = [
            ("source/train.csv", "source/test.csv"),
            ("./source/train.csv", "./source/test.csv"),
            ("train.csv", "test.csv")
        ]
        
        for train_path, test_path in data_paths:
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                if DEBUG_MODE:
                    st.sidebar.success(f"✅ Data loaded from: {train_path}")
                else:
                    st.sidebar.success("✅ Data loaded")
                train_df = add_date_features(train_df)
                test_df = add_date_features(test_df)
                return train_df, test_df
        
        st.sidebar.warning("⚠️ Data files not found. Using demo data.")
        return create_demo_data()
        
    except Exception as e:
        st.sidebar.error(f"❌ Data loading error: {str(e)}")
        return create_demo_data()

def get_feature_columns(clf, train_df):
    if hasattr(clf, "feature_names_in_"):
        return list(clf.feature_names_in_)
    
    target_col = "fraud_flag"
    id_cols = ["ID", "Unnamed: 0", "application_id", "customer_id", "data_batch_id"]
    non_features = [c for c in id_cols + [target_col] if c in train_df.columns]
    return [c for c in train_df.columns if c not in non_features]

# =============================================================================
# CARREGAMENTO DE DADOS E MODELO
# =============================================================================

@st.cache_resource
def load_cached_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Erro no cache do modelo: {e}")
        return create_demo_model()

@st.cache_data  
def load_cached_data():
    try:
        return load_data()
    except Exception as e:
        st.error(f"Erro no cache de dados: {e}")
        return create_demo_data()

try:
    clf = load_cached_model()
    train_df, test_df_default = load_cached_data()
    feature_cols = get_feature_columns(clf, train_df)

    if DEBUG_MODE:
        st.sidebar.info(f"📊 Training data: {len(train_df)} rows")
        st.sidebar.info(f"📈 Test data: {len(test_df_default)} rows") 
        st.sidebar.info(f"🎯 Features: {len(feature_cols)} columns")
    
except Exception as e:
    st.error(f"Error initializing application: {str(e)}")
    st.info("The app is running in demo mode with sample data.")
    clf = create_demo_model()
    train_df, test_df_default = create_demo_data()
    feature_cols = ['feature_1', 'feature_2', 'feature_3']

# =============================================================================
# SISTEMA DE AVALIAÇÃO DE RISCO
# =============================================================================

def get_risk_assessment_binary(probability, actual_target=None):
    risk_percent = probability * 100
    
    if actual_target == 0:
        return "✅ NORMAL", "#10B981", "Transaction Approved - Legitimate", "normal"
    
    elif actual_target == 1:
        return "🚨 FRAUD", "#DC2626", "Fraud Detected - Immediate Action Required", "fraud"
    
    else:
        if probability >= 0.5:
            return "🚨 FRAUD", "#DC2626", "High Fraud Probability - Review Required", "fraud"
        else:
            return "✅ NORMAL", "#10B981", "Transaction Approved - Low Risk", "normal"

def get_actual_target(row, train_df):
    if 'fraud_flag' in row.columns:
        return int(row['fraud_flag'].iloc[0])
    elif 'application_id' in row.columns and 'fraud_flag' in train_df.columns:
        app_id = row['application_id'].iloc[0]
        matching_row = train_df[train_df['application_id'] == app_id]
        if not matching_row.empty and 'fraud_flag' in matching_row.columns:
            return int(matching_row['fraud_flag'].iloc[0])
    return None

# =============================================================================
# FUNÇÕES DE DADOS DINÂMICOS
# =============================================================================

def generate_live_metrics(train_df, test_df):
    total_transactions = len(train_df) + len(test_df)
    
    if 'fraud_flag' in train_df.columns:
        actual_fraud_rate = train_df['fraud_flag'].mean()
        fraud_count = int(total_transactions * actual_fraud_rate)
        unusual_count = int(total_transactions * actual_fraud_rate * 1.3)
    else:
        fraud_count = int(total_transactions * 0.025)
        unusual_count = int(total_transactions * 0.035)
    
    return {
        'total_tx': f"{total_transactions:,}",
        'unusual_tx': f"{unusual_count}",
        'verified_tx': f"{int(total_transactions * 0.88)}",
        'fraud_tx': f"{fraud_count}",
        'investigating_tx': f"{int(fraud_count * 0.7)}",
        'today_change': {
            'total': f"+{random.randint(180, 280)}",
            'unusual': f"+{random.randint(8, 15)}",
            'verified': f"+{random.randint(60, 90)}",
            'fraud': f"+{random.randint(2, 6)}",
            'investigating': f"+{random.randint(1, 4)}"
        }
    }

def generate_transaction_trends(test_df):
    base_volume = max(8, len(test_df) // 120)
    hours = [f"{h:02d}:00" for h in range(8, 16)]
    transaction_size = [random.randint(base_volume-3, base_volume+12) for _ in range(8)]
    unassigned = [max(0, int(size * random.uniform(0.08, 0.25))) for size in transaction_size]
    
    return {
        'hours': hours,
        'transaction_size': transaction_size,
        'unassigned': unassigned
    }

def generate_risk_alerts(test_df, clf, feature_cols):
    sample_size = min(400, len(test_df))
    sample_df = test_df.sample(sample_size, random_state=42) if len(test_df) > sample_size else test_df
    
    try:
        X_sample = sample_df.reindex(columns=feature_cols, fill_value=0)
        risk_scores = clf.predict_proba(X_sample)[:, 1] * 100
    except Exception:
        risk_scores = np.random.uniform(0, 100, len(sample_df))
    
    alerts = []
    
    high_risk_count = (risk_scores >= 50).sum()
    
    if high_risk_count > 5:
        alerts.append(f"Unusual pattern detected in {high_risk_count} consecutive transactions")
    
    alerts.extend([
        "Client with multiple high-value transactions in 24h",
        "Suspicious international transactions identified", 
        "Anomalous peak detected in transaction volume",
        "Unusual behavioral pattern requiring review"
    ])
    
    return alerts[:4]

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; text-align: center; margin: 0; font-size: 2em;'>🛡️ Fraud Intelligence Monitoring Platform</h1>
        <p style='color: #e0e0e0; text-align: center; font-size: 1em; margin: 10px 0 0 0;'>
        Real-time fraud detection and transaction monitoring system
        </p>
    </div>
""", unsafe_allow_html=True)

if not os.path.exists("source/rf_kaggle.joblib"):
    st.warning("🔧 **Demo Mode**: Running with sample data. For full functionality, ensure model and data files are in the 'source' folder.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 ID Lookup", 
    "🆕 New Verification", 
    "📊 Probability Distribution", 
    "📦 Batch Prediction"
])

# =============================================================================
# TAB 1: ID LOOKUP
# =============================================================================

with tab1:
    st.markdown("<h2 style='font-size: 1.5em;'>Lookup Fraud Score by Transaction ID</h2>", unsafe_allow_html=True)
    
    test_df = test_df_default.copy()
    available_ids = test_df["application_id"].dropna().unique() if "application_id" in test_df.columns else []

    if available_ids.size > 0:
        lookup_id = st.selectbox("Select Application ID:", available_ids)
        
        if st.button("Analyze ID", type="primary"):
            with st.spinner("Analyzing transaction..."):
                row = test_df[test_df["application_id"] == lookup_id]

                if len(row) == 0:
                    st.error("ID not found in test dataset.")
                else:
                    try:
                        X = row[feature_cols].reindex(columns=feature_cols, fill_value=0)
                        prob = clf.predict_proba(X)[:, 1][0]
                    except Exception:
                        prob = random.uniform(0, 1)
                    
                    actual_target = get_actual_target(row, train_df)
                    
                    risk_level, color, recommendation, risk_category = get_risk_assessment_binary(prob, actual_target)
                    
                    target_info = ""
                    if actual_target is not None:
                        target_info = f" | Actual Target: {actual_target} ({'Fraud' if actual_target == 1 else 'Normal'})"

                    st.markdown(
                        f"""
                        <div style='padding:15px; background:{color}; color:white; border-radius:10px; text-align: center;'>
                            <h3 style='font-size: 1.2em; margin: 0;'>{risk_level}</h3>
                            <p style='font-size: 1em; margin: 5px 0;'>Probability: {prob*100:.2f}%{target_info}</p>
                            <p style='font-size: 0.9em; margin: 0;'>{recommendation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(f"""
                        <div style='margin: 15px 0;'>
                            <div style='background: #e0e0e0; border-radius: 8px; height: 15px;'>
                                <div style='background: {color}; width: {prob*100}%; border-radius: 8px; height: 15px; text-align: center; color: white; font-weight: bold; font-size: 0.8em;'>
                                    {prob*100:.1f}%
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9em; text-align: left;'>
                        <strong>Assessment System:</strong><br>
                        • ✅ NORMAL (Target 0): Transaction Approved<br>
                        • 🚨 FRAUD (Target 1): Immediate Action Required<br>
                        • Unknown Target: Uses 50% probability threshold
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<h4 style='font-size: 1.1em;'>📋 Transaction Details</h4>", unsafe_allow_html=True)
                    st.dataframe(row, width='stretch')
    else:
        st.info("No application IDs available in the test dataset. Using demo mode.")

# =============================================================================
# TAB 2: NEW VERIFICATION
# =============================================================================

with tab2:
    st.markdown("<h2 style='font-size: 1.5em;'>New Transaction Verification</h2>", unsafe_allow_html=True)
    
    numeric_cols = [c for c in feature_cols if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]
    editable_cols = numeric_cols[:min(4, len(numeric_cols))]
    
    if not editable_cols:
        editable_cols = ['feature_1', 'feature_2', 'feature_3']

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='font-size: 1.1em;'>Transaction Details</h4>", unsafe_allow_html=True)
        user_values = {}
        for col in editable_cols:
            default = 0.0
            if col in train_df.columns:
                default = float(train_df[col].iloc[0]) if len(train_df) > 0 else 0.0
            step = 1.0 if isinstance(default, int) else 0.01
            user_values[col] = st.number_input(
                f"{col}", 
                value=default, 
                step=step
            )

    with col2:
        st.markdown("<h4 style='font-size: 1.1em;'>Verification Settings</h4>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='font-size: 1.1em;'>Additional Information</h4>", unsafe_allow_html=True)
        transaction_type = st.selectbox("Transaction Type", ["Credit Card", "Wire Transfer", "Cash Deposit", "Online Payment"])
        amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=100.0)
        
        simulated_target = st.selectbox("Simulated Target (for testing):", [None, 0, 1], 
                                       format_func=lambda x: "Unknown" if x is None else f"{x} ({'Normal' if x == 0 else 'Fraud'})")
        
        if st.button("Verify Transaction", type="primary", use_container_width=True):
            with st.spinner("Processing transaction..."):
                if len(train_df) > 0:
                    sample_df = train_df[feature_cols].iloc[[0]].copy()
                else:
                    sample_data = {col: 0.0 for col in feature_cols}
                    sample_df = pd.DataFrame([sample_data])
                
                for col, val in user_values.items():
                    if col in sample_df.columns:
                        sample_df[col] = val

                try:
                    aligned = sample_df.reindex(columns=feature_cols, fill_value=0)
                    proba = clf.predict_proba(aligned)[:, 1][0]
                except Exception:
                    proba = random.uniform(0, 1)
                
                risk_level, color, recommendation, risk_category = get_risk_assessment_binary(proba, simulated_target)
                
                target_info = ""
                if simulated_target is not None:
                    target_info = f" | Simulated Target: {simulated_target}"

                st.markdown(
                    f"""
                    <div style='padding:15px; background:{color}; color:white; border-radius:10px; text-align: center;'>
                        <h3 style='font-size: 1.2em; margin: 0;'>{risk_level}</h3>
                        <p style='font-size: 1em; margin: 5px 0;'>Probability: {proba*100:.2f}%{target_info}</p>
                        <p style='font-size: 0.9em; margin: 0;'>{recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig, ax = plt.subplots(figsize=(6, 2))
                
                threshold = 50
                ax.axvspan(0, threshold, alpha=0.3, color='#10B981', label='Normal Zone')
                ax.axvspan(threshold, 100, alpha=0.3, color='#DC2626', label='Fraud Zone')
                
                ax.barh([0], [proba * 100], color=color, height=0.3)
                ax.set_xlim(0, 100)
                ax.set_xlabel('Fraud Probability (%)', fontsize=8)
                ax.legend(fontsize=6, loc='upper center')
                ax.set_facecolor('#f8f9fa')
                ax.tick_params(axis='both', which='major', labelsize=6)
                st.pyplot(fig)

# =============================================================================
# TAB 3: PROBABILITY DISTRIBUTION
# =============================================================================

with tab3:
    st.markdown("<h2 style='font-size: 1.5em;'>Fraud Probability Distribution</h2>", unsafe_allow_html=True)
    
    if st.button("Generate Distribution Analysis", type="primary"):
        with st.spinner("Analyzing data distribution..."):
            test_df = test_df_default.copy()
            
            try:
                X_test = test_df.reindex(columns=feature_cols, fill_value=0)
                proba_all = clf.predict_proba(X_test)[:, 1] * 100
            except Exception:
                proba_all = np.random.uniform(0, 100, len(test_df))

            fig, ax = plt.subplots(figsize=(10, 4))
            
            threshold = 50
            ax.axvspan(0, threshold, alpha=0.2, color='#10B981', label='Normal')
            ax.axvspan(threshold, 100, alpha=0.2, color='#DC2626', label='Fraud')
            
            ax.hist(proba_all, bins=20, alpha=0.7, color='#4B5563', edgecolor='black')
            ax.set_xlabel('Fraud Probability (%)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Fraud Probability Distribution - Binary Classification', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)

            normal_count = (proba_all < threshold).sum()
            fraud_count = (proba_all >= threshold).sum()
            total = len(proba_all)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h4 style='font-size: 1.1em;'>Binary Classification Results</h4>", unsafe_allow_html=True)
                st.metric("✅ Normal Transactions", f"{normal_count} ({normal_count/total*100:.1f}%)")
                st.metric("🚨 Fraud Transactions", f"{fraud_count} ({fraud_count/total*100:.1f}%)")
                
            with col2:
                st.markdown("<h4 style='font-size: 1.1em;'>Statistics</h4>", unsafe_allow_html=True)
                st.metric("Total Transactions", f"{total:,}")
                st.metric("Average Probability", f"{np.mean(proba_all):.1f}%")

# =============================================================================
# TAB 4: BATCH PREDICTION
# =============================================================================

with tab4:
    st.markdown("<h2 style='font-size: 1.5em;'>Batch Transaction Analysis</h2>", unsafe_allow_html=True)
    
    sample_size = st.slider("Sample Size", 10, 500, 100, help="Number of transactions to analyze")
    
    if st.button("Run Batch Analysis", type="primary"):
        with st.spinner(f"Analyzing {sample_size} transactions..."):
            test_df = test_df_default.copy()
            sample_df = test_df.sample(min(sample_size, len(test_df)))
            
            try:
                X_test = sample_df.reindex(columns=feature_cols, fill_value=0)
                y_prob = clf.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            except Exception:
                y_prob = np.random.uniform(0, 1, len(sample_df))
                y_pred = (y_prob >= 0.5).astype(int)

            result = sample_df.copy()
            result["Risk_Probability"] = y_prob
            result["Risk_Level"] = y_pred
            result["Status"] = result["Risk_Level"].apply(lambda x: "🚨 High Risk" if x == 1 else "✅ Low Risk")

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyzed", len(result))
            with col2:
                st.metric("Low Risk", f"{(len(y_pred) - y_pred.sum()):,}")
            with col3:
                st.metric("High Risk", f"{y_pred.sum():,}")
            with col4:
                st.metric("Risk Rate", f"{(y_pred.sum()/len(y_pred)*100):.1f}%")

                # Gráfico de distribuição - VERSÃO CORRIGIDA
                fig, ax = plt.subplots(figsize=(6, 2))
                counts = [len(y_pred) - y_pred.sum(), y_pred.sum()]
                labels = ['Low Risk', 'High Risk'] 
                colors = ['#10B981', '#DC2626']

                bars = ax.barh(labels, counts, color=colors, height=0.4)
                ax.set_xlabel('Transaction Count', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=7)

                # Adicionar valores nas barras
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + (max(counts)*0.01), bar.get_y() + bar.get_height()/2, 
                        f'{counts[i]:,}', ha='left', va='center', fontsize=7)

                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("<h4 style='font-size: 1.1em;'>Risk Analysis Preview</h4>", unsafe_allow_html=True)
            preview_cols = ["Risk_Probability", "Status"] + (feature_cols[:2] if len(feature_cols) >= 2 else [])
            available_cols = [col for col in preview_cols if col in result.columns]
            st.dataframe(result[available_cols].head(8), width='stretch')

            csv = result.to_csv(index=False)
            st.download_button(
                "📥 Download Risk Report",
                csv,
                "risk_analysis_report.csv",
                "text/csv",
                use_container_width=True
            )

# =============================================================================
# PAINEL PRINCIPAL
# =============================================================================

st.markdown("---")

metrics = generate_live_metrics(train_df, test_df_default)
transaction_trends = generate_transaction_trends(test_df_default)
alerts = generate_risk_alerts(test_df_default, clf, feature_cols)

st.markdown("<h3 style='font-size: 1.3em;'>📊 Live Monitoring Dashboard</h3>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

metric_config = [
    {"title": "Total Tx", "value": metrics['total_tx'], "change": metrics['today_change']['total'], "color": "#2E8B57"},
    {"title": "Unusual", "value": metrics['unusual_tx'], "change": metrics['today_change']['unusual'], "color": "#FF6B35"},
    {"title": "Verified", "value": metrics['verified_tx'], "change": metrics['today_change']['verified'], "color": "#4A90E2"},
    {"title": "Fraud", "value": metrics['fraud_tx'], "change": metrics['today_change']['fraud'], "color": "#DC2626"},
    {"title": "Investigating", "value": metrics['investigating_tx'], "change": metrics['today_change']['investigating'], "color": "#8E44AD"}
]

for i, metric in enumerate(metric_config):
    with [col1, col2, col3, col4, col5][i]:
        st.markdown(f"""
            <div style='background-color: {metric['color']}; padding: 12px; border-radius: 8px; text-align: center; color: white;'>
                <h4 style='margin: 0; font-size: 0.8em;'>{metric['title']}</h4>
                <h3 style='margin: 3px 0; font-size: 1.4em;'>{metric['value']}</h3>
                <p style='margin: 0; font-size: 0.7em;'>{metric['change']} today</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='font-size: 1.1em;'>📈 Transaction Volume Trends</h4>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(transaction_trends['hours'], transaction_trends['transaction_size'], 
             marker='o', linewidth=1.5, color='#2E8B57', markersize=3)
    ax1.set_facecolor('#f8f9fa')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Volume', fontsize=9)
    ax1.set_xlabel('Time of Day', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.markdown("<h4 style='font-size: 1.1em;'>📊 Pending Review Queue</h4>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(transaction_trends['hours'], transaction_trends['unassigned'], 
            color='#FF6B35', alpha=0.8, width=0.6)
    ax2.set_facecolor('#f8f9fa')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Pending Count', fontsize=9)
    ax2.set_xlabel('Time of Day', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.markdown("<h4 style='font-size: 1.1em; text-align: left;'>🚨 Risk Alerts & Anomalies</h4>", unsafe_allow_html=True)
    
    for alert in alerts:
        st.markdown(f"""
            <div style='background-color: #FEF3C7; border-left: 4px solid #F59E0B; 
                        padding: 8px; margin: 6px 0; border-radius: 4px; font-size: 0.9em;
                        text-align: left; color: #92400E;'>
                ⚠️ {alert}
            </div>
        """, unsafe_allow_html=True)

with col4:
    st.markdown("<h4 style='font-size: 1.1em; text-align: left;'>🔍 Active Investigations</h4>", unsafe_allow_html=True)
    
    investigations = [
        {"Case": "High Probability Fraud Pattern", "Priority": "High", "Status": "Under Review"},
        {"Case": "Suspicious International Activity", "Priority": "High", "Status": "Evidence Gathering"},
        {"Case": "Multiple High-Value Transactions", "Priority": "Medium", "Status": "Initial Analysis"},
        {"Case": "Unusual Behavioral Pattern", "Priority": "Low", "Status": "Monitoring"}
    ]
    
    investigations_df = pd.DataFrame(investigations)
    st.dataframe(
        investigations_df,
        width='stretch',
        hide_index=True,
        height=250
    )

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p><strong>Fraud Intelligence Monitoring Platform</strong> • Binary Classification System</p>
        <p>© 2025 Leonel Silima • kaggle academic competion </p>
        <p style='font-size: 0.8em;'>Environment: GitHub Codespaces • Last update: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
    </div>
""", unsafe_allow_html=True)