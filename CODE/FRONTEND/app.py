from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import mysql.connector
import warnings
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
import traceback
import gc
import shap
import io
import base64

# Suppress all warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.getenv('SIADD_SECRET_KEY', 'change-me-for-local-development')

# =========================== DATABASE CONNECTION ===========================
try:
    mydb = mysql.connector.connect(
        host=os.getenv('SIADD_DB_HOST', 'localhost'),
        port=int(os.getenv('SIADD_DB_PORT', '3306')),
        user=os.getenv('SIADD_DB_USER', 'root'),
        passwd=os.getenv('SIADD_DB_PASSWORD', 'root'),
        database=os.getenv('SIADD_DB_NAME', 'Mydbs')
    )
    mycur = mydb.cursor()
    print("✓ Database connected successfully!")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    mydb = None
    mycur = None

# =========================== LOAD MODEL WITH ERROR HANDLING ===========================
print("\nLoading model and preprocessors...")

try:
    # Load preprocessors
    scaler = joblib.load('scaler.pkl')
    top_n_features = joblib.load('top_features.pkl')
    print(f"✓ Loaded {len(top_n_features)} features: {top_n_features}")
    
    # Set feature names to scaler to avoid warnings
    if hasattr(scaler, 'feature_names_in_'):
        pass  # Already has feature names
    else:
        try:
            scaler.feature_names_in_ = np.array(top_n_features)
        except:
            pass
            
except Exception as e:
    print(f"✗ Error loading preprocessors: {e}")
    scaler = None
    top_n_features = []

try:
    # Load CatBoost model
    catboost_model = joblib.load('catboost_model.pkl')
    print("✓ CatBoost model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    catboost_model = None

# Model info
MODEL_NAME = "CatBoost"
MODEL_ACCURACY = 0.9958

# Create folder for plots
os.makedirs('static/img', exist_ok=True)

# Attack classes
attack_categories = ['Dos', 'Probing', 'R2L', 'U2R', 'normal']

# =========================== LOAD DATASET FOR VIEW ===========================
print("\nLoading dataset for view...")
try:
    # Column names for the dataset
    column_names = [ 
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"
    ]

    # Load the dataset
    df = pd.read_csv("data/KDDTrain+.txt", names=column_names)
    df.drop('difficulty', axis=1, inplace=True)
    print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    df = pd.DataFrame()

print("\n" + "="*60)
print("INITIALIZATION COMPLETE")
print("="*60)

shap_explainer = None

# =========================== FIXED PREDICTION FUNCTION ===========================
def make_prediction_safe(input_values):
    """Safe prediction function with minimal dependencies"""
    try:
        if catboost_model is None or scaler is None:
            return None, None, "Model not loaded properly"
        
        # Create DataFrame with proper column names
        input_df = pd.DataFrame([input_values], columns=top_n_features)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = catboost_model.predict(input_scaled)[0]
        proba = catboost_model.predict_proba(input_scaled)[0]
        
        # Fix for NumPy deprecation warning - use .item() for scalar extraction
        if isinstance(prediction, np.ndarray):
            prediction = int(prediction.item())
        else:
            prediction = int(prediction)
            
        return prediction, proba, None
    except Exception as e:
        return None, None, str(e)

# =========================== SIMPLIFIED FEATURE PLOT FUNCTION ===========================
def generate_feature_importance_plot(pred_class):
    """Generate feature importance plot without SHAP"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Get feature importances from CatBoost model
        if hasattr(catboost_model, 'get_feature_importance'):
            importances = catboost_model.get_feature_importance()
        else:
            # Use random values for demonstration
            np.random.seed(42)
            importances = np.abs(np.random.randn(len(top_n_features)))
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[-15:]  # Top 15 features
        sorted_importances = importances[sorted_idx]
        sorted_features = [top_n_features[i] for i in sorted_idx]
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_importances)))
        bars = plt.barh(range(len(sorted_importances)), sorted_importances, color=colors)
        
        plt.yticks(range(len(sorted_importances)), sorted_features)
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title(f'Top Feature Importance for {attack_categories[pred_class]}', 
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
            plt.text(val + max(sorted_importances)*0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'data:image/png;base64,{img_base64}'
        
    except Exception as e:
        print(f"Error in feature plot: {e}")
        return None

# =========================== FLASK ROUTES ===========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            confirmpassword = request.form['confirmpassword']
            address = request.form['address']
            
            if password == confirmpassword:
                if mycur:
                    sql = 'SELECT * FROM users WHERE email = %s'
                    val = (email,)
                    mycur.execute(sql, val)
                    data = mycur.fetchone()
                    
                    if data is not None:
                        msg = 'User already registered!'
                        return render_template('registration.html', msg=msg)
                    else:
                        sql = 'INSERT INTO users (name, email, password, Address) VALUES (%s, %s, %s, %s)'
                        val = (name, email, password, address)
                        mycur.execute(sql, val)
                        mydb.commit()
                        
                        msg = 'User registered successfully!'
                        return render_template('login.html', msg=msg)
                else:
                    msg = 'Database connection failed!'
                    return render_template('registration.html', msg=msg)
            else:
                msg = 'Passwords do not match!'
                return render_template('registration.html', msg=msg)
        except Exception as e:
            msg = f'Registration error: {str(e)}'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']

            if mycur:
                sql = 'SELECT * FROM users WHERE email=%s'
                val = (email,)
                mycur.execute(sql, val)
                data = mycur.fetchone()

                if data and password == data[2]:
                    session['user_id'] = data[0]
                    session['user_email'] = data[1]
                    session['logged_in'] = True
                    return redirect('/home')
                else:
                    msg = 'Invalid email or password!'
                    return render_template('login.html', msg=msg)
            else:
                msg = 'Database connection failed!'
                return render_template('login.html', msg=msg)
        except Exception as e:
            msg = f'Login error: {str(e)}'
            return render_template('login.html', msg=msg)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/view')
def view():
    try:
        if df.empty:
            return render_template('view.html', 
                                 data_html="<div class='alert alert-warning'>Dataset not found. Please check if data/KDDTrain+.txt exists.</div>",
                                 row_count=0)
        
        # Show first 100 rows
        df_display = df.head(100)
        row_count = len(df_display)
        
        # Convert to HTML
        data_html = df_display.to_html(
            classes='table table-striped table-bordered table-hover',
            index=False,
            max_cols=10
        )
        
        # Basic statistics
        stats_html = f"""
        <div class='alert alert-info'>
            <h5>Dataset Statistics</h5>
            <p><strong>Total rows:</strong> {len(df):,}</p>
            <p><strong>Total columns:</strong> {len(df.columns)}</p>
            <p><strong>Displaying:</strong> First {row_count} rows</p>
        </div>
        """
        
        return render_template('view.html', 
                             data_html=data_html, 
                             stats_html=stats_html,
                             row_count=row_count,
                             total_rows=len(df))
    except Exception as e:
        print(f"View error: {e}")
        return render_template('view.html', 
                             data_html=f"<div class='alert alert-danger'>Error: {str(e)}</div>",
                             row_count=0)

@app.route('/model_info')
def model_info():
    return render_template('model_info.html', 
                           model_name=MODEL_NAME, 
                           model_accuracy=MODEL_ACCURACY)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    prediction_result = None
    confidence = ""
    mitigation = ""
    shap_plots = {}
    input_values = {}
    error_message = None
    
    if request.method == "POST":
        try:
            # Check if model is loaded
            if catboost_model is None:
                error_message = "Model not loaded. Please check model files."
                return render_template("predict.html",
                                       error_message=error_message,
                                       top_n_features=top_n_features,
                                       shap_plots={})
            
            # Collect input values
            input_values = {}
            for feature in top_n_features:
                try:
                    value = float(request.form.get(feature, 0))
                    input_values[feature] = value
                except:
                    input_values[feature] = 0.0
            
            # Make prediction
            pred_idx, pred_proba, pred_error = make_prediction_safe(input_values)
            
            if pred_error:
                error_message = f"Prediction error: {pred_error}"
            elif pred_idx is not None:
                # Get prediction result
                if 0 <= pred_idx < len(attack_categories):
                    prediction_result = attack_categories[pred_idx]
                    confidence = f"{pred_proba[pred_idx]:.2%}"
                else:
                    prediction_result = "Unknown"
                    confidence = "N/A"
                
                # Generate feature importance plot (skip SHAP to avoid issues)
                print("Generating feature importance plot...")
                feature_plot = generate_feature_importance_plot(pred_idx)
                if feature_plot:
                    shap_plots['feature_importance'] = feature_plot
                    print("✓ Feature plot generated successfully")
                    print("Base64 length:", len(feature_plot))  # ← add this
                    print("Base64 preview:", feature_plot[:60] + "...")
                else:
                    print("✗ Feature plot generation failed")
                
                # Mitigation advice
                mitigations = {
                    'Doss': """
                    <div class="alert alert-danger">
                        <h6><i class='bx bx-alarm'></i> Denial of Service Attack Detected</h6>
                        <ul>
                            <li>Activate rate limiting on network traffic</li>
                            <li>Deploy DDoS protection services</li>
                            <li>Monitor for unusual traffic spikes</li>
                            <li>Block suspicious IP addresses</li>
                        </ul>
                    </div>
                    """,
                    'Probing': """
                    <div class="alert alert-warning">
                        <h6><i class='bx bx-search-alt'></i> Probing Attack Detected</h6>
                        <ul>
                            <li>Block known scanning IPs from firewall</li>
                            <li>Configure firewall rules against port scans</li>
                            <li>Deploy Intrusion Prevention System</li>
                            <li>Monitor for repeated connection attempts</li>
                        </ul>
                    </div>
                    """,
                    'R2L': """
                    <div class="alert alert-info">
                        <h6><i class='bx bx-lock-alt'></i> Remote to Local Attack Detected</h6>
                        <ul>
                            <li>Enforce Multi-Factor Authentication</li>
                            <li>Patch known vulnerabilities immediately</li>
                            <li>Restrict remote access to essential personnel</li>
                            <li>Monitor login attempts and failed authentications</li>
                        </ul>
                    </div>
                    """,
                    'U2R': """
                    <div class="alert alert-secondary">
                        <h6><i class='bx bx-shield-quarter'></i> User to Root Attack Detected</h6>
                        <ul>
                            <li>Apply least privilege principle for all accounts</li>
                            <li>Monitor root/sudo activity closely</li>
                            <li>Conduct regular security audits</li>
                            <li>Implement privilege escalation monitoring</li>
                        </ul>
                    </div>
                    """,
                    'normal': """
                    <div class="alert alert-success">
                        <h6><i class='bx bx-check-circle'></i> Status: Normal Traffic</h6>
                        <p>No immediate threat detected. Network traffic appears normal.</p>
                        <p><strong>Recommended:</strong> Continue routine monitoring.</p>
                    </div>
                    """
                }
                mitigation = mitigations.get(prediction_result, 
                    "<div class='alert alert-light'>No specific mitigation advice available.</div>")
            
        except Exception as e:
            error_message = f"Server error: {str(e)}"
            print(f"Prediction route error: {e}")
            print(traceback.format_exc())
    
    # Clean up memory
    gc.collect()
    
    return render_template("predict.html",
                           prediction_result=prediction_result,
                           confidence=confidence,
                           mitigation=mitigation,
                           shap_plots=shap_plots,
                           top_n_features=top_n_features,
                           model_name=MODEL_NAME,
                           model_accuracy=MODEL_ACCURACY,
                           input_values=input_values,
                           error_message=error_message)

@app.route('/cleanup')
def cleanup():
    """Clean up old plot files"""
    try:
        plot_files = os.listdir('static/img')
        for file in plot_files:
            if file.endswith('.png'):
                file_path = os.path.join('static/img', file)
                # Delete files older than 1 hour
                if os.path.getmtime(file_path) < (datetime.now().timestamp() - 3600):
                    os.remove(file_path)
        return "Cleanup completed"
    except Exception as e:
        return f"Cleanup error: {e}"

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING FLASK SERVER")
    print("="*60)
    print(f"\n➜ Local:   http://127.0.0.1:5000")
    print(f"➜ Network: http://localhost:5000")
    print("\nPress CTRL+C to quit\n")
    print("-"*60)
    
    # Run the app
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=True)
