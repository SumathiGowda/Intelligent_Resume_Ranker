from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson import ObjectId
import os
import time

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devkey") # needed for sessions

# === CONFIG ===
STATIC_UPLOAD_ROOT = os.path.join('static', 'uploads')
UPLOAD_FOLDER_PROFILE = os.path.join(STATIC_UPLOAD_ROOT, 'hr_profiles')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'svg'}

# ensure directories exist
os.makedirs(UPLOAD_FOLDER_PROFILE, exist_ok=True)

app.config['UPLOAD_FOLDER_PROFILE'] = UPLOAD_FOLDER_PROFILE
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# === DATABASE ===
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_db"]
hr_accounts = db["hr_accounts"]

# === HELPERS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_company_email(email):
    personal_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'}
    domain = email.split('@')[-1].lower()
    return domain not in personal_domains

def make_unique_filename(orig_filename):
    name = secure_filename(orig_filename)
    ts = int(time.time() * 1000)
    return f"{ts}_{name}"

def get_hr_from_session():
    hr_id = session.get('hr_id')
    if not hr_id:
        return None
    try:
        hr = hr_accounts.find_one({'_id': ObjectId(hr_id)})
    except Exception:
        hr = None
    return hr

# === ROUTES ===
@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/signin')
def signin_page():
    return render_template('signin.html')

# === HR SIGNUP ===
@app.route('/hr_signup', methods=['POST'])
def hr_signup():
    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirmPassword', '')
        company_name = request.form.get('companyName', '').strip()
        job_title = request.form.get('jobTitle', '').strip()
        company_website = request.form.get('companyWebsite', '').strip()

        # basic validation
        if not all([name, email, password, confirm, company_name, job_title]):
            return render_template('signin.html', error='All required fields must be filled')

        if password != confirm:
            return render_template('signin.html', error='Passwords do not match')

        if not is_company_email(email):
            return render_template('signin.html', error='Please use your company email address')

        if hr_accounts.find_one({'email': email}):
            return render_template('signin.html', error='Email already registered')

        hashed_pw = generate_password_hash(password)
        hr_data = {
            'name': name,
            'email': email,
            'password': hashed_pw,
            'company_name': company_name,
            'job_title': job_title,
            'company_website': company_website,
            'profile_pic': '',
            'phone': '',
            'verified': False
        }
        hr_accounts.insert_one(hr_data)
        return redirect(url_for('login_page'))

    except Exception as e:
        print("Error in /hr_signup:", e)
        return render_template('signin.html', error='Internal server error')

# === HR LOGIN ===
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            return render_template('login.html', error='Please enter both email and password.')

        hr = hr_accounts.find_one({'email': email})
        if not hr:
            return render_template('login.html', error='No account found with this email.')

        if not hr.get('verified', False):
            return render_template('login.html', error='Your account is pending admin verification.')

        if not check_password_hash(hr['password'], password):
            return render_template('login.html', error='Incorrect password.')

        # login successful
        session['hr_id'] = str(hr['_id'])
        session['hr_name'] = hr.get('name', '')
        session['hr_company'] = hr.get('company_name', '')
        session['hr_pic'] = hr.get('profile_pic', '')

        return redirect(url_for('dashboard'))

    return render_template('login.html')

# === HR DASHBOARD ===
@app.route('/dashboard')
def dashboard():
    hr = get_hr_from_session()
    if not hr:
        return redirect(url_for('login_page'))

    profile_image_url = (
        url_for('static', filename=hr['profile_pic'])
        if hr.get('profile_pic')
        else url_for('static', filename='images/default-avatar.svg')
    )

    metrics = {
        "open_positions": {"value": 0, "trend": 0},
        "applications_week": {"value": 0, "trend": 0},
        "manage_positions": {"value": 0, "trend": 0}
    }

    analytics = {
        "hiring_trends": "No data available",
        "department_distribution": "No data available"
    }

    return render_template('dashboard.html', hr=hr, metrics=metrics, analytics=analytics, profile_image_url=profile_image_url)

# === HR SETTINGS ===
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    hr = get_hr_from_session()
    if not hr:
        return redirect(url_for('login_page'))

    # Profile picture or default avatar
    profile_image_url = (
        url_for('static', filename=hr['profile_pic'])
        if hr.get('profile_pic')
        else url_for('static', filename='images/default-avatar.svg')
    )

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        job_title = request.form.get('job_title', '').strip()
        company_name = request.form.get('company_name', '').strip()
        company_website = request.form.get('company_website', '').strip()

        update_data = {
            'name': name,
            'phone': phone,
            'job_title': job_title,
            'company_name': company_name,
            'company_website': company_website
        }

        # ---- Handle remove image flag ----
        if request.form.get('remove_image') == '1':
            old = hr.get('profile_pic') or ''
            if old and old.startswith('uploads/'):
                old_path = os.path.join('static', old)
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except Exception:
                    pass
            update_data['profile_pic'] = ''

        # ---- Handle new profile image upload ----
        file = request.files.get('profile_image') or request.files.get('profile_pic')
        if file and file.filename != '':
            if allowed_file(file.filename):
                old = hr.get('profile_pic') or ''
                if old and old.startswith('uploads/'):
                    old_path = os.path.join('static', old)
                    try:
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    except Exception:
                        pass

                unique_name = make_unique_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER_PROFILE'], unique_name)
                file.save(save_path)
                update_data['profile_pic'] = f"uploads/hr_profiles/{unique_name}"
            else:
                flash("Invalid profile image type. Allowed: jpg, jpeg, png, svg.", "error")
                return redirect(url_for('settings'))

        # ---- Update DB ----
        hr_accounts.update_one({'_id': ObjectId(hr['_id'])}, {'$set': update_data})
        hr = hr_accounts.find_one({'_id': ObjectId(hr['_id'])})

        # ---- Refresh session ----
        session['hr_name'] = hr.get('name', '')
        session['hr_company'] = hr.get('company_name', '')
        session['hr_pic'] = hr.get('profile_pic', '')

        flash("Profile updated successfully!", "success")
        return redirect(url_for('settings'))

    # GET request
    return render_template('settings.html', hr=hr, profile_image_url=profile_image_url)

# === ADMIN ROUTES ===
@app.route('/admin/pending_hr')
def pending_hr():
    pending = list(hr_accounts.find({'verified': False}))
    return render_template('admin_pending.html', pending=pending)

@app.route('/admin/approve_hr/<hr_id>', methods=['POST'])
def approve_hr(hr_id):
    hr_accounts.update_one({'_id': ObjectId(hr_id)}, {'$set': {'verified': True}})
    return redirect(url_for('pending_hr'))

# === Serve Uploaded Files ===
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('static', filename)

# === LOGOUT ===
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)
