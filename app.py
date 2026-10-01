import streamlit as st
import os, uuid
from PIL import Image
from database import (init_db, create_user, verify_user, save_report,
                      get_user_reports, get_stats)
from model_utils import (load_models, preprocess, predict,
                         get_gradcam, overlay_heatmap)
from report_generator import generate_report

st.set_page_config(page_title="Lungs AI", page_icon="🫁", layout="wide")

# Create folders
for folder in ["data", "uploads", "heatmaps", "reports"]:
    os.makedirs(folder, exist_ok=True)

init_db()

# -------- Session State --------
if "user" not in st.session_state:
    st.session_state.user = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "models" not in st.session_state:
    st.session_state.models = load_models()


# ==================================================
# LANDING PAGE
# ==================================================
def landing():
    st.markdown("""
        <div style='text-align:center; padding:40px'>
            <h1>🫁 AI Lungs Health Analyzer</h1>
            <h4>Detect Lungs Cancer & Pneumonia with AI + Grad-CAM Explainability</h4>
            <p>Upload a chest X-ray → Get instant analysis, heatmap, recommendations,
            and a downloadable PDF report.</p>
            <hr>
            <p style='color:#888; font-size:13px'>⚠️ For screening assistance only.
            Not a substitute for a doctor's diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔐 Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
        if st.button("📝 Sign Up", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()


# ==================================================
# LOGIN
# ==================================================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        uid = verify_user(u, p)
        if uid:
            st.session_state.user = u
            st.session_state.user_id = uid
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid username or password")
    if st.button("← Back"):
        st.session_state.page = "landing"
        st.rerun()


# ==================================================
# SIGNUP
# ==================================================
def signup():
    st.title("📝 Sign Up")
    u = st.text_input("Username")
    e = st.text_input("Email")
    p = st.text_input("Password", type="password")
    if st.button("Create Account", use_container_width=True):
        if not u or not e or not p:
            st.error("Please fill all fields")
        elif create_user(u, e, p):
            st.success("Account created! Please login.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username or email already exists")
    if st.button("← Back"):
        st.session_state.page = "landing"
        st.rerun()


# ==================================================
# SIDEBAR
# ==================================================
def sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user}")
        st.markdown("---")
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        if st.button("🔬 New Analysis", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
        if st.button("📜 History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["user", "user_id", "models"]:
                st.session_state.pop(k, None)
            st.session_state.page = "landing"
            st.rerun()
        st.markdown("---")
        st.caption("⚠️ AI result may have errors. Always consult a doctor.")


# ==================================================
# DASHBOARD
# ==================================================
def dashboard():
    st.title("📊 Dashboard")
    stats = get_stats(st.session_state.user_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tests", stats["total"])
    c2.metric("Cancer Detected", stats["cancer"])
    c3.metric("Pneumonia Detected", stats["pneumonia"])

    st.markdown("### 📜 Recent Reports")
    reports = get_user_reports(st.session_state.user_id)[:5]
    if not reports:
        st.info("No reports yet. Start your first analysis!")
    else:
        for r in reports:
            st.write(f"**{r[8]}** | Cancer: {r[3]} ({r[4]*100:.1f}%) | "
                     f"Pneumonia: {r[5]} ({r[6]*100:.1f}%)")


# ==================================================
# ANALYSIS HELPER
# ==================================================
def analyze_image(uploaded_file):
    img = Image.open(uploaded_file)
    img_path = f"uploads/{uuid.uuid4().hex}.png"
    img.save(img_path)

    cancer_model, pneumonia_model = st.session_state.models
    arr, _ = preprocess(img)

    c_res, c_conf = predict(cancer_model, arr)
    p_res, p_conf = predict(pneumonia_model, arr)

    hm = get_gradcam(cancer_model, arr)
    overlay = overlay_heatmap(img, hm)
    hm_path = f"heatmaps/{uuid.uuid4().hex}.png"
    Image.fromarray(overlay).save(hm_path)

    return img_path, hm_path, c_res, c_conf, p_res, p_conf


# ==================================================
# MAIN ANALYSIS PAGE
# ==================================================
def main_page():
    st.title("🔬 New Analysis")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📤 Upload Chest X-Ray")
        uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
        analyze = st.button("🧠 Analyze with AI", use_container_width=True)
        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with right:
        st.subheader("📋 Results")
        if uploaded and analyze:
            with st.spinner("Analyzing..."):
                img_path, hm_path, c_res, c_conf, p_res, p_conf = analyze_image(uploaded)

            save_report(st.session_state.user_id, img_path, c_res, c_conf,
                        p_res, p_conf, hm_path)

            st.success("✅ Analysis Complete")
            st.markdown("### 🔥 Grad-CAM Heatmap")
            st.image(hm_path, caption="Red box = AI-detected affected region",
                     use_container_width=True)

            st.markdown("### 🧾 Detection Results")
            st.write(f"**Lungs Cancer:** {c_res} ({c_conf*100:.1f}% confidence)")
            st.write(f"**Pneumonia:** {p_res} ({p_conf*100:.1f}% confidence)")

            st.warning("⚠️ This result may contain errors. Please consult a doctor.")

            # Recommendations
            lifestyle = [
                "Quit smoking and avoid secondhand smoke",
                "Exercise 30 minutes daily (walking, yoga)",
                "Eat fruits, vegetables, and whole grains",
                "Drink 8+ glasses of water daily",
                "Avoid polluted areas; wear a mask outdoors",
                "Get 7-8 hours of sleep every night",
            ]
            critical = []
            if c_res == "Detected":
                critical.append("🚨 Immediate oncologist consultation required")
                critical.append("Get CT scan / biopsy scheduled urgently")
            if p_res == "Detected":
                critical.append("🚨 Consult pulmonologist immediately")
                critical.append("Start antibiotics only after doctor's advice")
            if not critical:
                critical.append("✅ No critical signs — maintain routine checkups")

            with st.expander("🥗 Lifestyle Recommendations", expanded=True):
                for l in lifestyle:
                    st.write(f"- {l}")
            with st.expander("🚨 Critical Recommendations", expanded=True):
                for l in critical:
                    st.write(f"- {l}")

            st.markdown("### 👨‍⚕️ Doctor Consultation")
            st.info("Visit a pulmonologist or oncologist. Carry this report and "
                    "the heatmap image for reference. If symptoms are severe, "
                    "go to the nearest emergency room immediately.")

            # PDF Report
            report_data = {
                "cancer_result": c_res,
                "cancer_conf": c_conf,
                "pneumonia_result": p_res,
                "pneumonia_conf": p_conf,
                "heatmap_path": hm_path,
                "lifestyle": lifestyle,
                "critical": critical,
            }
            pdf_path = f"reports/{uuid.uuid4().hex}.pdf"
            generate_report(st.session_state.user, img_path, report_data, pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Report (PDF)",
                    f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )


# ==================================================
# HISTORY
# ==================================================
def history():
    st.title("📜 Test History")
    reports = get_user_reports(st.session_state.user_id)
    if not reports:
        st.info("No history yet.")
        return
    for r in reports:
        with st.expander(f"🧾 {r[8]} — Cancer: {r[3]} | Pneumonia: {r[5]}"):
            st.write(f"Cancer confidence: {r[4]*100:.1f}%")
            st.write(f"Pneumonia confidence: {r[6]*100:.1f}%")
            if os.path.exists(r[2]):
                st.image(r[2], width=200, caption="Original")
            if os.path.exists(r[7]):
                st.image(r[7], width=200, caption="Heatmap")


# ==================================================
# ROUTER
# ==================================================
page = st.session_state.page

if page == "landing":
    landing()
elif page == "login":
    login()
elif page == "signup":
    signup()
elif page == "dashboard":
    sidebar()
    dashboard()
elif page == "main":
    sidebar()
    main_page()
elif page == "history":
    sidebar()
    history()
