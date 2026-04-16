
import streamlit as st
from Bio import Entrez
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import google.generativeai as genai
from docx import Document
from io import BytesIO
import urllib.parse
import hashlib

# --- CONFIGURATION & CSS ---
SCIHUB_BASE_URL = "https://www.sci-hub.se/" 
NOTEBOOK_LM_URL = "https://notebooklm.google.com/"

st.set_page_config(page_title="Research Assistant", layout="wide")

# Soothing Pastel Green Academic Styling (Inter & Lora fonts, Sage & Forest Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Lora:ital,wght@0,400;0,500;0,600;1,400&display=swap');

    html, body,[class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #2D3A33; /* Dark muted green-grey for general text */
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Lora', serif !important;
        color: #1A3628 !important; /* Deep Forest Green for professional contrast */
        font-weight: 500 !important;
        letter-spacing: -0.02em;
    }

    /* Primary Nav & Action Buttons */
    div.stButton > button {
        border-radius: 4px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
        padding: 0.5rem 1rem !important;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #608F79 !important; /* Sage Green */
        color: #FFFFFF !important;
        border: none !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #4A6B58 !important; /* Muted Forest Green */
        border: 1px solid #8EB69B !important; /* Light Sage Border */
    }
    
    div.stButton > button[kind="secondary"]:hover, div.stLinkButton > a:hover {
        border-color: #608F79 !important;
        color: #1A3628 !important;
        background-color: #F4F9F4 !important; /* Very pale mint hover */
    }

    div.stLinkButton > a {
        color: #4A6B58 !important;
        border: 1px solid #8EB69B !important;
    }

    /* Paper Metadata Bar */
    .paper-metadata {
        background-color: #F4F9F4; /* Off-white Mint */
        border-left: 3px solid #8EB69B; /* Light Sage Accent */
        padding: 12px 16px;
        margin: 12px 0px 20px 0px;
        font-size: 0.9rem;
        color: #3E5A4B;
        line-height: 1.6;
        border-radius: 0px 4px 4px 0px;
    }
    
    /* AI Summary Box */
    .ai-summary-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #608F79; /* Sage Green Accent */
        padding: 24px;
        margin-top: 16px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1E293B;
        border-radius: 4px;
        box-shadow: 0 4px 6px -1px rgba(96, 143, 121, 0.08); /* Soft green-tinted shadow */
    }
    
    /* API Instructions */
    .api-instructions {
        background-color: #EAF2EB; /* Pale Mint */
        padding: 16px;
        border-radius: 4px;
        border-left: 3px solid #608F79; /* Sage Green Accent */
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #2C4C3B;
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE CONNECTION (POSTGRESQL - PRODUCTION READY) ---
@st.cache_resource(ttl=300) # TTL clears idle connections every 5 mins to align with Supabase Pooler
def get_db():
    """Establish connection to Cloud PostgreSQL securely via Streamlit Secrets."""
    try:
        db_url = st.secrets["DATABASE_URL"]
        
        # SQLAlchemy and newer Psycopg2 require postgresql:// instead of postgres://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
            
        conn = psycopg2.connect(db_url)
        
        # Autocommit prevents idle-in-transaction timeouts with connection poolers
        conn.set_session(autocommit=True)
        return conn
        
    except Exception as e:
        st.error(f"🔌 **Database connection failed.** Please verify your Supabase Transaction Pooler settings.\n\n*Error details: {e}*")
        st.stop()

def init_db():
    conn = get_db()
    with conn.cursor() as c:
        # Users Table (Stores API keys privately per user)
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (email VARCHAR(255) PRIMARY KEY, name VARCHAR(255), password_hash VARCHAR(255), keywords TEXT, authors TEXT, api_key TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        # Reading List (Isolated by user_email)
        c.execute('''CREATE TABLE IF NOT EXISTS reading_list 
                     (id SERIAL PRIMARY KEY, user_email VARCHAR(255), pmid TEXT, title TEXT, journal TEXT, authors TEXT, date TEXT, notes TEXT, last_edited TEXT)''')
        # General Notes (Isolated by user_email)
        c.execute('''CREATE TABLE IF NOT EXISTS general_notes 
                     (id SERIAL PRIMARY KEY, user_email VARCHAR(255), content TEXT, date TEXT)''')
        # Audit Logs
        c.execute('''CREATE TABLE IF NOT EXISTS login_history 
                     (id SERIAL PRIMARY KEY, user_email VARCHAR(255), login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def load_user_profile(email):
    conn = get_db()
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute("SELECT name, email, keywords, authors, api_key FROM users WHERE email=%s", (email,))
        return c.fetchone()

def log_audit_trail(email):
    conn = get_db()
    with conn.cursor() as c:
        c.execute("INSERT INTO login_history (user_email) VALUES (%s)", (email,))

# --- AI LOGIC ---
def get_safe_model():
    api_key = st.session_state.get("api_key")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        available =[m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        targets =['models/gemini-1.5-flash', 'gemini-1.5-flash', 'models/gemini-pro']
        for t in targets:
            if t in available: return genai.GenerativeModel(t)
        return genai.GenerativeModel(available[0]) if available else None
    except: return None

def generate_ai_summary(abstract):
    model = get_safe_model()
    if not model: return "API Key Configuration Error. Please review Settings."
    prompt = f"""Summarize this abstract into 4 structured points for a brain researcher:
    1. OBJECTIVE: Main research question.
    2. METHODOLOGY: Tools used (EEG, animal models, fMRI, etc).
    3. FINDINGS: Primary results/mechanisms.
    4. SIGNIFICANCE: Implications for neuroscience/psychopathology.
    Abstract: {abstract}"""
    try:
        return model.generate_content(prompt).text
    except Exception as e: return f"AI Processing Error: {str(e)}"

# --- PUBMED TOOLS ---
def get_summaries(kw_list, author_search, days, logic="OR"):
    if not kw_list and not author_search: return[]
    query = ""
    if kw_list: query += "(" + f" {logic} ".join([f'"{k}"[Title]' for k in kw_list]) + ")"
    if author_search:
        if query: query += " AND "
        query += f"{author_search}[Author]"
    try:
        h = Entrez.esearch(db="pubmed", term=query, retmax=12, reldate=days)
        ids = Entrez.read(h)["IdList"]; h.close()
        if not ids: return[]
        h = Entrez.esummary(db="pubmed", id=",".join(ids))
        res = Entrez.read(h); h.close(); return res
    except: return[]

def fetch_abstract(pmid):
    try:
        h = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        recs = Entrez.read(h); h.close()
        article = recs['PubmedArticle'][0]['MedlineCitation']['Article']
        return " ".join(article['Abstract']['AbstractText']) if 'Abstract' in article else "Abstract not available."
    except: return "Error retrieving abstract data."

# --- DB HELPERS (Multi-User Safe) ---
def toggle_reading_list(pmid, title, journal, authors, date):
    email = st.session_state.user_email
    conn = get_db()
    with conn.cursor() as c:
        c.execute("SELECT id FROM reading_list WHERE pmid=%s AND user_email=%s", (pmid, email))
        if c.fetchone():
            c.execute("DELETE FROM reading_list WHERE pmid=%s AND user_email=%s", (pmid, email))
            msg = "Document removed from Reading Room."
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            c.execute("INSERT INTO reading_list (user_email, pmid, title, journal, authors, date, notes, last_edited) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
                      (email, pmid, title, journal, authors, date, "", now))
            msg = "Document saved to Reading Room."
    return msg

def export_notes_to_word():
    doc = Document()
    doc.add_heading(f"Research Notebook - {datetime.now().strftime('%Y-%m-%d')}", 0)
    email = st.session_state.user_email
    
    conn = get_db()
    with conn.cursor() as c:
        doc.add_heading("Literature Notes", level=1)
        c.execute("SELECT title, notes, authors, date FROM reading_list WHERE notes != '' AND user_email=%s", (email,))
        for title, notes, authors, date in c.fetchall():
            year = date[:4] if date else "n.d."
            doc.add_heading(f"{authors} ({year}). {title}.", level=2)
            doc.add_paragraph(notes)
        
        doc.add_heading("General Research Ideas", level=1)
        c.execute("SELECT content, date FROM general_notes WHERE user_email=%s ORDER BY date DESC", (email,))
        for content, date in c.fetchall():
            p = doc.add_paragraph()
            p.add_run(f"[{date}] ").bold = True
            p.add_run(content)
            
    bio = BytesIO(); doc.save(bio); return bio.getvalue()

# --- APP INITIALIZATION ---
init_db()

# --- AUTHENTICATION PORTAL ---
if "user_email" not in st.session_state:
    st.markdown("<h1 style='text-align: center; margin-top: 80px; font-size: 3rem;'>Research Workstation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4A6B58; font-size: 1.2rem; margin-bottom: 40px;'>Neuroscience & Psychopathology Literature Management</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_reg = st.tabs(["Authenticate", "New Investigator Registration"])
        
        with tab_login:
            with st.form("login_form"):
                log_email = st.text_input("Academic Email:")
                log_pass = st.text_input("Access Protocol (Password):", type="password")
                if st.form_submit_button("Initiate Session", type="primary", use_container_width=True):
                    conn = get_db()
                    with conn.cursor() as c:
                        c.execute("SELECT password_hash FROM users WHERE email=%s", (log_email,))
                        res = c.fetchone()
                        
                        if res and res[0] == hash_password(log_pass):
                            st.session_state.user_email = log_email
                            log_audit_trail(log_email)
                            st.rerun()
                        else:
                            st.error("Authentication Failed. Invalid credentials.")
                                
        with tab_reg:
            with st.form("reg_form"):
                reg_name = st.text_input("Investigator Name:")
                reg_email = st.text_input("Academic Email:")
                reg_pass = st.text_input("Define Access Protocol (Password):", type="password")
                st.markdown('<div class="api-instructions"><b>AI Integration:</b> Obtain your private Gemini API Key from <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a>.</div>', unsafe_allow_html=True)
                reg_key = st.text_input("Gemini API Credential:", type="password")
                
                if st.form_submit_button("Register Profile", type="primary", use_container_width=True):
                    if "@" in reg_email and reg_name and reg_pass:
                        try:
                            conn = get_db()
                            with conn.cursor() as c:
                                c.execute("INSERT INTO users (email, name, password_hash, keywords, authors, api_key) VALUES (%s, %s, %s, %s, %s, %s)", 
                                          (reg_email, reg_name, hash_password(reg_pass), "", "", reg_key))
                            st.success("Registration successful. Please proceed to Authenticate.")
                        except psycopg2.IntegrityError:
                            st.error("This email is already registered.")
                    else:
                        st.error("Complete all required fields.")
    st.stop()

# --- LOAD USER STATE ---
if "profile_loaded" not in st.session_state:
    prof = load_user_profile(st.session_state.user_email)
    st.session_state.name = prof['name']
    st.session_state.keywords =[k for k in prof['keywords'].split(",") if k] if prof['keywords'] else[]
    st.session_state.authors = prof['authors']
    st.session_state.api_key = prof['api_key']
    st.session_state.feed_results =[]
    st.session_state.current_page = "Dashboard"
    st.session_state.profile_loaded = True

Entrez.email = st.session_state.user_email
email = st.session_state.user_email

# --- TOP NAVIGATION BAR ---
nav_options =["Dashboard", "Recent Publications", "Literature Discovery", "Reading Room", "Notebook", "Settings"]

# Inject Admin Console dynamically
is_admin = (email == st.secrets.get("ADMIN_EMAIL", ""))
if is_admin:
    nav_options.append("Admin Console")

nav_cols = st.columns(len(nav_options))
for i, option in enumerate(nav_options):
    bt_type = "primary" if st.session_state.current_page == option else "secondary"
    if nav_cols[i].button(option, type=bt_type, use_container_width=True):
        st.session_state.current_page = option
        st.rerun()

st.divider()
page = st.session_state.current_page

# --- PAGE: DASHBOARD ---
if page == "Dashboard":
    st.markdown(f"## Welcome, {st.session_state.name}")
    st.markdown("System Overview & Active Metrics")
    
    conn = get_db()
    with conn.cursor() as c:
        c.execute("SELECT COUNT(*) FROM reading_list WHERE user_email=%s", (email,))
        saved_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM general_notes WHERE user_email=%s", (email,))
        notes_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM reading_list WHERE notes != '' AND user_email=%s", (email,))
        pnotes_count = c.fetchone()[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracked Keywords", len(st.session_state.keywords))
    m2.metric("Saved Documents", saved_count)
    m3.metric("Literature Notes", pnotes_count)
    m4.metric("Independent Ideas", notes_count)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_info, col_tools = st.columns([2, 1])
    with col_info:
        with st.container(border=True):
            st.markdown("### Workstation Protocol")
            st.markdown("""
            **1. Monitor:** Define tracking keywords in *Recent Publications* to receive continuous 48-hour literature updates from PubMed.  
            **2. Acquire:** Run deep historical queries in the *Literature Discovery* targeting specific authors or complex terminology.  
            **3. Synthesize:** Save relevant papers to the *Reading Room* to write notes and utilize Gemini AI for immediate methodological breakdowns.  
            **4. Consolidate:** Review your collective notes and hypotheses in the *Notebook*, and export a formatted APA manuscript directly to Word.
            """)
    with col_tools:
        with st.container(border=True):
            st.markdown("### External Research Tools")
            st.link_button("NotebookLM Access", NOTEBOOK_LM_URL, use_container_width=True)
            st.link_button("Perplexity AI Engine", "https://www.perplexity.ai/", use_container_width=True)
            st.link_button("Google AI Studio (API)", "https://aistudio.google.com/app/apikey", use_container_width=True)

# --- PAGE: RECENT Publications ---
elif page == "Recent Publications":
    st.markdown("## Active Tracking for Recent Publications")
    st.markdown("Automated 48-hour sweep based on active tracking filters.")
    
    with st.container(border=True):
        new_kw = st.text_input("Add Tracking Filter (Press Enter):", placeholder="Enter specific terminology (e.g., Default Mode Network)")
        if new_kw and new_kw not in st.session_state.keywords:
            st.session_state.keywords.append(new_kw)
            
            conn = get_db()
            with conn.cursor() as c:
                c.execute("UPDATE users SET keywords=%s WHERE email=%s", (",".join(st.session_state.keywords), email))
            st.rerun()

        if st.session_state.keywords:
            st.caption("Active Filters (Click to remove):")
            tag_cols = st.columns(6)
            for i, kw in enumerate(st.session_state.keywords):
                with tag_cols[i % 6]:
                    if st.button(f"Remove: {kw}", key=f"del_{kw}"):
                        st.session_state.keywords.remove(kw)
                        
                        conn = get_db()
                        with conn.cursor() as c:
                            c.execute("UPDATE users SET keywords=%s WHERE email=%s", (",".join(st.session_state.keywords), email))
                        st.rerun()

    recent = get_summaries(st.session_state.keywords, st.session_state.authors, 2)
    
    if not recent and st.session_state.keywords:
        st.info("No novel publications identified within the last 48 hours for the current filters.")
        
    for art in recent:
        with st.container(border=True):
            pid, ttl = str(art['Id']), art['Title']
            auths, jrnl, pdate = ", ".join(art['AuthorList']), art['Source'], art['PubDate']
            st.markdown(f"#### {ttl}")
            st.markdown(f"<div class='paper-metadata'><b>Authors:</b> {auths}<br><b>Publication:</b> <i>{jrnl}</i> | <b>Date:</b> {pdate}</div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.link_button("Library Access", f"https://pubmed-ncbi-nlm-nih-gov.ezproxy.haifa.ac.il/{pid}/", use_container_width=True)
            with c2: st.link_button("Sci-Hub Proxy", f"{SCIHUB_BASE_URL}{pid}", use_container_width=True)
            with c3:
                conn = get_db()
                with conn.cursor() as c:
                    c.execute("SELECT id FROM reading_list WHERE pmid=%s AND user_email=%s", (pid, email))
                    is_sv = c.fetchone()
                    
                if st.button("Save Document" if not is_sv else "Remove Document", key=f"al_sv_{pid}", use_container_width=True):
                    st.toast(toggle_reading_list(pid, ttl, jrnl, auths, pdate)); st.rerun()
            with c4:
                if st.button("Generate AI Summary", key=f"btn_sum_{pid}", use_container_width=True): st.session_state[f"show_sum_{pid}"] = True
            
            if st.session_state.get(f"show_sum_{pid}"):
                with st.spinner("Executing AI analysis..."):
                    summary = generate_ai_summary(fetch_abstract(pid))
                    st.markdown(f"<div class='ai-summary-box'>{summary}</div>", unsafe_allow_html=True)
                if st.button("Dismiss Analysis", key=f"cls_{pid}"): del st.session_state[f"show_sum_{pid}"]; st.rerun()

# --- PAGE: Literature Discovery ---
elif page == "Literature Discovery":
    st.markdown("## Literature Discovery")
    
    with st.container(border=True):
        st.markdown("### Filters")
        c1, c2 = st.columns(2)
        with c1: skws = st.text_input("Subject Keywords (comma separated):", value=",".join(st.session_state.keywords))
        with c2: sauth = st.text_input("Primary Author Target:", value=st.session_state.authors)
        
        c3, c4 = st.columns(2)
        with c3: slogic = st.radio("Boolean Operator:",["OR", "AND"], horizontal=True)
        with c4: timeframe = st.selectbox("Historical Range:",["Week", "Month", "Year", "10 Years"], index=1)
        
        if st.button("Execute Database Query", type="primary"):
            t_map = {"Week": 7, "Month": 30, "Year": 365, "10 Years": 3650}
            with st.spinner("Querying PubMed via Entrez API..."):
                st.session_state.feed_results = get_summaries([k.strip() for k in skws.split(",") if k.strip()], sauth, t_map[timeframe], slogic)

    st.markdown("<br>", unsafe_allow_html=True)

    for art in st.session_state.get('feed_results',[]):
        with st.container(border=True):
            pid, ttl = str(art['Id']), art['Title']
            auths, jrnl, pdate = ", ".join(art['AuthorList']), art['Source'], art['PubDate']
            st.markdown(f"#### {ttl}")
            st.markdown(f"<div class='paper-metadata'><b>Authors:</b> {auths}<br><b>Publication:</b> <i>{jrnl}</i> | <b>Date:</b> {pdate}</div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.link_button("Library Access", f"https://pubmed-ncbi-nlm-nih-gov.ezproxy.haifa.ac.il/{pid}/", use_container_width=True)
            with c2: st.link_button("Sci-Hub Proxy", f"{SCIHUB_BASE_URL}{pid}", use_container_width=True)
            with c3:
                conn = get_db()
                with conn.cursor() as c:
                    c.execute("SELECT id FROM reading_list WHERE pmid=%s AND user_email=%s", (pid, email))
                    is_sv = c.fetchone()
                    
                if st.button("Save Document" if not is_sv else "Remove Document", key=f"sv_{pid}", use_container_width=True):
                    st.toast(toggle_reading_list(pid, ttl, jrnl, auths, pdate)); st.rerun()
            with c4:
                if st.button("Generate AI Summary", key=f"ai_{pid}", use_container_width=True): st.session_state[f"show_sum_{pid}"] = True
            
            if st.session_state.get(f"show_sum_{pid}"):
                with st.spinner("Executing AI analysis..."):
                    summary = generate_ai_summary(fetch_abstract(pid))
                    st.markdown(f"<div class='ai-summary-box'>{summary}</div>", unsafe_allow_html=True)
                if st.button("Dismiss Analysis", key=f"cls_s_{pid}"): del st.session_state[f"show_sum_{pid}"]; st.rerun()

# --- PAGE: READING ROOM ---
elif page == "Reading Room":
    st.markdown("## Reading Room")
    
    conn = get_db()
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute("SELECT * FROM reading_list WHERE user_email=%s ORDER BY last_edited DESC", (email,))
        items = c.fetchall()
    
    if not items:
        st.info("The repository is currently empty. Transfer documents from Recent Publications or Literature Discovery to begin processing.")
        
    for item in items:
        pmid = item['pmid']
        with st.container(border=True):
            st.markdown(f"#### {item['title']}")
            st.markdown(f"<div class='paper-metadata'><b>Authors:</b> {item['authors']}<br><b>Publication:</b> <i>{item['journal']}</i> | <b>Date:</b> {item['date']}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                if st.button("Purge Document", key=f"rm_{pmid}", use_container_width=True): 
                    toggle_reading_list(pmid, "", "", "", "")
                    st.rerun()
            with c2: st.link_button("Library Access", f"https://pubmed-ncbi-nlm-nih-gov.ezproxy.haifa.ac.il/{pmid}/", use_container_width=True)
            with c3:
                q = urllib.parse.quote(f"Findings of: {item['title']}")
                st.link_button("Investigate via Perplexity", f"https://www.perplexity.ai/search?q={q}", use_container_width=True)
            
            st.markdown("##### Analytical Notes")
            new_note = st.text_area("Record methodological insights, critiques, or hypotheses:", value=item['notes'], key=f"nt_rr_{pmid}", height=120, label_visibility="collapsed")
            if st.button("Commit Notes to Database", key=f"sv_rr_{pmid}"):
                
                conn = get_db()
                with conn.cursor() as c:
                    c.execute("UPDATE reading_list SET notes=%s, last_edited=%s WHERE pmid=%s AND user_email=%s", 
                              (new_note, datetime.now().strftime("%Y-%m-%d %H:%M"), pmid, email))
                st.toast("Notes recorded.")
                st.rerun()

# --- PAGE: NOTEBOOK ---
elif page == "Notebook":
    st.markdown("## Research Notebook")
    
    with st.container(border=True):
        col_text, col_btn = st.columns([3, 1])
        with col_text:
            st.markdown("Compile your insights into a standardized format. The export engine generates an APA-structured manuscript containing your general hypotheses and specific literature notes.")
        with col_btn:
            st.download_button("Export Manuscript to Word", export_notes_to_word(), f"Research_Notebook_{datetime.now().strftime('%Y%m%d')}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Independent Hypotheses & Notes", "Literature Notes"])
    
    with tab1:
        with st.container(border=True):
            st.markdown("### Formulate New Hypothesis")
            new_gen_c = st.text_area("Content:", key="new_gen_note", height=120, label_visibility="collapsed", placeholder="Enter methodology adjustments, theoretical ideas, or supervisor meeting notes here...")
            if st.button("Append to Database", type="primary"):
                if new_gen_c:
                    conn = get_db()
                    with conn.cursor() as c:
                        c.execute("INSERT INTO general_notes (user_email, content, date) VALUES (%s, %s, %s)", 
                                  (email, new_gen_c, datetime.now().strftime("%Y-%m-%d %H:%M")))
                    st.rerun()
        
        conn = get_db()
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT id, content, date FROM general_notes WHERE user_email=%s ORDER BY id DESC", (email,))
            g_notes = c.fetchall()
                
        for note in g_notes:
            nid = note['id']
            with st.expander(f"Record Entry: {note['date']}"):
                ed_gen = st.text_area("Revise Entry:", value=note['content'], key=f"ed_gen_{nid}", height=100, label_visibility="collapsed")
                if st.button("Update Record", key=f"up_g_{nid}"):
                    conn = get_db()
                    with conn.cursor() as c:
                        c.execute("UPDATE general_notes SET content=%s WHERE id=%s AND user_email=%s", (ed_gen, nid, email))
                    st.toast("Record modified.")
                    st.rerun()
                    
    with tab2:
        conn = get_db()
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT pmid, title, notes, last_edited FROM reading_list WHERE notes != '' AND user_email=%s ORDER BY last_edited DESC", (email,))
            p_notes = c.fetchall()
                
        if not p_notes:
            st.info("No active literature notes detected in the repository.")
            
        for note in p_notes:
            pmid = note['pmid']
            with st.expander(f"Source Analysis: {note['title'][:80]}..."):
                col_t, col_l = st.columns([0.85, 0.15])
                with col_t:
                    new_pnote = st.text_area("Notes Data:", value=note['notes'], key=f"ed_p_{pmid}", height=150, label_visibility="collapsed")
                    if st.button("Update Notes Data", key=f"up_p_{pmid}"):
                        conn = get_db()
                        with conn.cursor() as c:
                            c.execute("UPDATE reading_list SET notes=%s, last_edited=%s WHERE pmid=%s AND user_email=%s", 
                                      (new_pnote, datetime.now().strftime("%Y-%m-%d %H:%M"), pmid, email))
                        st.toast("Notes modified.")
                        st.rerun()
                with col_l: 
                    st.link_button("Source URL", f"https://pubmed-ncbi-nlm-nih-gov.ezproxy.haifa.ac.il/{pmid}/", use_container_width=True)

# --- PAGE: SETTINGS ---
elif page == "Settings":
    st.markdown("## Configuration & Access Control")
    
    with st.container(border=True):
        with st.form("prof"):
            st.markdown("### Investigator Identity")
            n_up = st.text_input("Full Name:", value=st.session_state.name)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("### Engine Connectivity")
            st.markdown('<div class="api-instructions">Verify AI synchronization. Navigate to <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a> to manage cryptographic keys.</div>', unsafe_allow_html=True)
            k_up = st.text_input("Gemini API Credential:", value=st.session_state.api_key, type="password")
            
            if st.form_submit_button("Commit Configuration Updates", type="primary"):
                conn = get_db()
                with conn.cursor() as c:
                    c.execute("UPDATE users SET name=%s, api_key=%s WHERE email=%s", (n_up, k_up, email))
                st.session_state.name = n_up
                st.session_state.api_key = k_up
                st.success("System configuration updated successfully.")
                st.rerun()

        st.divider()
        if st.button("Terminate Session (Log Out)"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --- PAGE: ADMIN CONSOLE (Exclusive View) ---
elif page == "Admin Console" and is_admin:
    st.markdown("## System Administration Console")
    st.markdown("Executive telemetry and user oversight.")
    
    conn = get_db()
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        # Metrics
        c.execute("SELECT COUNT(*) as c FROM users")
        total_users = c.fetchone()['c']
        c.execute("SELECT COUNT(*) as c FROM reading_list")
        total_docs = c.fetchone()['c']
        c.execute("SELECT COUNT(*) as c FROM login_history")
        total_logins = c.fetchone()['c']
        
        # Tables
        c.execute("SELECT email, name, created_at FROM users ORDER BY created_at DESC")
        users_df = c.fetchall()
        
        c.execute("SELECT user_email, login_time FROM login_history ORDER BY login_time DESC LIMIT 100")
        logins_df = c.fetchall()

    m1, m2, m3 = st.columns(3)
    m1.metric("Registered Investigators", total_users)
    m2.metric("Total Documents Processed", total_docs)
    m3.metric("Total Authentication Events", total_logins)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_users, tab_audit = st.tabs(["User Directory", "Audit Logs (Last 100)"])
    
    with tab_users:
        st.dataframe(users_df, use_container_width=True)
        
    with tab_audit:
        st.dataframe(logins_df, use_container_width=True)