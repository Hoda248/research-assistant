import streamlit as st
from Bio import Entrez
from supabase import create_client, Client
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

# --- SESSION STATE INITIALIZATION ---
# This strictly prevents "AttributeError: st.session_state has no attribute..."
session_defaults = {
    "user_email": None,
    "profile_loaded": False,
    "name": "",
    "keywords":[],
    "authors": "",
    "api_key": "",
    "feed_results":[],
    "current_page": "Dashboard"
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- SUPABASE DATABASE CONNECTION ---
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"Failed to connect to Supabase API. Check your SUPABASE_URL and SUPABASE_KEY in secrets. Error: {e}")
    st.stop()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def load_user_profile(email):
    res = supabase.table('users').select('*').eq('email', email).execute()
    return res.data[0] if res.data else None

def log_audit_trail(email):
    supabase.table('login_history').insert({'user_email': email}).execute()

# --- AI LOGIC ---
def get_safe_model():
    api_key = st.session_state.api_key
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
        ids = Entrez.read(h)["IdList"]
        h.close()
        if not ids: return[]
        h = Entrez.esummary(db="pubmed", id=",".join(ids))
        res = Entrez.read(h)
        h.close()
        return res
    except: return[]

def fetch_abstract(pmid):
    try:
        h = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        recs = Entrez.read(h)
        h.close()
        article = recs['PubmedArticle'][0]['MedlineCitation']['Article']
        return " ".join(article['Abstract']['AbstractText']) if 'Abstract' in article else "Abstract not available."
    except: return "Error retrieving abstract data."

# --- DB HELPERS ---
def toggle_reading_list(pmid, title, journal, authors, date):
    email = st.session_state.user_email
    res = supabase.table('reading_list').select('id').eq('pmid', pmid).eq('user_email', email).execute()
    
    if res.data:
        supabase.table('reading_list').delete().eq('pmid', pmid).eq('user_email', email).execute()
        return "Document removed from Reading Room."
    else:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        data = {
            "user_email": email, "pmid": pmid, "title": title, "journal": journal, 
            "authors": authors, "date": date, "notes": "", "last_edited": now
        }
        supabase.table('reading_list').insert(data).execute()
        return "Document saved to Reading Room."

def export_notes_to_word():
    doc = Document()
    doc.add_heading(f"Research Notebook - {datetime.now().strftime('%Y-%m-%d')}", 0)
    email = st.session_state.user_email
    
    doc.add_heading("Literature Notes", level=1)
    res_notes = supabase.table('reading_list').select('title, notes, authors, date').neq('notes', '').eq('user_email', email).execute()
    for item in res_notes.data:
        year = item['date'][:4] if item['date'] else "n.d."
        doc.add_heading(f"{item['authors']} ({year}). {item['title']}.", level=2)
        doc.add_paragraph(item['notes'])
    
    doc.add_heading("General Research Ideas", level=1)
    res_gen = supabase.table('general_notes').select('content, date').eq('user_email', email).order('date', desc=True).execute()
    for item in res_gen.data:
        p = doc.add_paragraph()
        p.add_run(f"[{item['date']}] ").bold = True
        p.add_run(item['content'])
        
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- AUTHENTICATION PORTAL ---
if not st.session_state.user_email:
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
                    res = supabase.table('users').select('password_hash').eq('email', log_email).execute()
                    if res.data and res.data[0]['password_hash'] == hash_password(log_pass):
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
                        check_res = supabase.table('users').select('email').eq('email', reg_email).execute()
                        if check_res.data:
                            st.error("This email is already registered.")
                        else:
                            new_user = {
                                "email": reg_email, "name": reg_name, "password_hash": hash_password(reg_pass),
                                "keywords": "", "authors": "", "api_key": reg_key
                            }
                            supabase.table('users').insert(new_user).execute()
                            st.success("Registration successful. Please proceed to Authenticate.")
                    else:
                        st.error("Complete all required fields.")
    st.stop()

# --- LOAD USER STATE ---
if not st.session_state.profile_loaded:
    prof = load_user_profile(st.session_state.user_email)
    if prof:
        st.session_state.name = prof.get('name', '')
        st.session_state.keywords =[k for k in prof.get('keywords', '').split(",") if k] if prof.get('keywords') else[]
        st.session_state.authors = prof.get('authors', '')
        st.session_state.api_key = prof.get('api_key', '')
    st.session_state.feed_results =[]
    st.session_state.current_page = "Dashboard"
    st.session_state.profile_loaded = True

Entrez.email = st.session_state.user_email
email = st.session_state.user_email

# --- TOP NAVIGATION BAR ---
nav_options =["Dashboard", "Recent Publications", "Literature Discovery", "Reading Room", "Notebook", "Settings"]

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
    
    res_saved = supabase.table('reading_list').select('id', count='exact').eq('user_email', email).execute()
    saved_count = res_saved.count if res_saved.count else 0
    
    res_notes = supabase.table('general_notes').select('id', count='exact').eq('user_email', email).execute()
    notes_count = res_notes.count if res_notes.count else 0
    
    res_pnotes = supabase.table('reading_list').select('id', count='exact').neq('notes', '').eq('user_email', email).execute()
    pnotes_count = res_pnotes.count if res_pnotes.count else 0

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
            kw_str = ",".join(st.session_state.keywords)
            supabase.table('users').update({'keywords': kw_str}).eq('email', email).execute()
            st.rerun()

        if st.session_state.keywords:
            st.caption("Active Filters (Click to remove):")
            tag_cols = st.columns(6)
            for i, kw in enumerate(st.session_state.keywords):
                with tag_cols[i % 6]:
                    if st.button(f"Remove: {kw}", key=f"del_{kw}"):
                        st.session_state.keywords.remove(kw)
                        kw_str = ",".join(st.session_state.keywords)
                        supabase.table('users').update({'keywords': kw_str}).eq('email', email).execute()
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
                check_sv = supabase.table('reading_list').select('id').eq('pmid', pid).eq('user_email', email).execute()
                is_sv = bool(check_sv.data)
                    
                if st.button("Save Document" if not is_sv else "Remove Document", key=f"al_sv_{pid}", use_container_width=True):
                    st.toast(toggle_reading_list(pid, ttl, jrnl, auths, pdate))
                    st.rerun()
            with c4:
                if st.button("Generate AI Summary", key=f"btn_sum_{pid}", use_container_width=True): 
                    st.session_state[f"show_sum_{pid}"] = True
            
            if st.session_state.get(f"show_sum_{pid}"):
                with st.spinner("Executing AI analysis..."):
                    summary = generate_ai_summary(fetch_abstract(pid))
                    st.markdown(f"<div class='ai-summary-box'>{summary}</div>", unsafe_allow_html=True)
                if st.button("Dismiss Analysis", key=f"cls_{pid}"): 
                    del st.session_state[f"show_sum_{pid}"]
                    st.rerun()

# --- PAGE: Literature Discovery ---
elif page == "Literature Discovery":
    st.markdown("## Literature Discovery")
    
    with st.container(border=True):
        st.markdown("### Filters")
        c1, c2 = st.columns(2)
        with c1: skws = st.text_input("Subject Keywords (comma separated):", value=",".join(st.session_state.keywords))
        with c2: sauth = st.text_input("Primary Author Target:", value=st.session_state.authors)
        
        c3, c4 = st.columns(2)
        with c3: slogic = st.radio("Boolean Operator:", ["OR", "AND"], horizontal=True)
        with c4: timeframe = st.selectbox("Historical Range:",["Week", "Month", "Year", "10 Years"], index=1)
        
        if st.button("Execute Database Query", type="primary"):
            t_map = {"Week": 7, "Month": 30, "Year": 365, "10 Years": 3650}
            with st.spinner("Querying PubMed via Entrez API..."):
                st.session_state.feed_results = get_summaries([k.strip() for k in skws.split(",") if k.strip()], sauth, t_map[timeframe], slogic)

    st.markdown("<br>", unsafe_allow_html=True)

    for art in st.session_state.feed_results:
        with st.container(border=True):
            pid, ttl = str(art['Id']), art['Title']
            auths, jrnl, pdate = ", ".join(art['AuthorList']), art['Source'], art['PubDate']
            st.markdown(f"#### {ttl}")
            st.markdown(f"<div class='paper-metadata'><b>Authors:</b> {auths}<br><b>Publication:</b> <i>{jrnl}</i> | <b>Date:</b> {pdate}</div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.link_button("Library Access", f"https://pubmed-ncbi-nlm-nih-gov.ezproxy.haifa.ac.il/{pid}/", use_container_width=True)
            with c2: st.link_button("Sci-Hub Proxy", f"{SCIHUB_BASE_URL}{pid}", use_container_width=True)
            with c3:
                check_sv = supabase.table('reading_list').select('id').eq('pmid', pid).eq('user_email', email).execute()
                is_sv = bool(check_sv.data)
                    
                if st.button("Save Document" if not is_sv else "Remove Document", key=f"sv_{pid}", use_container_width=True):
                    st.toast(toggle_reading_list(pid, ttl, jrnl, auths, pdate))
                    st.rerun()
            with c4:
                if st.button("Generate AI Summary", key=f"ai_{pid}", use_container_width=True): 
                    st.session_state[f"show_sum_{pid}"] = True
            
            if st.session_state.get(f"show_sum_{pid}"):
                with st.spinner("Executing AI analysis..."):
                    summary = generate_ai_summary(fetch_abstract(pid))
                    st.markdown(f"<div class='ai-summary-box'>{summary}</div>", unsafe_allow_html=True)
                if st.button("Dismiss Analysis", key=f"cls_s_{pid}"): 
                    del st.session_state[f"show_sum_{pid}"]
                    st.rerun()

# --- PAGE: READING ROOM ---
elif page == "Reading Room":
    st.markdown("## Reading Room")
    
    res = supabase.table('reading_list').select('*').eq('user_email', email).order('last_edited', desc=True).execute()
    items = res.data
    
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
            new_note = st.text_area("Record methodological insights, critiques, or hypotheses:", value=item.get('notes', ''), key=f"nt_rr_{pmid}", height=120, label_visibility="collapsed")
            if st.button("Commit Notes to Database", key=f"sv_rr_{pmid}"):
                supabase.table('reading_list').update({
                    'notes': new_note, 
                    'last_edited': datetime.now().strftime("%Y-%m-%d %H:%M")
                }).eq('pmid', pmid).eq('user_email', email).execute()
                
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
                    data = {"user_email": email, "content": new_gen_c, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                    supabase.table('general_notes').insert(data).execute()
                    st.rerun()
        
        res_g = supabase.table('general_notes').select('*').eq('user_email', email).order('id', desc=True).execute()
        for note in res_g.data:
            nid = note['id']
            with st.expander(f"Record Entry: {note['date']}"):
                ed_gen = st.text_area("Revise Entry:", value=note['content'], key=f"ed_gen_{nid}", height=100, label_visibility="collapsed")
                if st.button("Update Record", key=f"up_g_{nid}"):
                    supabase.table('general_notes').update({'content': ed_gen}).eq('id', nid).eq('user_email', email).execute()
                    st.toast("Record modified.")
                    st.rerun()
                    
    with tab2:
        res_p = supabase.table('reading_list').select('*').neq('notes', '').eq('user_email', email).order('last_edited', desc=True).execute()
        p_notes = res_p.data
                
        if not p_notes:
            st.info("No active literature notes detected in the repository.")
            
        for note in p_notes:
            pmid = note['pmid']
            with st.expander(f"Source Analysis: {note['title'][:80]}..."):
                col_t, col_l = st.columns([0.85, 0.15])
                with col_t:
                    new_pnote = st.text_area("Notes Data:", value=note['notes'], key=f"ed_p_{pmid}", height=150, label_visibility="collapsed")
                    if st.button("Update Notes Data", key=f"up_p_{pmid}"):
                        supabase.table('reading_list').update({
                            'notes': new_pnote, 
                            'last_edited': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }).eq('pmid', pmid).eq('user_email', email).execute()
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
                supabase.table('users').update({'name': n_up, 'api_key': k_up}).eq('email', email).execute()
                st.session_state.name = n_up
                st.session_state.api_key = k_up
                st.success("System configuration updated successfully.")
                st.rerun()

        st.divider()
        if st.button("Terminate Session (Log Out)"):
            st.session_state.clear()
            st.rerun()

# --- PAGE: ADMIN CONSOLE (Exclusive View) ---
elif page == "Admin Console" and is_admin:
    st.markdown("## System Administration Console")
    st.markdown("Executive telemetry and user oversight.")
    
    c_users = supabase.table('users').select('email', count='exact').execute()
    c_docs = supabase.table('reading_list').select('id', count='exact').execute()
    c_logins = supabase.table('login_history').select('id', count='exact').execute()

    m1, m2, m3 = st.columns(3)
    m1.metric("Registered Investigators", c_users.count if c_users.count else 0)
    m2.metric("Total Documents Processed", c_docs.count if c_docs.count else 0)
    m3.metric("Total Authentication Events", c_logins.count if c_logins.count else 0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_users, tab_audit = st.tabs(["User Directory", "Audit Logs (Last 100)"])
    
    with tab_users:
        users_df = supabase.table('users').select('email, name, created_at').order('created_at', desc=True).execute().data
        st.dataframe(users_df, use_container_width=True)
        
    with tab_audit:
        logins_df = supabase.table('login_history').select('user_email, login_time').order('login_time', desc=True).limit(100).execute().data
        st.dataframe(logins_df, use_container_width=True)