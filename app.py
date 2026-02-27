import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import datetime
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="+Statistics · Running Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DM = st.session_state.dark_mode

# Paletas
if DM:
    BG       = "#060608"
    BG2      = "#0d0d10"
    CARD     = "#111116"
    BORDER   = "#1e1e28"
    BORDER2  = "#2a2a38"
    TEXT     = "#f0f0f5"
    SUBTEXT  = "#5a5a72"
    ACCENT   = "#C8FF00"   # lima eléctrico
    ACCENT2  = "#00E5FF"   # cyan
    GOOD     = "#39D98A"
    WARN     = "#FFCB47"
    BAD      = "#FF5C5C"
    CHART_BG = "#0d0d10"
    SHADOW   = "rgba(200,255,0,0.08)"
else:
    BG       = "#F5F5F0"
    BG2      = "#EBEBЕ4"
    CARD     = "#FFFFFF"
    BORDER   = "#E0E0D8"
    BORDER2  = "#CCCCС0"
    TEXT     = "#111116"
    SUBTEXT  = "#888880"
    ACCENT   = "#5C8A00"
    ACCENT2  = "#0077AA"
    GOOD     = "#1A8A55"
    WARN     = "#B8880A"
    BAD      = "#CC2222"
    CHART_BG = "#FFFFFF"
    SHADOW   = "rgba(92,138,0,0.10)"

# ─────────────────────────────────────────────
# CSS PREMIUM
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {{
  --bg:      {BG};
  --bg2:     {BG2};
  --card:    {CARD};
  --border:  {BORDER};
  --text:    {TEXT};
  --sub:     {SUBTEXT};
  --accent:  {ACCENT};
  --accent2: {ACCENT2};
  --good:    {GOOD};
  --warn:    {WARN};
  --bad:     {BAD};
  --shadow:  {SHADOW};
}}

/* ── Reset & Base ── */
html, body, [class*="css"] {{
  font-family: 'Space Grotesk', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}}
.stApp {{ background: var(--bg) !important; }}
.main .block-container {{ padding: 1.5rem 2rem !important; max-width: 1400px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--sub) !important; }}
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
  gap: 0.2rem;
  display: flex;
  flex-direction: column;
}}
[data-testid="stSidebar"] .stRadio label {{
  display: flex !important;
  align-items: center;
  padding: 0.6rem 0.8rem !important;
  border-radius: 6px !important;
  cursor: pointer;
  transition: all 0.15s;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.05em;
  color: var(--sub) !important;
  border: 1px solid transparent !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
  background: var(--border) !important;
  color: var(--text) !important;
}}
[data-testid="stSidebar"] .stRadio input:checked + div {{
  color: var(--accent) !important;
}}

/* ── Botón principal ── */
.stButton > button {{
  background: var(--accent) !important;
  color: #000 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.12em !important;
  border: none !important;
  border-radius: 4px !important;
  padding: 0.65rem 2rem !important;
  transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
  box-shadow: 0 0 24px var(--shadow) !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 32px var(--shadow) !important;
  opacity: 0.92 !important;
}}
.stButton > button:active {{
  transform: translateY(0) !important;
}}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {{
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: 'Space Grotesk', sans-serif !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--shadow) !important;
}}
.stSelectbox > div > div {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
}}
.stFileUploader {{
  background: var(--card) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: 8px !important;
  transition: border-color 0.2s;
}}
.stFileUploader:hover {{
  border-color: var(--accent) !important;
}}

/* ── Checkbox ── */
.stCheckbox label {{ color: var(--sub) !important; font-size: 0.85rem !important; }}

/* ── Dataframe ── */
.stDataFrame {{ background: var(--card) !important; border-radius: 8px; }}
[data-testid="stDataFrame"] th {{
  background: var(--bg2) !important;
  color: var(--accent) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em;
}}
[data-testid="stDataFrame"] td {{
  color: var(--text) !important;
  font-size: 0.8rem !important;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
  background: var(--accent) !important;
}}

/* ── Alerts ── */
.stSuccess {{ background: rgba(57,217,138,0.08) !important; border-color: var(--good) !important; }}
.stWarning {{ background: rgba(255,203,71,0.08) !important; border-color: var(--warn) !important; }}
.stError   {{ background: rgba(255,92,92,0.08) !important;  border-color: var(--bad)  !important; }}
.stInfo    {{ background: rgba(0,229,255,0.06) !important;  border-color: var(--accent2) !important; }}

/* ── Divider ── */
hr {{ border-color: var(--border) !important; margin: 1.5rem 0 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
  color: var(--sub) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.08em;
  padding: 0.7rem 1.2rem !important;
}}
.stTabs [aria-selected="true"] {{
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
  background: transparent !important;
}}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: var(--accent) !important; }}

/* ── Animaciones de entrada ── */
@keyframes fadeSlideUp {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
  from {{ opacity: 0; }}
  to   {{ opacity: 1; }}
}}
@keyframes pulse {{
  0%,100% {{ box-shadow: 0 0 0 0 var(--shadow); }}
  50%      {{ box-shadow: 0 0 20px 4px var(--shadow); }}
}}
@keyframes shimmer {{
  0%   {{ background-position: -200% center; }}
  100% {{ background-position: 200% center; }}
}}

.animate-in {{
  animation: fadeSlideUp 0.5s cubic-bezier(0.4,0,0.2,1) both;
}}
.animate-in-delay-1 {{ animation-delay: 0.08s; }}
.animate-in-delay-2 {{ animation-delay: 0.16s; }}
.animate-in-delay-3 {{ animation-delay: 0.24s; }}
.animate-in-delay-4 {{ animation-delay: 0.32s; }}
.animate-in-delay-5 {{ animation-delay: 0.40s; }}
.animate-in-delay-6 {{ animation-delay: 0.48s; }}

/* ── Tarjeta métrica ── */
.mcard {{
  background: var(--card);
  border: 1px solid var(--border);
  border-top: 2px solid;
  border-radius: 8px;
  padding: 1.4rem 1rem 1.1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
  cursor: default;
}}
.mcard:hover {{
  transform: translateY(-3px);
  box-shadow: 0 8px 32px var(--shadow);
}}
.mcard::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 60px;
  background: linear-gradient(180deg, currentColor 0%, transparent 100%);
  opacity: 0.04;
  pointer-events: none;
}}
.mcard .mlabel {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  color: var(--sub);
  margin-bottom: 0.5rem;
  text-transform: uppercase;
}}
.mcard .mvalue {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.8rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
}}
.mcard .munit {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.72rem;
  color: var(--sub);
  margin-top: 0.3rem;
}}
.mcard .mstatus {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  margin-top: 0.6rem;
  padding: 0.2rem 0.6rem;
  border-radius: 20px;
  display: inline-block;
  border: 1px solid currentColor;
  opacity: 0.85;
}}
.mcard .mbar-bg {{
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  margin: 0.8rem 0.5rem 0;
  overflow: hidden;
}}
.mcard .mbar-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 1s cubic-bezier(0.4,0,0.2,1);
}}

/* ── Rec card ── */
.rec-card {{
  background: var(--card);
  border-left: 3px solid;
  border-radius: 0 8px 8px 0;
  padding: 1rem 1.2rem;
  margin-bottom: 0.6rem;
  animation: fadeSlideUp 0.4s both;
}}
.rec-title {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  margin-bottom: 0.3rem;
}}
.rec-text {{
  font-size: 0.82rem;
  color: var(--sub);
  line-height: 1.55;
}}

/* ── Section title ── */
.stitle {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  color: var(--sub);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
  margin-bottom: 1.2rem;
  text-transform: uppercase;
}}

/* ── Header logo ── */
.logo-wrap {{
  padding: 1.2rem 1rem 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.2rem;
}}
.logo-symbol {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem;
  font-weight: 900;
  color: var(--accent);
  letter-spacing: -0.02em;
  line-height: 1;
}}
.logo-name {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  color: var(--sub);
  margin-top: 2px;
  text-transform: uppercase;
}}

/* ── Stat badge ── */
.stat-badge {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.8rem 1rem;
  text-align: center;
}}
.stat-badge .sv {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--accent);
}}
.stat-badge .sl {{
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  color: var(--sub);
  margin-top: 2px;
  text-transform: uppercase;
}}

/* ── Trend badge ── */
.trend-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-top: 2px solid;
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
  animation: fadeSlideUp 0.4s both;
}}

/* ── Toggle theme button ── */
.theme-btn {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--sub) !important;
  font-size: 0.75rem !important;
  padding: 0.3rem 0.8rem !important;
  border-radius: 20px !important;
  width: auto !important;
  box-shadow: none !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{
  background: var(--accent) !important;
}}

/* ── Splash overlay ── */
#splash {{
  position: fixed;
  inset: 0;
  background: {BG};
  z-index: 9999;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  animation: fadeIn 0.3s ease, fadeOut 0.5s ease 1.8s forwards;
  pointer-events: none;
}}
@keyframes fadeOut {{
  to {{ opacity: 0; visibility: hidden; }}
}}
#splash-logo {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 5rem;
  font-weight: 900;
  color: {ACCENT};
  letter-spacing: -0.04em;
  animation: fadeSlideUp 0.6s cubic-bezier(0.4,0,0.2,1) both;
}}
#splash-sub {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.75rem;
  letter-spacing: 0.35em;
  color: {SUBTEXT};
  margin-top: 0.5rem;
  animation: fadeSlideUp 0.6s cubic-bezier(0.4,0,0.2,1) 0.15s both;
}}
#splash-line {{
  width: 60px;
  height: 2px;
  background: {ACCENT};
  margin-top: 1.5rem;
  animation: shimmer 1.5s linear infinite;
  background-size: 200% auto;
  background-image: linear-gradient(90deg, transparent, {ACCENT}, transparent);
}}
</style>

<!-- SPLASH SCREEN -->
<div id="splash">
  <div id="splash-logo">+Statistics</div>
  <div id="splash-sub">RUNNING BIOMECHANICS ANALYZER</div>
  <div id="splash-line"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES Y HELPERS
# ─────────────────────────────────────────────
SAMPLE_RATE  = 100
HISTORY_FILE = "rba_sessions.json"
PROFILE_FILE = "rba_profile.json"

DEVICE_POSITIONS = {
    "Pecho / Arnés":      {"gss_good": (0, 3),  "gss_warn": (3, 6)},
    "Brazo / Muñeca":     {"gss_good": (1, 4),  "gss_warn": (4, 8)},
    "Bolsillo / Cintura": {"gss_good": (2, 6),  "gss_warn": (6, 10)},
    "Espalda / Canguro":  {"gss_good": (4, 9),  "gss_warn": (9, 13)},
    "Mano (sostenido)":   {"gss_good": (3, 8),  "gss_warn": (8, 14)},
}

def scolor(val, good, warn, invert=False):
    lo_g, hi_g = good; lo_w, hi_w = warn
    if not invert:
        if lo_g <= val <= hi_g: return GOOD
        if lo_w <= val <= hi_w: return WARN
        return BAD
    else:
        if val <= hi_g: return GOOD
        if val <= hi_w: return WARN
        return BAD

def slabel(c):
    return {GOOD: "ÓPTIMO", WARN: "MODERADO", BAD: "REVISAR"}.get(c, "—")

def mcard(label, value, unit, color, sublabel="", bar_frac=None, delay=0):
    bar_html = ""
    if bar_frac is not None:
        pct = int(np.clip(bar_frac, 0, 1) * 100)
        bar_html = f"""
        <div class="mbar-bg">
          <div class="mbar-fill" style="width:{pct}%; background:{color};"></div>
        </div>"""
    return f"""
    <div class="mcard animate-in animate-in-delay-{delay}" style="border-top-color:{color}; color:{color}">
      <div class="mlabel">{label}</div>
      <div class="mvalue" style="color:{color}">{value}</div>
      <div class="munit">{unit}</div>
      <div class="mstatus" style="color:{color}; border-color:{color}">{sublabel}</div>
      {bar_html}
    </div>"""

# ─────────────────────────────────────────────
# PERSISTENCIA
# ─────────────────────────────────────────────
def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f: return json.load(f)
    return {"name":"","weight":70,"height":170,"goal":"Mejorar resistencia","level":"Intermedio","default_device":"Espalda / Canguro"}

def save_profile(p):
    with open(PROFILE_FILE,"w") as f: json.dump(p, f, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f: return json.load(f)
    return []

def append_history(s):
    h = load_history(); h.append(s)
    with open(HISTORY_FILE,"w") as f: json.dump(h, f, indent=2)

# ─────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────
def butter_bp(data, lo, hi, fs, order=4):
    nyq = fs/2
    b,a = butter(order, [lo/nyq, min(hi/nyq,0.99)], btype='band')
    return filtfilt(b,a,data)

def butter_lp(data, cutoff, fs, order=4):
    nyq=fs/2; b,a=butter(order, min(cutoff/nyq,0.99), btype='low')
    return filtfilt(b,a,data)

def est_fs(df):
    if 'time' in df.columns and len(df)>10:
        dt=np.median(np.diff(df['time'].values))
        if dt>0: return round(1/dt)
    return SAMPLE_RATE

def load_csv(f):
    try:
        df=pd.read_csv(f); df.columns=[c.strip().lower() for c in df.columns]
        if 'seconds_elapsed' in df.columns:
            df['time']=pd.to_numeric(df['seconds_elapsed'],errors='coerce')
        else:
            tc=next((c for c in df.columns if 'time' in c),None)
            if tc:
                raw=pd.to_numeric(df[tc],errors='coerce')
                if raw.median()>1e12: raw=raw/1e9
                df['time']=raw-raw.iloc[0]
        am={}
        for c in df.columns:
            if c in ['x','accel_x','acceleration x (m/s^2)']: am[c]='x'
            if c in ['y','accel_y','acceleration y (m/s^2)']: am[c]='y'
            if c in ['z','accel_z','acceleration z (m/s^2)']: am[c]='z'
            if c in ['speed','velocity']: am[c]='speed'
            if c in ['altitude','alt','elevation']: am[c]='altitude'
        df.rename(columns=am,inplace=True)
        for ax in ['x','y','z']:
            if ax in df.columns: df[ax]=pd.to_numeric(df[ax],errors='coerce')
        df.dropna(subset=['time'],inplace=True); df.reset_index(drop=True,inplace=True)
        return df
    except Exception as e:
        st.error(f"Error: {e}"); return None

def demo_data(dur=600, fs=100):
    t=np.linspace(0,dur,dur*fs); np.random.seed(42)
    fat=np.linspace(1,1.4,len(t)); ch=2.83
    z=np.sin(2*np.pi*ch*t)*0.8*fat+np.random.normal(0,.15,len(t))+9.81
    x=np.sin(2*np.pi*ch*t+np.pi/4)*0.3*fat+np.random.normal(0,.1,len(t))
    y=np.sin(2*np.pi*ch*t+np.pi/2)*0.15+np.random.normal(0,.08,len(t))
    accel=pd.DataFrame({'time':t,'x':x,'y':y,'z':z})
    gt=np.arange(0,dur,1)
    gps=pd.DataFrame({'time':gt,'speed':3+0.5*np.sin(2*np.pi*gt/120)})
    return accel, gps

def preprocess(df):
    fs=est_fs(df)
    for ax in ['x','y','z']:
        if ax in df.columns:
            df[ax+'_filt']=butter_lp(df[ax].fillna(0), min(20,fs/2-1), fs)
    if 'x_filt' in df.columns:
        df['magnitude']=np.sqrt(df['x_filt']**2+df['y_filt']**2+(df['z_filt']-9.81)**2)
    df['_fs']=fs; return df

def detect_steps(accel):
    fs=int(accel['_fs'].iloc[0]) if '_fs' in accel.columns else SAMPLE_RATE
    raw=(np.sqrt(accel['x_filt']**2+accel['y_filt']**2+accel['z_filt']**2).values
         if 'x_filt' in accel.columns else accel.get('magnitude',accel['z']).values)
    sig=butter_bp(raw,1.5,4,fs)
    env=uniform_filter1d(np.abs(sig),size=int(fs*0.1))
    peaks,_=find_peaks(env,height=np.percentile(env,65),
                       distance=int(fs*0.27),prominence=np.std(env)*0.5)
    if len(peaks)<4:
        peaks,_=find_peaks(env,height=np.mean(env),distance=int(fs*0.27),
                           prominence=np.std(env)*0.2)
    accel['step_signal']=sig; accel['step_envelope']=env
    return peaks, accel['time'].values[peaks], np.abs(sig[peaks])

def calc_rei(accel, peak_values):
    z=(accel.get('z_filt',accel.get('z',pd.Series([9.81]*len(accel)))).values-9.81)
    sr=max(np.percentile(np.abs(z),95),1e-6)
    sv=max(0,1-np.var(z)/sr**2*3)
    sc=max(0,1-np.std(peak_values)/(np.mean(np.abs(peak_values))+1e-6)*2) if len(peak_values)>4 else 0.5
    return round((sv*0.6+sc*0.4)*100,1)

def calc_gss(pv): return round(np.mean(np.abs(pv)),2)

def calc_cad_asym(pt):
    if len(pt)<4: return 0.0,0.0
    iv=np.diff(pt); vc=(iv>=0.25)&(iv<=1.0)
    ic=iv[vc] if vc.sum()>2 else iv
    med=np.median(ic)
    if med<=0: return 0.0,0.0
    cad=round(60/med,1)
    l,r=ic[0::2],ic[1::2]; n=min(len(l),len(r))
    asym=abs(np.mean(l[:n])-np.mean(r[:n]))/med*100 if n>0 else 0
    return cad, round(asym,2)

def cad_over_time(pt, ws=40):
    if len(pt)<5: return np.array([]),np.array([])
    if len(pt)<ws+1: ws=max(10,len(pt)//3)
    iv=np.diff(pt); tm=pt[1:]; cads,tout=[],[]
    for i in range(len(iv)-ws+1):
        w=iv[i:i+ws]; vl=(w>=0.27)&(w<=1.0)
        if vl.sum()>ws*0.5:
            cads.append(60/np.median(w[vl])); tout.append(tm[i+ws//2])
    if len(cads)<3: return np.array([]),np.array([])
    from scipy.ndimage import gaussian_filter1d
    return np.array(tout), np.clip(gaussian_filter1d(np.array(cads),sigma=6),100,230)

def calc_fi(accel, pt, pv, wm=2):
    tt=accel['time'].max(); w=wm*60; ft,fv=[],[]
    for ws in np.arange(0,min(tt,3600),w):
        m=(pt>=ws)&(pt<ws+w); wp=pv[m]
        if len(wp)<4: continue
        fv.append(round(np.mean(np.abs(wp))*0.6+np.std(wp)*0.4,3))
        ft.append(ws/60)
    return ft,fv

def analyze(accel_df, gps_df, dev_name):
    dp=DEVICE_POSITIONS[dev_name]
    accel=preprocess(accel_df.copy())
    _,pt,pv=detect_steps(accel)
    rei=calc_rei(accel,pv)
    gss=calc_gss(pv)
    cad,asym=calc_cad_asym(pt)
    ft,fv=calc_fi(accel,pt,pv)
    dur=accel['time'].max()/60
    spd=(np.mean(gps_df['speed'].values) if gps_df is not None and 'speed' in gps_df.columns
         else len(pt)/2/(dur*60) if dur>0 else 0)
    slope=float(np.polyfit(ft,fv,1)[0]) if len(fv)>=2 else 0
    return {
        "accel":accel,"gps":gps_df,"pt":pt,"pv":pv,
        "rei":rei,"gss":gss,"cadence":cad,"asymmetry":asym,
        "fi_times":ft,"fi_values":fv,"dur":dur,"speed":spd,
        "device":dev_name,"gss_good":dp["gss_good"],"gss_warn":dp["gss_warn"],
        "fatigue_slope":slope,"steps":len(pt),
        "date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

# ─────────────────────────────────────────────
# GRÁFICOS INTERACTIVOS CON PLOTLY
# ─────────────────────────────────────────────
def plotly_charts(r):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        pt=r["pt"]; pv=r["pv"]
        ft=r["fi_times"]; fv=r["fi_values"]
        cad=r["cadence"]; gps=r["gps"]
        t_cad, cad_v = cad_over_time(pt)

        plot_bg   = CHART_BG
        grid_col  = BORDER
        text_col  = SUBTEXT
        acc_col   = ACCENT
        good_col  = GOOD
        warn_col  = WARN

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["CADENCIA EN EL TIEMPO","FATIGUE INDEX","VELOCIDAD GPS"],
            horizontal_spacing=0.06
        )

        layout_font = dict(family="Barlow Condensed, monospace", color=text_col, size=11)

        # ── Cadencia ──
        if len(t_cad)>2:
            fig.add_trace(go.Scatter(
                x=t_cad/60, y=cad_v,
                mode='lines',
                line=dict(color=good_col, width=2),
                fill='tozeroy', fillcolor=f"rgba(57,217,138,0.07)",
                name="Cadencia", hovertemplate="%{y:.0f} ppm<extra></extra>"
            ), row=1, col=1)
            fig.add_hrect(y0=170, y1=185, row=1, col=1,
                          fillcolor=f"rgba(57,217,138,0.06)", line_width=0)
            fig.add_hline(y=cad, row=1, col=1,
                          line=dict(color=acc_col, dash='dot', width=1),
                          annotation_text=f"  {cad:.0f} ppm",
                          annotation_font=dict(color=acc_col, size=10))

        # ── Fatigue ──
        if ft and fv:
            fi_arr=np.array(fv); fi_t=np.array(ft)
            sl=np.polyfit(fi_t,fi_arr,1)
            tc=warn_col if sl[0]>0.0001 else (good_col if sl[0]<-0.0001 else acc_col)
            fig.add_trace(go.Scatter(
                x=fi_t, y=fi_arr,
                mode='lines+markers',
                line=dict(color=tc, width=2),
                marker=dict(size=6, color=plot_bg, line=dict(color=tc, width=1.5)),
                fill='tozeroy', fillcolor=f"rgba(255,203,71,0.07)",
                name="Fatigue", hovertemplate="%{y:.3f}<extra></extra>"
            ), row=1, col=2)
            trend=np.poly1d(sl)(fi_t)
            fig.add_trace(go.Scatter(
                x=fi_t, y=trend,
                mode='lines', line=dict(color=acc_col, dash='dash', width=1),
                name="Tendencia", showlegend=False,
                hovertemplate="tendencia: %{y:.3f}<extra></extra>"
            ), row=1, col=2)

        # ── Velocidad ──
        if gps is not None and 'speed' in gps.columns:
            tg=gps['time'].values/60; sp=gps['speed'].values
            fig.add_trace(go.Scatter(
                x=tg, y=sp,
                mode='lines',
                line=dict(color=acc_col, width=2),
                fill='tozeroy', fillcolor=f"rgba(200,255,0,0.07)",
                name="Velocidad", hovertemplate="%{y:.2f} m/s<extra></extra>"
            ), row=1, col=3)
            fig.add_hline(y=np.mean(sp), row=1, col=3,
                          line=dict(color=text_col, dash='dot', width=1),
                          annotation_text=f"  {np.mean(sp):.2f} m/s",
                          annotation_font=dict(color=text_col, size=10))

        fig.update_layout(
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=layout_font,
            showlegend=False,
            height=280,
            margin=dict(l=10, r=10, t=40, b=30),
        )
        for i in range(1,4):
            fig.update_xaxes(
                showgrid=True, gridcolor=grid_col, gridwidth=0.5,
                zeroline=False, tickfont=dict(size=9, color=text_col),
                title_text="min", title_font=dict(size=9, color=text_col),
                row=1, col=i
            )
            fig.update_yaxes(
                showgrid=True, gridcolor=grid_col, gridwidth=0.5,
                zeroline=False, tickfont=dict(size=9, color=text_col),
                row=1, col=i
            )
        for ann in fig.layout.annotations:
            ann.font.color = ACCENT
            ann.font.size  = 10
        return fig
    except ImportError:
        return None

def plotly_comparison(history, n):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        h=history[-n:]
        labels=[s["date"][5:16] for s in h]
        x=list(range(len(labels)))
        metrics=[
            ("REI",      [s["rei"]       for s in h], GOOD,  "/100"),
            ("GROUND SHOCK",[s["gss"]    for s in h], BAD,   "m/s²"),
            ("CADENCIA", [s["cadence"]   for s in h], ACCENT,"ppm"),
            ("ASIMETRÍA",[s["asymmetry"] for s in h], WARN,  "%"),
        ]
        fig=make_subplots(rows=2,cols=2,
                          subplot_titles=[m[0] for m in metrics],
                          horizontal_spacing=0.08, vertical_spacing=0.14)
        positions=[(1,1),(1,2),(2,1),(2,2)]
        for (row,col),(title,vals,color,unit) in zip(positions,metrics):
            fig.add_trace(go.Bar(
                x=labels, y=vals, marker_color=color,
                opacity=0.3, showlegend=False,
                hovertemplate=f"%{{y}}{unit}<extra></extra>"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=labels, y=vals, mode='lines+markers',
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=CHART_BG, line=dict(color=color,width=2)),
                showlegend=False,
                hovertemplate=f"%{{y}}{unit}<extra></extra>"
            ), row=row, col=col)
            if len(vals)>=2:
                tr=np.poly1d(np.polyfit(x,vals,1))(x)
                fig.add_trace(go.Scatter(
                    x=labels, y=tr, mode='lines',
                    line=dict(color=TEXT, dash='dot', width=1), opacity=0.3,
                    showlegend=False
                ), row=row, col=col)
        fig.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(family="Barlow Condensed", color=SUBTEXT, size=11),
            height=500, margin=dict(l=10,r=10,t=50,b=20)
        )
        for ann in fig.layout.annotations:
            ann.font.color=ACCENT; ann.font.size=11
        fig.update_xaxes(showgrid=False, tickfont=dict(size=8, color=SUBTEXT),
                         tickangle=-30)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.5,
                         zeroline=False, tickfont=dict(size=8, color=SUBTEXT))
        return fig
    except ImportError:
        return None

# ─────────────────────────────────────────────
# MATPLOTLIB DASHBOARD
# ─────────────────────────────────────────────
def build_fig(r):
    pt=r["pt"]; rei=r["rei"]; gss=r["gss"]
    cad=r["cadence"]; asym=r["asymmetry"]
    ft=r["fi_times"]; fv=r["fi_values"]
    spd=r["speed"]; dur=r["dur"]
    gss_g=r["gss_good"]; gss_w=r["gss_warn"]
    gps=r["gps"]; date_str=r["date"]

    rc=scolor(rei,(65,100),(40,65))
    gc=scolor(gss,gss_g,gss_w,invert=True)
    cc=scolor(cad,(170,185),(160,195))
    ac=scolor(asym,(0,5),(5,10),invert=True)

    fi_arr=np.array(fv) if fv else np.array([])
    fi_t  =np.array(ft) if ft else np.array([])
    if len(fi_arr)>1:
        slope=np.polyfit(fi_t,fi_arr,1)[0]
        ftc=WARN if slope>0.0001 else (GOOD if slope<-0.0001 else ACCENT)
        fvs=f"{fi_arr[-1]:.2f}"; fsym="▲" if slope>0.0001 else ("▼" if slope<-0.0001 else "—")
    else:
        ftc=SUBTEXT; fvs="N/A"; fsym="—"

    plt.rcParams.update({'font.family':'monospace','figure.facecolor':BG,'axes.facecolor':BG})
    fig=plt.figure(figsize=(20,11),facecolor=BG)
    gs=gridspec.GridSpec(3,1,figure=fig,height_ratios=[0.09,0.44,0.44],
                         hspace=0.08,left=0.03,right=0.97,top=0.96,bottom=0.04)
    gsc=gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=gs[1],wspace=0.025)

    def dcard(ax,label,vstr,unit,color,sub='',bf=None):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(0.5)
        ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.add_patch(Rectangle((0,0.945),1,0.055,color=color,alpha=0.9,
                               transform=ax.transAxes,clip_on=False,zorder=5))
        ax.text(0.5,0.82,label,ha='center',va='center',color=SUBTEXT,
                fontsize=7.5,fontfamily='monospace',transform=ax.transAxes)
        ax.text(0.5,0.50,vstr,ha='center',va='center',color=color,
                fontsize=28,fontweight='bold',fontfamily='monospace',
                transform=ax.transAxes)
        ax.text(0.5,0.24,unit,ha='center',va='center',color=SUBTEXT,
                fontsize=8.5,fontfamily='monospace',transform=ax.transAxes)
        if sub:
            ax.text(0.5,0.09,sub,ha='center',va='center',color=color,
                    fontsize=7,fontfamily='monospace',transform=ax.transAxes)
        if bf is not None:
            bf=float(np.clip(bf,0,1))
            ax.add_patch(Rectangle((0.06,0.022),0.88,0.04,
                                   color=BORDER,transform=ax.transAxes,clip_on=False))
            ax.add_patch(Rectangle((0.06,0.022),0.88*bf,0.04,
                                   color=color,alpha=0.7,transform=ax.transAxes,clip_on=False))

    # HEADER
    ax_h=fig.add_subplot(gs[0]); ax_h.axis('off'); ax_h.set_facecolor(BG)
    ax_h.add_line(Line2D([0,0],[0,1],color=ACCENT,linewidth=3,
                         transform=ax_h.transAxes,clip_on=False))
    ax_h.text(0.022,0.62,'RUNNING BIOMECHANICS',ha='left',va='center',
              color=TEXT,fontsize=18,fontweight='bold',fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.022,0.18,f"{date_str}  ·  {r['steps']:,} STEPS  ·  {dur:.1f} MIN",
              ha='left',va='center',color=SUBTEXT,fontsize=8,fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.98,0.55,f"{rei:.0f}",ha='right',va='center',color=rc,
              fontsize=38,fontweight='bold',fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.98,0.12,'REI SCORE',ha='right',va='center',color=SUBTEXT,
              fontsize=7.5,fontfamily='monospace',transform=ax_h.transAxes)
    ax_h.add_line(Line2D([0,1],[0,0],color=BORDER,linewidth=0.8,
                         transform=ax_h.transAxes,clip_on=False))

    # 6 TARJETAS
    ca=[fig.add_subplot(gsc[i]) for i in range(6)]
    mp=1000/spd/60 if spd>0 else 0
    mp_m,mp_s=int(mp),int((mp%1)*60)
    dcard(ca[0],'RUNNING ECONOMY',f'{rei:.0f}','/100',rc,
          sub='BUENO' if rc==GOOD else 'MODERADO' if rc==WARN else 'MEJORAR',
          bf=rei/100)
    dcard(ca[1],'GROUND SHOCK',f'{gss:.1f}','m/s²',gc,
          sub='BUENO' if gc==GOOD else 'MODERADO' if gc==WARN else 'ALTO',
          bf=1-np.clip(gss/20,0,1))
    dcard(ca[2],'CADENCIA',f'{cad:.0f}','ppm',cc,
          sub='ÓPTIMA' if cc==GOOD else 'REVISAR',
          bf=np.clip((cad-120)/100,0,1))
    dcard(ca[3],'ASIMETRÍA',f'{asym:.1f}','%',ac,
          sub='OK' if ac==GOOD else 'LEVE' if ac==WARN else 'ALTO',
          bf=1-np.clip(asym/20,0,1))
    dcard(ca[4],'VELOCIDAD',f'{spd:.2f}','m/s',ACCENT2,
          sub=f'{mp_m}m {mp_s:02d}s /km',
          bf=np.clip(spd/5,0,1))
    dcard(ca[5],'FATIGUE INDEX',fvs,fsym,ftc,
          sub='ESTABLE' if fsym=='—' else 'AUMENTANDO' if fsym=='▲' else 'BAJANDO')

    # FILA INFERIOR — 3 mini gráficos matplotlib
    gsb=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[2],wspace=0.04)

    def schart(ax,title,accent):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(0.4)
        ax.tick_params(labelsize=7,colors=SUBTEXT,length=2)
        ax.grid(True,color=BORDER,linewidth=0.4,alpha=0.8)
        ax.add_line(Line2D([0,1],[1,1],color=accent,linewidth=1.5,
                           transform=ax.transAxes,clip_on=False))
        ax.text(0.015,0.93,title,transform=ax.transAxes,color=accent,
                fontsize=8,fontweight='bold',va='top',fontfamily='monospace')

    ax_c=fig.add_subplot(gsb[0])
    tc,cv=cad_over_time(pt)
    if len(tc)>2:
        ax_c.fill_between(tc/60,cv,cv.min()-5,alpha=0.08,color=cc)
        ax_c.plot(tc/60,cv,color=cc,linewidth=1.5,alpha=0.9)
        ax_c.axhspan(170,185,alpha=0.06,color=GOOD)
        ax_c.axhline(cad,color=ACCENT,linewidth=0.8,linestyle='--',alpha=0.6)
        ax_c.set_ylim(max(130,cv.min()-10),min(220,cv.max()+10))
        ax_c.set_xlabel('min',fontsize=7,color=SUBTEXT)
    schart(ax_c,'CADENCIA  [ppm]',cc)

    ax_f=fig.add_subplot(gsb[1])
    if len(fi_arr)>1:
        ax_f.fill_between(fi_t,fi_arr,fi_arr.min()-.01,alpha=0.10,color=ftc)
        ax_f.plot(fi_t,fi_arr,color=ftc,linewidth=1.5,marker='o',markersize=4,
                  markerfacecolor=CARD,markeredgecolor=ftc,markeredgewidth=1.2)
        ax_f.plot(fi_t,np.poly1d(np.polyfit(fi_t,fi_arr,1))(fi_t),
                  '--',color=ACCENT,linewidth=0.9,alpha=0.6)
        ax_f.set_xlabel('min',fontsize=7,color=SUBTEXT)
    schart(ax_f,'FATIGUE INDEX',ftc)

    ax_s=fig.add_subplot(gsb[2])
    if gps is not None and 'speed' in gps.columns:
        tg=gps['time'].values/60; sp=gps['speed'].values
        ax_s.fill_between(tg,sp,sp.min()*.98,alpha=0.10,color=ACCENT)
        ax_s.plot(tg,sp,color=ACCENT,linewidth=1.5,alpha=0.9)
        ax_s.axhline(np.mean(sp),color=SUBTEXT,linewidth=0.8,linestyle='--',alpha=0.6)
        ax_s.set_xlabel('min',fontsize=7,color=SUBTEXT)
        schart(ax_s,'VELOCIDAD  [m/s]',ACCENT)
    elif len(pv)>4:
        imp=np.abs(pv)
        ax_s.hist(imp,bins=35,color=ACCENT,alpha=0.5,edgecolor='none')
        ax_s.axvline(np.mean(imp),color=TEXT,linewidth=1,linestyle='--',alpha=0.6)
        ax_s.set_xlabel('m/s²',fontsize=7,color=SUBTEXT)
        schart(ax_s,'DISTRIBUCIÓN IMPACTO',ACCENT)
    else:
        schart(ax_s,'VELOCIDAD  [m/s]',ACCENT)

    plt.tight_layout(pad=0); return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
profile = load_profile()

with st.sidebar:
    st.markdown(f"""
    <div class="logo-wrap">
      <div class="logo-symbol">+Statistics</div>
      <div class="logo-name">Running Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏃  Nueva sesión",
        "📊  Historial",
        "👥  Comparar",
        "👤  Perfil",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Toggle tema
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("☀️ Claro" if DM else "🌙 Oscuro", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    if profile.get("name"):
        st.markdown(f"""
        <div style="margin-top:1.5rem; padding:0.8rem; background:{CARD};
             border:1px solid {BORDER}; border-radius:6px;">
          <div style="color:{SUBTEXT}; font-size:0.6rem; letter-spacing:0.15em;
               margin-bottom:3px; font-family:'Barlow Condensed';">ATLETA</div>
          <div style="color:{TEXT}; font-size:0.9rem; font-weight:600;
               font-family:'Barlow Condensed';">{profile['name'].upper()}</div>
          <div style="color:{ACCENT}; font-size:0.65rem; margin-top:2px;
               font-family:'Barlow Condensed';">{profile.get('level','—').upper()}</div>
        </div>""", unsafe_allow_html=True)

    # Stats rápidas en sidebar
    history = load_history()
    if history:
        st.markdown(f"""
        <div style="margin-top:1rem; padding:0.8rem; background:{CARD};
             border:1px solid {BORDER}; border-radius:6px;">
          <div style="color:{SUBTEXT}; font-size:0.6rem; letter-spacing:0.15em;
               margin-bottom:6px; font-family:'Barlow Condensed';">RESUMEN</div>
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="text-align:center">
              <div style="color:{ACCENT}; font-size:1.2rem; font-weight:800;
                   font-family:'Barlow Condensed'">{len(history)}</div>
              <div style="color:{SUBTEXT}; font-size:0.58rem; font-family:'Barlow Condensed'">SESIONES</div>
            </div>
            <div style="text-align:center">
              <div style="color:{ACCENT}; font-size:1.2rem; font-weight:800;
                   font-family:'Barlow Condensed'">{np.mean([s['rei'] for s in history]):.0f}</div>
              <div style="color:{SUBTEXT}; font-size:0.58rem; font-family:'Barlow Condensed'">REI MEDIO</div>
            </div>
            <div style="text-align:center">
              <div style="color:{ACCENT}; font-size:1.2rem; font-weight:800;
                   font-family:'Barlow Condensed'">{np.mean([s['cadence'] for s in history]):.0f}</div>
              <div style="color:{SUBTEXT}; font-size:0.58rem; font-family:'Barlow Condensed'">CAD MEDIA</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PÁGINA: NUEVA SESIÓN
# ═══════════════════════════════════════════════
if "Nueva" in page:
    st.markdown('<div class="stitle animate-in">// NUEVA SESIÓN</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown(f'<div style="color:{SUBTEXT}; font-size:0.75rem; letter-spacing:0.1em; margin-bottom:0.6rem; font-family:\'Barlow Condensed\'">ARCHIVOS DE SENSOR</div>', unsafe_allow_html=True)
        accel_f = st.file_uploader("Acelerómetro (CSV)", type=["csv"], key="au",
                                    label_visibility="collapsed",
                                    help="Archivo CSV del acelerómetro — columnas: time, x, y, z")
        st.markdown(f'<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        gps_f   = st.file_uploader("GPS / Velocidad — opcional", type=["csv"], key="gu",
                                    label_visibility="collapsed",
                                    help="Opcional: columnas time, speed (m/s)")

    with col2:
        st.markdown(f'<div style="color:{SUBTEXT}; font-size:0.75rem; letter-spacing:0.1em; margin-bottom:0.6rem; font-family:\'Barlow Condensed\'">CONFIGURACIÓN</div>', unsafe_allow_html=True)
        dev = st.selectbox("Posición del dispositivo",
                           list(DEVICE_POSITIONS.keys()),
                           index=list(DEVICE_POSITIONS.keys()).index(
                               profile.get("default_device","Espalda / Canguro")),
                           label_visibility="collapsed")
        use_demo = st.checkbox("⚡  Usar datos demo", value=False)
        st.markdown(f'<div style="height:0.3rem"></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("▶  ANALIZAR SESIÓN", use_container_width=True):
        prog = st.progress(0, text="Iniciando análisis...")
        import time

        prog.progress(15, "Cargando datos...")
        time.sleep(0.2)

        if use_demo or accel_f is None:
            adf, gdf = demo_data()
            st.info("⚡ Usando datos de demostración")
        else:
            adf = load_csv(accel_f)
            gdf = load_csv(gps_f) if gps_f else None

        if adf is not None:
            prog.progress(35, "Preprocesando señal...")
            time.sleep(0.2)
            prog.progress(55, "Detectando pisadas...")
            time.sleep(0.2)
            r = analyze(adf, gdf, dev)
            prog.progress(80, "Calculando métricas...")
            time.sleep(0.2)
            st.session_state["last_result"] = r
            append_history({
                "date":r["date"],"duration":round(r["dur"],1),
                "steps":r["steps"],"device":r["device"],
                "rei":r["rei"],"gss":r["gss"],"cadence":r["cadence"],
                "asymmetry":r["asymmetry"],"fatigue_slope":r["fatigue_slope"],
                "speed":round(r["speed"],2),
            })
            prog.progress(100, "¡Listo!")
            time.sleep(0.3)
            prog.empty()
            st.success("✓  Análisis completado")

    # ── RESULTADOS ──
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        rc=scolor(r["rei"],(65,100),(40,65))
        gc=scolor(r["gss"],r["gss_good"],r["gss_warn"],invert=True)
        cc=scolor(r["cadence"],(170,185),(160,195))
        ac=scolor(r["asymmetry"],(0,5),(5,10),invert=True)
        spd=r["speed"]; mp=1000/spd/60 if spd>0 else 0
        mp_m,mp_s=int(mp),int((mp%1)*60)
        fi_arr=np.array(r["fi_values"]) if r["fi_values"] else np.array([])
        fi_t  =np.array(r["fi_times"])  if r["fi_times"]  else np.array([])
        if len(fi_arr)>1:
            sl=np.polyfit(fi_t,fi_arr,1)[0]
            ftc=WARN if sl>0.0001 else (GOOD if sl<-0.0001 else ACCENT)
            fvs=f"{fi_arr[-1]:.2f}"; fsym="▲" if sl>0.0001 else("▼" if sl<-0.0001 else "—")
        else:
            ftc=SUBTEXT; fvs="N/A"; fsym="—"

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">// MÉTRICAS DE SESIÓN</div>', unsafe_allow_html=True)

        cols = st.columns(6, gap="small")
        cards = [
            ("RUNNING ECONOMY", f"{r['rei']:.0f}", "/100", rc,
             "BUENO" if rc==GOOD else "MODERADO" if rc==WARN else "MEJORAR",
             r["rei"]/100, 1),
            ("GROUND SHOCK", f"{r['gss']:.1f}", "m/s²", gc,
             "BUENO" if gc==GOOD else "MODERADO" if gc==WARN else "ALTO",
             1-np.clip(r["gss"]/20,0,1), 2),
            ("CADENCIA", f"{r['cadence']:.0f}", "ppm", cc,
             "ÓPTIMA" if cc==GOOD else "REVISAR",
             np.clip((r["cadence"]-120)/100,0,1), 3),
            ("ASIMETRÍA", f"{r['asymmetry']:.1f}", "%", ac,
             "OK" if ac==GOOD else "LEVE" if ac==WARN else "ALTO",
             1-np.clip(r["asymmetry"]/20,0,1), 4),
            ("VELOCIDAD", f"{spd:.2f}", f"m/s  ·  {mp_m}m{mp_s:02d}s/km", ACCENT2,
             "", np.clip(spd/5,0,1), 5),
            ("FATIGUE INDEX", fvs, fsym, ftc,
             "ESTABLE" if fsym=="—" else "AUMENTANDO" if fsym=="▲" else "BAJANDO",
             None, 6),
        ]
        for col,(label,val,unit,color,sub,bf,delay) in zip(cols,cards):
            with col:
                st.markdown(mcard(label,val,unit,color,sub,bf,delay), unsafe_allow_html=True)

        # ── GRÁFICOS INTERACTIVOS ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">// ANÁLISIS TEMPORAL</div>', unsafe_allow_html=True)

        fig_plotly = plotly_charts(r)
        if fig_plotly:
            st.plotly_chart(fig_plotly, use_container_width=True)
        else:
            # Fallback matplotlib
            mfig = build_fig(r)
            st.pyplot(mfig, use_container_width=True)
            plt.close(mfig)

        # ── DESCARGAR DASHBOARD ──
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        buf = io.BytesIO()
        dfig = build_fig(r)
        dfig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close(dfig); buf.seek(0)
        st.download_button("⬇  Descargar dashboard PNG", buf,
                           f"rba_{r['date'][:10]}.png", "image/png",
                           use_container_width=True)

        # ── RECOMENDACIONES ──
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">// RECOMENDACIONES</div>', unsafe_allow_html=True)

        recs=[]
        if cc!=GOOD:
            if r["cadence"]<160:
                recs.append(("⚠  CADENCIA BAJA",
                    f"Tu cadencia ({r['cadence']:.0f} ppm) está por debajo del rango óptimo. "
                    "Intenta acortar la zancada y correr con música de 175-180 BPM.", WARN))
            else:
                recs.append(("💡  CADENCIA MEJORABLE",
                    f"Estás cerca del rango ideal (170-185 ppm). Pequeños ajustes pueden marcar la diferencia.", ACCENT))
        if gc==BAD:
            recs.append(("⚠  IMPACTO ELEVADO",
                f"GSS de {r['gss']:.1f} m/s² es alto para tu posición de sensor. "
                "Aterriza con el pie bajo el centro de masa y aumenta la cadencia.", WARN))
        if ac!=GOOD:
            recs.append(("🔄  ASIMETRÍA DETECTADA",
                f"Diferencia del {r['asymmetry']:.1f}% entre piernas. "
                "Considera ejercicios unilaterales y revisa posibles compensaciones.", WARN if ac==WARN else BAD))
        if r["rei"]<50:
            recs.append(("📉  REI BAJO",
                "Movimiento vertical excesivo. Trabaja la técnica: caída desde tobillos y activación del core.", BAD))
        if r["fatigue_slope"]>0.005:
            recs.append(("🔋  FATIGA PROGRESIVA",
                "El índice de fatiga aumenta significativamente. Revisa nutrición e hidratación pre-carrera.", WARN))
        if not recs:
            recs.append(("✅  SESIÓN EXCELENTE",
                "Todos los indicadores en rangos óptimos. Mantén la consistencia y sigue monitoreando.", GOOD))

        for i,(title,text,color) in enumerate(recs):
            st.markdown(f"""
            <div class="rec-card animate-in animate-in-delay-{i+1}" style="border-color:{color}">
              <div class="rec-title" style="color:{color}">{title}</div>
              <div class="rec-text">{text}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PÁGINA: HISTORIAL
# ═══════════════════════════════════════════════
elif "Historial" in page:
    history = load_history()
    st.markdown('<div class="stitle animate-in">// HISTORIAL DE SESIONES</div>', unsafe_allow_html=True)

    if not history:
        st.markdown(f"""
        <div style="text-align:center; padding:4rem; color:{SUBTEXT};">
          <div style="font-size:3rem; margin-bottom:0.5rem">📭</div>
          <div style="font-family:'Barlow Condensed'; font-size:1rem; letter-spacing:0.1em">
            SIN SESIONES REGISTRADAS
          </div>
          <div style="font-size:0.8rem; margin-top:0.5rem">
            Analiza tu primera sesión para empezar
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # KPIs
        c1,c2,c3,c4 = st.columns(4, gap="small")
        kpis = [
            (str(len(history)), "SESIONES TOTALES"),
            (f"{np.mean([s['rei'] for s in history]):.1f}", "REI PROMEDIO"),
            (f"{np.mean([s['cadence'] for s in history]):.0f} ppm", "CADENCIA MEDIA"),
            (f"{max(history,key=lambda x:x['rei'])['rei']:.0f}", "MEJOR REI"),
        ]
        for col,(val,label) in zip([c1,c2,c3,c4],kpis):
            with col:
                st.markdown(f"""
                <div class="stat-badge animate-in">
                  <div class="sv">{val}</div>
                  <div class="sl">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Tabla
        df_h=pd.DataFrame(history)
        show_cols=[c for c in ["date","duration","steps","device","rei","gss","cadence","asymmetry","speed"] if c in df_h.columns]
        rename={"date":"Fecha","duration":"Min","steps":"Pasos","device":"Dispositivo",
                "rei":"REI","gss":"GSS","cadence":"Cadencia","asymmetry":"Asim%","speed":"m/s"}
        st.dataframe(df_h[show_cols].rename(columns=rename).iloc[::-1].reset_index(drop=True),
                     use_container_width=True, height=420)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("🗑  Borrar historial", use_container_width=False):
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.rerun()

# ═══════════════════════════════════════════════
# PÁGINA: COMPARAR
# ═══════════════════════════════════════════════
elif "Comparar" in page:
    history = load_history()
    st.markdown('<div class="stitle animate-in">// COMPARAR SESIONES</div>', unsafe_allow_html=True)

    if len(history) < 2:
        st.markdown(f"""
        <div style="text-align:center; padding:4rem; color:{SUBTEXT};">
          <div style="font-size:3rem; margin-bottom:0.5rem">📊</div>
          <div style="font-family:'Barlow Condensed'; font-size:1rem; letter-spacing:0.1em">
            NECESITAS AL MENOS 2 SESIONES
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        n = st.slider("Sesiones a comparar (más recientes)", 2, min(10,len(history)),
                      min(5,len(history)), label_visibility="visible")
        sel = history[-n:]

        fig_comp = plotly_comparison(history, n)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)

        # Tendencias
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">// TENDENCIAS</div>', unsafe_allow_html=True)

        trend_cols = st.columns(4, gap="small")
        mtrends = [
            ("REI",       [s["rei"]       for s in sel], GOOD,  "/100", True),
            ("CADENCIA",  [s["cadence"]   for s in sel], ACCENT,"ppm",  True),
            ("GSS",       [s["gss"]       for s in sel], BAD,   "m/s²", False),
            ("ASIMETRÍA", [s["asymmetry"] for s in sel], WARN,  "%",    False),
        ]
        for col,(name,vals,color,unit,higher_better) in zip(trend_cols,mtrends):
            sl=np.polyfit(range(len(vals)),vals,1)[0]
            improving=(sl>0 and higher_better) or (sl<0 and not higher_better)
            arrow="↑" if sl>0 else "↓"
            tc=GOOD if improving else (WARN if abs(sl)<0.5 else BAD)
            with col:
                st.markdown(f"""
                <div class="trend-card" style="border-top-color:{color}">
                  <div style="font-family:'Barlow Condensed'; font-size:0.65rem;
                       letter-spacing:0.15em; color:{SUBTEXT}; margin-bottom:0.4rem">{name}</div>
                  <div style="font-family:'Barlow Condensed'; font-size:2.2rem;
                       font-weight:800; color:{color}">{arrow}</div>
                  <div style="font-size:0.72rem; color:{SUBTEXT}; margin-top:0.2rem">
                    {abs(sl):.2f} {unit}/sesión
                  </div>
                  <div style="font-family:'Barlow Condensed'; font-size:0.7rem;
                       color:{tc}; margin-top:0.4rem; letter-spacing:0.1em">
                    {'MEJORANDO' if improving else 'REVISAR'}
                  </div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PÁGINA: PERFIL
# ═══════════════════════════════════════════════
elif "Perfil" in page:
    st.markdown('<div class="stitle animate-in">// PERFIL DE ATLETA</div>', unsafe_allow_html=True)

    with st.form("pf"):
        c1,c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(f'<div style="color:{SUBTEXT}; font-size:0.7rem; letter-spacing:0.1em; margin-bottom:0.3rem; font-family:\'Barlow Condensed\'">DATOS PERSONALES</div>', unsafe_allow_html=True)
            name   = st.text_input("Nombre", value=profile.get("name",""), placeholder="Tu nombre")
            weight = st.number_input("Peso (kg)", 30, 200, int(profile.get("weight",70)))
            height = st.number_input("Altura (cm)", 140, 220, int(profile.get("height",170)))
        with c2:
            st.markdown(f'<div style="color:{SUBTEXT}; font-size:0.7rem; letter-spacing:0.1em; margin-bottom:0.3rem; font-family:\'Barlow Condensed\'">PREFERENCIAS</div>', unsafe_allow_html=True)
            level  = st.selectbox("Nivel",
                                  ["Principiante","Intermedio","Avanzado","Élite"],
                                  index=["Principiante","Intermedio","Avanzado","Élite"].index(
                                      profile.get("level","Intermedio")))
            goal   = st.selectbox("Objetivo principal",
                                  ["Mejorar resistencia","Perder peso","Preparar maratón",
                                   "Reducir lesiones","Mejorar velocidad","Mantenimiento"])
            ddev   = st.selectbox("Dispositivo habitual",
                                  list(DEVICE_POSITIONS.keys()),
                                  index=list(DEVICE_POSITIONS.keys()).index(
                                      profile.get("default_device","Espalda / Canguro")))
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        saved = st.form_submit_button("GUARDAR PERFIL", use_container_width=True)
        if saved:
            save_profile({"name":name,"weight":weight,"height":height,
                          "level":level,"goal":goal,"default_device":ddev})
            st.success("✓  Perfil guardado")
            st.rerun()

    # Stats acumuladas
    history = load_history()
    if history and profile.get("name"):
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">// TUS ESTADÍSTICAS</div>', unsafe_allow_html=True)

        total_h   = sum(s.get("duration",0) for s in history) / 60
        total_km  = sum(s.get("speed",0)*s.get("duration",0)*60/1000 for s in history)
        best_rei  = max(s["rei"] for s in history)
        streak    = len(history)

        sc1,sc2,sc3,sc4 = st.columns(4, gap="small")
        stats=[
            (f"{len(history)}", "SESIONES"),
            (f"{total_h:.1f}h", "TIEMPO TOTAL"),
            (f"{total_km:.0f}km", "KM ESTIMADOS"),
            (f"{best_rei:.0f}", "MEJOR REI"),
        ]
        for col,(val,label) in zip([sc1,sc2,sc3,sc4],stats):
            with col:
                st.markdown(f"""
                <div class="stat-badge animate-in">
                  <div class="sv" style="color:{ACCENT}">{val}</div>
                  <div class="sl">{label}</div>
                </div>""", unsafe_allow_html=True)
