import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ⚡  TEAM TECH TITANS — ZERO-G CINEMATIC LOADING SCREEN
# ============================================================
# KEY: st.markdown strips <script> tags → no JS → no animation.
# FIX: components.html() RUNS JS. The JS injects the overlay into
#      window.parent.document (the real Streamlit page), so
#      position:fixed covers 100vw × 100vh correctly.
if "_splash_shown" not in st.session_state:
    st.session_state["_splash_shown"] = True
    components.html("""
<script>
(function(){
  var D = window.parent.document;
  var W = window.parent;

  /* ---- Font ---- */
  if(!D.getElementById('ttt-font')){
    var lk=D.createElement('link');
    lk.id='ttt-font'; lk.rel='stylesheet';
    lk.href='https://fonts.googleapis.com/css2?family=Anton&family=Black+Han+Sans&family=Inter:wght@900&display=swap';
    D.head.appendChild(lk);
  }

  /* ---- CSS ---- */
  if(!D.getElementById('ttt-style')){
    var st2=D.createElement('style'); st2.id='ttt-style';
    st2.textContent=[
      '#ttt-splash{position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:2147483647!important;background:radial-gradient(ellipse at center,#A80000 0%,#5C0000 60%,#2a0000 100%)!important;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;overflow:hidden!important;animation:tttOut 0.9s ease 6s forwards!important;}',
      '#ttt-splash::before{content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 60px,rgba(180,210,255,0.07) 60px,rgba(180,210,255,0.07) 61px),repeating-linear-gradient(90deg,transparent,transparent 80px,rgba(180,210,255,0.07) 80px,rgba(180,210,255,0.07) 81px);animation:tttGrid 9s linear infinite;pointer-events:none;}',
      '@keyframes tttGrid{0%{background-position:0 0,0 0}100%{background-position:0 64px,0 0}}',
      '.ttt-dust{position:absolute;border-radius:50%;background:rgba(255,215,80,.75);pointer-events:none;animation:tttDust linear infinite;}',
      '@keyframes tttDust{0%{transform:translateY(0) translateX(0) scale(1);opacity:.7}50%{transform:translateY(-40vh) translateX(var(--dx)) scale(.7);opacity:.3}100%{transform:translateY(-90vh) translateX(calc(var(--dx)*1.8)) scale(.2);opacity:0}}',
      '#ttt-dolly{position:relative;z-index:10;display:flex;flex-direction:column;align-items:center;animation:tttZoom 6.2s cubic-bezier(.22,1,.36,1) forwards;}',
      '@keyframes tttZoom{0%{transform:scale(.75)}100%{transform:scale(1)}}',
      '#ttt-rows{display:flex;flex-direction:column;align-items:center;gap:8px;}',
      '.ttt-row{display:flex;align-items:center;justify-content:center;gap:5px;}',
      '.ttt-sp{display:inline-block;width:34px;}',
      /* --- DUDE-style letter: Anton italic, thick stacked outline, rough SVG filter --- */
      '.ttt-L{display:inline-block;font-family:"Anton","Black Han Sans","Impact",sans-serif;font-size:clamp(60px,11vw,148px);font-style:italic;letter-spacing:1px;color:#FFE033;-webkit-text-stroke:8px #0a0a0a;text-shadow:1px 1px 0 #0a0a0a,2px 2px 0 #0a0a0a,3px 3px 0 #0a0a0a,4px 4px 0 #0a0a0a,5px 5px 0 #0a0a0a,6px 6px 0 #0a0a0a,7px 7px 0 #0a0a0a,8px 8px 0 #0a0a0a,10px 10px 4px rgba(0,0,0,.55);line-height:0.95;opacity:0;filter:blur(4px) url(#ttt-rough);transition:transform 1.15s cubic-bezier(.34,1.56,.64,1),opacity .6s ease,filter .6s ease;}',
      '.ttt-L.on{opacity:1!important;filter:blur(0) url(#ttt-rough)!important;transform:translate(0,0) rotate(0deg)!important;}',
      '.ttt-L.glow{animation:tttGlow 2.3s ease-in-out infinite;}',
      '@keyframes tttGlow{0%,100%{color:#FFE033;-webkit-text-stroke:8px #0a0a0a;text-shadow:1px 1px 0 #0a0a0a,2px 2px 0 #0a0a0a,3px 3px 0 #0a0a0a,4px 4px 0 #0a0a0a,5px 5px 0 #0a0a0a,6px 6px 0 #0a0a0a,7px 7px 0 #0a0a0a,8px 8px 0 #0a0a0a,10px 10px 4px rgba(0,0,0,.55)}40%{color:#FFFFF0;-webkit-text-stroke:7px #00DFFF;text-shadow:1px 1px 0 #0a0a0a,2px 2px 0 #0a0a0a,3px 3px 0 #0a0a0a,0 0 16px #00DFFF,0 0 42px rgba(0,191,255,.9),0 0 85px rgba(0,191,255,.6),0 0 145px rgba(0,120,255,.35)}75%{color:#FFE033;-webkit-text-stroke:7px rgba(0,191,255,.5);text-shadow:1px 1px 0 #0a0a0a,2px 2px 0 #0a0a0a,0 0 12px rgba(0,191,255,.55),0 0 28px rgba(0,191,255,.35)}}' ,
      '#ttt-sub{font-family:"Inter",sans-serif;font-weight:900;font-size:clamp(10px,1.5vw,19px);color:rgba(255,255,255,.9);letter-spacing:7px;text-transform:uppercase;opacity:0;transform:translateY(20px);transition:opacity .8s ease 3.2s,transform .8s ease 3.2s;margin-top:20px;text-shadow:0 0 20px rgba(0,191,255,.7);}',
      '#ttt-sub.show{opacity:1;transform:translateY(0);}',
      '#ttt-bw{width:clamp(160px,36vw,460px);height:4px;background:rgba(255,255,255,.15);border-radius:4px;margin-top:24px;overflow:hidden;opacity:0;transition:opacity .4s ease 3.4s;}',
      '#ttt-bw.show{opacity:1;}',
      '#ttt-bi{height:100%;width:0;background:linear-gradient(90deg,#00DFFF,#FFD700,#00DFFF);background-size:200% 100%;border-radius:4px;transition:width 2s cubic-bezier(.4,0,.2,1) 3.5s;animation:tttBar 1.4s linear infinite;}',
      '#ttt-bi.full{width:100%;}',
      '@keyframes tttBar{0%{background-position:200% 0}100%{background-position:-200% 0}}',
      '@keyframes tttOut{0%{opacity:1;pointer-events:all}100%{opacity:0;pointer-events:none;visibility:hidden}}'
    ].join('');
    D.head.appendChild(st2);
  }

  /* ---- SVG Distortion Filter (DUDE rough-paint texture) ---- */
  if(!D.getElementById('ttt-svg-filter')){
    var svgEl=D.createElementNS('http://www.w3.org/2000/svg','svg');
    svgEl.id='ttt-svg-filter';
    svgEl.setAttribute('style','position:absolute;width:0;height:0;overflow:hidden;');
    svgEl.innerHTML='<defs>'
      +'<filter id="ttt-rough" x="-8%" y="-8%" width="116%" height="116%">'
      +'<feTurbulence type="fractalNoise" baseFrequency="0.028 0.055" numOctaves="3" seed="2" result="noise"/>'
      +'<feDisplacementMap in="SourceGraphic" in2="noise" scale="3.5" xChannelSelector="R" yChannelSelector="G"/>'
      +'</filter>'
      +'<filter id="ttt-rough-strong" x="-10%" y="-10%" width="120%" height="120%">'
      +'<feTurbulence type="fractalNoise" baseFrequency="0.04 0.08" numOctaves="4" seed="5" result="noise2"/>'
      +'<feDisplacementMap in="SourceGraphic" in2="noise2" scale="5" xChannelSelector="R" yChannelSelector="G"/>'
      +'</filter>'
      +'</defs>';
    D.body.appendChild(svgEl);
  }

  /* ---- Build DOM ---- */
  if(D.getElementById('ttt-splash')) return;
  var splash=D.createElement('div'); splash.id='ttt-splash';
  var dolly=D.createElement('div'); dolly.id='ttt-dolly';
  var rowsEl=D.createElement('div'); rowsEl.id='ttt-rows';
  var sub=D.createElement('div'); sub.id='ttt-sub';
  sub.innerHTML='KFGSC, Tiptur &nbsp;&bull;&nbsp; BCA Data Science &nbsp;&bull;&nbsp; 2024&ndash;2027';
  var bw=D.createElement('div'); bw.id='ttt-bw';
  var bi=D.createElement('div'); bi.id='ttt-bi';
  bw.appendChild(bi);
  dolly.appendChild(rowsEl); dolly.appendChild(sub); dolly.appendChild(bw);
  splash.appendChild(dolly);
  D.body.appendChild(splash);

  /* ---- Particles ---- */
  for(var k=0;k<65;k++){
    var dp=D.createElement('div'); dp.className='ttt-dust';
    var sz=Math.random()*5+2;
    dp.style.cssText='width:'+sz+'px;height:'+sz+'px;left:'+(Math.random()*100)+'%;top:'+(55+Math.random()*55)+'%;'
      +'--dx:'+((Math.random()-.5)*140)+'px;'
      +'animation-duration:'+(3+Math.random()*6)+'s;'
      +'animation-delay:'+(Math.random()*4)+'s;'
      +'opacity:'+(0.25+Math.random()*0.6)+';';
    splash.appendChild(dp);
  }

  /* ---- Letters ---- */
  var lines=[['TEAM','TECH'],['TITANS']];
  var all=[]; var vw=W.innerWidth; var vh=W.innerHeight;
  lines.forEach(function(words){
    var row=D.createElement('div'); row.className='ttt-row';
    words.forEach(function(word,wi){
      if(wi>0){var spc=D.createElement('span');spc.className='ttt-sp';row.appendChild(spc);}
      word.split('').forEach(function(ch){
        var s=D.createElement('span'); s.className='ttt-L'; s.textContent=ch;
        var tx=(Math.random()-.5)*vw*0.86;
        var ty=(Math.random()-.5)*vh*0.86;
        var rot=(Math.random()-.5)*140;
        s.style.transform='translate('+tx+'px,'+ty+'px) rotate('+rot+'deg)';
        row.appendChild(s); all.push(s);
      });
    });
    rowsEl.appendChild(row);
  });

  /* Phase 1 — float reveal (150ms) */
  setTimeout(function(){
    all.forEach(function(l,i){setTimeout(function(){l.style.opacity='0.5';l.style.filter='blur(2px)';},i*42);});
  },150);

  /* Phase 2 — gravity snap (1.1s) */
  setTimeout(function(){
    all.forEach(function(l,i){setTimeout(function(){l.classList.add('on');},i*36);});
  },1100);

  /* Phase 3 — blue glow + sub-text (2.9s) */
  setTimeout(function(){
    all.forEach(function(l,i){setTimeout(function(){l.classList.add('glow');},i*30);});
    sub.classList.add('show');
  },2900);

  /* Phase 4 — loading bar (3.4s) */
  setTimeout(function(){
    bw.classList.add('show');
    setTimeout(function(){bi.classList.add('full');},60);
  },3400);

  /* Phase 5 — cleanup (7s) */
  setTimeout(function(){
    var el=D.getElementById('ttt-splash');
    var es=D.getElementById('ttt-style');
    if(el) el.remove();
    if(es) es.remove();
  },7000);
})();
</script>
""", height=0)



# --- Custom CSS (Theme-Agnostic with Variable Colors) ---
st.markdown("""
<div class="fixed-bg"></div>
<style>
    /* Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Rotating California Background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: transparent !important; 
    }
    
    /* 
     * Increased opacity for permanent Dark mode 
     */
    div.fixed-bg {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: -999;
        animation: californiaSlideshow 24s infinite;
        background-size: cover;
        background-position: center;
        opacity: 0.28; 
    }

    @keyframes californiaSlideshow {
        0%, 100% { background-image: url('https://images.unsplash.com/photo-1542223616-740d5dff7f56?auto=format&fit=crop&w=1920&q=80'); }
        25%  { background-image: url('https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=1920&q=80'); }
        50%  { background-image: url('https://images.unsplash.com/photo-1501594907352-04cda38ebc29?auto=format&fit=crop&w=1920&q=80'); }
        75%  { background-image: url('https://images.unsplash.com/photo-1440847899694-90043f91c7f9?auto=format&fit=crop&w=1920&q=80'); }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color) !important;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Standard Cards */
    .card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); 
        margin-bottom: 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        color: var(--text-color);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); 
        flex: 1;
        text-align: left;
        border-top: 4px solid #f8cb46; /* Blinkit Yellow */
        position: relative;
        color: var(--text-color);
    }
    .metric-card-green {
        border-top: 4px solid #0c831f; /* Blinkit Green */
    }
    .metric-card-blue {
        border-top: 4px solid var(--primary-color);
    }
    .metric-card h3 {
        margin: 5px 0 0 0;
        font-size: 1.8em;
        color: var(--text-color);
    }
    .metric-card p {
        margin: 0;
        font-size: 0.9em;
        opacity: 0.7;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    /* Universal Titles/Text Adapter */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0c831f; /* Blinkit Green */
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: 0.2s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        background-color: #0a6918;
        color: white;
        transform: scale(1.02);
    }
    
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* Prediction Output Box */
    .output-card {
        background: var(--secondary-background-color);
        border-left: 6px solid #0c831f;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 15px;
        color: var(--text-color);
    }
    .output-card-val-usd {
        color: var(--text-color);
        font-size: 2.4em; 
        font-weight: bold;
        margin-bottom: 0px;
    }
    .output-card-val-inr {
        color: #0c831f; 
        font-size: 2em; 
        font-weight: bold;
        margin-top: 5px;
    }
    
    /* ⚡ Custom Spark / Glow Animation for Sidebar Logo */
    @keyframes sparkGlowPulse {
        0% { box-shadow: 0 0 10px 0px rgba(0, 191, 255, 0.5); filter: drop-shadow(0 0 5px rgba(0,191,255,0.6)); }
        15% { box-shadow: 0 0 25px 10px rgba(0, 191, 255, 0.8), 0 0 5px 2px white; filter: drop-shadow(0 0 15px rgba(0,255,255,0.9)) brightness(1.1); transform: scale(1.01); }
        30% { box-shadow: 0 0 15px 5px rgba(0, 191, 255, 0.6); filter: drop-shadow(0 0 8px rgba(0,191,255,0.7)); transform: scale(1); }
        60% { box-shadow: 0 0 45px 20px rgba(0, 191, 255, 0.9), 0 0 90px 30px rgba(0, 255, 255, 0.5); filter: drop-shadow(0 0 25px rgba(0,255,255,1)) brightness(1.2); transform: scale(1.02); }
        100% { box-shadow: 0 0 10px 0px rgba(0, 191, 255, 0.5); filter: drop-shadow(0 0 5px rgba(0,191,255,0.6)); transform: scale(1); }
    }
    
    [data-testid="stSidebar"] div[data-testid="stImage"] img {
        border-radius: 50%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        animation: sparkGlowPulse 3s infinite;
        border: 2px solid rgba(0, 191, 255, 0.5);
        max-width: 160px !important;
        margin: 0 auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Function to trigger diwali firework blast using canvas-confetti
def diwali_blast():
    fireworks_code = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
    var duration = 5 * 1000;
    var animationEnd = Date.now() + duration;
    var defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 999999 };

    function randomInRange(min, max) {
      return Math.random() * (max - min) + min;
    }

    var interval = setInterval(function() {
      var timeLeft = animationEnd - Date.now();

      if (timeLeft <= 0) {
        return clearInterval(interval);
      }

      var particleCount = 50 * (timeLeft / duration);
      confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 }, colors: ['#ff0000', '#00ff00', '#f8cb46', '#ffffff'] }));
      confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 }, colors: ['#ff0000', '#0c831f', '#f8cb46', '#ffffff'] }));
    }, 250);
    </script>
    """
    components.html(fireworks_code, height=0)

# --- Constants & Initialization ---
@st.cache_resource
def load_model():
    model_path = os.path.join(SCRIPT_DIR, 'model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    cache_path = os.path.join(SCRIPT_DIR, 'dataset_cache.pkl')
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    return None

model = load_model()
df = load_data()

# --- SIDEBAR NAV ---
st.sidebar.image(os.path.join(SCRIPT_DIR, "custom_user_logo.jpg"), use_container_width=True)
st.sidebar.markdown("---")

menu = {
    "🏠 Master Dashboard": "dashboard",
    "⚡ Live Predictor": "predict",
    "📊 Dataset Center": "dataset",
    "🤖 Models & ML": "model",
    "ℹ️ Team Overview": "about"
}
choice = st.sidebar.radio("MAIN MENU", list(menu.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Quick Actions**")
if st.sidebar.button("🔄 Sync Database"):
    st.sidebar.success("Database synced successfully!")

# --- ROUTER ---

if menu[choice] == "dashboard":
    st.markdown("<p style='text-align: center; font-size: 0.85em; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0px;'>KFGSC, Tiptur - 572201 • BCA (Data Science) 2024-2027 • Machine Learning Academic Project Work • 4th semester</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-weight: 900; margin-top: 10px; margin-bottom: 0px;'>California House Price Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; opacity: 0.7; font-weight: 400; margin-top: 5px; margin-bottom: 30px;'>House price prediction using Regression</h4>", unsafe_allow_html=True)
    
    if df is not None:
        total_listings = len(df)
        avg_price = df['median_house_value'].mean()
        max_price = df['median_house_value'].max()
        top_ocean = df['ocean_proximity'].mode()[0]
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card"><p>Total Listings 🏢</p><h3>{total_listings:,}</h3></div>
            <div class="metric-card metric-card-green"><p>Avg Valuation 💵</p><h3>${avg_price:,.0f}</h3></div>
            <div class="metric-card metric-card-blue"><p>Max Property 💎</p><h3>${max_price:,.0f}</h3></div>
            <div class="metric-card"><p>Top Region 🏖️</p><h3>{top_ocean}</h3></div>
        </div>
        """, unsafe_allow_html=True)
        
        # When plotting with Seaborn/Matplotlib during dark mode, transparent backgrounds are best
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("#### 📈 Price Distribution Overview")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_alpha(0.0) # Transparent chart background
            ax.patch.set_alpha(0.0)
            sns.histplot(df['median_house_value'], bins=50, kde=True, color='#f8cb46', ax=ax)
            ax.set_xlabel("Price in USD")
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')
            ax.tick_params(colors='gray')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("#### 🗺️ Demographic Income vs Price")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_alpha(0.0) # Transparent chart background
            ax2.patch.set_alpha(0.0)
            sns.scatterplot(x=df['median_income'], y=df['median_house_value'], alpha=0.3, color='#0c831f', ax=ax2)
            ax2.set_xlabel("Median Income")
            ax2.set_ylabel("House Value")
            ax2.xaxis.label.set_color('gray')
            ax2.yaxis.label.set_color('gray')
            ax2.tick_params(colors='gray')
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Dataset not found. Please place dataset in correct path.")

elif menu[choice] == "predict":
    st.markdown("<h2>⚡ Lightning Fast Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Configure the parameters below to compute an accurate real-time valuation of the property.")
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🗺️ Location")
        longitude = st.number_input("Longitude", value=-122.23, format="%.2f", step=0.1)
        latitude = st.number_input("Latitude", value=37.88, format="%.2f", step=0.1)
        ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])
        
    with col2:
        st.markdown("#### 🏠 Property details")
        housing_median_age = st.slider("Median Age (Years)", min_value=1, max_value=100, value=30)
        total_rooms = st.number_input("Total Rooms", min_value=1, value=500, step=10)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=100, step=5)
        
    with col3:
        st.markdown("#### 👥 Demographics")
        population = st.number_input("Population", min_value=1, value=300, step=10)
        households = st.number_input("Households", min_value=1, value=100, step=5)
        median_income = st.slider("Median Income (x$10,000)", min_value=0.0, max_value=15.0, value=5.0, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🚀 Predict Market Value")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_button:
        if model is not None:
            input_data = pd.DataFrame([{
                'longitude': longitude,
                'latitude': latitude,
                'housing_median_age': housing_median_age,
                'total_rooms': total_rooms,
                'total_bedrooms': total_bedrooms,
                'population': population,
                'households': households,
                'median_income': median_income,
                'ocean_proximity': ocean_proximity
            }])
            
            with st.spinner('Accessing ML Engine...'):
                prediction = model.predict(input_data)[0]
                pred_inr = prediction * EXCHANGE_RATE
            
            # Fire the Diwali Blast animation
            diwali_blast()
            
            st.markdown(
                f"""
                <div class="output-card">
                    <p style="opacity: 0.7; font-weight:bold; letter-spacing: 1px; text-transform: uppercase;">Final Assessment Ready</p>
                    <p class="output-card-val-usd">${prediction:,.2f} USD</p>
                    <p class="output-card-val-inr">₹ {pred_inr:,.2f} INR</p>
                    <p style="opacity: 0.5; font-size: 0.9em;">Conversion Rate applied: 1 USD = {EXCHANGE_RATE} INR 📈</p>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Prepare CSV report
            report_df = input_data.copy()
            report_df['Predicted_Price_USD'] = round(prediction, 2)
            report_df['Predicted_Price_INR'] = round(pred_inr, 2)
            csv_data = report_df.to_csv(index=False).encode('utf-8')
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Prediction Report (CSV)",
                data=csv_data,
                file_name="house_price_prediction_report.csv",
                mime="text/csv",
            )
            
        else:
            st.error("Model engine is currently offline. Please run the model training sequence.")

elif menu[choice] == "dataset":
    st.markdown("<h2>📊 Dataset Control Center</h2>", unsafe_allow_html=True)
    if df is not None:
        col1, col2 = st.columns([1, 1])
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 📂 Raw Data Browser")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 📉 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Data source is currently unreachable.")

elif menu[choice] == "model":
    st.markdown("<h2>🤖 Machine Learning Core</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <h3 style="color: #f8cb46 !important;">Model Specifications</h3>
            <p>Our system uses an advanced <b>Random Forest Regressor</b> ensemble algorithm trained across 20,000+ data points.</p>
            <ul>
                <li><b>Framework:</b> Scikit-Learn</li>
                <li><b>Pipeline:</b> Automatic Null-Imputation, Feature Scaling (StandardScaler), Categorical OneHotEncoding</li>
                <li><b>Performance:</b> Non-linear logic capable of interpreting complex real-estate market patterns dynamically.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="card">
            <h3 style="color: #0c831f !important;">System Features</h3>
            <button style="background-color: var(--background-color); color: var(--text-color); border: 1px solid rgba(128,128,128,0.3); border-radius: 5px; padding: 5px 10px; margin-right: 10px;">Robust against outliers</button>
            <button style="background-color: var(--background-color); color: var(--text-color); border: 1px solid rgba(128,128,128,0.3); border-radius: 5px; padding: 5px 10px; margin-right: 10px;">High accuracy</button>
            <button style="background-color: var(--background-color); color: var(--text-color); border: 1px solid rgba(128,128,128,0.3); border-radius: 5px; padding: 5px 10px;">Real-time validation</button>
        </div>
        """, unsafe_allow_html=True
    )

elif menu[choice] == "about":
    st.markdown("<h2>ℹ️ Management & Administration</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card" style="border-top: 4px solid #0071E3;">
            <h3>Academic Division: Team Tech Titans</h3>
            <p><b>Course:</b> BCA (Data Science)</p>
            <p><b>Institute:</b> Kalpataru First Grade Science College Tiptur</p>
            <hr style="border-color:rgba(128,128,128,0.2);">
            <h4>👨‍💻 Core Committee:</h4>
            <ul>
                <li><b>Deepu . B</b> (Team Leader)</li>
                <li><b>Bhaskar N . S</b></li>
                <li><b>Thrupthi K S</b></li>
                <li><b>Sandhya T M</b></li>
            </ul>
            <hr style="border-color:rgba(128,128,128,0.2);">
            <h4>🎓 Academic Guidance:</h4>
            <p><b>Subject Teacher:</b> Ravikiran S J, MCA, SOC Analyst</p>
            <p><b>Technical Advisor:</b> Omkar K P, MCA, Python Developer</p>
            <hr style="border-color:rgba(128,128,128,0.2);">
            <h4>💻 Software & Technologies:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px;">
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="24" height="24" alt="Python">
                    <b style="color: var(--text-color);">Python</b>
                </div>
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/RStudio_logo_flat.svg" width="24" height="24" alt="R Studio">
                    <b style="color: var(--text-color);">R Studio</b>
                </div>
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Spyder_logo.svg" width="24" height="24" alt="Spyder">
                    <b style="color: var(--text-color);">Spyder</b>
                </div>
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" width="24" height="24" alt="Jupyter">
                    <b style="color: var(--text-color);">Jupyter Notebook</b>
                </div>
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="24" height="24" alt="Google Colab">
                    <b style="color: var(--text-color);">Google Colab</b>
                </div>
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/kaggle/kaggle-original.svg" width="24" height="24" alt="Kaggle">
                    <b style="color: var(--text-color);">Kaggle</b>
                </div>
                <div style="background-color: var(--background-color); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px;">🤖</span>
                    <b style="color: var(--text-color);">Cursor AI</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )
