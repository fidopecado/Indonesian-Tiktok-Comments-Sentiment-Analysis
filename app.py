import os
import random
import re
import urllib.parse
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Setup NLTK stopwords secara aman untuk cloud runtime
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

setup_nltk()
list_stopwords = stopwords.words('indonesian')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. THEME DEFINITION (KDMP Prabowo)
# ==========================================
current_theme = {
    "sidebar_bg": "#C8102E",
    "sidebar_text": "#FFFFFF",
    "main_bg": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text_color": "#0F172A",
    "input_bg": "#FFFFFF",
    "input_text": "#0F172A",
    "Positif": "#16A34A",
    "Netral": "#D97706",
    "Negatif": "#DC2626",
    "accent": "#C8102E"
}

# ==========================================
# 3. SIDEBAR — BRANDING BLOCK
# ==========================================
st.sidebar.markdown("""
<div style="
    padding: 18px 16px 14px 16px;
    margin-bottom: 4px;
    text-align: center;
">
    <div style="font-size: 22px; font-weight: 900; letter-spacing: -0.5px; line-height: 1.2; color: #F8FAFC;">
        📊 Sentiment<br>Analysis
    </div>
    <div style="font-size: 11px; opacity: 1; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; color: #0F172A;">
        TikTok · Indonesia
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin: 0 0 12px 0; opacity:0.3;'>", unsafe_allow_html=True)

# ==========================================
# 5. SIDEBAR — DATASET SELECTOR
# ==========================================
st.sidebar.markdown("**📂 Dataset**", unsafe_allow_html=False)

dataset_mapping = {
    "Koperasi Desa Merah Putih (KDMP)": os.path.join(BASE_DIR, "KDMP.csv"),
    "Makan Bergizi Gratis (MBG)": os.path.join(BASE_DIR, "MBG_PROCESSED.csv")
}

selected_project = st.sidebar.selectbox(
    "Pilih topik:",
    options=list(dataset_mapping.keys()),
    label_visibility="visible"
)

# ==========================================
# 6. LOAD DATA
# ==========================================
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['create_time'] = pd.to_datetime(df['create_time'])
    df['date_only'] = df['create_time'].dt.date

    if 'is_empty_after_cleaning' in df.columns:
        df = df[df['is_empty_after_cleaning'] == False]

    if 'final_sentiment_voting' in df.columns:
        df = df[df['final_sentiment_voting'].notna() & (df['final_sentiment_voting'] != 'empty')]
    return df

df = load_data(dataset_mapping[selected_project])

st.sidebar.markdown("<hr style='margin: 12px 0; opacity:0.3;'>", unsafe_allow_html=True)

# ==========================================
# 7. SIDEBAR — FILTERS
# ==========================================
st.sidebar.markdown("**🔍 Filter Data**", unsafe_allow_html=False)

min_date_dataset = df['date_only'].min()
max_date_dataset = df['date_only'].max()
video_list = sorted(df['video_id'].dropna().unique())

date_range = st.sidebar.date_input(
    "Periode komentar:",
    value=(min_date_dataset, max_date_dataset),
    min_value=min_date_dataset,
    max_value=max_date_dataset
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_range[0], max_date_dataset

selected_videos = st.sidebar.multiselect(
    "Filter by Video ID (opsional):",
    options=video_list,
    default=[],
    placeholder="Semua video"
)

st.sidebar.markdown("<hr style='margin: 12px 0; opacity:0.3;'>", unsafe_allow_html=True)

# ==========================================
# 8. SIDEBAR — MODEL SELECTOR
# ==========================================
st.sidebar.markdown("**🤖 Model**", unsafe_allow_html=False)

model_mapping = {
    "Final Voting (Ensemble)": "final_sentiment_voting",
    "HuggingFace: taufiqdp": "sentiment",
    "AI: Aardiiiiy": "aardiiiiy_sentiment",
    "AI: Agufsamudra": "agufsamudra_sentiment",
    "AI: Mdhugol": "mdhugol_sentiment"
}

selected_model_name = st.sidebar.selectbox(
    "Sentiment model:",
    options=list(model_mapping.keys()),
    index=0
)
active_column = model_mapping[selected_model_name]

# ==========================================
# 9. DATA PROCESSING
# ==========================================
mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
if selected_videos:
    mask = mask & (df['video_id'].isin(selected_videos))

filtered_df = df[mask]
clean_filtered_df = filtered_df[
    filtered_df[active_column].notna() & (filtered_df[active_column] != 'empty')
]
kpi_stats = clean_filtered_df[active_column].value_counts()

# ==========================================
# 10. SIDEBAR — ACTIVE FILTER BADGE
# ==========================================
st.sidebar.markdown("<hr style='margin: 12px 0; opacity:0.3;'>", unsafe_allow_html=True)

video_label = f"{len(selected_videos)} video dipilih" if selected_videos else "Semua video"
comment_count = len(clean_filtered_df)

st.sidebar.markdown(f"""
<div style="
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12px;
    line-height: 1.7;
">
    <div style="color: #F8FAFC; font-weight: 800; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; opacity: 1; margin-bottom: 6px;">
        ✅ Filter Aktif
    </div>
    <div style="color: #F8FAFC;">📅 {start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}</div>
    <div style="color: #F8FAFC;">🎬 {video_label}</div>
    <div style="color: #F8FAFC;">🖥️ {selected_model_name}</div>
    <div style="margin-top:6px; font-weight:700; color: #F8FAFC;">💬 {comment_count:,} comments loaded</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 11. THEME VARIABLES & CSS INJECTION
# ==========================================
THEME_BG = current_theme["main_bg"]
THEME_TEXT = current_theme["text_color"]
THEME_CARD_BG = current_theme["card_bg"]
PLOT_FONT_COLOR = current_theme["text_color"]

color_map = {
    'positif': current_theme['Positif'],
    'netral': current_theme['Netral'],
    'negatif': current_theme['Negatif']
}

st.markdown(f"""
<style>
    :root {{
        --background-color: {THEME_BG} !important;
        --secondary-background-color: {THEME_CARD_BG} !important;
        --text-color: {THEME_TEXT} !important;
        --primary-color: {current_theme['accent']} !important;
    }}

    [data-testid="stMainMenu"] {{
        visibility: hidden !important;
    }}

    /* Main background & text */
    [data-testid="stAppViewContainer"] {{
        background-color: {THEME_BG} !important;
        color: {THEME_TEXT} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}

    /* Sidebar Background */
    [data-testid="stSidebar"] {{
        background-color: {current_theme['sidebar_bg']} !important;
    }}

    /* Sidebar Headings, Labels, Markdown Text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown b {{
        color: {current_theme['sidebar_text']} !important;
    }}

    /* Buttons in Sidebar (Only st.button widgets, not BaseWeb internal controls) */
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {current_theme['input_bg']} !important;
        color: {current_theme['input_text']} !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }}

    [data-testid="stSidebar"] .stButton > button * {{
        color: {current_theme['input_text']} !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {current_theme['accent']} !important;
        color: #ffffff !important;
        border-color: {current_theme['accent']} !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover * {{
        color: #ffffff !important;
    }}

    /* Clean, Symmetric Sidebar Input Controls & Selectboxes */
    [data-testid="stSidebar"] div[data-baseweb="input"],
    [data-testid="stSidebar"] div[data-baseweb="base-input"],
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: {current_theme['input_bg']} !important;
        color: {current_theme['input_text']} !important;
        border-radius: 8px !important;
        min-height: 42px !important;
        border: 1px solid rgba(15, 23, 42, 0.18) !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
    [data-testid="stSidebar"] div[data-baseweb="input"]:hover,
    [data-testid="stSidebar"] div[data-baseweb="base-input"]:hover {{
        border-color: {current_theme['accent']} !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="input"] input,
    [data-testid="stSidebar"] div[data-baseweb="base-input"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] input {{
        color: {current_theme['input_text']} !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }}

    /* Dropdown Clear-All and Arrow Icons Reset (Remove floating white boxes) */
    [data-testid="stSidebar"] div[data-baseweb="select"] button,
    [data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="select"] svg,
    [data-testid="stSidebar"] div[data-baseweb="input"] svg {{
        fill: {current_theme['input_text']} !important;
        color: {current_theme['input_text']} !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div:last-child {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding-right: 6px !important;
        gap: 2px !important;
    }}

    /* Aesthetic Multiselect Tag / Pill Styling & Tag X Button */
    [data-testid="stSidebar"] span[data-baseweb="tag"],
    [data-testid="stSidebar"] div[data-baseweb="tag"] {{
        background-color: {current_theme['accent']} !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 3px 10px !important;
        margin: 2px !important;
        height: 26px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
    }}

    [data-testid="stSidebar"] span[data-baseweb="tag"] *,
    [data-testid="stSidebar"] div[data-baseweb="tag"] * {{
        color: #ffffff !important;
        fill: #ffffff !important;
        background: transparent !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        box-shadow: none !important;
        border: none !important;
    }}

    [data-testid="stSidebar"] span[data-baseweb="tag"] [role="button"],
    [data-testid="stSidebar"] div[data-baseweb="tag"] [role="button"],
    [data-testid="stSidebar"] span[data-baseweb="tag"] svg,
    [data-testid="stSidebar"] div[data-baseweb="tag"] svg {{
        cursor: pointer !important;
        opacity: 0.85 !important;
        transition: transform 0.15s ease, opacity 0.15s ease !important;
    }}

    [data-testid="stSidebar"] span[data-baseweb="tag"] [role="button"]:hover,
    [data-testid="stSidebar"] div[data-baseweb="tag"] [role="button"]:hover,
    [data-testid="stSidebar"] span[data-baseweb="tag"] svg:hover,
    [data-testid="stSidebar"] div[data-baseweb="tag"] svg:hover {{
        opacity: 1 !important;
        transform: scale(1.15) !important;
    }}

    /* Calendar & Date Picker Popover Dropdown Fix */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="calendar"],
    div[data-baseweb="calendar"] *,
    div[role="dialog"],
    div[role="dialog"] * {{
        background-color: #ffffff !important;
        color: #0f172a !important;
    }}

    div[data-baseweb="calendar"] button[aria-selected="true"],
    div[data-baseweb="calendar"] button[aria-selected="true"] * {{
        background-color: {current_theme['accent']} !important;
        color: #ffffff !important;
    }}

    /* Typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .responsive-title {{
        font-size: clamp(22px, 3.5vw, 38px) !important;
        font-weight: 900;
        line-height: 1.2;
        color: {THEME_TEXT} !important;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }}
    .responsive-caption {{
        font-size: clamp(12px, 1.2vw, 15px) !important;
        color: {THEME_TEXT} !important;
        opacity: 0.7;
    }}

    hr {{
        border-top: 2px solid {THEME_TEXT} !important;
        opacity: 0.1;
    }}

    /* Section headers */
    .section-header {{
        font-size: clamp(16px, 1.8vw, 22px);
        font-weight: 800;
        color: {THEME_TEXT};
        margin: 4px 0 12px 0;
        letter-spacing: -0.3px;
    }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 12. KPI CARD COMPONENT
# ==========================================
def create_kpi_card(title, value, color_top_border):
    st.markdown(f"""
    <div style="
        background-color: {THEME_CARD_BG};
        padding: clamp(10px, 1.5vw, 20px);
        border-radius: 12px;
        border-top: 5px solid {color_top_border};
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    ">
        <p style="
            font-size: clamp(11px, 1.1vw, 13px);
            color: {THEME_TEXT};
            margin: 0;
            font-weight: 700;
            text-transform: uppercase;
            opacity: 0.7;
            letter-spacing: 0.8px;
            word-wrap: break-word;
        ">{title}</p>
        <p style="
            font-size: clamp(20px, 2.3vw, 34px);
            color: {THEME_TEXT};
            margin: 6px 0 0 0;
            font-weight: 800;
            line-height: 1.1;
            word-wrap: break-word;
        ">{value}</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 13. SECTION 1 — HEADER & KPI CARDS
# ==========================================
st.markdown(f'<p class="responsive-title">📊 Sentiment Dashboard — {selected_project}</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="responsive-caption">Jelajahi opini publik TikTok secara interaktif, dianalisis menggunakan model <b>{selected_model_name}</b>.</p>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# Row 1: General overview
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    create_kpi_card("💬 Total Comments", f"{len(clean_filtered_df):,}", current_theme['Negatif'])
with row1_col2:
    create_kpi_card("📹 Videos Analyzed", f"{clean_filtered_df['video_id'].nunique()}", current_theme['accent'])
with row1_col3:
    create_kpi_card("👥 Unique Users", f"{clean_filtered_df['username'].nunique():,}", current_theme['accent'])

# Row 2: Sentiment stats
row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    create_kpi_card("🟢 Positive", f"{kpi_stats.get('positif', 0):,}", current_theme['Positif'])
with row2_col2:
    create_kpi_card("🟡 Neutral", f"{kpi_stats.get('netral', 0):,}", current_theme['Netral'])
with row2_col3:
    create_kpi_card("🔴 Negative", f"{kpi_stats.get('negatif', 0):,}", current_theme['Negatif'])

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 14. SECTION 2 — SENTIMENT DISTRIBUTION CHARTS
# ==========================================
st.markdown(f'<p class="section-header">📈 Sentiment Distribution — {selected_model_name}</p>', unsafe_allow_html=True)

chart_data = clean_filtered_df[active_column].value_counts().reset_index()
chart_data.columns = ['Sentimen', 'Total']

col_left, col_right = st.columns(2)

with col_left:
    fig_pie = px.pie(
        chart_data, values='Total', names='Sentimen', hole=0.5,
        color='Sentimen', color_discrete_map=color_map,
        title="Sentiment Breakdown"
    )
    fig_pie.update_traces(
        textposition='inside', textinfo='percent+label',
        hovertemplate="<b>Kategori:</b> %{label}<br><b>Porsi:</b> %{percent}<br><b>Jumlah:</b> %{value:,} comments<extra></extra>"
    )
    fig_pie.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13),
        title_font=dict(color=PLOT_FONT_COLOR),
        legend=dict(font=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_pie, width='stretch')

with col_right:
    fig_bar = px.bar(
        chart_data, x='Sentimen', y='Total',
        color='Sentimen', color_discrete_map=color_map,
        title="Comment Volume by Sentiment"
    )
    fig_bar.update_traces(
        hovertemplate="<b>Sentiment:</b> %{x}<br><b>Volume:</b> %{y:,} comments<extra></extra>"
    )
    fig_bar.update_layout(
        xaxis_title="Sentiment", yaxis_title="Total Comments",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13),
        title_font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        legend=dict(font=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_bar, width='stretch')

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 15. SECTION 3 — MODEL BENCHMARKING
# ==========================================
st.markdown(f'<p class="section-header">🔄 Model Benchmarking — Semua Model Dibandingkan</p>', unsafe_allow_html=True)

comparison_list = []
for m_name, m_col in model_mapping.items():
    clean_m_df = filtered_df[filtered_df[m_col].notna() & (filtered_df[m_col] != 'empty')]
    m_counts = clean_m_df[m_col].value_counts()
    for sentiment_type in ['positif', 'netral', 'negatif']:
        comparison_list.append({
            'Model': m_name,
            'Sentiment': sentiment_type,
            'Count': m_counts.get(sentiment_type, 0)
        })

comp_df = pd.DataFrame(comparison_list)
fig_comp = px.bar(
    comp_df, x='Model', y='Count', color='Sentiment',
    barmode='group', color_discrete_map=color_map,
    title="Sentiment Distribution Across All Models"
)
fig_comp.update_traces(
    hovertemplate="<b>Model:</b> %{x}<br><b>Sentiment:</b> %{hovertext}<br><b>Count:</b> %{y:,} comments<extra></extra>",
    hovertext=comp_df['Sentiment']
)
fig_comp.update_layout(
    xaxis_title="Model", yaxis_title="Comment Count", legend_title="Sentiment",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color=PLOT_FONT_COLOR, size=13),
    title_font=dict(color=PLOT_FONT_COLOR),
    xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
    yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
    legend=dict(font=dict(color=PLOT_FONT_COLOR))
)
st.plotly_chart(fig_comp, width='stretch')

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 16. TEXT PREPROCESSING
# ==========================================
df_wordcloud = clean_filtered_df.copy()
df_wordcloud['clean_comment'] = df_wordcloud['clean_comment'].fillna('').astype(str)

text_corpus = ' '.join(df_wordcloud['clean_comment'].tolist())

custom_stopwords = [
    'bos', 'bang', 'kak', 'gan', 'min', 'om', 'bro', 'guys',
    'sih', 'ya', 'nih', 'ni', 'dong', 'deh', 'lah', 'kok',
    'yg', 'utk', 'dgn', 'ga', 'gak', 'nggak', 'ngga', 'nya',
    'aja', 'biar', 'gue', 'gw', 'gua', 'klo', 'kalo', 'kalau',
    'udh', 'udah', 'sdh', 'sudah', 'tdk', 'tak', 'tp', 'tapi',
    'krn', 'karna', 'karena', 'dr', 'dari', 'ke', 'di',
    'sticker', 'stiker', 'stickers', 'stikers',
    'koperasi', 'desa', 'merah', 'putih', 'kdmp', 'kopdes',
    'makan', 'gratis', 'bergizi', 'program', 'mbg'
]
list_stopwords.extend(custom_stopwords)
list_stopwords = set(list_stopwords)

words = re.findall(r'\b\w+\b', text_corpus.lower())
filtered_words = [
    word for word in words
    if word not in list_stopwords
    and not word.isdigit()
    and len(word) > 2
]
word_counts = Counter(filtered_words)


# ==========================================
# 17. SECTION 4A — WORD CLOUD
# ==========================================
st.markdown('<p class="section-header">☁️ Word Cloud — Topik yang Sering Muncul</p>', unsafe_allow_html=True)

if filtered_words:
    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return random.choice([current_theme['text_color'], current_theme['Positif'], current_theme['Negatif']])

    wordcloud = WordCloud(
        width=1200,
        height=500,
        background_color=current_theme['card_bg'],
        collocations=False,
        color_func=custom_color_func
    ).generate_from_frequencies(word_counts)

    fig_wc, ax_wc = plt.subplots(figsize=(14, 6))
    fig_wc.patch.set_facecolor(current_theme['card_bg'])
    ax_wc.set_facecolor(current_theme['card_bg'])
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wc)
else:
    st.info("💡 Tidak ada data yang cocok dengan filter ini. Coba sesuaikan rentang tanggal atau pilih video lain.")

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 18. SECTION 4B — TOP 10 KEYWORDS
# ==========================================
st.markdown('<p class="section-header">📊 Top 10 Keywords</p>', unsafe_allow_html=True)

if filtered_words:
    top_words = pd.DataFrame(word_counts.most_common(10), columns=['Kata', 'Frekuensi'])
    top_words = top_words.sort_values(by='Frekuensi', ascending=True)

    fig_words = px.bar(
        top_words, x='Frekuensi', y='Kata', orientation='h',
        color_discrete_sequence=[current_theme['accent']],
        title="Keyword Frequency (after preprocessing)"
    )
    fig_words.update_traces(
        hovertemplate="<b>Kata:</b> %{y}<br><b>Frekuensi:</b> %{x:,}x<extra></extra>"
    )
    fig_words.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13), title_font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_words, width='stretch')
else:
    st.info("💡 Tidak ada data yang cocok dengan filter ini. Coba sesuaikan rentang tanggal atau pilih video lain.")

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 19. SECTION 4C — TOP 10 MOST ACTIVE USERS
# ==========================================
st.markdown('<p class="section-header">👥 Top 10 Most Active Users</p>', unsafe_allow_html=True)

if not clean_filtered_df.empty and 'username' in clean_filtered_df.columns:
    user_counts = clean_filtered_df['username'].value_counts().head(10).reset_index()
    user_counts.columns = ['Username', 'Jumlah Komentar']
    user_counts = user_counts.sort_values(by='Jumlah Komentar', ascending=True)

    fig_users = px.bar(
        user_counts, x='Jumlah Komentar', y='Username', orientation='h',
        color_discrete_sequence=[current_theme['Negatif']],
        title="Most Active Commenters"
    )
    fig_users.update_traces(
        hovertemplate="<b>Username:</b> @%{y}<br><b>Total:</b> %{x:,} comments<extra></extra>"
    )
    fig_users.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13), title_font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_users, width='stretch')
else:
    st.info("💡 Data username tidak tersedia pada filter ini.")