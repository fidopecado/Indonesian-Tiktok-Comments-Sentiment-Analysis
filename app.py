import os
import random
import re
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
# 1. KONFIGURASI HALAMAN WEBSITE
# ==========================================
st.set_page_config(
    page_title="Public Opinion Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definisi Presets Tema Warna (Original, E-Ink, MBG Prabowo, KDMP Prabowo, Dark Mode)
THEMES = {
    "Original Light (Terracotta & Violet)": {
        "sidebar_bg": "#CA2E16",
        "sidebar_text": "#EEF0F2",
        "main_bg": "#EEF0F2",
        "card_bg": "#FFFFFF",
        "text_color": "#271F30",
        "input_bg": "#FFFFFF",
        "input_text": "#1E293B",
        "positif": "#566E3D",
        "netral": "#F9C22E",
        "negatif": "#CA2E16",
        "accent": "#CA2E16"
    },
    "E-Ink (Paper & Charcoal Monochrome)": {
        "sidebar_bg": "#1C1C1E",
        "sidebar_text": "#F4F4F0",
        "main_bg": "#F5F5F0",
        "card_bg": "#FFFFFF",
        "text_color": "#111111",
        "input_bg": "#FFFFFF",
        "input_text": "#111111",
        "positif": "#2D5A27",
        "netral": "#7E6C46",
        "negatif": "#8B0000",
        "accent": "#1C1C1E"
    },
    "MBG Prabowo Color (Biru Muda / Sky Blue)": {
        "sidebar_bg": "#0080FF",
        "sidebar_text": "#FFFFFF",
        "main_bg": "#F0F8FF",
        "card_bg": "#FFFFFF",
        "text_color": "#0F172A",
        "input_bg": "#FFFFFF",
        "input_text": "#0F172A",
        "positif": "#10B981",
        "netral": "#F59E0B",
        "negatif": "#EF4444",
        "accent": "#0080FF"
    },
    "KDMP Prabowo Color (Merah Putih Patriotic)": {
        "sidebar_bg": "#C8102E",
        "sidebar_text": "#FFFFFF",
        "main_bg": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "text_color": "#0F172A",
        "input_bg": "#FFFFFF",
        "input_text": "#0F172A",
        "positif": "#16A34A",
        "netral": "#D97706",
        "negatif": "#DC2626",
        "accent": "#C8102E"
    },
    "Dark Mode (Midnight Slate)": {
        "sidebar_bg": "#0F172A",
        "sidebar_text": "#38BDF8",
        "main_bg": "#020617",
        "card_bg": "#1E293B",
        "text_color": "#F8FAFC",
        "input_bg": "#0F172A",
        "input_text": "#F8FAFC",
        "positif": "#34D399",
        "netral": "#FBBF24",
        "negatif": "#F87171",
        "accent": "#38BDF8"
    }
}

# Selector Tema di Sidebar
st.sidebar.markdown("<h2 style='margin-top:0; font-weight:800;'>🎨 Pengaturan Tema</h2>", unsafe_allow_html=True)
selected_theme_name = st.sidebar.selectbox("Pilih Tema Warna Website:", options=list(THEMES.keys()), index=0)
current_theme = THEMES[selected_theme_name]

THEME_BG = current_theme["main_bg"]
THEME_TEXT = current_theme["text_color"]
THEME_CARD_BG = current_theme["card_bg"]
PLOT_FONT_COLOR = current_theme["text_color"]

color_map = {
    'positif': current_theme['positif'],
    'netral': current_theme['netral'],
    'negatif': current_theme['negatif']
}

# Injeksi CSS Dinamis Berdasarkan Tema Terpilih & Perbaikan Range Filter Color
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
    
    /* Background & Teks Utama */
    [data-testid="stAppViewContainer"] {{
        background-color: {THEME_BG} !important;
        color: {THEME_TEXT} !important;
    }}
    
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {current_theme['sidebar_bg']} !important;
    }}
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] .stSelectbox label p,
    [data-testid="stSidebar"] .stMultiSelect label p,
    [data-testid="stSidebar"] .stDateInput label p {{
        color: {current_theme['sidebar_text']} !important;
        font-weight: 600 !important;
    }}
    
    /* FIX: Input Controls, Selectboxes, Multiselect, & Date Inputs in Sidebar */
    [data-testid="stSidebar"] div[data-baseweb="input"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] div,
    [data-testid="stSidebar"] div[data-baseweb="base-input"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] span {{
        color: {current_theme['input_text']} !important;
        background-color: {current_theme['input_bg']} !important;
    }}

    /* FIX: Date Picker Dialog & Calendar Popover (Prevent White text on White BG) */
    div[data-baseweb="popover"],
    div[data-baseweb="calendar"],
    div[role="dialog"] {{
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.15) !important;
    }}
    div[data-baseweb="popover"] *,
    div[role="dialog"] * {{
        color: #0f172a !important;
    }}
    div[data-baseweb="calendar"] button[aria-selected="true"] {{
        background-color: {current_theme['accent']} !important;
        color: #ffffff !important;
    }}
    
    /* Tipografi Responsif */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .responsive-title {{
        font-size: clamp(24px, 4vw, 40px) !important;
        font-weight: 800;
        line-height: 1.2;
        color: {THEME_TEXT} !important;
        margin-bottom: 5px;
    }}
    .responsive-caption {{
        font-size: clamp(12px, 1.2vw, 15px) !important;
        color: {THEME_TEXT} !important;
        opacity: 0.8;
    }}
    
    hr {{
        border-top: 2px solid {THEME_TEXT} !important;
        opacity: 0.15;
    }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD DATA DENGAN CACHING MURNI
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


# ==========================================
# FUNCTION: KOTAK KPI
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
            opacity: 0.8;
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
# 3. SIDEBAR SELEKSI DATASET & FILTER
# ==========================================
st.sidebar.markdown(f"<h2 style='color:{current_theme['sidebar_text']}; margin-top:15px; font-weight:800;'>📂 Proyek Data</h2>", unsafe_allow_html=True)

dataset_mapping = {
    "Koperasi Desa Merah Putih (KDMP)": os.path.join(BASE_DIR, "KDMP.csv"),
    "Makan Bergizi Gratis (MBG)": os.path.join(BASE_DIR, "MBG_PROCESSED.csv")
}

selected_project = st.sidebar.selectbox("Topik Analisis:", options=list(dataset_mapping.keys()))
df = load_data(dataset_mapping[selected_project])

st.sidebar.markdown("<hr style='border-top: 2px solid opacity:0.3;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h2 style='color:{current_theme['sidebar_text']}; font-weight:800;'>🕹️ Kendali Filter</h2>", unsafe_allow_html=True)

min_date_dataset = df['date_only'].min()
max_date_dataset = df['date_only'].max()
video_list = sorted(df['video_id'].dropna().unique())

date_range = st.sidebar.date_input("Rentang Waktu:", value=(min_date_dataset, max_date_dataset), min_value=min_date_dataset, max_value=max_date_dataset)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_range[0], max_date_dataset

selected_videos = st.sidebar.multiselect("Pilih ID Video (Opsional):", options=video_list, default=[])

model_mapping = {
    "Final Sentiment Voting (Konsensus)": "final_sentiment_voting",
    "Model HuggingFace: taufiqdp": "sentiment",
    "Model AI: Aardiiiiy": "aardiiiiy_sentiment",
    "Model AI: Agufsamudra": "agufsamudra_sentiment",
    "Model AI: Mdhugol": "mdhugol_sentiment"
}

selected_model_name = st.sidebar.selectbox("Model Klasifikasi Utama:", options=list(model_mapping.keys()), index=0)
active_column = model_mapping[selected_model_name]


# ==========================================
# 4. LOGIKA PROSES DATA & PEMETAAN WARNA
# ==========================================
mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
if selected_videos:
    mask = mask & (df['video_id'].isin(selected_videos))

filtered_df = df[mask]
clean_filtered_df = filtered_df[filtered_df[active_column].notna() & (filtered_df[active_column] != 'empty')]
kpi_stats = clean_filtered_df[active_column].value_counts()


# ==========================================
# 5. SECTION 1: HEADER & KPI CARDS (ATAS)
# ==========================================
st.markdown(f'<p class="responsive-title">📊 Analisis Opini Publik: {selected_project}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="responsive-caption">Eksplorasi data interaktif menggunakan pemrosesan <b>{selected_model_name}</b>.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Baris 1: Ringkasan Umum
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    create_kpi_card("💬 Total Komentar Bersih", f"{len(clean_filtered_df):,}", current_theme['negatif'])
with row1_col2:
    create_kpi_card("📹 Distribusi Video", f"{clean_filtered_df['video_id'].nunique()}", current_theme['accent'])
with row1_col3:
    create_kpi_card("👥 Pengguna Unik", f"{clean_filtered_df['username'].nunique():,}", current_theme['accent'])

# Baris 2: Statistik Sentimen
row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    create_kpi_card("🟢 Respon Positif", f"{kpi_stats.get('positif', 0):,}", current_theme['positif'])
with row2_col2:
    create_kpi_card("🟡 Respon Netral", f"{kpi_stats.get('netral', 0):,}", current_theme['netral'])
with row2_col3:
    create_kpi_card("🔴 Respon Negatif", f"{kpi_stats.get('negatif', 0):,}", current_theme['negatif'])

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 6. SECTION 2: GRAFIK UTAMA
# ==========================================
st.markdown(f"<h3 style='color: {THEME_TEXT}; font-weight:800;'>📈 Visualisasi Distribusi: {selected_model_name}</h3>", unsafe_allow_html=True)

chart_data = clean_filtered_df[active_column].value_counts().reset_index()
chart_data.columns = ['Sentimen', 'Total']

col_left, col_right = st.columns(2)

with col_left:
    fig_pie = px.pie(chart_data, values='Total', names='Sentimen', hole=0.5, color='Sentimen', color_discrete_map=color_map, title="Persentase Opini Publik")
    fig_pie.update_traces(
        textposition='inside', textinfo='percent+label',
        hovertemplate="<b>Kategori Opini:</b> %{label}<br><b>Porsi Persentase:</b> %{percent}<br><b>Jumlah Riil:</b> %{value:,} komentar<extra></extra>"
    )
    fig_pie.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13),
        title_font=dict(color=PLOT_FONT_COLOR),
        legend=dict(font=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
with col_right:
    fig_bar = px.bar(chart_data, x='Sentimen', y='Total', color='Sentimen', color_discrete_map=color_map, title="Volume Perbandingan Komentar")
    fig_bar.update_traces(hovertemplate="<b>Sentimen:</b> %{x}<br><b>Volume Data:</b> %{y:,} komentar<extra></extra>")
    fig_bar.update_layout(
        xaxis_title="Jenis Sentimen", yaxis_title="Total Komentar", 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13),
        title_font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        legend=dict(font=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 7. SECTION 3: KOMPARASI LINTAS MODEL (BENCHMARKING)
# ==========================================
st.markdown(f"<h3 style='color: {THEME_TEXT}; font-weight:800;'>🔄 Benchmarking: Komparasi Seluruh Model AI</h3>", unsafe_allow_html=True)

comparison_list = []
for m_name, m_col in model_mapping.items():
    clean_m_df = filtered_df[filtered_df[m_col].notna() & (filtered_df[m_col] != 'empty')]
    m_counts = clean_m_df[m_col].value_counts()
    for sentiment_type in ['positif', 'netral', 'negatif']:
        comparison_list.append({'Model AI': m_name, 'Kategori Sentimen': sentiment_type, 'Jumlah Komentar': m_counts.get(sentiment_type, 0)})

comp_df = pd.DataFrame(comparison_list)
fig_comp = px.bar(comp_df, x='Model AI', y='Jumlah Komentar', color='Kategori Sentimen', barmode='group', color_discrete_map=color_map, title="Komparasi Distribusi Sentimen Antar Model")
fig_comp.update_traces(hovertemplate="<b>Model AI:</b> %{x}<br><b>Jenis Sentimen:</b> %{hovertext}<br><b>Hasil Hitung:</b> %{y:,} komentar<extra></extra>", hovertext=comp_df['Kategori Sentimen'])
fig_comp.update_layout(
    xaxis_title="Nama Model", yaxis_title="Jumlah Komentar", legend_title="Kategori", 
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color=PLOT_FONT_COLOR, size=13),
    title_font=dict(color=PLOT_FONT_COLOR),
    xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
    yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
    legend=dict(font=dict(color=PLOT_FONT_COLOR))
)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# 8. TEXT PROCESSING & GENERATION LOGIC
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
# SECTION 4A: WORDCLOUD (1 BARIS PENUH)
# ==========================================
st.markdown(f"<h3 style='color: {THEME_TEXT}; font-weight:800;'>☁️ Ringkasan Topik Utama (Word Cloud)</h3>", unsafe_allow_html=True)

if filtered_words:
    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return random.choice([current_theme['text_color'], current_theme['positif'], current_theme['negatif']])

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
    st.info("💡 Tidak ada data teks komentar yang memadai pada filter rentang waktu atau video saat ini.")

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# SECTION 4B: TOP 10 WORDS (1 BARIS PENUH)
# ==========================================
st.markdown(f"<h3 style='color: {THEME_TEXT}; font-weight:800;'>📊 10 Kata Kunci Paling Sering Muncul</h3>", unsafe_allow_html=True)

if filtered_words:
    top_words = pd.DataFrame(word_counts.most_common(10), columns=['Kata', 'Frekuensi'])
    top_words = top_words.sort_values(by='Frekuensi', ascending=True)
    
    fig_words = px.bar(
        top_words, x='Frekuensi', y='Kata', orientation='h',
        color_discrete_sequence=[current_theme['accent']],
        title="Distribusi Frekuensi Kata Kunci Hasil Preprocessing"
    )
    fig_words.update_traces(hovertemplate="<b>Kata:</b> %{y}<br><b>Frekuensi:</b> %{x:,} kali<extra></extra>")
    fig_words.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13), title_font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_words, use_container_width=True)
else:
    st.info("💡 Data kata tidak tersedia pada filter saat ini.")

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# SECTION 4C: TOP 10 COMMENTERS (1 BARIS PENUH)
# ==========================================
st.markdown(f"<h3 style='color: {THEME_TEXT}; font-weight:800;'>👥 10 Pengguna (User) Paling Aktif Berkomentar</h3>", unsafe_allow_html=True)

if not clean_filtered_df.empty and 'username' in clean_filtered_df.columns:
    user_counts = clean_filtered_df['username'].value_counts().head(10).reset_index()
    user_counts.columns = ['Username', 'Jumlah Komentar']
    user_counts = user_counts.sort_values(by='Jumlah Komentar', ascending=True)
    
    fig_users = px.bar(
        user_counts, x='Jumlah Komentar', y='Username', orientation='h',
        color_discrete_sequence=[current_theme['negatif']],
        title="Volume Keaktifan Akun Pengguna Media Sosial"
    )
    fig_users.update_traces(hovertemplate="<b>Username:</b> @%{y}<br><b>Total Post:</b> %{x:,} komentar<extra></extra>")
    fig_users.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_FONT_COLOR, size=13), title_font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR)),
        yaxis=dict(title_font=dict(color=PLOT_FONT_COLOR), tickfont=dict(color=PLOT_FONT_COLOR))
    )
    st.plotly_chart(fig_users, use_container_width=True)
else:
    st.info("💡 Data nama pengguna tidak ditemukan.")