import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analysis import load_and_clean_data

# Page config
st.set_page_config(
    page_title="Marvel vs DC: Cinematic Showdown",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background adjustment & main title */
    .main {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    
    /* Custom Card Style for KPIs */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #8b949e;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .marvel-text {
        color: #E23636;
        text-shadow: 0 0 10px rgba(226, 54, 54, 0.3);
    }
    
    .dc-text {
        color: #004B87;
        text-shadow: 0 0 10px rgba(0, 75, 135, 0.3);
    }

    .overall-text {
        color: #9d4edd;
        text-shadow: 0 0 10px rgba(157, 78, 221, 0.3);
    }
    
    /* Title layout */
    .title-container {
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        background: radial-gradient(circle, rgba(226,54,54,0.08) 0%, rgba(0,75,135,0.08) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .title-container h1 {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: -webkit-linear-gradient(45deg, #E23636, #004B87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .title-container p {
        color: #8b949e;
        font-size: 18px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load data dynamically
@st.cache_data
def get_clean_data():
    return load_and_clean_data("marvel_dc_movies.csv")

df = get_clean_data()

# Page header
st.markdown("""
<div class="title-container">
    <h1>Marvel vs DC: Cinematic Showdown</h1>
    <p>An interactive, data-driven analysis of Box Office Performance, Ratings, and ROI trends.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b9/Marvel_Logo.svg", width=120)
st.sidebar.markdown("<h3 style='text-align: center; margin: 0;'>VS</h3>", unsafe_allow_html=True)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3d/DC_Comics_logo.svg", width=80)
st.sidebar.markdown("---")

st.sidebar.subheader("🎛️ Filter Control Center")

# Franchise Filter
selected_franchises = st.sidebar.multiselect(
    "Franchises to Display",
    options=df["Franchise"].unique(),
    default=list(df["Franchise"].unique())
)

# Year Filter
min_year = int(df["Release_Year"].min())
max_year = int(df["Release_Year"].max())
year_range = st.sidebar.slider(
    "Release Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Rating Filters
min_imdb = st.sidebar.slider(
    "Minimum IMDb Rating",
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

min_rt = st.sidebar.slider(
    "Minimum Rotten Tomatoes (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)

# Filter Data
filtered_df = df[
    (df["Franchise"].isin(selected_franchises)) &
    (df["Release_Year"].between(year_range[0], year_range[1])) &
    (df["IMDb_Rating"] >= min_imdb) &
    (df["Rotten_Tomatoes_Pct"] >= min_rt)
]

# Quick metrics computations
m_df = filtered_df[filtered_df["Franchise"] == "Marvel"]
dc_df = filtered_df[filtered_df["Franchise"] == "DC"]

# Colors mapping
palette = {'Marvel': '#E23636', 'DC': '#004B87'}

# Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Dashboard", "📈 Financial Analytics", "⭐ Reception Analysis", "🔍 Data Explorer"])

with tab1:
    st.subheader("💡 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Total Box Office gross
        total_box = filtered_df["Box_Office_Million"].sum()
        m_box = m_df["Box_Office_Million"].sum()
        dc_box = dc_df["Box_Office_Million"].sum()
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Global Box Office</div>
            <div class="metric-value overall-text">${total_box:,.1f}M</div>
            <div style="font-size:12px; color:#8b949e;">
                <span class="marvel-text">Marvel: ${m_box:,.1f}M</span> | <span class="dc-text">DC: ${dc_box:,.1f}M</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Avg ROI %
        avg_roi = filtered_df["ROI_Pct"].mean()
        m_roi = m_df["ROI_Pct"].mean()
        dc_roi = dc_df["ROI_Pct"].mean()
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Return on Investment</div>
            <div class="metric-value overall-text">{avg_roi:.1f}%</div>
            <div style="font-size:12px; color:#8b949e;">
                <span class="marvel-text">Marvel: {m_roi:.1f}%</span> | <span class="dc-text">DC: {dc_roi:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Avg IMDb Rating
        avg_imdb = filtered_df["IMDb_Rating"].mean()
        m_imdb = m_df["IMDb_Rating"].mean()
        dc_imdb = dc_df["IMDb_Rating"].mean()
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average IMDb Rating</div>
            <div class="metric-value overall-text">{avg_imdb:.2f}/10</div>
            <div style="font-size:12px; color:#8b949e;">
                <span class="marvel-text">Marvel: {m_imdb:.2f}</span> | <span class="dc-text">DC: {dc_imdb:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
   # =========================================================
# FINANCIAL BATTLEGROUND SCATTER PLOT
# =========================================================

st.subheader("🕸️ Financial Battleground: Budget vs. Box Office Gross")

# ---------------------------------------------------------
# Create safe marker sizes for Plotly
# Plotly does NOT allow negative marker sizes
# ---------------------------------------------------------

# Convert ROI into positive visualization scale
filtered_df["ROI_Size"] = filtered_df["ROI_Pct"].clip(lower=0) + 10

# Optional: Prevent extremely large bubbles
filtered_df["ROI_Size"] = filtered_df["ROI_Size"].clip(upper=500)

# ---------------------------------------------------------
# Interactive Scatter Plot
# ---------------------------------------------------------

fig_scatter = px.scatter(
    filtered_df,
    x="Budget_Million",
    y="Box_Office_Million",
    color="Franchise",
    size="ROI_Size",  # SAFE positive values
    hover_name="Movie",
    
    hover_data={
        "Release_Year": True,
        "IMDb_Rating": ':.1f',
        "Rotten_Tomatoes_Pct": ':.0f',
        "Profit_Million": ':.1f',
        "ROI_Pct": ':.1f',
        "ROI_Size": False,   # hide helper column
        "Budget_Million": ':.1f',
        "Box_Office_Million": ':.1f'
    },

    color_discrete_map=palette,

    labels={
        "Budget_Million": "Production Budget (Millions USD)",
        "Box_Office_Million": "Worldwide Box Office (Millions USD)",
        "ROI_Size": "ROI Strength"
    },

    size_max=45,
    opacity=0.80
)

# ---------------------------------------------------------
# Layout Styling
# ---------------------------------------------------------

fig_scatter.update_traces(
    marker=dict(
        line=dict(width=1, color='white')
    )
)

fig_scatter.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#f0f6fc',

    title={
        'text': 'Marvel vs DC: Budget vs Box Office Performance',
        'x': 0.5,
        'xanchor': 'center'
    },

    xaxis=dict(
        title='Production Budget (Millions USD)',
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False
    ),

    yaxis=dict(
        title='Worldwide Box Office (Millions USD)',
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),

    hoverlabel=dict(
        bgcolor="#161b22",
        font_size=13,
        font_family="Outfit"
    ),

    margin=dict(l=20, r=20, t=60, b=20)
)

# ---------------------------------------------------------
# Display Chart
# ---------------------------------------------------------

st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("💵 Financial Deep-Dive")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("#### Total & Average Financial Overview")
        fin_agg = filtered_df.groupby("Franchise")[["Budget_Million", "Box_Office_Million", "Profit_Million"]].mean().reset_index()
        
        fig_bar_fin = go.Figure()
        for col_name, color in zip(["Budget_Million", "Box_Office_Million", "Profit_Million"], ["#4b5563", "#10b981", "#3b82f6"]):
            fig_bar_fin.add_trace(go.Bar(
                name=col_name.replace("_Million", " (Avg)").replace("_", " "),
                x=fin_agg["Franchise"],
                y=fin_agg[col_name],
                marker_color=color,
                text=fin_agg[col_name].round(1).apply(lambda val: f"${val}M"),
                textposition='auto',
            ))
            
        fig_bar_fin.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f6fc',
            yaxis=dict(title="Millions USD", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar_fin, use_container_width=True)
        
    with col_f2:
        st.markdown("#### Return on Investment (ROI %) Distribution")
        
        fig_roi_box = px.box(
            filtered_df,
            x="Franchise",
            y="ROI_Pct",
            color="Franchise",
            color_discrete_map=palette,
            points="all",
            hover_name="Movie"
        )
        fig_roi_box.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven Point")
        fig_roi_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f6fc',
            yaxis=dict(title="ROI %", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False
        )
        st.plotly_chart(fig_roi_box, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Box Office Revenue Evolution Over Time")
    
    # Calculate yearly average
    yearly_agg = filtered_df.groupby(["Release_Year", "Franchise"])["Box_Office_Million"].mean().reset_index()
    
    fig_time = px.line(
        yearly_agg,
        x="Release_Year",
        y="Box_Office_Million",
        color="Franchise",
        color_discrete_map=palette,
        markers=True,
        labels={"Release_Year": "Year", "Box_Office_Million": "Avg Box Office (Millions USD)"}
    )
    fig_time.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f6fc',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    st.subheader("⭐ Audience & Critics Reception")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("#### IMDb Audience Rating Distribution")
        fig_imdb_violin = px.violin(
            filtered_df,
            y="IMDb_Rating",
            x="Franchise",
            color="Franchise",
            color_discrete_map=palette,
            box=True,
            points="all",
            hover_name="Movie"
        )
        fig_imdb_violin.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f6fc',
            yaxis=dict(title="IMDb Rating (out of 10)", range=[2, 10], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False
        )
        st.plotly_chart(fig_imdb_violin, use_container_width=True)
        
    with col_r2:
        st.markdown("#### Rotten Tomatoes Critics Rating Distribution")
        fig_rt_violin = px.violin(
            filtered_df,
            y="Rotten_Tomatoes_Pct",
            x="Franchise",
            color="Franchise",
            color_discrete_map=palette,
            box=True,
            points="all",
            hover_name="Movie"
        )
        fig_rt_violin.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f6fc',
            yaxis=dict(title="Rotten Tomatoes Score (%)", range=[0, 105], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False
        )
        st.plotly_chart(fig_rt_violin, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Leaderboard Standings")
    
    l_col1, l_col2, l_col3 = st.columns(3)
    
    with l_col1:
        st.markdown("##### Highest Return on Investment (ROI %)")
        top_roi_table = filtered_df.sort_values(by="ROI_Pct", ascending=False).head(5)[["Movie", "Franchise", "ROI_Pct"]]
        top_roi_table["ROI_Pct"] = top_roi_table["ROI_Pct"].round(0).apply(lambda val: f"{val:,.0f}%")
        st.dataframe(top_roi_table, hide_index=True, use_container_width=True)
        
    with l_col2:
        st.markdown("##### Highest Box Office Grossing")
        top_box_table = filtered_df.sort_values(by="Box_Office_Million", ascending=False).head(5)[["Movie", "Franchise", "Box_Office_Million"]]
        top_box_table["Box_Office_Million"] = top_box_table["Box_Office_Million"].apply(lambda val: f"${val:,.1f}M")
        st.dataframe(top_box_table, hide_index=True, use_container_width=True)
        
    with l_col3:
        st.markdown("##### Highest IMDb Audience Rated")
        top_imdb_table = filtered_df.sort_values(by="IMDb_Rating", ascending=False).head(5)[["Movie", "Franchise", "IMDb_Rating"]]
        top_imdb_table["IMDb_Rating"] = top_imdb_table["IMDb_Rating"].apply(lambda val: f"{val:.1f}/10")
        st.dataframe(top_imdb_table, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("🔍 Cleaned Dataset Explorer")
    
    search_query = st.text_input("🔍 Search movies by title or director:", "")
    
    disp_df = filtered_df.copy()
    if search_query:
        disp_df = disp_df[
            disp_df["Movie"].str.contains(search_query, case=False, na=False) |
            disp_df["Director"].str.contains(search_query, case=False, na=False)
        ]
        
    st.dataframe(
        disp_df.style.format({
            "Budget_Million": "${:,.1f}M",
            "Box_Office_Million": "${:,.1f}M",
            "Profit_Million": "${:,.1f}M",
            "ROI_Pct": "{:,.1f}%",
            "IMDb_Rating": "{:.1f}",
            "Rotten_Tomatoes_Pct": "{:.0f}%",
            "Release_Year": "{:.0f}"
        }),
        use_container_width=True
    )
    
    # Download option
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV Data",
        data=csv_bytes,
        file_name="marvel_vs_dc_cleaned_movies.csv",
        mime="text/csv"
    )
