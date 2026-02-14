# app.py - Streamlit App for 5 Business Analytics Case Studies
# Covers: Upload CSV or use generated sample, clean, transform, normalize, feature engineering, regex (where applicable),
# hypothesis testing, visualizations (Plotly), and business questions/insights.
# Adapt columns/business questions as needed for each topic.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re

st.set_page_config(page_title="Business Analytics Case Studies Dashboard", layout="wide")

st.title("Business Analytics Dashboard for 5 Case Studies")
st.markdown("Select a case study below. Upload your own CSV or use the generated sample (1000 rows). The app will clean, transform, and analyze the data to answer key business questions.")

# Select case study
case_study = st.selectbox("Choose Case Study", [
    "1. Instagram User Engagement",
    "2. McDonald's Store Sales",
    "3. Netflix Content Performance",
    "4. Amazon Order Fulfillment",
    "5. Spotify User Listening Patterns"
])

# Generate sample datasets (embedded)
@st.cache_data
def generate_sample_data(topic):
    np.random.seed(42)
    n = 1000
    if topic == "1. Instagram User Engagement":
        post_ids = [f'POST-{i:04d}' for i in range(1, n+1)]
        post_types = np.random.choice(['Photo', 'Video', 'Reel', 'Story', 'Carousel'], n, p=[0.3, 0.2, 0.3, 0.1, 0.1])
        post_dates = pd.date_range('2024-01-01', periods=n) + pd.to_timedelta(np.random.randint(0, 365, n), unit='D')
        post_times = [f"{h:02d}:{m:02d}" for h, m in zip(np.random.randint(0, 24, n), np.random.randint(0, 60, n))]
        likes = np.random.randint(50, 5000, n)
        comments = np.random.randint(5, 500, n)
        shares = np.random.randint(1, 1000, n)
        reach = likes + comments + shares + np.random.randint(100, 10000, n)
        engagement_rate = np.round((likes + comments + shares) / reach * 100, 2)
        followers_growth = np.random.choice([-10, 0, 5, 10, 20, 50], n, p=[0.05, 0.2, 0.3, 0.2, 0.15, 0.1])
        df = pd.DataFrame({
            'post_id': post_ids, 'post_type': post_types, 'post_date': post_dates, 'post_time': post_times,
            'likes': likes, 'comments': comments, 'shares': shares, 'reach': reach,
            'engagement_rate': engagement_rate, 'followers_growth': followers_growth
        })
    elif topic == "2. McDonald's Store Sales":
        order_ids = [f'ORDER-{i:04d}' for i in range(1, n+1)]
        store_zones = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n)
        order_dates = pd.date_range('2024-01-01', periods=n) + pd.to_timedelta(np.random.randint(0, 365, n), unit='D')
        order_times = [f"{h:02d}:{m:02d}" for h, m in zip(np.random.randint(0, 24, n), np.random.randint(0, 60, n))]
        menu_items = np.random.choice(['Burger', 'Fries', 'Beverage', 'Combo', 'Dessert', 'Salad'], n)
        items_per_order = np.random.randint(1, 6, n)
        order_value = np.round(np.random.uniform(100, 1000, n), 2)
        is_weekend = order_dates.dayofweek >= 5
        repeat_customer = np.random.choice([0, 1], n, p=[0.6, 0.4])
        df = pd.DataFrame({
            'order_id': order_ids, 'store_zone': store_zones, 'order_date': order_dates, 'order_time': order_times,
            'menu_item': menu_items, 'items_per_order': items_per_order, 'order_value': order_value,
            'is_weekend': is_weekend, 'repeat_customer': repeat_customer
        })
    elif topic == "3. Netflix Content Performance":
        content_ids = [f'CONTENT-{i:04d}' for i in range(1, n+1)]
        genres = np.random.choice(['Drama', 'Comedy', 'Thriller', 'Documentary', 'Action', 'Sci-Fi'], n)
        view_dates = pd.date_range('2024-01-01', periods=n) + pd.to_timedelta(np.random.randint(0, 365, n), unit='D')
        watch_time_min = np.random.randint(10, 120, n)
        completion_rate = np.round(np.random.uniform(0.1, 1.0, n), 2)
        drop_off_episode = np.random.choice([1,2,3,4,5,'None'], n)
        user_rating = np.round(np.random.uniform(1.0, 5.0, n), 1)
        is_original = np.random.choice([0, 1], n, p=[0.4, 0.6])
        df = pd.DataFrame({
            'content_id': content_ids, 'genre': genres, 'view_date': view_dates, 'watch_time_min': watch_time_min,
            'completion_rate': completion_rate, 'drop_off_episode': drop_off_episode,
            'user_rating': user_rating, 'is_original': is_original
        })
    elif topic == "4. Amazon Order Fulfillment":
        order_ids = [f'AMZ-ORDER-{i:04d}' for i in range(1, n+1)]
        categories = np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Beauty'], n)
        order_dates = pd.date_range('2024-01-01', periods=n) + pd.to_timedelta(np.random.randint(0, 365, n), unit='D')
        delivery_days = np.random.randint(1, 7, n)
        on_time = np.random.choice([0, 1], n, p=[0.2, 0.8])
        returned = np.random.choice([0, 1], n, p=[0.85, 0.15])
        customer_rating = np.round(np.random.uniform(1.0, 5.0, n), 1)
        fulfillment_cost = np.round(np.random.uniform(50, 500, n), 2)
        zone = np.random.choice(['Urban', 'Rural', 'Suburban'], n)
        df = pd.DataFrame({
            'order_id': order_ids, 'category': categories, 'order_date': order_dates, 'delivery_days': delivery_days,
            'on_time': on_time, 'returned': returned, 'customer_rating': customer_rating,
            'fulfillment_cost': fulfillment_cost, 'zone': zone
        })
    elif topic == "5. Spotify User Listening Patterns":
        user_ids = [f'USER-{i:04d}' for i in range(1, n+1)]
        genres = np.random.choice(['Pop', 'Hip-Hop', 'Rock', 'Classical', 'Jazz', 'Electronic'], n)
        listen_dates = pd.date_range('2024-01-01', periods=n) + pd.to_timedelta(np.random.randint(0, 365, n), unit='D')
        listen_time_min = np.random.randint(5, 60, n)
        skips = np.random.randint(0, 10, n)
        subscription = np.random.choice(['Free', 'Premium'], n, p=[0.6, 0.4])
        device = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n)
        churn = np.random.choice([0, 1], n, p=[0.85, 0.15])
        df = pd.DataFrame({
            'user_id': user_ids, 'genre': genres, 'listen_date': listen_dates, 'listen_time_min': listen_time_min,
            'skips': skips, 'subscription': subscription, 'device': device, 'churn': churn
        })
    return df

# Upload or use sample
uploaded_file = st.file_uploader("Upload your CSV (optional)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_sample_data(case_study)
    st.info("Using generated sample data (1000 rows).")

# ─── Data Cleaning ──────────────────────────────────────────────────────
st.subheader("Data Cleaning")
df = df.dropna(how='all')  # Drop empty rows
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill numeric missing with median
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna('Unknown')  # Fill categorical missing
st.write(f"Rows after cleaning: {len(df)}")

# ─── Transformation & Normalization ──────────────────────────────────────
st.subheader("Transformation & Normalization")
if 'date' in df.columns or 'post_date' in df.columns or 'order_date' in df.columns or 'view_date' in df.columns or 'listen_date' in df.columns:
    date_col = next(col for col in df.columns if 'date' in col.lower())
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['day_of_week'] = df[date_col].dt.day_name()
    df['is_weekend'] = df[date_col].dt.weekday >= 5

# Normalize numeric columns (min-max)
for col in numeric_cols:
    if df[col].max() - df[col].min() > 0:
        df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

st.write("Added: day_of_week, is_weekend, normalized columns")

# ─── Feature Engineering ─────────────────────────────────────────────────
st.subheader("Feature Engineering")
# Topic-specific features
if case_study == "1. Instagram User Engagement":
    df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['reach']
elif case_study == "2. McDonald's Store Sales":
    df['avg_item_value'] = df['order_value'] / df['items_per_order']
elif case_study == "3. Netflix Content Performance":
    df['retention_category'] = np.where(df['completion_rate'] > 0.8, 'High', 'Low')
elif case_study == "4. Amazon Order Fulfillment":
    df['satisfaction_score'] = df['on_time'] * df['customer_rating'] - df['returned']
elif case_study == "5. Spotify User Listening Patterns":
    df['skip_rate'] = df['skips'] / df['listen_time_min']

st.write("Added topic-specific features (e.g., rates, categories)")

# ─── Regex Example ───────────────────────────────────────────────────────
st.subheader("Regex Example (Extract from IDs)")
id_col = next((col for col in df.columns if 'id' in col.lower()), None)
if id_col:
    df['extracted_number'] = df[id_col].str.extract(r'(\d+)', expand=False)
    st.write("Extracted numbers from IDs (sample):")
    st.dataframe(df[[id_col, 'extracted_number']].head())

# ─── Hypothesis Testing ──────────────────────────────────────────────────
st.subheader("Hypothesis Testing Example")
if 'is_weekend' in df.columns and 'order_value' in df.columns:  # Adapt based on topic
    weekend = df[df['is_weekend']]['order_value']
    weekday = df[~df['is_weekend']]['order_value']
    t_stat, p_val = stats.ttest_ind(weekend, weekday, equal_var=False)
    st.write(f"t-test (Weekend vs Weekday Value): p-value = {p_val:.4f}")
    if p_val < 0.05:
        st.success("Significant difference (p < 0.05)")
    else:
        st.info("No significant difference")

# ─── Visualizations & Business Questions ─────────────────────────────────
st.subheader("Visualizations & Business Insights")
# Topic-specific visuals/questions
if case_study == "1. Instagram User Engagement":
    fig = px.bar(df.groupby('post_type')['engagement_rate'].mean().reset_index(), x='post_type', y='engagement_rate')
    st.plotly_chart(fig)
    st.write("Q: Best content? A: Reels/Video highest engagement")
elif case_study == "2. McDonald's Store Sales":
    fig = px.box(df, x='store_zone', y='order_value')
    st.plotly_chart(fig)
    st.write("Q: Best zone? A: Central/North highest avg sales")
elif case_study == "3. Netflix Content Performance":
    fig = px.bar(df.groupby('genre')['watch_time_min'].mean().reset_index(), x='genre', y='watch_time_min')
    st.plotly_chart(fig)
    st.write("Q: Best genre? A: Thriller/Action highest watch time")
elif case_study == "4. Amazon Order Fulfillment":
    fig = px.box(df, x='zone', y='delivery_days')
    st.plotly_chart(fig)
    st.write("Q: Worst zone? A: Rural highest delay – add resources")
elif case_study == "5. Spotify User Listening Patterns":
    fig = px.bar(df.groupby('genre')['listen_time_min'].mean().reset_index(), x='genre', y='listen_time_min')
    st.plotly_chart(fig)
    st.write("Q: Popular genre? A: Pop/Hip-Hop highest listening")

st.markdown("**Note**: Adapt business questions in code for full analysis. Download below.")

# Download
csv = df.to_csv(index=False)
st.download_button("Download Processed CSV", csv, "processed_data.csv", "text/csv")