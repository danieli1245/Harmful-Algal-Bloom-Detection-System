import streamlit as st
import pandas as pd

st.set_page_config(page_title="HAB Detection System", layout="centered")

st.title("HAB Detection System")
st.markdown("Upload your dataset to explore harmful algal bloom events.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    if all(col in df.columns for col in ['region', 'season', 'year']):
        region = st.selectbox("Filter by Region", df['region'].dropna().unique())
        season = st.selectbox("Filter by Season", df['season'].dropna().unique())
        year = st.selectbox("Filter by Year", sorted(df['year'].dropna().unique()))

        filtered_df = df[
            (df['region'] == region) &
            (df['season'] == season) &
            (df['year'] == year)
        ]

        st.subheader("Filtered Data")
        st.dataframe(filtered_df)

        if 'severity' in df.columns:
            st.subheader("Severity Distribution")
            st.bar_chart(filtered_df['severity'].value_counts())

        if 'lat' in df.columns and 'lon' in df.columns:
            st.subheader("Detection Locations")
            st.map(filtered_df[['lat', 'lon']])
    else:
        st.warning("Your dataset must contain 'region', 'season', and 'year' columns for filtering.")