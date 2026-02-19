import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os

st.set_page_config(page_title="Tomato PCA Explorer", layout="wide")

# Function to load data from the same directory as the script
@st.cache_data
def load_data():
    # This looks for the file in the same folder as app.py
    file_name = 'Tomato metabolite data FULL Clean.xlsx - SD1_Original_Data.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        return df
    else:
        st.error(f"Data file '{file_name}' not found in the repository!")
        return None

df = load_data()

if df is not None:
    st.title("🍅 Tomato Breeding: 3D Metabolite Biplot")
    
    # --- Sidebar Controls ---
    st.sidebar.header("Plot Settings")
    show_labels = st.sidebar.checkbox("Show Sample ID Labels", value=False)
    loading_scalar = st.sidebar.slider("Loadings Line Length", 5.0, 100.0, 40.0)
    num_loadings = st.sidebar.number_input("Number of Loadings to Display", 5, 50, 15)

    # --- PCA Processing ---
    # IDs are in col 0, Metabolites start at col 5
    ids = df.iloc[:, 0].astype(str)
    metabs = df.iloc[:, 5:].fillna(df.iloc[:, 5:].mean())
    metabs = metabs.loc[:, metabs.std() > 0] # Remove zero-variance

    # Pareto Scaling
    scaled = (metabs - metabs.mean()) / np.sqrt(metabs.std())
    
    pca = PCA(n_components=3)
    scores = pca.fit_transform(scaled)
    loadings = pca.components_.T
    var = pca.explained_variance_ratio_

    # --- Build 3D Biplot ---
    fig = go.Figure()

    # Scores
    fig.add_trace(go.Scatter3d(
        x=scores[:, 0], y=scores[:, 1], z=scores[:, 2],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(size=4, color='royalblue', opacity=0.6),
        text=ids, name='Samples'
    ))

    # Loadings
    loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'], index=metabs.columns)
    loadings_df['Mag'] = np.linalg.norm(loadings, axis=1)
    top_metabs = loadings_df.nlargest(num_loadings, 'Mag')

    lx, ly, lz = [], [], []
    for _, row in top_metabs.iterrows():
        lx.extend([0, row['PC1']*loading_scalar, None])
        ly.extend([0, row['PC2']*loading_scalar, None])
        lz.extend([0, row['PC3']*loading_scalar, None])

    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='red', width=3), name='Metabolites'))
    fig.add_trace(go.Scatter3d(
        x=top_metabs['PC1']*loading_scalar, 
        y=top_metabs['PC2']*loading_scalar, 
        z=top_metabs['PC3']*loading_scalar,
        mode='text', text=top_metabs.index, name='Metabolite Labels'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({var[0]:.1%})',
            yaxis_title=f'PC2 ({var[1]:.1%})',
            zaxis_title=f'PC3 ({var[2]:.1%})'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)
