# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.title("3D 지반 모델링 & 단면 Slice Viewer")

# 0. 엑셀 파일 업로드
uploaded_file = st.sidebar.file_uploader(
    "시추공·말뚝 엑셀 파일 업로드 (.xlsx)", 
    type=["xlsx"]
)
if not uploaded_file:
    st.warning("왼쪽 사이드바에서 `.xlsx` 파일을 업로드해주세요.")
    st.stop()

# 1. 데이터 로드
@st.cache_data
def load_data(file_buffer):
    df_loc      = pd.read_excel(file_buffer, sheet_name='시추주상도 위치')
    df_info_raw = pd.read_excel(file_buffer, sheet_name='시추주상도 정보', header=0)
    df_pile     = pd.read_excel(file_buffer, sheet_name='말뚝 정보')
    # 정보 시트 전처리
    df_info = df_info_raw.drop(index=0).apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    pts = []
    for col in df_info.columns:
        if '.1' not in col:
            depth = df_info[col].values
            spt   = df_info[f"{col}.1"].values
            loc   = df_loc[df_loc.iloc[:,0]==col].iloc[0]
            x0,y0,topz = loc['X좌표(m)'], loc['Y좌표(m)'], loc['상단높이(m)']
            z_vals = topz - depth
            for xi, yi, zi, si in zip([x0]*len(z_vals), [y0]*len(z_vals), z_vals, spt):
                pts.append((xi, yi, zi, si))
    df_pts = pd.DataFrame(pts, columns=['X','Y','Z','SPT'])
    return df_pts, df_pile

df_pts, df_pile = load_data(uploaded_file)
X, Y, Z, SPT = df_pts['X'], df_pts['Y'], df_pts['Z'], df_pts['SPT']
Xp, Yp, Zp = df_pile['X'], df_pile['Y'], df_pile['상단높이(m)']

# 2. 그리드 보간 (한 번만 계산)
@st.cache_data
def make_grid(x, y, z, val, nx=50, ny=50, nz=25):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    zi = np.linspace(z.min(), z.max(), nz)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='xy')
    gv = griddata((x, y, z), val, (gx, gy, gz), method='linear')
    return xi, yi, zi, gx, gy, gz, gv

xi, yi, zi, gx, gy, gz, gv = make_grid(X, Y, Z, SPT)

# 3. 사이드바: Z 슬라이스 선택
z_sel = st.sidebar.slider(
    "단면 Z 높이 선택 (m)",
    float(Z.min()), float(Z.max()),
    float((Z.min()+Z.max())/2)
)

# 4. Plotly 3D + Slice
vol = go.Volume(
    x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
    value=gv.flatten(),
    isomin=np.nanpercentile(gv, 5),
    isomax=np.nanpercentile(gv, 95),
    opacity=0.1, surface_count=15, colorscale='Viridis'
)
idx = int(np.abs(zi - z_sel).argmin())
slice_val = gv[:, :, idx]
slice_surf = go.Surface(
    x=xi, y=yi,
    z=np.full_like(slice_val, zi[idx]),
    surfacecolor=slice_val,
    colorscale='Viridis',
    cmin=np.nanmin(gv), cmax=np.nanmax(gv),
    showscale=False
)
pts3d = go.Scatter3d(
    x=X, y=Y, z=Z,
    mode='markers',
    marker=dict(size=3, color=SPT, colorscale='Viridis')
)
piles3d = go.Scatter3d(
    x=Xp, y=Yp, z=Zp,
    mode='markers',
    marker=dict(size=5, symbol='diamond', color='red')
)

fig = go.Figure(data=[vol, slice_surf, pts3d, piles3d])
fig.update_layout(
    scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Elevation (m)'),
    height=700, margin=dict(r=10, l=10, b=10, t=50)
)

st.plotly_chart(fig, use_container_width=True)
