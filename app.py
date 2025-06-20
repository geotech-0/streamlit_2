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

# 1. 데이터 로드 및 전처리
@st.cache_data
def load_data(file_buffer):
    df_loc      = pd.read_excel(file_buffer, sheet_name='시추주상도 위치')
    df_info_raw = pd.read_excel(file_buffer, sheet_name='시추주상도 정보', header=0)
    df_pile     = pd.read_excel(file_buffer, sheet_name='말뚝 정보')
    # 시추정보 전처리: borehole name 포함
    df_info = df_info_raw.drop(index=0).apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    pts = []
    for col in df_info.columns:
        if '.1' not in col:
            depth_arr = df_info[col].values
            spt_arr   = df_info[f"{col}.1"].values
            loc       = df_loc[df_loc.iloc[:,0] == col].iloc[0]
            x0, y0, topz = loc['X좌표(m)'], loc['Y좌표(m)'], loc['상단높이(m)']
            z_vals = topz - depth_arr
            for z, s in zip(z_vals, spt_arr):
                pts.append((col, x0, y0, z, s))
    df_pts = pd.DataFrame(pts, columns=['BH','X','Y','Z','SPT'])
    return df_pts, df_pile

# Load
df_pts, df_pile = load_data(uploaded_file)
X, Y, Z, SPT = df_pts['X'], df_pts['Y'], df_pts['Z'], df_pts['SPT']
Xp, Yp, Zp = df_pile['X'], df_pile['Y'], df_pile['상단높이(m)']

# 2. 그리드 보간
@st.cache_data
def make_grid(x, y, z, val, nx=50, ny=50, nz=25):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    zi = np.linspace(z.min(), z.max(), nz)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='xy')
    gv = griddata((x, y, z), val, (gx, gy, gz), method='linear')
    return xi, yi, zi, gx, gy, gz, gv

xi, yi, zi, gx, gy, gz, gv = make_grid(X, Y, Z, SPT)

# 3. 사이드바 컨트롤
z_sel = st.sidebar.slider(
    "단면 Z 높이 선택 (m)",
    float(Z.min()), float(Z.max()),
    float((Z.min()+Z.max())/2)
)
bh_radius = st.sidebar.number_input(
    "시추공 반경 설정 (m)",
    min_value=0.01, max_value=5.0, value=0.1, step=0.01
)
pile_radius = st.sidebar.number_input(
    "말뚝 반경 설정 (m)",
    min_value=0.01, max_value=5.0, value=0.2, step=0.01
)
pile_bottom = st.sidebar.number_input(
    "말뚝 하단 높이 (m)",
    min_value=float(Z.min()), max_value=float(Z.max()), 
    value=float(Z.min())
)

# 4. Plotly 시각화
# 4.1 Volume
vol = go.Volume(
    x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
    value=gv.flatten(),
    isomin=np.nanpercentile(gv,5), isomax=np.nanpercentile(gv,95),
    opacity=0.1, surface_count=15, colorscale='Viridis', name='SPT Volume'
)

# 4.2 Slice Surface
idx = int(np.abs(zi - z_sel).argmin())
slice_val = gv[:, :, idx]
slice_surf = go.Surface(
    x=xi, y=yi,
    z=np.full_like(slice_val, zi[idx]),
    surfacecolor=slice_val,
    colorscale='Viridis', cmin=np.nanmin(gv), cmax=np.nanmax(gv),
    showscale=False, name=f'Slice Z≈{zi[idx]:.2f}'
)

# 4.3 시추공 Cylinder
cylinder_bh = []
theta = np.linspace(0, 2*np.pi, 30)
for bh in df_pts['BH'].unique():
    grp = df_pts[df_pts['BH']==bh].sort_values('Z')
    x0, y0 = grp[['X','Y']].iloc[0]
    z_vals = grp['Z'].values
    spt_vals = grp['SPT'].values
    # cylinder grid
    zg, tg = np.meshgrid(z_vals, theta, indexing='ij')
    xg = x0 + bh_radius * np.cos(tg)
    yg = y0 + bh_radius * np.sin(tg)
    # SPT interpolation along Z
    sptg = np.interp(zg.flatten(), z_vals, spt_vals).reshape(zg.shape)
    cyl = go.Surface(
        x=xg.T, y=yg.T, z=zg.T,
        surfacecolor=sptg.T,
        colorscale='Viridis', cmin=np.nanmin(SPT), cmax=np.nanmax(SPT),
        showscale=False, opacity=0.8, name='Borehole'
    )
    cylinder_bh.append(cyl)

# 4.4 말뚝 Cylinder
cylinder_piles = []
for x0, y0, z_top in zip(Xp, Yp, Zp):
    z_cyl = np.array([pile_bottom, z_top])
    tg, zg = np.meshgrid(theta, z_cyl, indexing='xy')
    xg = x0 + pile_radius * np.cos(tg)
    yg = y0 + pile_radius * np.sin(tg)
    cyl = go.Surface(
        x=xg, y=yg, z=zg,
        surfacecolor=np.full_like(zg, np.nan), # 단색 레드로 설정
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False, opacity=0.6, name='Pile'
    )
    cylinder_piles.append(cyl)

# 4.5 Figure 구성
fig = go.Figure(data=[vol, slice_surf] + cylinder_bh + cylinder_piles)
fig.update_layout(
    scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Elevation (m)'),
    height=700, margin=dict(r=10, l=10, b=10, t=50)
)

st.plotly_chart(fig, use_container_width=True)
