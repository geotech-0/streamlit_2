# app.py (Optimized)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("⚡️ Optimized 3D 지반 모델링 & Slice Viewer")

# --- 사이드바 UI ---
uploaded = st.sidebar.file_uploader("📁 엑셀 업로드", type="xlsx")
if not uploaded:
    st.sidebar.info("엑셀 파일을 업로드하세요.")
    st.stop()

use_volume = st.sidebar.checkbox("볼륨 렌더링 사용", value=False)
z_sel      = st.sidebar.slider("단면 Z 높이 (m)", 0.0, 1.0, 0.5)  # 실제 min/max로 교체
pile_r     = st.sidebar.number_input("말뚝 반경 (m)", 0.01, 5.0, 0.2, 0.01)
pile_bot   = st.sidebar.number_input("말뚝 하단 높이 (m)", 0.0, 1.0, 0.0)  # 실제 min/max
bh_r       = st.sidebar.number_input("시추공 반경 (m)", 0.01, 5.0, 0.1, 0.01)

# --- 데이터 로드 & 전처리 (캐시) ---
@st.cache_data(show_spinner=False)
def load_and_prepare(buffer):
    df_loc      = pd.read_excel(buffer, sheet_name='시추주상도 위치')
    df_info_raw = pd.read_excel(buffer, sheet_name='시추주상도 정보', header=0)
    df_pile     = pd.read_excel(buffer, sheet_name='말뚝 정보')
    df_info = df_info_raw.drop(index=0).apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

    pts = []
    for col in df_info.columns:
        if '.1' in col: continue
        depths = df_info[col].values
        spts   = df_info[f"{col}.1"].values
        loc    = df_loc[df_loc.iloc[:,0]==col].iloc[0]
        x0,y0,ztop = loc['X좌표(m)'], loc['Y좌표(m)'], loc['상단높이(m)']
        for d, s in zip(depths, spts):
            pts.append((col, x0, y0, ztop-d, s, ztop))
    df_pts = pd.DataFrame(pts, columns=['BH','X','Y','Z','SPT','top_z'])
    return df_pts, df_pile

df_pts, df_pile = load_and_prepare(uploaded)
X, Y, Z, SPT = df_pts[['X','Y','Z','SPT']].values.T
Xp, Yp, Zp      = df_pile[['X','Y','상단높이(m)']].values.T

# --- 그리드 보간 (캐시) ---
@st.cache_data(show_spinner=False)
def build_grid(x, y, z, val, nx=30, ny=30, nz=15):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    zi = np.linspace(z.min(), z.max(), nz)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='xy')
    gv = griddata((x, y, z), val, (gx, gy, gz), method='linear')
    return xi, yi, zi, gx, gy, gz, gv

xi, yi, zi, gx, gy, gz, gv = build_grid(X, Y, Z, SPT)

# --- 3D Figure 생성 ---
traces = []

# 1) 볼륨 or 아이소서페이스
if use_volume:
    vol = go.Volume(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=gv.ravel(),
        isomin=np.nanpercentile(gv,10),
        isomax=np.nanpercentile(gv,90),
        opacity=0.05, surface_count=10, colorscale='Viridis',
        name='SPT Volume'
    )
    traces.append(vol)
else:
    iso = go.Isosurface(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=gv.ravel(),
        isomin=np.nanpercentile(gv,50),
        isomax=np.nanpercentile(gv,50),
        surface_count=1, caps=dict(x_show=False,y_show=False,z_show=False),
        colorscale='Viridis', opacity=0.4,
        name='SPT Isosurface'
    )
    traces.append(iso)

# 2) Slice Surface
idx = int((np.abs(zi - z_sel)).argmin())
z_plane = zi[idx]
slice_val = gv[:,:,idx]
slice_surf = go.Surface(
    x=xi, y=yi, z=np.full_like(slice_val, z_plane),
    surfacecolor=slice_val, cmin=np.nanmin(gv), cmax=np.nanmax(gv),
    showscale=False, opacity=0.8,
    name=f"Slice Z={z_plane:.2f}"
)
traces.append(slice_surf)

# 3) Borehole & Pile Cylinders (저해상도로)
theta = np.linspace(0,2*np.pi,16)
for x0,y0,zt in zip(Xp, Yp, Zp):
    zc = np.array([pile_bot, zt])
    th, zgr = np.meshgrid(theta, zc)
    traces.append(go.Surface(
        x=(x0 + pile_r*np.cos(th)).T,
        y=(y0 + pile_r*np.sin(th)).T,
        z=zgr.T, showscale=False, opacity=0.5,
        surfacecolor=zgr.T*0 + 1,  # 단색 레벨
        colorscale=[[0,'red'],[1,'red']],
        name='Pile'
    ))

for bh, grp in df_pts.groupby('BH'):
    x0, y0, ztop = grp[['X','Y','top_z']].iloc[0]
    zbot = grp['Z'].min()
    zc = np.array([zbot, ztop])
    th, zgr = np.meshgrid(theta, zc)
    traces.append(go.Surface(
        x=(x0 + bh_r*np.cos(th)).T,
        y=(y0 + bh_r*np.sin(th)).T,
        z=zgr.T, showscale=False, opacity=0.3,
        surfacecolor=zgr.T*0 + 1,
        colorscale=[[0,'gray'],[1,'gray']],
        name='Borehole'
    ))

# 4) 최종 레이아웃
fig = go.Figure(traces)
fig.update_layout(
    scene=dict(
        xaxis_title='X(m)', yaxis_title='Y(m)', zaxis_title='Elev(m)',
        aspectmode='data'
    ),
    margin=dict(l=0,r=0,b=0,t=30), height=700
)
st.plotly_chart(fig, use_container_width=True)
