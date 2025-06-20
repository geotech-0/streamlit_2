# app.py (Optimized & Fixed Z Slider + Borehole Fix)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(layout="wide")
st.title("⚡️ Optimized 3D 지반 모델링 & Slice Viewer")

# 0. 파일 업로드
uploaded = st.sidebar.file_uploader("📁 엑셀 업로드 (.xlsx)", type="xlsx")
if not uploaded:
    st.sidebar.info("시추공·말뚝 엑셀 파일을 업로드하세요.")
    st.stop()

# 1. 데이터 로드 & 전처리 (캐시)
@st.cache_data(show_spinner=False)
def load_and_prepare(buffer):
    df_loc      = pd.read_excel(buffer, sheet_name='시추주상도 위치')
    df_info_raw = pd.read_excel(buffer, sheet_name='시추주상도 정보', header=0)
    df_pile     = pd.read_excel(buffer, sheet_name='말뚝 정보')
    df_info = df_info_raw.drop(index=0).apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

    pts = []
    for col in df_info.columns:
        if '.1' in col:
            continue
        depths = df_info[col].values
        spts   = df_info[f"{col}.1"].values
        loc    = df_loc[df_loc.iloc[:,0] == col].iloc[0]
        x0, y0, ztop = loc['X좌표(m)'], loc['Y좌표(m)'], loc['상단높이(m)']
        for d, s in zip(depths, spts):
            pts.append((col, x0, y0, ztop - d, s, ztop))
    df_pts = pd.DataFrame(pts, columns=['BH','X','Y','Z','SPT','top_z'])
    return df_pts, df_pile

# 데이터 로드
df_pts, df_pile = load_and_prepare(uploaded)
# 배열 추출
X, Y, Z, SPT = df_pts[['X','Y','Z','SPT']].values.T
Xp, Yp, Zp    = df_pile[['X','Y','상단높이(m)']].values.T

# 실제 Z 범위
z_min, z_max = float(Z.min()), float(Z.max())

# 2. 사이드바 컨트롤
use_volume = st.sidebar.checkbox("볼륨 렌더링 사용", value=False)
z_sel      = st.sidebar.slider(
    "단면 Z 높이 (m)",
    z_min, z_max,
    (z_min + z_max) / 2.0
)
pile_r   = st.sidebar.number_input("말뚝 반경 (m)", min_value=0.01, max_value=5.0, value=0.2, step=0.01)
pile_bot = st.sidebar.number_input("말뚝 하단 높이 (m)", min_value=z_min, max_value=z_max, value=z_min)
bh_r     = st.sidebar.number_input("시추공 반경 (m)", min_value=0.01, max_value=5.0, value=0.1, step=0.01)

# 3. 그리드 보간 (캐시)
@st.cache_data(show_spinner=False)
def build_grid(x, y, z, val, nx=30, ny=30, nz=15):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    zi = np.linspace(z.min(), z.max(), nz)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='xy')
    gv = griddata((x, y, z), val, (gx, gy, gz), method='linear')
    return xi, yi, zi, gx, gy, gz, gv

xi, yi, zi, gx, gy, gz, gv = build_grid(X, Y, Z, SPT)

# 4. 3D Figure 생성
traces = []

# 4.1 볼륨 또는 아이소서페이스
if use_volume:
    traces.append(
        go.Volume(
            x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=gv.ravel(),
            isomin=np.nanpercentile(gv,10),
            isomax=np.nanpercentile(gv,90),
            opacity=0.05, surface_count=10, colorscale='Viridis',
            name='SPT Volume'
        )
    )
else:
    traces.append(
        go.Isosurface(
            x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=gv.ravel(),
            isomin=np.nanpercentile(gv,50),
            isomax=np.nanpercentile(gv,50),
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale='Viridis', opacity=0.4,
            name='SPT Isosurface'
        )
    )

# 4.2 단면 Surface
idx = int((np.abs(zi - z_sel)).argmin())
z_plane = zi[idx]
slice_val = gv[:,:,idx]
traces.append(
    go.Surface(
        x=xi, y=yi, z=np.full_like(slice_val, z_plane),
        surfacecolor=slice_val,
        cmin=np.nanmin(gv), cmax=np.nanmax(gv),
        showscale=False, opacity=0.8,
        name=f"Slice Z={z_plane:.2f}"
    )
)

# 4.3 말뚝 원통 (저해상도)
theta = np.linspace(0, 2*np.pi, 16)
for x0, y0, zt in zip(Xp, Yp, Zp):
    zc = np.array([pile_bot, zt])
    th, zgr = np.meshgrid(theta, zc)
    traces.append(
        go.Surface(
            x=(x0 + pile_r * np.cos(th)).T,
            y=(y0 + pile_r * np.sin(th)).T,
            z=zgr.T,
            showscale=False,
            opacity=0.5,
            surfacecolor=zgr.T*0 + 1,
            colorscale=[[0,'red'],[1,'red']],
            name='Pile'
        )
    )

# 4.4 보어홀 원통 (수정)
for bh, grp in df_pts.groupby('BH'):
    x0, y0, ztop = grp[['X','Y','top_z']].iloc[0]
    zbot = grp['Z'].min()
    zc = np.array([zbot, ztop])
    th, zgr = np.meshgrid(theta, zc)
    traces.append(
        go.Surface(
            x=(x0 + bh_r * np.cos(th)).T,
            y=(y0 + bh_r * np.sin(th)).T,
            z=zgr.T,
            showscale=False,
            opacity=0.3,
            surfacecolor=zgr.T*0 + 1,
            colorscale=[[0,'gray'],[1,'gray']],
            name='Borehole'
        )
    )

# 5. 레이아웃 & 출력
fig = go.Figure(traces)
fig.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Elevation (m)',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=700
)

st.plotly_chart(fig, use_container_width=True)
