# app.py (Optimized & Borehole Top Surface Fix)
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
        if '.1' in col: continue
        depths = df_info[col].values
        spts   = df_info[f"{col}.1"].values
        loc    = df_loc[df_loc.iloc[:,0] == col].iloc[0]
        x0, y0, ztop = loc['X좌표(m)'], loc['Y좌표(m)'], loc['상단높이(m)']
        for d, s in zip(depths, spts):
            pts.append((col, x0, y0, ztop - d, s, ztop))
    df_pts = pd.DataFrame(pts, columns=['BH','X','Y','Z','SPT','top_z'])
    return df_pts, df_pile

df_pts, df_pile = load_and_prepare(uploaded)
# 배열 추출
X, Y, Z, SPT = df_pts[['X','Y','Z','SPT']].values.T
Xp, Yp, Zp    = df_pile[['X','Y','상단높이(m)']].values.T

# 2D Top surface interpolation (cache) using borehole tops
@st.cache_data(show_spinner=False)
def build_top_surface(xp, yp, zp, xi, yi):
    grid_x2, grid_y2 = np.meshgrid(xi, yi, indexing='xy')
    grid_z2 = griddata((xp, yp), zp, (grid_x2, grid_y2), method='linear')
    return grid_z2

# 2. 실제 Z 범위
z_min, z_max = float(Z.min()), float(Z.max())

# 3. 사이드바 컨트롤
render_modes = ['None', 'Volume', 'Top Surface']
render_mode = st.sidebar.selectbox("렌더링 모드 선택", render_modes, index=0)
show_slice   = st.sidebar.checkbox("단면 표시", value=True)
z_sel        = st.sidebar.slider("단면 Z 높이 (m)", z_min, z_max, (z_min+z_max)/2)
pile_r       = st.sidebar.number_input("말뚝 반경 (m)", 0.01, 5.0, 0.2, 0.01)
pile_bot     = st.sidebar.number_input("말뚝 하단 높이 (m)", z_min, z_max, z_min)
bh_r         = st.sidebar.number_input("시추공 반경 (m)", 0.01, 5.0, 0.1, 0.01)

# 4. 그리드 보간 (캐시)
@st.cache_data(show_spinner=False)
def build_grid(x, y, z, val, nx=30, ny=30, nz=15):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    zi = np.linspace(z.min(), z.max(), nz)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='xy')
    gv = griddata((x, y, z), val, (gx, gy, gz), method='linear')
    return xi, yi, zi, gx, gy, gz, gv

xi, yi, zi, gx, gy, gz, gv = build_grid(X, Y, Z, SPT)
# borehole top coordinates
df_bh_tops = df_pts.groupby('BH')[['X','Y','top_z']].first().reset_index(drop=True)
xb = df_bh_tops['X'].values
yb = df_bh_tops['Y'].values
zb = df_bh_tops['top_z'].values
top_surface = build_top_surface(xb, yb, zb, xi, yi)

# 5. 3D Figure 생성
traces = []

# 5.1 Volume 렌더링
if render_mode == 'Volume':
    traces.append(go.Volume(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=gv.ravel(),
        isomin=np.nanpercentile(gv,10), isomax=np.nanpercentile(gv,90),
        opacity=0.05, surface_count=10, colorscale='Viridis', name='SPT Volume'
    ))

# 5.2 Top Surface 연결 (시추공 최상단)
elif render_mode == 'Top Surface':
    traces.append(go.Surface(
        x=xi, y=yi, z=top_surface,
        colorscale='Greys', showscale=False, opacity=0.7,
        name='Top Borehole Surface'
    ))

# 5.3 단면 Surface
if show_slice:
    idx = int((np.abs(zi - z_sel)).argmin())
    z_plane = zi[idx]
    slice_val = gv[:,:,idx]
    traces.append(go.Surface(
        x=xi, y=yi, z=np.full_like(slice_val, z_plane),
        surfacecolor=slice_val, cmin=np.nanmin(gv), cmax=np.nanmax(gv),
        showscale=True, colorbar=dict(title='SPT N', len=0.5, y=0.7), opacity=0.8,
        name=f"Slice Z={z_plane:.2f}"
    ))

# 5.4 말뚝 & 보어홀 원통
theta = np.linspace(0, 2*np.pi, 16)
# 말뚝
for x0, y0, zt in zip(Xp, Yp, Zp):
    zc = np.array([pile_bot, zt])
    th, zgr = np.meshgrid(theta, zc)
    traces.append(go.Surface(
        x=(x0 + pile_r*np.cos(th)).T, y=(y0 + pile_r*np.sin(th)).T, z=zgr.T,
        surfacecolor=zgr.T*0, colorscale=[[0,'red'],[1,'red']], showscale=False, opacity=0.5,
        name='Pile'
    ))
# 보어홀
for bh, grp in df_pts.groupby('BH'):
    x0, y0, ztop = grp[['X','Y','top_z']].iloc[0]
    zbot = grp['Z'].min()
    zc = np.array([zbot, ztop])
    th, zgr = np.meshgrid(theta, zc)
    traces.append(go.Surface(
        x=(x0 + bh_r*np.cos(th)).T, y=(y0 + bh_r*np.sin(th)).T, z=zgr.T,
        surfacecolor=zgr.T*0, colorscale=[[0,'gray'],[1,'gray']], showscale=False, opacity=0.3,
        name='Borehole'
    ))

# 6. 레이아웃 & 출력
fig = go.Figure(traces)
fig.update_layout(
    scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Elev (m)', aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=30), height=700
)
st.plotly_chart(fig, use_container_width=True)
