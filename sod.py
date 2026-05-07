import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from case_description import OpenFOAMCase

rootdir="/Volumes/SD/ChemPhysRAS/TEA/"

case_paths=[x for x in Path('/Volumes/SD/ChemPhysRAS/TEA/data/sod/rhoRFoam/air').iterdir() if x.is_dir() and x.name.startswith('k_')]


case=[OpenFOAMCase.from_path(path) for path in case_paths] 
for c in case:
    c.load_fields(["p", "U","rho", "T", "SoundSpeed"])

from scipy.optimize import brentq
import math
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- gas + initial state ----
gamma = 1.4
R     = 287.05
T0    = 293
p1    = 101325.0
p4    = 33.17 * p1
rho1  = p1 / (R * T0)
rho4  = p4 / (R * T0)
a1    = np.sqrt(gamma * p1 / rho1)
a4    = np.sqrt(gamma * p4 / rho4)

def window(df, min=0, max=10):
    return df.loc[(df['x_m'] > min) & (df['x_m'] < max)]

def sod_exact(x, t, x0=0.0):
    g = gamma
    def f(p2p1):
        rhs = p2p1 * (1 - (g-1)*(a1/a4)*(p2p1 - 1) /
                      np.sqrt(2*g*(2*g + (g+1)*(p2p1 - 1))))**(-2*g/(g-1))
        return rhs - p4/p1
    p2p1 = brentq(f, 1.0+1e-9, p4/p1)
    p2   = p2p1 * p1
    Ms   = np.sqrt((g+1)/(2*g)*p2p1 + (g-1)/(2*g))
    rho2 = rho1 * ((g+1)*p2p1 + (g-1)) / ((g-1)*p2p1 + (g+1))
    u2   = (a1/g) * (p2p1 - 1) * np.sqrt(2*g/(g+1) / (p2p1 + (g-1)/(g+1)))
    p3, u3 = p2, u2
    rho3 = rho4 * (p3/p4)**(1/g)
    a3   = np.sqrt(g*p3/rho3)
    xs = x0 + Ms * a1 * t
    xc = x0 + u2 * t
    xt = x0 + (u3 - a3) * t
    xh = x0 - a4 * t
    p = np.empty_like(x); rho = np.empty_like(x); u = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi < xh:
            p[i], rho[i], u[i] = p4, rho4, 0.0
        elif xi < xt:
            u[i]   = 2/(g+1) * (a4 + (xi - x0)/t)
            a_loc  = a4 - (g-1)/2 * u[i]
            p[i]   = p4 * (a_loc/a4)**(2*g/(g-1))
            rho[i] = rho4 * (a_loc/a4)**(2/(g-1))
        elif xi < xc:
            p[i], rho[i], u[i] = p3, rho3, u3
        elif xi < xs:
            p[i], rho[i], u[i] = p2, rho2, u2
        else:
            p[i], rho[i], u[i] = p1, rho1, 0.0
    T    = p / (R * rho)
    Mach = u / np.sqrt(gamma * p / rho)
    return p, rho, u, T, Mach

def tidy(df, var, x0, t_snap):
    d = df[np.isclose(df['t'], t_snap)].copy()
    d['xi'] = (d['x_m'] - x0) / d['t']
    return d[['xi', var]].sort_values('xi')

# ---- load all cases ----
case_data = []
for i, c in enumerate(case):
    c.load_fields(["p", "U", "rho", "T", "SoundSpeed"])
    middle   = int(c.blocks[0].ny / 2)
    membrane = 9
    delta    = membrane / 9

    p_df   = window(c.data_p[c.data_p['j']     == middle], min=membrane-delta, max=membrane+delta)
    rho_df = window(c.data_rho[c.data_rho['j'] == middle], min=membrane-delta, max=membrane+delta)
    u_df   = window(c.data_U[c.data_U['j']     == middle], min=membrane-delta, max=membrane+delta).rename(columns={"U_x": "u"})
    ss_df  = window(c.data_SoundSpeed[c.data_SoundSpeed['j'] == middle], min=membrane-delta, max=membrane+delta)
    T_df   = window(c.data_T[c.data_T['j']     == middle], min=membrane-delta, max=membrane+delta)

    mach_df = u_df[['x_m', 't', 'j']].copy()
    mach_df['Mach'] = u_df['u'].values / ss_df['SoundSpeed'].values

    name = 'cell = ' + str(abs(1000 * c.blocks[0].z_min)) + ' mm'
    case_data.append({
        'name': name,
        'x0':   9,
        'dfs':  {'p': p_df, 'rho': rho_df, 'u': u_df, 'T': T_df, 'Mach': mach_df},
    })
    case_data=sorted(case_data, key=lambda cd: cd['name'], reverse=True)  # sort by name for consistent legend order
# ---- shared time axis ----
all_t = None
for cd in case_data:
    t_set = set(np.round(cd['dfs']['p']['t'].unique(), decimals=8))
    all_t = t_set if all_t is None else all_t & t_set
all_t = sorted(all_t)

vars_list = ['p', 'rho', 'u', 'T', 'Mach']
ylabels   = {'p': 'p [Pa]', 'rho': 'ρ [kg/m³]', 'u': 'u [m/s]', 'T': 'T [K]', 'Mach': 'Mach [-]'}
n_vars    = len(vars_list)
n_cases   = len(case_data)

# one distinct colour per case, reused in both figures
CASE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# -----------------------------------------------------------------------
# Figure 1: sim vs exact for ONE case, with case name in each subplot title
# -----------------------------------------------------------------------
def build_figure(c_idx, t_idx):
    cd    = case_data[c_idx]
    x0    = cd['x0']
    t_val = all_t[t_idx]

    x_dense = np.linspace(cd['dfs']['p']['x_m'].min(),
                          cd['dfs']['p']['x_m'].max(), 2000)
    p_ex, rho_ex, u_ex, T_ex, Mach_ex = sod_exact(x_dense, t_val, x0)
    ex_map = {'p': p_ex, 'rho': rho_ex, 'u': u_ex, 'T': T_ex, 'Mach': Mach_ex}
    ex_xi  = (x_dense - x0) / t_val

    specs = [[{"secondary_y": True}] for _ in range(n_vars)]

    # Case name + variable in every subplot title
    subplot_titles = [f'{var}  —  {cd["name"]}' for var in vars_list]

    fig = make_subplots(
        rows=n_vars, cols=1,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    for v_idx, var in enumerate(vars_list):
        s       = tidy(cd['dfs'][var], var, x0, t_val)
        ex_vals = ex_map[var]

        exact_on_sim = np.interp(s['xi'].values, ex_xi, ex_vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = 100.0 * (s[var].values - exact_on_sim) / exact_on_sim
            pct[~np.isfinite(pct)] = np.nan
            pct = np.clip(pct, -50, 50)

        showlegend = (v_idx == 0)

        fig.add_trace(go.Scatter(
            x=s['xi'], y=s[var], mode='lines', name='sim',
            line=dict(color='steelblue'),
            showlegend=showlegend, legendgroup='sim'),
            row=v_idx+1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=ex_xi, y=ex_vals, mode='lines', name='exact',
            line=dict(color='black'),
            showlegend=showlegend, legendgroup='exact'),
            row=v_idx+1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=s['xi'], y=pct, mode='lines', name='Δ%',
            line=dict(color='crimson', dash='dot'),
            showlegend=showlegend, legendgroup='delta'),
            row=v_idx+1, col=1, secondary_y=True)

        fig.update_xaxes(title_text='(x−x₀)/t  [m/s]', row=v_idx+1, col=1)
        fig.update_yaxes(title_text=ylabels[var], row=v_idx+1, col=1, secondary_y=False)
        fig.update_yaxes(title_text='error [%]',  row=v_idx+1, col=1, secondary_y=True)

    fig.update_layout(
        title=f'rhoReactingFoam — {cd["name"]}  |  t = {t_val:.6f} s',
        height=300 * n_vars,
        width=950,
        margin=dict(t=60),
    )
    return fig

# -----------------------------------------------------------------------
# Figure 2: error % for ALL cases on the same subplots, one row per variable
# -----------------------------------------------------------------------
def build_error_figure(t_idx):
    t_val = all_t[t_idx]

    fig = make_subplots(
        rows=n_vars, cols=1,
        subplot_titles=[f'{var} error  |  all cases' for var in vars_list],
        vertical_spacing=0.08,
    )

    for v_idx, var in enumerate(vars_list):
        for c_idx, cd in enumerate(case_data):
            x0      = cd['x0']
            x_dense = np.linspace(cd['dfs']['p']['x_m'].min(),
                                  cd['dfs']['p']['x_m'].max(), 2000)
            p_ex, rho_ex, u_ex, T_ex, Mach_ex = sod_exact(x_dense, t_val, x0)
            ex_map  = {'p': p_ex, 'rho': rho_ex, 'u': u_ex, 'T': T_ex, 'Mach': Mach_ex}
            ex_xi   = (x_dense - x0) / t_val
            ex_vals = ex_map[var]

            s = tidy(cd['dfs'][var], var, x0, t_val)

            exact_on_sim = np.interp(s['xi'].values, ex_xi, ex_vals)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = 100.0 * (s[var].values - exact_on_sim) / exact_on_sim
                pct[~np.isfinite(pct)] = np.nan
                pct = np.clip(pct, -50, 50)

            fig.add_trace(go.Scatter(
                x=s['xi'], y=pct,
                mode='lines',
                name=cd['name'],
                line=dict(color=CASE_COLORS[c_idx % len(CASE_COLORS)]),
                # show legend only on first variable row to avoid duplication
                showlegend=(v_idx == 0),
                legendgroup=cd['name'],
            ), row=v_idx+1, col=1)

        fig.add_hline(y=0, line=dict(color='black', width=0.8, dash='dot'),
                      row=v_idx+1, col=1)
        fig.update_xaxes(title_text='(x−x₀)/t  [m/s]', row=v_idx+1, col=1)
        fig.update_yaxes(title_text='error [%]', row=v_idx+1, col=1)

    fig.update_layout(
        title=f'Error vs exact — all cases  |  t = {t_val:.6f} s',
        height=250 * n_vars,
        width=950,
        margin=dict(t=60),
    )
    return fig

# -----------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------
app = Dash(__name__)

controls = html.Div([
    html.Label('Case:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
    dcc.Dropdown(
        id='case-dropdown',
        options=[{'label': cd['name'], 'value': i} for i, cd in enumerate(case_data)],
        value=0,
        clearable=False,
        style={'width': '260px', 'display': 'inline-block', 'verticalAlign': 'middle'},
    ),
    html.Label('Time:', style={
        'fontWeight': 'bold', 'marginLeft': '24px', 'marginRight': '8px'}),
    dcc.Dropdown(
        id='time-dropdown',
        options=[{'label': f'{t:.6f} s', 'value': i} for i, t in enumerate(all_t)],
        value=0,
        clearable=False,
        style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'},
    ),
], style={
    'position': 'fixed',      # stick to viewport
    'top': 0,
    'left': 0,
    'right': 0,
    'zIndex': 9999,           # stay above plots when scrolling
    'padding': '12px 16px',
    'background': '#f5f5f5',
    'borderBottom': '1px solid #ddd',
    'boxShadow': '0 2px 6px rgba(0,0,0,0.15)',  # subtle shadow to separate from content
})

# ---- parameter panel ----
param_panel = html.Div([
    html.H4('Initial State', style={'marginTop': 0, 'marginBottom': '12px',
                                    'borderBottom': '1px solid #ccc', 'paddingBottom': '6px'}),
    html.Table([
        html.Tr([html.Td('γ',    style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{gamma}')]),
        html.Tr([html.Td('R',    style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{R} J/(kg·K)')]),
        html.Tr([html.Td('T₀',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{T0} K')]),
        html.Tr([html.Td('p₁',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{p1:.2f} Pa')]),
        html.Tr([html.Td('p₄',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{p4:.2f} Pa')]),
        html.Tr([html.Td('p₄/p₁', style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{p4/p1:.2f}')]),
        html.Tr([html.Td('ρ₁',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{rho1:.4f} kg/m³')]),
        html.Tr([html.Td('ρ₄',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{rho4:.4f} kg/m³')]),
        html.Tr([html.Td('a₁',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{a1:.2f} m/s')]),
        html.Tr([html.Td('a₄',   style={'paddingRight': '12px', 'color': '#555'}),
                 html.Td(f'{a4:.2f} m/s')]),
    ], style={'borderCollapse': 'collapse', 'fontSize': '13px', 'lineHeight': '2.0'}),
], style={
    'position': 'sticky',       # stays in view while scrolling within the flex row
    'top': '70px',              # clears the fixed controls bar
    'alignSelf': 'flex-start',  # prevents stretching to full column height
    'width': '220px',
    'minWidth': '220px',
    'background': '#fafafa',
    'border': '1px solid #ddd',
    'borderRadius': '6px',
    'padding': '16px',
    'marginTop': '16px',
    'marginRight': '16px',
    'boxShadow': '0 1px 4px rgba(0,0,0,0.08)',
    'fontFamily': 'monospace',
})

# ---- graphs column ----
graphs_col = html.Div([
    html.H3('Simulation vs Exact', style={'margin': '16px 16px 0'}),
    dcc.Graph(id='main-graph', figure=build_figure(0, 0)),
    html.Hr(),
    html.H3('Error comparison — all cases', style={'margin': '16px 16px 0'}),
    dcc.Graph(id='error-graph', figure=build_error_figure(0)),
], style={'flex': '1', 'minWidth': 0})   # minWidth:0 prevents flex overflow

app.layout = html.Div([
    # ---- fixed top bar ----
    controls,
    html.Div(style={'height': '60px'}),   # spacer

    # ---- body: graphs + param panel side by side ----
    html.Div([
        graphs_col,
        param_panel,
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'flex-start',
    }),
])

@callback(
    Output('main-graph', 'figure'),
    Input('case-dropdown', 'value'),
    Input('time-dropdown', 'value'),
)
def update_main(c_idx, t_idx):
    return build_figure(int(c_idx), int(t_idx))

@callback(
    Output('error-graph', 'figure'),
    Input('time-dropdown', 'value'),   # only time matters — all cases always shown
)
def update_error(t_idx):
    return build_error_figure(int(t_idx))

app.run(jupyter_mode='external', debug=False)