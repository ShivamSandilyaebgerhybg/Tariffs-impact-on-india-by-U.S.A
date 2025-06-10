import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import textwrap
from matplotlib.widgets import Slider
import warnings

# Styles & warnings
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('ggplot')

sns.set_theme(style="whitegrid", palette="husl")
warnings.filterwarnings('ignore')

# Data setup
years = np.array(list(range(2018, 2025)))
us_exports = np.array([33.18, 34.21, 28.75, 32.89, 36.42, 39.15, 41.75])
us_imports = np.array([54.25, 56.32, 48.91, 65.23, 72.15, 80.11, 87.42])
trade_balance = np.array([-21.07, -22.11, -20.16, -32.34, -35.73, -40.96, -45.67])

trade_df = pd.DataFrame({
    'Year': years,
    'US_Exports_to_India': us_exports,
    'US_Imports_from_India': us_imports,
    'Trade_Balance': trade_balance
})

steel_exports = np.array([1.85, 1.20, 0.95, 1.10, 1.25, 1.35, 1.40])
aluminum_exports = np.array([0.82, 0.94, 0.78, 0.85, 0.89, 0.92, 0.95])

sector_df = pd.DataFrame({
    'Year': years,
    'Steel_Exports': steel_exports,
    'Aluminum_Exports': aluminum_exports
})

gdp_growth = [6.8, 6.0, -5.8, 9.1, 7.2, 6.9, 7.0]
inflation = [4.9, 4.8, 6.2, 5.5, 5.7, 5.4, 4.6]

macro_df = pd.DataFrame({
    'Year': years,
    'GDP_Growth': gdp_growth,
    'Inflation': inflation
})

policy_events = [
    {'Date': '2018-03', 'Event': 'Section 232 tariffs (25% steel, 10% aluminum)'},
    {'Date': '2019-06', 'Event': 'US removes India from GSP program'},
    {'Date': '2019-06', 'Event': 'India retaliates with tariffs on 28 US products'},
    {'Date': '2021-01', 'Event': 'USTR threatens Section 301 tariffs on digital tax'},
    {'Date': '2023-06', 'Event': 'US-India agree to ease steel/aluminum trade friction'}
]

events_df = pd.DataFrame(policy_events)
events_df['Date'] = pd.to_datetime(events_df['Date'])
events_df['Year'] = events_df['Date'].dt.year

# Merge dataframes
full_df = trade_df.merge(sector_df, on='Year').merge(macro_df, on='Year')

# Feature engineering
full_df['Total_Trade'] = full_df['US_Exports_to_India'] + full_df['US_Imports_from_India']
full_df['Export_Growth_Rate'] = full_df['US_Exports_to_India'].pct_change() * 100
full_df['Import_Growth_Rate'] = full_df['US_Imports_from_India'].pct_change() * 100
full_df['Trade_Balance_Pct_GDP'] = full_df['Trade_Balance'] / 2800 * 100

pre_tariff_steel = full_df.loc[full_df['Year'] == 2018, 'Steel_Exports'].values[0]
full_df['Steel_Export_Change'] = (full_df['Steel_Exports'] - pre_tariff_steel) / pre_tariff_steel * 100

# === 1. Enhanced Trade Trends Over Time with annotations and event callouts ===
plt.figure(figsize=(14, 7))
events_df['YearFraction'] = events_df['Date'].dt.year + (events_df['Date'].dt.month - 1)/12

plt.plot(full_df['Year'], full_df['US_Exports_to_India'], marker='o', linewidth=2.5, label='US Exports', color='#2a9df4')
plt.plot(full_df['Year'], full_df['US_Imports_from_India'], marker='o', linewidth=2.5, label='US Imports', color='#f4a261')
plt.plot(full_df['Year'], full_df['Trade_Balance'], marker='D', linestyle='--', linewidth=2.2, label='Trade Balance', color='#e63946')

plt.xlabel('Year')
plt.ylabel('Billions USD')
plt.title('US-India Trade & Policy Events (2018-2024)', fontsize=16, fontweight='bold')
plt.xticks(full_df['Year'], rotation=45)
plt.grid(True, linestyle='--', alpha=0.3)

mid_y = (plt.ylim()[0] + plt.ylim()[1]) / 2
grouped = events_df.groupby('YearFraction')

for year_frac, group in grouped:
    plt.axvline(x=year_frac, color='gray', linestyle='--', alpha=0.4)
    for i, (_, event) in enumerate(group.iterrows()):
        label = textwrap.fill(event['Event'], width=25)
        x_offset = -0.15 if i % 2 == 0 else 0.15
        ha = 'right' if i % 2 == 0 else 'left'
        y_pos = mid_y + (i - len(group)/2)*5
        plt.text(year_frac + x_offset, y_pos, label,
                 rotation=90, va='center', ha=ha,
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.6))

# Callouts for major inflection points on Trade Balance
for idx, row in full_df.iterrows():
    if abs(row['Trade_Balance']) > 35:
        plt.annotate(f"{row['Trade_Balance']:.1f}B deficit", 
                     (row['Year'], row['Trade_Balance']),
                     textcoords="offset points", xytext=(0,-30),
                     ha='center', fontsize=10, color='#e63946', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#e63946'))

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# === 2. Sectoral Impact with annotations on key changes ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.bar(full_df['Year'], full_df['Steel_Exports'], color='steelblue', alpha=0.9)
plt.axhline(y=pre_tariff_steel, color='red', linestyle='--', linewidth=1.5, label='Pre-Tariff Level (2018)')
plt.title('Indian Steel Exports to US (2018–2024)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Billions USD')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

# Annotate biggest drop
min_steel = full_df['Steel_Exports'].min()
min_year = full_df.loc[full_df['Steel_Exports'] == min_steel, 'Year'].values[0]
plt.annotate(f'Min Drop: {min_steel:.2f}B in {min_year}', xy=(min_year, min_steel), 
             xytext=(min_year, min_steel+0.3),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='darkred', fontweight='bold')

plt.subplot(1, 2, 2)
plt.bar(full_df['Year'], full_df['Aluminum_Exports'], color='lightblue', alpha=0.9)
plt.title('Indian Aluminum Exports to US (2018–2024)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Billions USD')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.annotate(f'Steady growth\nfrom {aluminum_exports[0]:.2f}B to {aluminum_exports[-1]:.2f}B', 
             xy=(2021, 0.85), xytext=(2021, 1.1),
             bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.4),
             fontsize=10)

plt.tight_layout()
plt.show()

# === 3. Trade Composition Stacked Bar + Line with clearer labels and shading ===
fig, ax1 = plt.subplots(figsize=(12, 7))

export_color = '#1a5276'
import_color = '#e67e22'
balance_color = '#c0392b'

bars_export = ax1.bar(full_df['Year'], full_df['US_Exports_to_India'], color=export_color, label='US Exports', alpha=0.8)
bars_import = ax1.bar(full_df['Year'], full_df['US_Imports_from_India'], 
                      bottom=full_df['US_Exports_to_India'], color=import_color, label='US Imports', alpha=0.8)

# Add bar labels with background for readability
for bar in bars_export:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height/2, f'{height:.1f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
for bar, base in zip(bars_import, full_df['US_Exports_to_India']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, base + height/2, f'{height:.1f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

ax2 = ax1.twinx()
line = ax2.plot(full_df['Year'], full_df['Trade_Balance'], color=balance_color, marker='o', linestyle='--', linewidth=2, markersize=8, label='Trade Balance')

for year, balance in zip(full_df['Year'], full_df['Trade_Balance']):
    va = 'bottom' if balance < 0 else 'top'
    ax2.text(year, balance, f'{balance:.1f}', ha='center', va=va, color=balance_color, fontsize=9, fontweight='bold')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Billions USD', fontsize=12)
ax2.set_ylabel('Trade Balance (Billions USD)', fontsize=12)
ax1.set_title('US-India Trade Composition (2018-2024)', fontsize=16, pad=20)

ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
ax2.yaxis.grid(False)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, shadow=True, facecolor='white')

plt.tight_layout()
plt.show()

# === 4. Macroeconomic Indicators with annotations on key years ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(full_df['Year'], full_df['GDP_Growth'], marker='o', color='green')
ax1.set_title('India GDP Growth Rate (2018–2024)', fontsize=14)
ax1.set_ylabel('Annual % Change', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=full_df['GDP_Growth'].mean(), color='green', linestyle='--', alpha=0.6)
ax1.text(full_df['Year'].iloc[-1], full_df['GDP_Growth'].mean() + 0.2, f'Average: {full_df["GDP_Growth"].mean():.1f}%', color='green', fontsize=10)

ax2.plot(full_df['Year'], full_df['Inflation'], marker='s', color='orange')
ax2.set_title('India Inflation Rate (CPI, 2018–2024)', fontsize=14)
ax2.set_ylabel('Annual % Change', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=full_df['Inflation'].mean(), color='orange', linestyle='--', alpha=0.6)
ax2.text(full_df['Year'].iloc[-1], full_df['Inflation'].mean() + 0.2, f'Average: {full_df["Inflation"].mean():.1f}%', color='orange', fontsize=10)

plt.xlabel('Year')
plt.tight_layout()
plt.show()

# === 5. Interactive Matplotlib Chart with Slider & Hover ===

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

line_exports, = ax.plot(years, us_exports, label='US Exports', color='#2a9df4', marker='o')
line_imports, = ax.plot(years, us_imports, label='US Imports', color='#f4a261', marker='o')
line_balance, = ax.plot(years, trade_balance, label='Trade Balance', color='#e63946', marker='D', linestyle='--')

ax.set_xlabel('Year')
ax.set_ylabel('Billions USD')
ax.set_title('Interactive US-India Trade Trends (2018-2024)')
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)

# Slider setup
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
year_slider = Slider(ax_slider, 'Start Year', years[0], years[-1], valinit=years[0], valstep=1)

def update(val):
    yr = year_slider.val
    mask = years >= yr
    line_exports.set_data(years[mask], us_exports[mask])
    line_imports.set_data(years[mask], us_imports[mask])
    line_balance.set_data(years[mask], trade_balance[mask])
    ax.set_xlim(yr - 0.5, years[-1] + 0.5)
    ymin = min(np.min(us_exports[mask]), np.min(us_imports[mask]), np.min(trade_balance[mask]))
    ymax = max(np.max(us_exports[mask]), np.max(us_imports[mask]), np.max(trade_balance[mask]))
    ax.set_ylim(ymin - 5, ymax + 5)
    fig.canvas.draw_idle()

year_slider.on_changed(update)

# Hover event
annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind, line):
    x, y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = f"{line.get_label()}\nYear: {int(x[ind['ind'][0]])}\nValue: {y[ind['ind'][0]]:.2f}B"
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(line.get_color())
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        for line in [line_exports, line_imports, line_balance]:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind, line)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    if vis:
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()

# === 6. Optional: Interactive Plotly Chart (uncomment to enable) ===


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(
    x=full_df['Year'], y=full_df['US_Exports_to_India'],
    mode='lines+markers', name='US Exports', line=dict(color='#2a9df4')))

fig.add_trace(go.Scatter(
    x=full_df['Year'], y=full_df['US_Imports_from_India'],
    mode='lines+markers', name='US Imports', line=dict(color='#f4a261')))

fig.add_trace(go.Scatter(
    x=full_df['Year'], y=full_df['Trade_Balance'],
    mode='lines+markers', name='Trade Balance', line=dict(color='#e63946'),
    yaxis='y2'))

for _, event in events_df.iterrows():
    fig.add_vline(x=event['Year'], line_dash='dot', line_color='gray')
    fig.add_annotation(
        x=event['Year'], y=max(full_df['US_Imports_from_India']),
        text=event['Event'], showarrow=True, arrowhead=1,
        ax=0, ay=-40, bgcolor="lightyellow", opacity=0.7,
        font=dict(size=9))

fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True, True, True]}]),
                dict(label="Exports Only",
                     method="update",
                     args=[{"visible": [True, False, False]}]),
                dict(label="Imports Only",
                     method="update",
                     args=[{"visible": [False, True, False]}]),
                dict(label="Trade Balance Only",
                     method="update",
                     args=[{"visible": [False, False, True]}]),
            ]),
            direction="down",
            showactive=True,
        )
    ]
)
fig.show()
# Insights
print("\n=== Key Insights ===")
print("1. Trade Trends:")
print(f"- US-India bilateral trade grew from ${full_df.loc[0, 'Total_Trade']:.1f}B in 2018 to ${full_df.loc[6, 'Total_Trade']:.1f}B in 2024")
print(f"- US trade deficit with India widened from ${full_df.loc[0, 'Trade_Balance']:.1f}B to ${full_df.loc[6, 'Trade_Balance']:.1f}B")

print("\n2. Sectoral Impacts:")
print(f"- Steel exports dropped by {full_df.loc[1, 'Steel_Export_Change']:.1f}% in 2019 after tariffs")
print(f"- Aluminum exports showed resilience, growing from ${aluminum_exports[0]:.2f}B to ${aluminum_exports[-1]:.2f}B")

print("\n3. Macroeconomic Impact:")
print(f"- India's GDP growth averaged {full_df['GDP_Growth'].mean():.1f}% despite tariffs")
print(f"- Inflation stayed within target range, averaging {full_df['Inflation'].mean():.1f}%")

print("\n4. Policy Events Impact:")
print("- Section 232 tariffs (2018) had immediate impact on steel exports")
print("- GSP removal (2019) affected $5.6B of Indian exports but overall trade continued growing")
print("- 2023 agreements helped normalize steel/aluminum trade flows")

# Save to CSV
full_df.to_csv('us_india_trade_analysis_2018_2024.csv', index=False)
print("\nAnalysis complete. Data saved to 'us_india_trade_analysis_2018_2024.csv'")

