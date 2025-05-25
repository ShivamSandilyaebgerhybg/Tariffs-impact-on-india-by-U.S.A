import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import warnings

# Set up visualization styles with robust fallbacks
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('ggplot')

sns.set_theme(style="whitegrid", palette="husl")
warnings.filterwarnings('ignore')

# Load trade data (simulated)
years = list(range(2018, 2025))
us_exports = [33.18, 34.21, 28.75, 32.89, 36.42, 39.15, 41.75]
us_imports = [54.25, 56.32, 48.91, 65.23, 72.15, 80.11, 87.42]
trade_balance = [-21.07, -22.11, -20.16, -32.34, -35.73, -40.96, -45.67]

trade_df = pd.DataFrame({
    'Year': years,
    'US_Exports_to_India': us_exports,
    'US_Imports_from_India': us_imports,
    'Trade_Balance': trade_balance
})

steel_exports = [1.85, 1.20, 0.95, 1.10, 1.25, 1.35, 1.40]
aluminum_exports = [0.82, 0.94, 0.78, 0.85, 0.89, 0.92, 0.95]

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

# Merge all data
full_df = trade_df.merge(sector_df, on='Year').merge(macro_df, on='Year')

# Data checks
print("=== Data Quality Checks ===")
print("Missing values check:")
print(full_df.isnull().sum())

print("\nData types:")
print(full_df.dtypes)

print("\nSummary statistics:")
print(full_df.describe())

# Feature engineering
full_df['Total_Trade'] = full_df['US_Exports_to_India'] + full_df['US_Imports_from_India']
full_df['Export_Growth_Rate'] = full_df['US_Exports_to_India'].pct_change() * 100
full_df['Import_Growth_Rate'] = full_df['US_Imports_from_India'].pct_change() * 100
full_df['Trade_Balance_Pct_GDP'] = full_df['Trade_Balance'] / 2800 * 100

pre_tariff_steel = full_df.loc[full_df['Year'] == 2018, 'Steel_Exports'].values[0]
full_df['Steel_Export_Change'] = (full_df['Steel_Exports'] - pre_tariff_steel) / pre_tariff_steel * 100

# 1. Trade Trends Over Time
plt.figure(figsize=(12, 6))
plt.plot(full_df['Year'], full_df['US_Exports_to_India'], marker='o', label='US Exports to India')
plt.plot(full_df['Year'], full_df['US_Imports_from_India'], marker='o', label='US Imports from India')
plt.plot(full_df['Year'], full_df['Trade_Balance'], marker='o', linestyle='--', label='Trade Balance (US Deficit)')

for _, event in events_df.iterrows():
    plt.axvline(x=event['Year'], color='gray', linestyle='--', alpha=0.5)
    plt.text(event['Year'], 90, event['Event'].split('(')[0], rotation=90, va='top', ha='right', fontsize=8)

plt.title('US-India Bilateral Trade (2018–2024)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Billions USD', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(full_df['Year'])
plt.tight_layout()
plt.show()

# 2. Sectoral Impact
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(full_df['Year'], full_df['Steel_Exports'], color='steelblue')
plt.axhline(y=pre_tariff_steel, color='red', linestyle='--', label='Pre-Tariff Level (2018)')
plt.title('Indian Steel Exports to US', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Billions USD', fontsize=10)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(full_df['Year'], full_df['Aluminum_Exports'], color='lightblue')
plt.title('Indian Aluminum Exports to US', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Billions USD', fontsize=10)

plt.tight_layout()
plt.show()

# 3. Trade Composition
plt.figure(figsize=(10, 6))
plt.stackplot(full_df['Year'],
              full_df['US_Exports_to_India'],
              full_df['US_Imports_from_India'],
              labels=['US Exports', 'US Imports'],
              colors=['#1f77b4', '#ff7f0e'])
plt.plot(full_df['Year'], full_df['Trade_Balance'], marker='o', color='red', label='Trade Balance')
plt.title('US-India Trade Composition (2018–2024)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Billions USD', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(full_df['Year'])
plt.tight_layout()
plt.show()

# 4. Macroeconomic Indicators
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(full_df['Year'], full_df['GDP_Growth'], marker='o', color='green')
ax1.set_title('India GDP Growth Rate (2018–2024)', fontsize=12)
ax1.set_ylabel('Annual % Change', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(full_df['Year'], full_df['Inflation'], marker='o', color='purple')
ax2.set_title('India Inflation Rate (2018–2024)', fontsize=12)
ax2.set_xlabel('Year', fontsize=10)
ax2.set_ylabel('Annual % Change', fontsize=10)
ax2.grid(True, alpha=0.3)

ax1.set_xticks(full_df['Year'])
ax2.set_xticks(full_df['Year'])

plt.tight_layout()
plt.show()

# 5. Correlation Matrix
corr_df = full_df[['US_Exports_to_India', 'US_Imports_from_India', 'Trade_Balance',
                   'Steel_Exports', 'Aluminum_Exports', 'GDP_Growth', 'Inflation']]
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Correlation Matrix of Key Indicators', fontsize=14)
plt.tight_layout()
plt.show()

# 6. Interactive Plotly Chart
fig = px.line(full_df, x='Year', y=['US_Exports_to_India', 'US_Imports_from_India', 'Trade_Balance'],
              title='US-India Trade Trends (2018–2024)',
              labels={'value': 'Billions USD', 'variable': 'Metric'},
              markers=True)

for _, event in events_df.iterrows():
    fig.add_vline(x=event['Year'], line_width=1, line_dash="dash", line_color="gray",
                  annotation_text=event['Event'].split('(')[0],
                  annotation_position="top right",
                  annotation_font_size=10)

fig.update_layout(hovermode="x unified")
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