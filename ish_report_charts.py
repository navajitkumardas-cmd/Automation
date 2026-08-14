import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Data from ISH weekly report
days = ['Sat 08-08', 'Sun 08-09', 'Mon 08-10', 'Tue 08-11', 'Wed 08-12', 'Thu 08-13', 'Fri 08-14']
short_days = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# Production data
issuance = [4553.3, 4946.5, 6142.3, 5121.4, 4697.7, 8351.3, 1616.1]
produced = [87.0, 138.2, 4.9, 105.5, 61.0, 245.4, 14.7]
dispatch = [5175.1, 6024.7, 5289.3, 6266.0, 5667.2, 5860.3, 5723.3]
wastage = [6248.1, 4059.7, 7072.4, 0.0, 9665.7, 4105.3, 0.0]
prm_packing = [5051.0, 5123.0, 6857.0, 5482.0, 5921.0, 6198.0, 1084.0]

# Machine scans
scans = [4289, 4449, 4338, 4385, 4504, 4332, 3103]

# Outlet orders
outlet_orders = [631, 461, 534, 502, 467, 570, 464]
outlet_lines = [9134, 6912, 8016, 7530, 7005, 8550, 6960]

# Vendor orders
vendor_orders = [14, 0, 24, 25, 26, 37, 2]

# Machine health data
machines = ['Savoury-WIS', 'Kaju-KIS', 'Barfi-KIS', 'Chaats SFG', 'Sweet SFG', 'Premium-KIS', 'Bengali-WIS', 'Packaging-KIS', 'test', 'Warehouse KIT', 'demo']
machine_scans = [8344, 5122, 3974, 3955, 2913, 2741, 2351, 0, 0, 0, 0]
machine_health = ['High latency', 'High latency', 'Offline', 'ok', 'Offline+High latency', 'Offline', 'Offline+High latency', 'idle', 'idle', 'idle', 'idle']
machine_sections = ['Chaat', 'Kaju', 'Barfi', 'Chaat', 'Bengali', 'Premium', 'Bengali', 'Packing', 'Kaju', 'Day Store', 'Bengali']

# Production plans
plans_created = [3, 3, 3, 5, 5, 5, 5]
items_planned = [288, 288, 288, 480, 480, 480, 480]

# Weekly totals
weekly_totals = {
    'Issuance': 35428.6,
    'Produced': 656.6,
    'Dispatch': 40005.9,
    'Wastage': 31151.2,
    'Prm Packing': 35716.0
}

# Line fill rate
lines_total = 16980
lines_full = 9512
lines_short = 7468
lines_over = 2433

os.makedirs('ish_report_charts', exist_ok=True)

# Color palette
colors = {
    'issuance': '#2E86AB',
    'produced': '#A23B72',
    'dispatch': '#F18F01',
    'wastage': '#C73E1D',
    'packing': '#6A994E',
    'scans': '#3B1F2B',
    'orders': '#BC4B51',
    'vendor': '#5C4B7A',
    'ok': '#2A9D8F',
    'warning': '#E9C46A',
    'danger': '#E76F51',
    'idle': '#8D99AE'
}

# ============================================================
# Chart 1: Daily Production Metrics (Issuance, Produced, Dispatch)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(days))
width = 0.25

bars1 = ax.bar(x - width, issuance, width, label='Issuance (Kg)', color=colors['issuance'], alpha=0.85)
bars2 = ax.bar(x, [p * 100 for p in produced], width, label='Produced (Qty ×100)', color=colors['produced'], alpha=0.85)
bars3 = ax.bar(x + width, [d / 100 for d in dispatch], width, label='Dispatch (Qty ÷100)', color=colors['dispatch'], alpha=0.85)

ax.set_xlabel('Day', fontsize=11)
ax.set_ylabel('Volume', fontsize=11)
ax.set_title('ISH Daily Production Overview — Week of Aug 8-14, 2026', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(short_days, fontsize=10)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:,.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig('ish_report_charts/01_daily_production.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 2: Wastage vs Issuance (stacked comparison)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(days))
width = 0.35

bars1 = ax.bar(x - width/2, issuance, width, label='Issuance (Kg)', color=colors['issuance'], alpha=0.85)
bars2 = ax.bar(x + width/2, wastage, width, label='Wastage (Qty)', color=colors['wastage'], alpha=0.85)

ax.set_xlabel('Day', fontsize=11)
ax.set_ylabel('Quantity', fontsize=11)
ax.set_title('ISH Daily Wastage vs Issuance — Week of Aug 8-14, 2026', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(short_days, fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Highlight days with zero wastage
for i, (iss, was) in enumerate(zip(issuance, wastage)):
    if was == 0:
        ax.annotate('Zero Wastage', xy=(i + width/2, 100), ha='center', fontsize=8, color=colors['danger'], fontweight='bold')

plt.tight_layout()
plt.savefig('ish_report_charts/02_wastage_vs_issuance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 3: Weekly Totals Summary (Pie + Bar)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart for weekly totals
labels = list(weekly_totals.keys())
sizes = list(weekly_totals.values())
pie_colors = [colors['issuance'], colors['produced'], colors['dispatch'], colors['wastage'], colors['packing']]
explode = (0.02, 0.02, 0.05, 0.05, 0.02)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=pie_colors,
                                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
ax1.set_title('Weekly Volume Distribution', fontsize=12, fontweight='bold')

# Bar chart for weekly totals
bars = ax2.barh(list(weekly_totals.keys()), list(weekly_totals.values()), 
                color=[colors['issuance'], colors['produced'], colors['dispatch'], colors['wastage'], colors['packing']], alpha=0.85)
ax2.set_xlabel('Quantity', fontsize=11)
ax2.set_title('Weekly Totals Comparison', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for bar in bars:
    width = bar.get_width()
    ax2.annotate(f'{width:,.1f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('ish_report_charts/03_weekly_totals.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 4: Machine Scans Per Day
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(short_days, scans, color=colors['scans'], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.plot(short_days, scans, color=colors['scans'], marker='o', linewidth=2, markersize=8)

# Highlight Friday dip
bars[-1].set_color(colors['danger'])
ax.annotate(f'Fri: {scans[-1]:,}\n(▼27.4% drop)', xy=(6, scans[-1]), xytext=(5.2, scans[-1] + 400),
            arrowprops=dict(arrowstyle='->', color=colors['danger']), fontsize=9, color=colors['danger'], fontweight='bold')

ax.set_xlabel('Day', fontsize=11)
ax.set_ylabel('Scans', fontsize=11)
ax.set_title('Machine Scans Per Day — Week of Aug 8-14, 2026', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add average line
avg_scans = np.mean(scans)
ax.axhline(y=avg_scans, color='red', linestyle='--', linewidth=1.5, label=f'Weekly Avg: {avg_scans:,.0f}')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('ish_report_charts/04_machine_scans.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 5: Machine Health Status
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Color mapping for health
health_colors = []
for h in machine_health:
    if h == 'ok':
        health_colors.append(colors['ok'])
    elif 'Offline' in h and 'High latency' in h:
        health_colors.append('#9B2226')  # dark red
    elif 'Offline' in h:
        health_colors.append(colors['danger'])
    elif 'High latency' in h:
        health_colors.append(colors['warning'])
    else:
        health_colors.append(colors['idle'])

y_pos = np.arange(len(machines))
bars = ax.barh(y_pos, machine_scans, color=health_colors, edgecolor='black', linewidth=0.5, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(machines, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Total Scans This Week', fontsize=11)
ax.set_title('Machine Activity & Health Status — Week of Aug 8-14, 2026', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add health status as text
for i, (bar, health) in enumerate(zip(bars, machine_health)):
    width = bar.get_width()
    if width > 0:
        ax.text(width + 100, bar.get_y() + bar.get_height()/2, health,
                va='center', fontsize=8, fontweight='bold')
    else:
        ax.text(100, bar.get_y() + bar.get_height()/2, f'{health} (0 scans)',
                va='center', fontsize=8, fontweight='bold', color=colors['idle'])

# Legend
legend_patches = [
    mpatches.Patch(color=colors['ok'], label='Healthy'),
    mpatches.Patch(color=colors['warning'], label='High Latency'),
    mpatches.Patch(color=colors['danger'], label='Offline'),
    mpatches.Patch(color='#9B2226', label='Offline + High Latency'),
    mpatches.Patch(color=colors['idle'], label='Idle (no scans)')
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('ish_report_charts/05_machine_health.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 6: Outlet Orders & Line Fill Rate
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# Outlet orders per day
ax1.bar(short_days, outlet_orders, color=colors['orders'], alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Orders Placed', fontsize=11)
ax1.set_title('Daily Outlet Orders — Week of Aug 8-14, 2026', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(outlet_orders):
    ax1.text(i, v + 10, str(v), ha='center', fontsize=9, fontweight='bold')

# Line fill rate breakdown
categories = ['Lines in Full', 'Lines Over-delivered', 'Lines Short']
values = [lines_full, lines_over, lines_short]
fill_colors = [colors['ok'], colors['packing'], colors['danger']]

wedges, texts, autotexts = ax2.pie(values, labels=categories, colors=fill_colors, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 10})
ax2.set_title(f'Line Fill Rate Breakdown — {lines_full:,} / {lines_total:,} lines in full (56.0%)', 
              fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('ish_report_charts/06_outlet_orders_fill_rate.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 7: Vendor Orders & Production Plans
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Vendor orders per day
ax1.bar(short_days, vendor_orders, color=colors['vendor'], alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Orders Raised', fontsize=11)
ax1.set_title('Daily Vendor Orders Raised — Week of Aug 8-14, 2026', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(vendor_orders):
    ax1.text(i, v + 0.5, str(v), ha='center', fontsize=9, fontweight='bold')

# Production plans
ax2.bar(short_days, plans_created, color='#457B9D', alpha=0.85, edgecolor='black', linewidth=0.5, label='Plans Created')
ax2.set_ylabel('Plans Created', fontsize=11)
ax2.set_title('Daily Production Plans Created — Week of Aug 8-14, 2026', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(short_days)))
ax2.set_xticklabels(short_days, fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(plans_created):
    ax2.text(i, v + 0.1, str(v), ha='center', fontsize=9, fontweight='bold')

# Add secondary axis for items planned
ax2_twin = ax2.twinx()
ax2_twin.plot(short_days, items_planned, color='#E63946', marker='o', linewidth=2, markersize=8, label='Items Planned')
ax2_twin.set_ylabel('Items Planned', fontsize=11, color='#E63946')
ax2_twin.tick_params(axis='y', labelcolor='#E63946')
ax2_twin.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('ish_report_charts/07_vendor_plans.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 8: Dashboard Summary (4-panel compact)
# ============================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle('ISH Weekly Report Dashboard — Aug 8-14, 2026', fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Daily Production Flow
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
x = np.arange(len(days))
width = 0.2
ax1.bar(x - 1.5*width, issuance, width, label='Issuance (Kg)', color=colors['issuance'], alpha=0.85)
ax1.bar(x - 0.5*width, [p * 500 for p in produced], width, label='Produced (×500)', color=colors['produced'], alpha=0.85)
ax1.bar(x + 0.5*width, [d / 100 for d in dispatch], width, label='Dispatch (÷100)', color=colors['dispatch'], alpha=0.85)
ax1.bar(x + 1.5*width, wastage, width, label='Wastage', color=colors['wastage'], alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(short_days, fontsize=9)
ax1.set_title('Daily Production Flow', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Weekly KPIs
ax2 = plt.subplot2grid((3, 3), (0, 2))
kpi_labels = ['Issuance\n(Kg)', 'Produced\n(Qty)', 'Dispatch\n(Qty)', 'Wastage\n(Qty)', 'Line Fill\nRate']
kpi_values = [35428.6, 656.6, 40005.9, 31151.2, 56.0]
kpi_colors = [colors['issuance'], colors['produced'], colors['dispatch'], colors['wastage'], colors['warning']]
bars = ax2.barh(kpi_labels, kpi_values, color=kpi_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_title('Weekly KPI Summary', fontsize=11, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, kpi_values):
    ax2.text(bar.get_width() + max(kpi_values)*0.02, bar.get_y() + bar.get_height()/2,
             f'{val:,.1f}' if isinstance(val, float) and val > 100 else f'{val}%',
             va='center', fontsize=9, fontweight='bold')

# Panel 3: Machine Scans Trend
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3.fill_between(short_days, scans, alpha=0.3, color=colors['scans'])
ax3.plot(short_days, scans, color=colors['scans'], marker='o', linewidth=2, markersize=8)
ax3.axhline(y=np.mean(scans), color='red', linestyle='--', linewidth=1.5, label=f'Avg: {np.mean(scans):,.0f}')
ax3.set_title('Machine Scans Trend', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Machine Health Distribution
ax4 = plt.subplot2grid((3, 3), (1, 2))
health_counts = {'ok': 1, 'Offline': 3, 'High Latency': 2, 'Offline+Latency': 2, 'idle': 3}
health_pie_colors = [colors['ok'], colors['danger'], colors['warning'], '#9B2226', colors['idle']]
wedges, texts, autotexts = ax4.pie(list(health_counts.values()), labels=list(health_counts.keys()),
                                    colors=health_pie_colors, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
ax4.set_title('Machine Health (11 total)', fontsize=11, fontweight='bold')

# Panel 5: Outlet Metrics
ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
ax5.bar(short_days, outlet_orders, color=colors['orders'], alpha=0.85, label='Orders Placed')
ax5.set_title('Daily Outlet Orders', fontsize=11, fontweight='bold')
ax5.set_xticks(range(len(short_days)))
ax5.set_xticklabels(short_days, fontsize=9)
ax5.grid(axis='y', alpha=0.3)
for i, v in enumerate(outlet_orders):
    ax5.text(i, v + 10, str(v), ha='center', fontsize=8, fontweight='bold')

# Panel 6: Vendor Orders
ax6 = plt.subplot2grid((3, 3), (2, 2))
ax6.bar(short_days, vendor_orders, color=colors['vendor'], alpha=0.85, edgecolor='black', linewidth=0.5)
ax6.set_title('Vendor Orders/Day', fontsize=11, fontweight='bold')
ax6.set_xticks(range(len(short_days)))
ax6.set_xticklabels(short_days, fontsize=8, rotation=45)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('ish_report_charts/08_dashboard_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("All charts generated successfully in 'ish_report_charts/' directory:")
print("  01_daily_production.png")
print("  02_wastage_vs_issuance.png")
print("  03_weekly_totals.png")
print("  04_machine_scans.png")
print("  05_machine_health.png")
print("  06_outlet_orders_fill_rate.png")
print("  07_vendor_plans.png")
print("  08_dashboard_summary.png")
