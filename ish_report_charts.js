const fs = require('fs');
const path = require('path');

const OUTPUT_DIR = 'ish_report_charts';
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const COLORS = {
    issuance: '#2E86AB',
    produced: '#A23B72',
    dispatch: '#F18F01',
    wastage: '#C73E1D',
    packing: '#6A994E',
    scans: '#3B1F2B',
    orders: '#BC4B51',
    vendor: '#5C4B7A',
    ok: '#2A9D8F',
    warning: '#E9C46A',
    danger: '#E76F51',
    idle: '#8D99AE',
    offlineLatency: '#9B2226'
};

const days = ['Sat 08-08', 'Sun 08-09', 'Mon 08-10', 'Tue 08-11', 'Wed 08-12', 'Thu 08-13', 'Fri 08-14'];
const shortDays = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const issuance = [4553.3, 4946.5, 6142.3, 5121.4, 4697.7, 8351.3, 1616.1];
const produced = [87.0, 138.2, 4.9, 105.5, 61.0, 245.4, 14.7];
const dispatch = [5175.1, 6024.7, 5289.3, 6266.0, 5667.2, 5860.3, 5723.3];
const wastage = [6248.1, 4059.7, 7072.4, 0.0, 9665.7, 4105.3, 0.0];
const scans = [4289, 4449, 4338, 4385, 4504, 4332, 3103];
const outletOrders = [631, 461, 534, 502, 467, 570, 464];
const vendorOrders = [14, 0, 24, 25, 26, 37, 2];
const plansCreated = [3, 3, 3, 5, 5, 5, 5];
const machines = ['Savoury-WIS', 'Kaju-KIS', 'Barfi-KIS', 'Chaats SFG', 'Sweet SFG', 'Premium-KIS', 'Bengali-WIS', 'Packaging-KIS', 'test', 'Warehouse KIT', 'demo'];
const machineScans = [8344, 5122, 3974, 3955, 2913, 2741, 2351, 0, 0, 0, 0];
const machineHealth = ['High latency', 'High latency', 'Offline', 'ok', 'Offline+High latency', 'Offline', 'Offline+High latency', 'idle', 'idle', 'idle', 'idle'];
const linesFull = 9512, linesShort = 7468, linesOver = 2433, linesTotal = 16980;

function svgHeader(w, h) {
    return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">\n<defs><style>.t{font-family:Arial,sans-serif;font-size:16px;font-weight:bold;fill:#1a1a2e}.s{font-family:Arial,sans-serif;font-size:12px;fill:#555}.l{font-family:Arial,sans-serif;font-size:11px;fill:#333}.a{font-family:Arial,sans-serif;font-size:10px;fill:#666}.v{font-family:Arial,sans-serif;font-size:10px;font-weight:bold;fill:#1a1a2e}.g{stroke:#e0e0e0;stroke-width:0.5}</style></defs>\n<rect width="${w}" height="${h}" fill="#fafafa"/>\n`;
}
function svgFooter() { return '</svg>'; }
function grid(svg, x, y, w, h, sx, sy) {
    for (let i = 0; i <= sy; i++) { const p = y + (h / sy) * i; svg += `<line x1="${x}" y1="${p}" x2="${x + w}" y2="${p}" class="g"/>\n`; }
    for (let i = 0; i <= sx; i++) { const p = x + (w / sx) * i; svg += `<line x1="${p}" y1="${y}" x2="${p}" y2="${y + h}" class="g"/>\n`; }
    return svg;
}
function title(svg, t, sub, x, y) {
    svg += `<text x="${x}" y="${y}" class="t">${t}</text>\n`;
    if (sub) svg += `<text x="${x}" y="${y + 18}" class="s">${sub}</text>\n`;
    return svg;
}

// ---- Chart 1: Daily Production ----
{
    const W = 1000, H = 500, m = { top: 60, right: 80, bottom: 60, left: 80 };
    const cw = W - m.left - m.right, ch = H - m.top - m.bottom;
    const bw = cw / days.length / 5, maxV = Math.max(...issuance, ...dispatch);
    let svg = svgHeader(W, H);
    svg = title(svg, 'Daily Production Overview - ISH Week (Aug 8-14, 2026)', 'Issuance, Produced, Dispatch by day', m.left, 30);
    svg = grid(svg, m.left, m.top, cw, ch, days.length - 1, 5);
    for (let i = 0; i <= 5; i++) { const v = (maxV / 5) * (5 - i); svg += `<text x="${m.left - 10}" y="${m.top + (ch / 5) * i + 4}" text-anchor="end" class="a">${v.toFixed(0)}</text>\n`; }
    days.forEach((d, i) => {
        const x = m.left + (cw / days.length) * i + cw / days.length / 2;
        const ih = (issuance[i] / maxV) * ch;
        svg += `<rect x="${x - bw * 2 - 2}" y="${m.top + ch - ih}" width="${bw}" height="${ih}" fill="${COLORS.issuance}" opacity="0.85"/>\n`;
        svg += `<text x="${x - bw * 2 - 2 + bw / 2}" y="${m.top + ch - ih - 5}" text-anchor="middle" class="v" font-size="8">${issuance[i].toFixed(0)}</text>\n`;
        const ph = (produced[i] * 100 / maxV) * ch;
        svg += `<rect x="${x - bw - 1}" y="${m.top + ch - ph}" width="${bw}" height="${Math.max(ph, 1)}" fill="${COLORS.produced}" opacity="0.85"/>\n`;
        const dh = (dispatch[i] / maxV) * ch;
        svg += `<rect x="${x + 1}" y="${m.top + ch - dh}" width="${bw}" height="${dh}" fill="${COLORS.dispatch}" opacity="0.85"/>\n`;
        const wh = (wastage[i] / maxV) * ch;
        if (wastage[i] > 0) svg += `<rect x="${x + bw + 2}" y="${m.top + ch - wh}" width="${bw}" height="${wh}" fill="${COLORS.wastage}" opacity="0.85"/>\n`;
        svg += `<text x="${x}" y="${m.top + ch + 20}" text-anchor="middle" class="a">${d}</text>\n`;
    });
    const ly = m.top + ch + 35;
    const items = [['Issuance (Kg)', COLORS.issuance], ['Produced (Qty)', COLORS.produced], ['Dispatch (Qty)', COLORS.dispatch], ['Wastage (Qty)', COLORS.wastage]];
    let lx = m.left;
    items.forEach(it => { svg += `<rect x="${lx}" y="${ly}" width="12" height="12" fill="${it[1]}" opacity="0.85"/>\n<text x="${lx + 16}" y="${ly + 10}" class="a">${it[0]}</text>\n`; lx += 150; });
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '01_daily_production.svg'), svg);
}

// ---- Chart 2: Wastage vs Issuance ----
{
    const W = 1000, H = 500, m = { top: 60, right: 80, bottom: 60, left: 80 };
    const cw = W - m.left - m.right, ch = H - m.top - m.bottom;
    const bw = cw / days.length / 3, maxV = Math.max(...issuance, ...wastage);
    let svg = svgHeader(W, H);
    svg = title(svg, 'Daily Wastage vs Issuance - ISH Week (Aug 8-14, 2026)', 'Wastage spikes Mon/Wed; zero Tue/Fri', m.left, 30);
    svg = grid(svg, m.left, m.top, cw, ch, days.length - 1, 5);
    for (let i = 0; i <= 5; i++) { const v = (maxV / 5) * (5 - i); svg += `<text x="${m.left - 10}" y="${m.top + (ch / 5) * i + 4}" text-anchor="end" class="a">${v.toFixed(0)}</text>\n`; }
    days.forEach((d, i) => {
        const x = m.left + (cw / days.length) * i + cw / days.length / 2;
        const ih = (issuance[i] / maxV) * ch;
        svg += `<rect x="${x - bw * 1.5}" y="${m.top + ch - ih}" width="${bw}" height="${ih}" fill="${COLORS.issuance}" opacity="0.85"/>\n`;
        svg += `<text x="${x - bw * 1.5 + bw / 2}" y="${m.top + ch - ih - 5}" text-anchor="middle" class="v" font-size="8">${issuance[i].toFixed(0)}</text>\n`;
        const wh = (wastage[i] / maxV) * ch;
        if (wastage[i] > 0) { svg += `<rect x="${x + bw * 0.5}" y="${m.top + ch - wh}" width="${bw}" height="${wh}" fill="${COLORS.wastage}" opacity="0.85"/>\n`; svg += `<text x="${x + bw * 0.5 + bw / 2}" y="${m.top + ch - wh - 5}" text-anchor="middle" class="v" font-size="8">${wastage[i].toFixed(0)}</text>\n`; }
        svg += `<text x="${x}" y="${m.top + ch + 20}" text-anchor="middle" class="a">${d}</text>\n`;
    });
    const ly = m.top + ch + 35;
    svg += `<rect x="${m.left}" y="${ly}" width="12" height="12" fill="${COLORS.issuance}" opacity="0.85"/><text x="${m.left + 16}" y="${ly + 10}" class="a">Issuance (Kg)</text>\n`;
    svg += `<rect x="${m.left + 140}" y="${ly}" width="12" height="12" fill="${COLORS.wastage}" opacity="0.85"/><text x="${m.left + 156}" y="${ly + 10}" class="a">Wastage (Qty)</text>\n`;
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '02_wastage_vs_issuance.svg'), svg);
}

// ---- Chart 3: Weekly Totals ----
{
    const W = 1000, H = 500, m = { top: 60, right: 150, bottom: 80, left: 120 };
    const ch = H - m.top - m.bottom;
    const labels = ['Issuance\n(Kg)', 'Produced\n(Qty)', 'Dispatch\n(Qty)', 'Wastage\n(Qty)', 'Prm Packing\n(Qty)'];
    const values = [35428.6, 656.6, 40005.9, 31151.2, 35716.0];
    const bc = [COLORS.issuance, COLORS.produced, COLORS.dispatch, COLORS.wastage, COLORS.packing];
    const maxV = Math.max(...values), bh = ch / values.length * 0.6;
    let svg = svgHeader(W, H);
    svg = title(svg, 'Weekly Volume Totals - ISH Week (Aug 8-14, 2026)', 'Aggregate comparison across all metrics', m.left, 30);
    values.forEach((v, i) => {
        const y = m.top + (ch / values.length) * i + (ch / values.length - bh) / 2;
        const bw = (v / maxV) * (W - m.left - m.right - 100);
        svg += `<rect x="${m.left}" y="${y}" width="${bw}" height="${bh}" fill="${bc[i]}" opacity="0.85"/>\n`;
        svg += `<text x="${m.left + bw + 8}" y="${y + bh / 2 + 4}" class="v">${v.toLocaleString()}</text>\n`;
        svg += `<text x="${m.left - 8}" y="${y + bh / 2 + 4}" text-anchor="end" class="a">${labels[i].replace('\n', ' ')}</text>\n`;
    });
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '03_weekly_totals.svg'), svg);
}

// ---- Chart 4: Machine Scans ----
{
    const W = 1000, H = 450, m = { top: 60, right: 80, bottom: 60, left: 80 };
    const cw = W - m.left - m.right, ch = H - m.top - m.bottom, maxV = Math.max(...scans);
    let svg = svgHeader(W, H);
    svg = title(svg, 'Machine Scans Per Day - ISH Week (Aug 8-14, 2026)', 'Total: 29,400 scans | 7 machines | 7 active days', m.left, 30);
    svg = grid(svg, m.left, m.top, cw, ch, days.length - 1, 5);
    for (let i = 0; i <= 5; i++) { const v = (maxV / 5) * (5 - i); svg += `<text x="${m.left - 10}" y="${m.top + (ch / 5) * i + 4}" text-anchor="end" class="a">${v.toFixed(0)}</text>\n`; }
    let area = `M ${m.left} ${m.top + ch}`;
    days.forEach((_, i) => { const x = m.left + (cw / (days.length - 1)) * i; const y = m.top + ch - (scans[i] / maxV) * ch; area += ` L ${x} ${y}`; });
    area += ` L ${m.left + cw} ${m.top + ch} Z`;
    svg += `<path d="${area}" fill="${COLORS.scans}" opacity="0.15"/>\n`;
    let line = '';
    days.forEach((_, i) => { const x = m.left + (cw / (days.length - 1)) * i; const y = m.top + ch - (scans[i] / maxV) * ch; line += (i === 0 ? 'M' : 'L') + ` ${x} ${y} `; });
    svg += `<path d="${line}" fill="none" stroke="${COLORS.scans}" stroke-width="2.5"/>\n`;
    days.forEach((_, i) => {
        const x = m.left + (cw / (days.length - 1)) * i; const y = m.top + ch - (scans[i] / maxV) * ch; const f = i === 6 ? COLORS.danger : COLORS.scans;
        svg += `<line x1="${x}" y1="${m.top + ch}" x2="${x}" y2="${y}" stroke="${f}" stroke-width="1" opacity="0.3"/>\n`;
        svg += `<circle cx="${x}" cy="${y}" r="6" fill="${f}" stroke="white" stroke-width="2"/>\n`;
        svg += `<text x="${x}" y="${y - 12}" text-anchor="middle" class="v" font-size="9" fill="${f}">${scans[i].toLocaleString()}</text>\n`;
        svg += `<text x="${x}" y="${m.top + ch + 20}" text-anchor="middle" class="a">${days[i]}</text>\n`;
    });
    const avg = scans.reduce((a, b) => a + b, 0) / scans.length, ay = m.top + ch - (avg / maxV) * ch;
    svg += `<line x1="${m.left}" y1="${ay}" x2="${m.left + cw}" y2="${ay}" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="5,5"/>\n`;
    svg += `<text x="${m.left + cw + 5}" y="${ay + 4}" class="a" fill="#e74c3c">Avg: ${avg.toFixed(0)}</text>\n`;
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '04_machine_scans.svg'), svg);
}

// ---- Chart 5: Machine Health ----
{
    const W = 1000, H = 650, m = { top: 60, right: 250, bottom: 40, left: 180 };
    const cw = W - m.left - m.right, ch = H - m.top - m.bottom, maxS = Math.max(...machineScans);
    let svg = svgHeader(W, H);
    svg = title(svg, 'Machine Activity & Health Status - ISH Week', '11 registered | 7 active | 10 need attention', m.left, 30);
    svg = grid(svg, m.left, m.top, cw, ch, 4, machines.length - 1);
    machines.forEach((mc, i) => {
        const y = m.top + (ch / machines.length) * i + (ch / machines.length) * 0.15, bh = (ch / machines.length) * 0.7;
        const bw = machineScans[i] > 0 ? (machineScans[i] / maxS) * cw : 5;
        let c; if (machineHealth[i] === 'ok') c = COLORS.ok; else if (machineHealth[i] === 'Offline+High latency') c = COLORS.offlineLatency; else if (machineHealth[i] === 'Offline') c = COLORS.danger; else if (machineHealth[i] === 'High latency') c = COLORS.warning; else c = COLORS.idle;
        svg += `<rect x="${m.left}" y="${y}" width="${Math.max(bw, 5)}" height="${bh}" fill="${c}" opacity="0.85"/>\n`;
        svg += `<text x="${m.left - 8}" y="${y + bh / 2 + 4}" text-anchor="end" class="a">${mc}</text>\n`;
        if (machineScans[i] > 0) svg += `<text x="${m.left + bw + 8}" y="${y + bh / 2 + 4}" class="v">${machineScans[i].toLocaleString()}</text>\n`;
        svg += `<text x="${m.left + bw + 100}" y="${y + bh / 2 + 4}" class="a" font-weight="bold" fill="${c}">${machineHealth[i]}</text>\n`;
    });
    const ly = m.top + ch + 15;
    const li = [['Healthy', COLORS.ok], ['High Latency', COLORS.warning], ['Offline', COLORS.danger], ['Offline+Latency', COLORS.offlineLatency], ['Idle', COLORS.idle]];
    let lx = m.left;
    li.forEach(it => { svg += `<rect x="${lx}" y="${ly}" width="12" height="12" fill="${it[1]}" opacity="0.85"/><text x="${lx + 16}" y="${ly + 10}" class="a">${it[0]}</text>\n`; lx += 140; });
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '05_machine_health.svg'), svg);
}

// ---- Chart 6: Outlet Orders + Fill Rate ----
{
    const W = 1000, H = 700, m = { top: 40, right: 40, bottom: 60, left: 60 };
    const cw = W - m.left - m.right, ch = (H - m.top - m.bottom - 40) / 2;
    let svg = svgHeader(W, H);
    svg = title(svg, 'Outlet Orders & Line Fill Rate - ISH Week', '3,629 orders | 16,981 line items | 56.0% fill', m.left, 20);
    const topY = m.top + 20, maxO = Math.max(...outletOrders);
    svg += `<text x="${m.left}" y="${topY - 5}" class="s" font-weight="bold">Daily Outlet Orders Placed</text>\n`;
    svg = grid(svg, m.left, topY, cw, ch, days.length - 1, 4);
    for (let i = 0; i <= 4; i++) { const v = (maxO / 4) * (4 - i); svg += `<text x="${m.left - 8}" y="${topY + (ch / 4) * i + 4}" text-anchor="end" class="a">${v.toFixed(0)}</text>\n`; }
    outletOrders.forEach((v, i) => { const x = m.left + (cw / outletOrders.length) * i + cw / outletOrders.length / 2; const bh = (v / maxO) * ch; const y = topY + ch - bh; svg += `<rect x="${x - 15}" y="${y}" width="30" height="${bh}" fill="${COLORS.orders}" opacity="0.85"/>\n<text x="${x}" y="${y - 5}" text-anchor="middle" class="v" font-size="9">${v}</text>\n<text x="${x}" y="${topY + ch + 15}" text-anchor="middle" class="a">${shortDays[i]}</text>\n`; });
    const pieY = topY + ch + 50, cx = W / 2 - 80, cy = pieY + 100, r = 80;
    svg += `<text x="${W / 2}" y="${pieY - 5}" text-anchor="middle" class="s" font-weight="bold">Line Fill Rate Breakdown</text>\n`;
    svg += `<text x="${W / 2}" y="${pieY + 10}" text-anchor="middle" class="a">${linesFull.toLocaleString()} of ${linesTotal.toLocaleString()} lines in full (56.0%)</text>\n`;
    const pd = [['In Full', linesFull, COLORS.ok], ['Over', linesOver, COLORS.packing], ['Short', linesShort, COLORS.danger]];
    const tot = pd.reduce((a, b) => a + b[1], 0); let sa = -Math.PI / 2;
    pd.forEach(sl => { const ang = (sl[1] / tot) * 2 * Math.PI; const x1 = cx + r * Math.cos(sa), y1 = cy + r * Math.sin(sa); const x2 = cx + r * Math.cos(sa + ang), y2 = cy + r * Math.sin(sa + ang); const la = ang > Math.PI ? 1 : 0; const ma = sa + ang / 2; const lx = cx + (r + 25) * Math.cos(ma), ly = cy + (r + 25) * Math.sin(ma); svg += `<path d="M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${la} 1 ${x2} ${y2} Z" fill="${sl[2]}" opacity="0.85"/>\n<text x="${lx}" y="${ly}" text-anchor="middle" class="v" font-size="10">${((sl[1] / tot) * 100).toFixed(1)}%</text>\n`; sa += ang; });
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '06_outlet_orders_fill_rate.svg'), svg);
}

// ---- Chart 7: Vendor + Plans ----
{
    const W = 1000, H = 600, m = { top: 40, right: 60, bottom: 60, left: 60 };
    const cw = W - m.left - m.right, ch = (H - m.top - m.bottom - 40) / 2;
    let svg = svgHeader(W, H);
    svg = title(svg, 'Vendor Orders & Production Plans - ISH Week', '128 vendor orders | 29 plans created', m.left, 20);
    const topY = m.top + 20, maxV = Math.max(...vendorOrders);
    svg += `<text x="${m.left}" y="${topY - 5}" class="s" font-weight="bold">Daily Vendor Orders Raised (Total: 128)</text>\n`;
    svg = grid(svg, m.left, topY, cw, ch, days.length - 1, 4);
    for (let i = 0; i <= 4; i++) { const v = (maxV / 4) * (4 - i); svg += `<text x="${m.left - 8}" y="${topY + (ch / 4) * i + 4}" text-anchor="end" class="a">${v.toFixed(0)}</text>\n`; }
    vendorOrders.forEach((v, i) => { const x = m.left + (cw / vendorOrders.length) * i + cw / vendorOrders.length / 2; const bh = maxV > 0 ? (v / maxV) * ch : 0; const y = topY + ch - bh; svg += `<rect x="${x - 15}" y="${y}" width="30" height="${Math.max(bh, 2)}" fill="${COLORS.vendor}" opacity="0.85"/>\n`; if (v > 0) svg += `<text x="${x}" y="${y - 5}" text-anchor="middle" class="v" font-size="9">${v}</text>\n`; svg += `<text x="${x}" y="${topY + ch + 15}" text-anchor="middle" class="a">${shortDays[i]}</text>\n`; });
    const botY = topY + ch + 50, maxP = Math.max(...plansCreated);
    svg += `<text x="${m.left}" y="${botY - 5}" class="s" font-weight="bold">Production Plans Created Per Day</text>\n`;
    svg = grid(svg, m.left, botY, cw, ch, days.length - 1, 4);
    for (let i = 0; i <= 4; i++) { const v = (maxP / 4) * (4 - i); svg += `<text x="${m.left - 8}" y="${botY + (ch / 4) * i + 4}" text-anchor="end" class="a">${v.toFixed(0)}</text>\n`; }
    plansCreated.forEach((v, i) => { const x = m.left + (cw / plansCreated.length) * i + cw / plansCreated.length / 2; const bh = (v / maxP) * ch; const y = botY + ch - bh; svg += `<rect x="${x - 15}" y="${y}" width="30" height="${bh}" fill="${COLORS.issuance}" opacity="0.85"/>\n<text x="${x}" y="${y - 5}" text-anchor="middle" class="v" font-size="9">${v}</text>\n<text x="${x}" y="${botY + ch + 15}" text-anchor="middle" class="a">${shortDays[i]}</text>\n`; });
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '07_vendor_plans.svg'), svg);
}

// ---- Chart 8: Dashboard ----
{
    const W = 1200, H = 900;
    let svg = svgHeader(W, H);
    svg += `<text x="${W / 2}" y="30" text-anchor="middle" class="t" font-size="18">ISH Weekly Report Dashboard - Aug 8-14, 2026</text>\n`;
    svg += `<text x="${W / 2}" y="50" text-anchor="middle" class="s">Production | Machines | Orders | Vendors</text>\n`;
    const kpi = [['Issuance', '35,429 Kg', '-58.4%', COLORS.issuance], ['Produced', '656.6 Qty', '+57.5%', COLORS.produced], ['Dispatch', '40,006 Qty', '+3.9%', COLORS.dispatch], ['Wastage', '31,151 Qty', '+8.8%', COLORS.wastage], ['Line Fill', '56.0%', '9,512/16,980', COLORS.warning], ['Machines', '7 Active', '10 need attn', COLORS.danger]];
    const bw = 180, bh = 80, gap = 15, sx = (W - (kpi.length * bw + (kpi.length - 1) * gap)) / 2, by = 70;
    kpi.forEach((k, i) => { const x = sx + i * (bw + gap); svg += `<rect x="${x}" y="${by}" width="${bw}" height="${bh}" fill="${k[3]}" opacity="0.15" rx="5"/><rect x="${x}" y="${by}" width="${bw}" height="4" fill="${k[3]}"/><text x="${x + bw / 2}" ..="..">`; svg += `<text x="${x + bw / 2}" y="${by + 25}" text-anchor="middle" class="s" font-size="10">${k[0]}</text>\n<text x="${x + bw / 2}" y="${by + 48}" text-anchor="middle" class="t" font-size="16">${k[1]}</text>\n<text x="${x + bw / 2}" y="${by + 65}" text-anchor="middle" class="a">${k[2]}</text>\n`; });
    // Panel 1: Production flow
    const p1x = 30, p1y = 180, p1w = 560, p1h = 280, pm = { top: 20, right: 60, bottom: 40, left: 60 };
    const p1cw = p1w - pm.left - pm.right, p1ch = p1h - pm.top - pm.bottom, p1bw = p1cw / days.length / 4, p1max = Math.max(...issuance, ...dispatch);
    svg += `<rect x="${p1x}" y="${p1y}" width="${p1w}" height="${p1h}" fill="white" stroke="#ddd" rx="5"/><text x="${p1x + 10}" y="${p1y + 18}" class="s" font-weight="bold">Daily Production Flow</text>\n`;
    svg = grid(svg, p1x + pm.left, p1y + pm.top, p1cw, p1ch, days.length - 1, 4);
    for (let i = 0; i <= 4; i++) { const v = (p1max / 4) * (4 - i); svg += `<text x="${p1x + pm.left - 8}" y="${p1y + pm.top + (p1ch / 4) * i + 4}" text-anchor="end" class="a" font-size="9">${v.toFixed(0)}</text>\n`; }
    days.forEach((_, i) => { const x = p1x + pm.left + (p1cw / days.length) * i + p1cw / days.length / 2; const ih = (issuance[i] / p1max) * p1ch; svg += `<rect x="${x - p1bw * 2 - 2}" y="${p1y + pm.top + p1ch - ih}" width="${p1bw}" height="${ih}" fill="${COLORS.issuance}" opacity="0.8"/>\n`; const dh = (dispatch[i] / p1max) * p1ch; svg += `<rect x="${x - p1bw}" y="${p1y + pm.top + p1ch - dh}" width="${p1bw}" height="${dh}" fill="${COLORS.dispatch}" opacity="0.8"/>\n`; const ph = (produced[i] * 100 / p1max) * p1ch; svg += `<rect x="${x}" y="${p1y + pm.top + p1ch - ph}" width="${p1bw}" height="${Math.max(ph, 1)}" fill="${COLORS.produced}" opacity="0.8"/>\n`; const wh = (wastage[i] / p1max) * p1ch; if (wastage[i] > 0) svg += `<rect x="${x + p1bw + 2}" y="${p1y + pm.top + p1ch - wh}" width="${p1bw}" height="${wh}" fill="${COLORS.wastage}" opacity="0.8"/>\n`; svg += `<text x="${x}" y="${p1y + pm.top + p1ch + 15}" text-anchor="middle" class="a" font-size="9">${shortDays[i]}</text>\n`; });
    svg += `<rect x="${p1x + 10}" y="${p1y + p1h - 12}" width="10" height="10" fill="${COLORS.issuance}" opacity="0.8"/><text x="${p1x + 24}" y="${p1y + p1h - 3}" class="a" font-size="9">Issuance</text>\n`;
    svg += `<rect x="${p1x + 100}" y="${p1y + p1h - 12}" width="10" height="10" fill="${COLORS.dispatch}" opacity="0.8"/><text x="${p1x + 114}" y="${p1y + p1h - 3}" class="a" font-size="9">Dispatch</text>\n`;
    svg += `<rect x="${p1x + 200}" y="${p1y + p1h - 12}" width="10" height="10" fill="${COLORS.produced}" opacity="0.8"/><text x="${p1x + 214}" y="${p1y + p1h - 3}" class="a" font-size="9">Produced</text></text>\n`;
    // Panel 2: Machine scans
    const p2x = 610, p2y = 180, p2w = 560, p2h = 280, p2m = { top: 20, right: 60, bottom: 40, left: 60 };
    const p2cw = p2w - p2m.left - p2m.right, p2ch = p2h - p2m.top - p2m.bottom, p2max = Math.max(...scans);
    svg += `<rect x="${p2x}" y="${p2y}" width="${p2w}" height="${p2h}" fill="white" stroke="#ddd" rx="5"/><text x="${p2x + 10}" y="${p2y + 18}" class="s" font-weight="bold">Machine Scans Trend</text>\n`;
    svg = grid(svg, p2x + p2m.left, p2y + p2m.top, p2cw, p2ch, days.length - 1, 4);
    let p2line = '';
    days.forEach((_, i) => { const x = p2x + p2m.left + (p2cw / (days.length - 1)) * i; const y = p2y + p2m.top + p2ch - (scans[i] / p2max) * p2ch; p2line += (i === 0 ? 'M' : 'L') + ` ${x} ${y} `; });
    svg += `<path d="${p2line}" fill="none" stroke="${COLORS.scans}" stroke-width="2.5"/>\n`;
    days.forEach((_, i) => { const x = p2x + p2m.left + (p2cw / (days.length - 1)) * i; const y = p2y + p2m.top + p2ch - (scans[i] / p2max) * p2ch; const f = i === 6 ? COLORS.danger : COLORS.scans; svg += `<circle cx="${x}" cy="${y}" r="6" fill="${f}" stroke="white" stroke-width="2"/><text x="${x}" y="${y - 12}" text-anchor="middle" class="v" font-size="9" fill="${f}">${scans[i].toLocaleString()}</text>\n<text x="${x}" y="${p2y + p2m.top + p2ch + 15}" text-anchor="middle" class="a" font-size="9">${shortDays[i]}</text>\n`; });
    // Panel 3: Machine health summary
    const p3x = 30, p3y = 480, p3w = 560, p3h = 240;
    svg += `<rect x="${p3x}" y="${p3y}" width="${p3w}" height="${p3h}" fill="white" stroke="#ddd" rx="5"/><text x="${p3x + 10}" y="${p3y + 18}" class="s" font-weight="bold">Machine Health Summary</text>\n`;
    const hc = [['Healthy', 1, COLORS.ok], ['High Latency', 2, COLORS.warning], ['Offline', 3, COLORS.danger], ['Offline+Latency', 2, COLORS.offlineLatency], ['Idle', 3, COLORS.idle]];
    const htot = hc.reduce((a, b) => a + b[1], 0); let hx = p3x + 30, hy = p3y + 50;
    hc.forEach(h => { const w = (h[1] / htot) * (p3w - 60); svg += `<rect x="${hx}" y="${hy}" width="${w}" height="30" fill="${h[2]}" opacity="0.85" rx="3"/><text x="${hx + w / 2}" y="${hy + 20}" text-anchor="middle" class="v" font-size="11">${h[0]} (${h[1]})</text>\n`; hx += w + 5; });
    svg += `<text x="${p3x + 10}" y="${p3y + p3h - 15}" class="a">11 machines: 7 active | 4 idle | 10 need attention</text>\n`;
    // Panel 4: Fill rate + vendor
    const p4x = 610, p4y = 480, p4w = 560, p4h = 240;
    svg += `<rect x="${p4x}" y="${p4y}" width="${p4w}" height="${p4h}" fill="white" stroke="#ddd" rx="5"/><text x="${p4x + 10}" y="${p4y + 18}" class="s" font-weight="bold">Key Metrics</text>\n`;
    svg += `<text x="${p4x + 20}" y="${p4y + 60}" class="l">Line Fill Rate: 56.0% (9,512 / 16,980 full)</text>\n`;
    svg += `<text x="${p4x + 20}" y="${p4y + 90}" class="l">Outlet Orders: 3,629 across 50 outlets</text>\n`;
    svg += `<text x="${p4x + 20}" y="${p4y + 120}" class="l">Vendor Orders: 128 (Rs 2.92 Cr), 300 lines short</text>\n`;
    svg += `<text x="${p4x + 20}" y="${p4y + 150}" class="l">Production Plans: 29 created, 288 items</text>\n`;
    svg += `<text x="${p4x + 20}" y="${p4y + 180}" class="l">Wastage: 31,151 Qty (8.8% above last week)</text>\n`;
    svg += `<text x="${p4x + 20}" y="${p4y + 210}" class="l">Friday: production collapse (14.7 units)</text>\n`;
    svg += svgFooter();
    fs.writeFileSync(path.join(OUTPUT_DIR, '08_dashboard_summary.svg'), svg);
}

console.log('All 8 SVG charts written to', OUTPUT_DIR + '/');
console.log('Files:');
console.log('  01_daily_production.svg');
console.log('  02_wastage_vs_issuance.svg');
console.log('  03_weekly_totals.svg');
console.log('  04_machine_scans.svg');
console.log('  05_machine_health.svg');
console.log('  06_outlet_orders_fill_rate.svg');
console.log('  07_vendor_plans.svg');
console.log('  08_dashboard_summary.svg');
