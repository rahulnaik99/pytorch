"""
prompts.py
──────────
Canonical system prompt and prompt builder shared by pipeline, splitter, serve, and train.
Import from here to guarantee the inference prompt matches the training prompt exactly.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a disciplined NSE equity trader. Protect capital first.
R:R must be >= 1.5. HOLD if setup unclear.
Ground every decision in the features given — never invent data.

Output ONLY valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "planned_entry": <float — limit price to wait for; equals current price for immediate entry>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "risk_reward": <float>,
  "reasoning": "<2-3 sentences citing specific prices and indicator values>",
  "confidence": <0.0-1.0>,
  "features_used": ["1h_ema_stack", "5m_structure", "1h_support", "5m_demand_zone"],
  "key_risk": "<one-line failure scenario>"
}"""


# ── Prompt builder ────────────────────────────────────────────────────

def _pct(a, b):
    if not b or b == 0: return None
    return round((a - b) / b * 100, 2)


def build_prompt(symbol: str, date: str, tech: dict, vc: dict, zc: dict,
                 decision_slot: str = "close") -> str:
    close = tech.get("close", 0)
    rc    = tech.get("recent_candles", [])
    prev_close = rc[-2]["close"] if len(rc) >= 2 else close
    day_chg = _pct(close, prev_close)

    d5_high = max(c["high"] for c in rc) if rc else close
    d5_low  = min(c["low"]  for c in rc) if rc else close
    rng     = d5_high - d5_low
    pct_pos = round((close - d5_low) / rng * 100) if rng > 0 else 50

    avg_vol  = sum(c.get("volume", 0) for c in rc) / len(rc) if rc else 1
    last_vol = rc[-1].get("volume", 0) if rc else 0
    vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1

    daily_ranges = [c["high"] - c["low"] for c in rc] if rc else [0]
    atr = round(sum(daily_ranges) / len(daily_ranges), 2)

    d3_chg = _pct(close, rc[-4]["close"]) if len(rc) >= 4 else None

    e50, e100, e200 = tech.get("ema_50"), tech.get("ema_100"), tech.get("ema_200")
    vwap  = tech.get("vwap")
    stack = tech.get("ema_stack", "—")
    c_vs_200  = _pct(close, e200) if e200 else None
    c_vs_vwap = _pct(close, vwap) if vwap else None

    def fmt_zones(zones, n=2):
        parts = []
        for z in (zones[:n] if zones else []):
            if z and z.get("low"):
                d = _pct(z["mid"], close) if close else 0
                parts.append(f"{z['low']:.1f}-{z['high']:.1f}({d:+.1f}%)")
        return "  ".join(parts) or "—"

    # Support zones: ordered low→high in DB, so [:2] gives FARTHEST zones.
    # Reverse so the LLM sees nearest first — it consistently cites the first
    # listed zone as "nearest support", which was wrong before this fix.
    sup_str = fmt_zones(list(reversed(tech.get("support_zones") or [])))
    # Resistance zones: already ordered low→high = nearest first.
    res_str = fmt_zones(tech.get("resistance_zones"))
    ns = tech.get("nearest_support");  dist_s = _pct(ns["mid"], close) if ns and close else None
    nr = tech.get("nearest_resistance"); dist_r = _pct(nr["mid"], close) if nr and close else None

    sh90 = tech.get("swing_highs", [])
    sl90 = tech.get("swing_lows",  [])
    sh30 = tech.get("swing_highs_30d", [])
    sl30 = tech.get("swing_lows_30d",  [])
    sh90_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sh90[-6:]) or "—"
    sl90_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sl90[-6:]) or "—"
    sh30_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sh30[-6:]) or "—"
    sl30_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sl30[-6:]) or "—"

    va = tech.get("volume_analysis", {})
    vol_trend_line = ""
    if va:
        avg5  = va.get("avg_5d",  0)
        avg20 = va.get("avg_20d", 0)
        ratio = va.get("ratio",   1.0)
        sentiment = va.get("sentiment", "")
        vol_trend_line = (f"VOL_TREND: 5d:{avg5/1e6:.1f}M 20d:{avg20/1e6:.1f}M "
                          f"({ratio:.2f}x) | {sentiment}\n")

    bias = ("bullish" if (stack or "").startswith("bullish") and (c_vs_vwap or 0) > 0 else
            "bearish" if (stack or "").startswith("bearish") and (c_vs_vwap or 0) < 0 else "neutral")

    lv   = tech.get("levels", {}) or {}
    headline_bits = [f"bias:{bias}", f"trend:{tech.get('trend','-')}"]
    if lv.get("pdh_state") in ("broke", "rejected"): headline_bits.append(f"PDH:{lv['pdh_state']}")
    if vc: headline_bits.append(f"VIX:{vc.get('regime')}")
    if (zc.get("confluence") or {}).get("demand"): headline_bits.append("confluent-demand")
    if (zc.get("confluence") or {}).get("supply"): headline_bits.append("confluent-supply")

    levels_line = ""
    if lv:
        parts = []
        if "pdh" in lv: parts.append(f"PDH {lv['pdh']}({lv.get('pdh_state','')})")
        if "pdl" in lv: parts.append(f"PDL {lv['pdl']}({lv.get('pdl_state','')})")
        if "two_day_high" in lv:
            b = "broke" if lv.get("two_day_high_broken") else "intact"
            parts.append(f"2DH {lv['two_day_high']}({b})")
        if "two_day_low" in lv:
            b = "broke" if lv.get("two_day_low_broken") else "intact"
            parts.append(f"2DL {lv['two_day_low']}({b})")
        if "three_day_high" in lv:
            b = "broke" if lv.get("three_day_high_broken") else "intact"
            parts.append(f"3DH {lv['three_day_high']}({b})")
        if "three_day_low" in lv:
            b = "broke" if lv.get("three_day_low_broken") else "intact"
            parts.append(f"3DL {lv['three_day_low']}({b})")
        if "week_high" in lv: parts.append(f"WkH {lv['week_high']}({lv.get('week_high_state','')})")
        if "week_low"  in lv: parts.append(f"WkL {lv['week_low']}({lv.get('week_low_state','')})")
        if parts: levels_line = "LEVELS: " + " | ".join(parts) + "\n"

    vix_line = ""
    if vc:
        rt = vc.get("regime_trading", "")
        t10 = vc.get("trend_10d", "")
        qb  = vc.get("quantity_bias", "")
        vix_line = (f"VIX:{vc.get('vix')} | {vc.get('pctile_1y')}pct(1y) | "
                    f"{vc.get('regime')}"
                    f"{f' | {rt}' if rt else ''} | "
                    f"5d:{vc.get('dir_5d')}"
                    f"{f' 10d:{t10}' if t10 else ''} | "
                    f"20dRange:{vc.get('pos_20d')}"
                    f"{f' | size:{qb}' if qb else ''}\n")

    zones_line = ""
    if zc:
        def fmt_z(z):
            if not z: return "—"
            tag = "fresh" if z.get("fresh") else f"tested({z.get('touches')})"
            return f"{z['low']}-{z['high']} {tag}({z.get('dist_pct',0):+.1f}%)"
        s5m = zc.get("tf_5min", {}); b1h = zc.get("tf_1h", {})
        zparts = [f"5m  DZ:{fmt_z(s5m.get('demand'))} SZ:{fmt_z(s5m.get('supply'))}",
                  f"1h  DZ:{fmt_z(b1h.get('demand'))} SZ:{fmt_z(b1h.get('supply'))}"]
        cf = [k for k in ("demand", "supply") if (zc.get("confluence") or {}).get(k)]
        if cf: zparts.append("CONFLUENCE:" + "+".join(cf))
        zones_line = "ZONES:\n  " + "\n  ".join(zparts) + "\n"

    rl = tech.get("round_levels", [])
    round_line = ""
    if rl:
        def fmt_rl(r):
            t = "M" if r.get("type") == "major" else "m"
            return f"{r['level']}({t},{r['dist_pct']:+.1f}%)"
        sup_rounds = [r for r in rl if r.get("side") == "support"][-2:]
        res_rounds = [r for r in rl if r.get("side") == "resistance"][:2]
        all_rounds = sup_rounds + res_rounds
        if all_rounds:
            round_line = "ROUND: " + "  ".join(fmt_rl(r) for r in
                         sorted(all_rounds, key=lambda r: abs(r["dist_pct"]))) + "\n"

    return (
        f"[{symbol} {date} @{decision_slot}]\n"
        f"HEADLINE: {' | '.join(headline_bits)}\n"
        f"C:{close} ({'+' if (day_chg or 0)>=0 else ''}{day_chg}%) | "
        f"5d:{d5_low}-{d5_high} C@{pct_pos}pct\n"
        f"ATR:{atr} | Vol:{last_vol/1e6:.1f}M({vol_ratio}x avg) | "
        f"3d:{'+' if (d3_chg or 0)>=0 else ''}{d3_chg if d3_chg is not None else 0}%\n"
        f"{vol_trend_line}"
        f"\nEMA: 50@{e50} {'<' if (e50 or 0)<(e100 or 0) else '>'} "
        f"100@{e100} {'<' if (e100 or 0)<(e200 or 0) else '>'} "
        f"200@{e200} [{stack.split()[0].upper()}]\n"
        f"VWAP:{vwap} C{'+' if (c_vs_vwap or 0)>=0 else ''}{c_vs_vwap}% | "
        f"C vs EMA200:{'+' if (c_vs_200 or 0)>=0 else ''}{c_vs_200}%\n"
        f"Trend:{tech.get('trend','-')} | "
        f"Struct:{','.join(tech.get('structure',[]) or ['-'])}\n\n"
        f"S:{sup_str}  (nearest {(dist_s or 0):+.1f}%)\n"
        f"R:{res_str}  (nearest {'+' if (dist_r or 0)>=0 else ''}{dist_r or 0}%)\n"
        f"{round_line}"
        f"SWH/L-90d: H:{sh90_str}  L:{sl90_str}\n"
        f"SWH/L-30d: H:{sh30_str}  L:{sl30_str}\n"
        f"{levels_line}"
        f"{vix_line}"
        f"{zones_line}"
    )


# ── Multi-timeframe prompt builder ────────────────────────────────────

def build_prompt_mtf(symbol: str, date: str, tf_1h: dict, tf_5m: dict,
                     vc: dict, zc: dict, decision_slot: str = "close") -> str:
    """Two-section prompt: 1H long context + 5min short setup."""

    def _bias(stack: str) -> str:
        if stack.startswith("bullish"): return "bullish"
        if stack.startswith("bearish"): return "bearish"
        return "neutral"

    def _fmt_zones_mtf(zones, n=2):
        parts = []
        for z in (zones[:n] if zones else []):
            if z and z.get("low"):
                close = tf_1h.get("close", 0)
                d = _pct(z["mid"], close) if close else 0
                parts.append(f"{z['low']:.1f}-{z['high']:.1f}({d:+.1f}%)")
        return "  ".join(parts) or "—"

    def _fmt_swings(swings):
        return "  ".join(f"{s['price']:.1f}({str(s['date'])[5:]})" for s in (swings or [])[-6:]) or "—"

    # ── 1H section ─────────────────────────────────────────────────────
    c1h   = tf_1h.get("close", 0)
    e_f1  = tf_1h.get("ema_fast"); e_m1 = tf_1h.get("ema_mid"); e_s1 = tf_1h.get("ema_slow")
    sp1   = tf_1h.get("spans", [9, 21, 50])
    stk1  = tf_1h.get("ema_stack", "—")
    vw1   = tf_1h.get("vwap")
    bias_1h = _bias(stk1)
    cvv1  = _pct(c1h, vw1) if vw1 else None
    cve1  = _pct(c1h, e_s1) if e_s1 else None

    sup1  = list(reversed(tf_1h.get("support_zones") or []))
    res1  = tf_1h.get("resistance_zones") or []
    ns1   = tf_1h.get("nearest_support");  ds1 = _pct(ns1["mid"], c1h) if ns1 and c1h else None
    nr1   = tf_1h.get("nearest_resistance"); dr1 = _pct(nr1["mid"], c1h) if nr1 and c1h else None

    def _fmt_z1(zones, n=2):
        parts = []
        for z in (zones[:n] if zones else []):
            if z and z.get("low"):
                d = _pct(z["mid"], c1h) if c1h else 0
                parts.append(f"{z['low']:.1f}-{z['high']:.1f}({d:+.1f}%)")
        return "  ".join(parts) or "—"

    sh1   = _fmt_swings(tf_1h.get("swing_highs"))
    sl1   = _fmt_swings(tf_1h.get("swing_lows"))
    lv1   = tf_1h.get("levels") or {}
    rl1   = tf_1h.get("round_levels") or []
    n1    = tf_1h.get("n_bars", 90)

    pdh_line = ""
    if lv1.get("pdh"):
        pdh_line = (f"PDH(sess):{lv1['pdh']}({lv1.get('pdh_state','')})"
                    f"  PDL(sess):{lv1['pdl']}({lv1.get('pdl_state','')})\n")

    def _fmt_round(rl, n=4):
        if not rl: return ""
        sup_r = [r for r in rl if r.get("side") == "support"][-2:]
        res_r = [r for r in rl if r.get("side") == "resistance"][:2]
        merged = sorted(sup_r + res_r, key=lambda r: abs(r["dist_pct"]))[:n]
        t = lambda r: "M" if r.get("type") == "major" else "m"
        return "ROUND: " + "  ".join(f"{r['level']}({t(r)},{r['dist_pct']:+.1f}%)" for r in merged) + "\n"

    round1 = _fmt_round(rl1)

    # ema ordering symbol for 1H
    def _es(a, b): return ">" if (a or 0) > (b or 0) else "<"
    e1_str = (f"EMA{sp1[0]}@{e_f1} {_es(e_f1,e_m1)} "
              f"EMA{sp1[1]}@{e_m1} {_es(e_m1,e_s1)} "
              f"EMA{sp1[2]}@{e_s1} [{stk1.split()[0].upper()}]\n") if e_f1 else ""

    block_1h = (
        f"━━ 1H ({n1} bars)  LONG BIAS: {bias_1h} ━━\n"
        f"Trend:{tf_1h.get('trend','-')} | "
        f"Struct:{','.join(tf_1h.get('structure',[]) or ['-'])}\n"
        f"{e1_str}"
        f"VWAP(sess):{vw1}  C{'+' if (cvv1 or 0)>=0 else ''}{cvv1}%"
        f" | C vs EMA{sp1[2]}:{'+' if (cve1 or 0)>=0 else ''}{cve1}%\n"
        f"S:{_fmt_z1(sup1)}  (nearest {(ds1 or 0):+.1f}%)\n"
        f"R:{_fmt_z1(res1)}  (nearest {'+' if (dr1 or 0)>=0 else ''}{dr1 or 0}%)\n"
        f"SWH: {sh1}\n"
        f"SWL: {sl1}\n"
        f"{pdh_line}"
        f"{round1}"
    )

    # ── 5min section ───────────────────────────────────────────────────
    c5m   = tf_5m.get("close", 0)
    e_f5  = tf_5m.get("ema_fast"); e_m5 = tf_5m.get("ema_mid"); e_s5 = tf_5m.get("ema_slow")
    sp5   = tf_5m.get("spans", [9, 21, 50])
    stk5  = tf_5m.get("ema_stack", "—")
    vw5   = tf_5m.get("vwap")
    bias_5m = _bias(stk5)
    cvv5  = _pct(c5m, vw5) if vw5 else None

    sup5  = list(reversed(tf_5m.get("support_zones") or []))
    res5  = tf_5m.get("resistance_zones") or []
    ns5   = tf_5m.get("nearest_support");  ds5 = _pct(ns5["mid"], c5m) if ns5 and c5m else None
    nr5   = tf_5m.get("nearest_resistance"); dr5 = _pct(nr5["mid"], c5m) if nr5 and c5m else None

    def _fmt_z5(zones, n=2):
        parts = []
        for z in (zones[:n] if zones else []):
            if z and z.get("low"):
                d = _pct(z["mid"], c5m) if c5m else 0
                parts.append(f"{z['low']:.1f}-{z['high']:.1f}({d:+.1f}%)")
        return "  ".join(parts) or "—"

    sh5  = _fmt_swings(tf_5m.get("swing_highs"))
    sl5  = _fmt_swings(tf_5m.get("swing_lows"))
    n5   = tf_5m.get("n_bars", 90)
    e5_str = (f"EMA{sp5[0]}@{e_f5} {_es(e_f5,e_m5)} "
              f"EMA{sp5[1]}@{e_m5} {_es(e_m5,e_s5)} "
              f"EMA{sp5[2]}@{e_s5} [{stk5.split()[0].upper()}]\n") if e_f5 else ""

    block_5m = (
        f"━━ 5M ({n5} bars)  SHORT BIAS: {bias_5m} ━━\n"
        f"Trend:{tf_5m.get('trend','-')} | "
        f"Struct:{','.join(tf_5m.get('structure',[]) or ['-'])}\n"
        f"{e5_str}"
        f"VWAP(sess):{vw5}  C{'+' if (cvv5 or 0)>=0 else ''}{cvv5}%\n"
        f"S:{_fmt_z5(sup5)}  (nearest {(ds5 or 0):+.1f}%)\n"
        f"R:{_fmt_z5(res5)}  (nearest {'+' if (dr5 or 0)>=0 else ''}{dr5 or 0}%)\n"
        f"SWH: {sh5}\n"
        f"SWL: {sl5}\n"
    )

    # ── Headline ────────────────────────────────────────────────────────
    hbits = [f"1h:{bias_1h}", f"5m:{bias_5m}"]
    if vc: hbits.append(f"VIX:{vc.get('regime')}")
    if (zc.get("confluence") or {}).get("demand"): hbits.append("confluent-demand")
    if (zc.get("confluence") or {}).get("supply"): hbits.append("confluent-supply")

    # ── Zones line (from compute_zones — demand/supply impulse zones) ──
    zones_line = ""
    if zc:
        def fmt_z_dz(z):
            if not z: return "—"
            tag = "fresh" if z.get("fresh") else f"tested({z.get('touches')})"
            return f"{z['low']}-{z['high']} {tag}({z.get('dist_pct',0):+.1f}%)"
        s5m = zc.get("tf_5min", {}); b1h = zc.get("tf_1h", {})
        zparts = [f"5m  DZ:{fmt_z_dz(s5m.get('demand'))} SZ:{fmt_z_dz(s5m.get('supply'))}",
                  f"1h  DZ:{fmt_z_dz(b1h.get('demand'))} SZ:{fmt_z_dz(b1h.get('supply'))}"]
        cf = [k for k in ("demand", "supply") if (zc.get("confluence") or {}).get(k)]
        if cf: zparts.append("CONFLUENCE:" + "+".join(cf))
        zones_line = "ZONES:\n  " + "\n  ".join(zparts) + "\n"

    # ── VIX line ────────────────────────────────────────────────────────
    vix_line = ""
    if vc:
        rt  = vc.get("regime_trading", "")
        t10 = vc.get("trend_10d", "")
        qb  = vc.get("quantity_bias", "")
        vix_line = (f"VIX:{vc.get('vix')} | {vc.get('pctile_1y')}pct(1y) | "
                    f"{vc.get('regime')}"
                    f"{f' | {rt}' if rt else ''} | "
                    f"5d:{vc.get('dir_5d')}"
                    f"{f' 10d:{t10}' if t10 else ''} | "
                    f"20dRange:{vc.get('pos_20d')}"
                    f"{f' | size:{qb}' if qb else ''}\n")

    return (
        f"[{symbol} {date} @{decision_slot}]\n"
        f"HEADLINE: {' | '.join(hbits)}\n\n"
        f"{block_1h}\n"
        f"{block_5m}\n"
        f"{zones_line}"
        f"{vix_line}"
    )
