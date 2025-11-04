import os
import json
import re
from datetime import timedelta, datetime

import pytz
from dateutil import parser as dtp
import streamlit as st
import pandas as pd
import altair as alt


# ============================ PAGE & SECRETS ============================

st.set_page_config(page_title="Production Scheduler", layout="wide")

# Pull OPENAI key from Streamlit secrets if available
try:
    os.environ["OPENAI_API_KEY"] = (
        os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    )
except Exception:
    pass

# ============================ DATA LOADING & GENERATION =============================


@st.cache_data
def load_and_generate_data():
    """Load orders and generate a denser schedule"""
    orders_df = pd.read_csv("data/orders.csv", parse_dates=["due_date"])
    products_df = pd.read_csv("data/vrac_products.csv")
    lines_df = pd.read_csv("data/lines.csv")
    
    # Create mapping for rates
    product_rates = products_df.set_index('sku_id').to_dict('index')
    line_map = {
        'MIX': 'MIX_1',
        'TRF': 'TRF_1', 
        'FILL': 'FILL_1',
        'FIN': 'FIN_1'
    }
    
    schedule_rows = []
    base_start = datetime(2025, 11, 3, 6, 0)  # Start Nov 3 at 6 AM
    
    # Track last end time per machine to pack operations densely
    machine_timeline = {v: base_start for v in line_map.values()}
    
    # Sort orders by due date to schedule efficiently
    orders_sorted = orders_df.sort_values(['due_date', 'order_id']).reset_index(drop=True)
    
    for idx, order in orders_sorted.iterrows():
        order_id = order['order_id']
        sku = order['sku_id']
        qty = order['qty_kg']
        due = order['due_date']
        
        # Get product rates
        rates = product_rates.get(sku, {})
        mix_rate = rates.get('mix_rate_kg_per_h', 5000)
        trf_rate = rates.get('transfer_rate_kg_per_h', 14000)
        fill_rate = rates.get('fill_rate_kg_per_h', 4500)
        fin_rate = rates.get('finish_rate_kg_per_h', 18000)
        
        # Calculate durations in hours
        operations = [
            ('MIX', mix_rate, 1),
            ('TRF', trf_rate, 2),
            ('FILL', fill_rate, 3),
            ('FIN', fin_rate, 4)
        ]
        
        for op_type, rate, seq in operations:
            machine = line_map[op_type]
            duration_hours = qty / rate
            
            # Start right after previous operation on this machine ends
            op_start = machine_timeline[machine]
            op_end = op_start + timedelta(hours=duration_hours)
            
            schedule_rows.append({
                'order_id': order_id,
                'operation': op_type,
                'sequence': seq,
                'machine': machine,
                'start': op_start,
                'end': op_end,
                'due_date': due,
                'wheel_type': sku  # Using sku as product type
            })
            
            # Update machine timeline
            machine_timeline[machine] = op_end
    
    schedule_df = pd.DataFrame(schedule_rows)
    return orders_df, schedule_df


orders, base_schedule = load_and_generate_data()

# Working schedule in session (so edits persist)
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = base_schedule.copy()

# ============================ FILTER & LOG STATE =======================

if "filt_max_orders" not in st.session_state:
    st.session_state.filt_max_orders = 20
if "filt_products" not in st.session_state:
    st.session_state.filt_products = sorted(
        base_schedule["wheel_type"].unique().tolist()
    )
if "filt_machines" not in st.session_state:
    st.session_state.filt_machines = sorted(
        base_schedule["machine"].unique().tolist()
    )
if "cmd_log" not in st.session_state:
    st.session_state.cmd_log = []

# ============================ COMPACT LAYOUT =============================

# Create two columns: filters on left, chart on right
col_filter, col_chart = st.columns([1, 4])

with col_filter:
    st.markdown("### ⚙️ Filters")
    
    st.session_state.filt_max_orders = st.number_input(
        "Orders",
        1,
        100,
        value=st.session_state.filt_max_orders,
        step=1,
        key="num_orders",
    )
    
    products_all = sorted(base_schedule["wheel_type"].unique().tolist())
    st.session_state.filt_products = st.multiselect(
        "Products",
        products_all,
        default=st.session_state.filt_products or products_all,
        key="product_ms",
    )
    
    machines_all = sorted(base_schedule["machine"].unique().tolist())
    st.session_state.filt_machines = st.multiselect(
        "Machines",
        machines_all,
        default=st.session_state.filt_machines or machines_all,
        key="machine_ms",
    )
    
    if st.button("Reset", key="reset_filters"):
        st.session_state.filt_max_orders = 20
        st.session_state.filt_products = products_all
        st.session_state.filt_machines = machines_all
        st.rerun()
    
    # Debug panel
    with st.expander("🐛 Debug"):
        if st.session_state.cmd_log:
            last = st.session_state.cmd_log[-1]
            st.caption(f"**Last:** {last['raw']}")
            st.caption(f"✓ {last['msg']}" if last['ok'] else f"✗ {last['msg']}")
        else:
            st.caption("No commands yet")

# Effective filter values
max_orders = int(st.session_state.filt_max_orders)
product_choice = st.session_state.filt_products or products_all
machine_choice = st.session_state.filt_machines or machines_all

# ============================ NLP / INTELLIGENCE =========================

DEFAULT_TZ = "Africa/Casablanca"
TZ = pytz.timezone(DEFAULT_TZ)

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _num_token_to_float(tok: str):
    t = tok.strip().lower().replace("-", " ").replace(",", ".")
    try:
        return float(t)
    except Exception:
        pass
    parts = [p for p in t.split() if p]
    if len(parts) == 1 and parts[0] in NUM_WORDS:
        return float(NUM_WORDS[parts[0]])
    if len(parts) == 2 and parts[0] in NUM_WORDS and parts[1] in NUM_WORDS:
        return float(NUM_WORDS[parts[0]] + NUM_WORDS[parts[1]])
    return None


def _parse_duration_chunks(text: str):
    d = {"days": 0.0, "hours": 0.0, "minutes": 0.0}
    for num, unit in re.findall(
        r"([\d\.,]+|\b\w+\b)\s*(days?|d|hours?|h|minutes?|mins?|m)\b",
        text,
        flags=re.I,
    ):
        n = _num_token_to_float(num)
        if n is None:
            continue
        u = unit.lower()
        if u.startswith("d"):
            d["days"] += n
        elif u.startswith("h"):
            d["hours"] += n
        else:
            d["minutes"] += n
    return d


def _regex_fallback(user_text: str):
    t = user_text.strip()
    low = t.lower()
    
    # SWAP
    m = re.search(
        r"(?:^|\b)(swap|switch)\s+(ord-\d{3})\s*(?:with|and|&)?\s*(ord-\d{3})\b",
        low,
    )
    if m:
        return {
            "intent": "swap_orders",
            "order_id": m.group(2).upper(),
            "order_id_2": m.group(3).upper(),
            "_source": "regex",
        }
    
    # DELAY/ADVANCE detection
    delay_sign = +1
    if re.search(r"\b(advance|bring\s+forward|pull\s+in)\b", low):
        delay_sign = -1
        low_norm = re.sub(r"\b(advance|bring\s+forward|pull\s+in)\b", "delay", low)
    else:
        low_norm = low
    
    # Delay with "by"
    m = re.search(r"(delay|push|postpone)\s+(ord-\d{3}).*?\bby\b\s+(.+)$", low_norm)
    if m:
        oid = m.group(2).upper()
        dur_text = m.group(3)
        dur = _parse_duration_chunks(dur_text)
        if any(v != 0 for v in dur.values()):
            return {
                "intent": "delay_order",
                "order_id": oid,
                "days": delay_sign * dur["days"],
                "hours": delay_sign * dur["hours"],
                "minutes": delay_sign * dur["minutes"],
                "_source": "regex",
            }
    
    # Delay without "by"
    m = re.search(
        r"(delay|push|postpone)\s+(ord-\d{3}).*?(days?|d|hours?|h|minutes?|mins?|m)\b",
        low_norm,
    )
    if m:
        oid = m.group(2).upper()
        dur = _parse_duration_chunks(low_norm)
        if any(v != 0 for v in dur.values()):
            return {
                "intent": "delay_order",
                "order_id": oid,
                "days": delay_sign * dur["days"],
                "hours": delay_sign * dur["hours"],
                "minutes": delay_sign * dur["minutes"],
                "_source": "regex",
            }
    
    return {"intent": "unknown", "raw": user_text, "_source": "regex"}


def extract_intent(user_text: str) -> dict:
    return _regex_fallback(user_text)


def validate_intent(payload: dict, orders_df, sched_df):
    intent = payload.get("intent")
    
    def order_exists(oid):
        return oid and (orders_df["order_id"] == oid).any()
    
    if intent not in ("delay_order", "swap_orders"):
        return False, "Unsupported intent"
    
    if intent in ("delay_order", "swap_orders"):
        oid = payload.get("order_id")
        if not order_exists(oid):
            return False, f"Unknown order: {oid}"
    
    if intent == "swap_orders":
        oid2 = payload.get("order_id_2")
        if not order_exists(oid2):
            return False, f"Unknown order: {oid2}"
        if oid2 == payload.get("order_id"):
            return False, "Cannot swap same order"
        return True, "ok"
    
    if intent == "delay_order":
        for k in ("days", "hours", "minutes"):
            if k in payload and payload[k] is not None:
                try:
                    payload[k] = float(payload[k])
                except Exception:
                    return False, f"{k} must be numeric"
        if not any(payload.get(k) for k in ("days", "hours", "minutes")):
            return False, "Need duration (days/hours/minutes)"
        return True, "ok"
    
    return False, "Invalid payload"


# ============================ APPLY FUNCTIONS =========================


def _repack_touched_machines(s: pd.DataFrame, touched_orders):
    machines = s.loc[s["order_id"].isin(touched_orders), "machine"].unique().tolist()
    for m in machines:
        block_idx = s.index[s["machine"] == m]
        block = s.loc[block_idx].sort_values(["start", "end"]).copy()
        last_end = None
        for idx, row in block.iterrows():
            if last_end is not None and row["start"] < last_end:
                dur = row["end"] - row["start"]
                s.at[idx, "start"] = last_end
                s.at[idx, "end"] = last_end + dur
            last_end = s.at[idx, "end"]
    return s


def apply_delay(schedule_df: pd.DataFrame, order_id: str, days=0, hours=0, minutes=0):
    s = schedule_df.copy()
    delta = timedelta(
        days=float(days or 0),
        hours=float(hours or 0),
        minutes=float(minutes or 0),
    )
    mask = s["order_id"] == order_id
    s.loc[mask, "start"] = s.loc[mask, "start"] + delta
    s.loc[mask, "end"] = s.loc[mask, "end"] + delta
    return _repack_touched_machines(s, [order_id])


def apply_swap(schedule_df: pd.DataFrame, a: str, b: str):
    s = schedule_df.copy()
    a0 = s.loc[s["order_id"] == a, "start"].min()
    b0 = s.loc[s["order_id"] == b, "start"].min()
    da = b0 - a0
    db = a0 - b0
    s = apply_delay(
        s, a,
        days=da.days,
        hours=da.seconds // 3600,
        minutes=(da.seconds % 3600) // 60,
    )
    s = apply_delay(
        s, b,
        days=db.days,
        hours=db.seconds // 3600,
        minutes=(db.seconds % 3600) // 60,
    )
    return s


# ============================ FILTER & CHART =========================

with col_chart:
    sched = st.session_state.schedule_df.copy()
    sched = sched[sched["wheel_type"].isin(product_choice)]
    sched = sched[sched["machine"].isin(machine_choice)]
    
    order_priority = (
        sched.groupby("order_id", as_index=False)["start"]
        .min()
        .sort_values("start")
    )
    keep_ids = order_priority["order_id"].head(max_orders).tolist()
    sched = sched[sched["order_id"].isin(keep_ids)].copy()
    
    if sched.empty:
        st.info("No operations match filters")
    else:
        # Generate color per order
        unique_orders = sorted(sched["order_id"].unique())
        color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
        ]
        # Extend palette if needed
        while len(color_palette) < len(unique_orders):
            color_palette.extend(color_palette[:10])
        
        order_color_map = {oid: color_palette[i % len(color_palette)] 
                          for i, oid in enumerate(unique_orders)}
        
        sched["order_color"] = sched["order_id"].map(order_color_map)
        
        select_order = alt.selection_point(
            fields=["order_id"], on="click", clear="dblclick"
        )
        y_machines_sorted = sorted(sched["machine"].unique().tolist())
        
        base_enc = {
            "y": alt.Y("machine:N", sort=y_machines_sorted, title=None),
            "x": alt.X("start:T", title=None, axis=alt.Axis(format="%b %d %H:%M")),
            "x2": "end:T",
        }
        
        bars = (
            alt.Chart(sched)
            .mark_bar(cornerRadius=2)
            .encode(
                color=alt.condition(
                    select_order,
                    alt.Color("order_color:N", scale=None, legend=None),
                    alt.value("#e0e0e0"),
                ),
                opacity=alt.condition(
                    select_order, alt.value(1.0), alt.value(0.3)
                ),
                tooltip=[
                    alt.Tooltip("order_id:N", title="Order"),
                    alt.Tooltip("operation:N", title="Op"),
                    alt.Tooltip("machine:N", title="Machine"),
                    alt.Tooltip("start:T", title="Start"),
                    alt.Tooltip("end:T", title="End"),
                    alt.Tooltip("due_date:T", title="Due"),
                ],
            )
        )
        
        labels = (
            alt.Chart(sched)
            .mark_text(align="left", dx=4, baseline="middle", fontSize=9, color="white")
            .encode(
                text="order_id:N",
                opacity=alt.condition(
                    select_order, alt.value(1.0), alt.value(0.7)
                ),
            )
        )
        
        gantt = (
            alt.layer(bars, labels, data=sched)
            .encode(**base_enc)
            .add_params(select_order)
            .properties(width="container", height=450)
            .configure_view(stroke=None)
        )
        
        st.altair_chart(gantt, use_container_width=True)

# ============================ INTELLIGENCE INPUT =========================

user_cmd = st.chat_input("Delay / Advance / Swap orders…", key="cmd_input")

if user_cmd:
    try:
        payload = extract_intent(user_cmd)
        ok, msg = validate_intent(payload, orders, st.session_state.schedule_df)
        
        log_payload = dict(payload)
        st.session_state.cmd_log.append({
            "raw": user_cmd,
            "payload": log_payload,
            "ok": bool(ok),
            "msg": msg,
            "source": payload.get("_source", "?"),
        })
        st.session_state.cmd_log = st.session_state.cmd_log[-50:]
        
        if not ok:
            st.error(f"❌ {msg}")
        else:
            if payload["intent"] == "delay_order":
                st.session_state.schedule_df = apply_delay(
                    st.session_state.schedule_df,
                    payload["order_id"],
                    days=payload.get("days", 0),
                    hours=payload.get("hours", 0),
                    minutes=payload.get("minutes", 0),
                )
                direction = "Advanced" if (payload.get("days", 0) < 0 or 
                           payload.get("hours", 0) < 0) else "Delayed"
                st.success(f"✅ {direction} {payload['order_id']}")
            elif payload["intent"] == "swap_orders":
                st.session_state.schedule_df = apply_swap(
                    st.session_state.schedule_df,
                    payload["order_id"],
                    payload["order_id_2"],
                )
                st.success(f"✅ Swapped {payload['order_id']} ↔ {payload['order_id_2']}")
        st.rerun()
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
