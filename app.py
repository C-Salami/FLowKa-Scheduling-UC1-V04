import os
import json
import re
from datetime import timedelta
import pytz
from dateutil import parser as dtp

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path


# ============================ PAGE CONFIG ============================
st.set_page_config(
    page_title="FlowKa – Factory Scheduling UC1",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Pull OPENAI key from Streamlit secrets if available
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ============================ DATA LOADING ============================
def _guess_column(cols, keywords):
    """Return first column name that contains any of keywords (case-insensitive)."""
    low = {c.lower(): c for c in cols}
    for k in keywords:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def _build_schedule_from_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple, dense demo schedule from orders.csv only.
    Assumptions (best-effort, robust to column naming):
      - There is some column for order_id
      - Optionally a due_date column
      - Optionally a product/vrac column
    """
    if orders.empty:
        # Return empty schedule with expected columns
        return pd.DataFrame(
            columns=[
                "order_id",
                "operation",
                "sequence",
                "machine",
                "start",
                "end",
                "due_date",
                "wheel_type",
            ]
        )

    cols = orders.columns.tolist()

    # Guess columns
    order_col = _guess_column(cols, ["order_id", "order", "cmd"])
    if order_col is None:
        order_col = cols[0]  # fallback

    due_col = _guess_column(cols, ["due", "delivery", "date"])
    product_col = _guess_column(cols, ["product", "vrac", "sku", "item"])

    # Normalize order_id
    orders = orders.copy()
    orders["order_id"] = orders[order_col].astype(str)

    # Parse due_date if present
    if due_col:
        orders["due_date"] = pd.to_datetime(orders[due_col], errors="coerce")
    else:
        orders["due_date"] = pd.NaT

    # Build base start date
    if orders["due_date"].notna().any():
        base_start = orders["due_date"].min() - pd.Timedelta(days=2)
    else:
        base_start = pd.Timestamp.today().normalize()

    # Machine pool (3 demo tanks)
    machines = ["TANK-1", "TANK-2", "TANK-3"]
    machine_times = {m: base_start for m in machines}

    rows = []
    for i, row in orders.sort_values("due_date").reset_index(drop=True).iterrows():
        order_id = row["order_id"]
        due_date = row["due_date"]

        # Choose machine round-robin
        machine = machines[i % len(machines)]
        start_time = machine_times[machine]
        # Dense schedule: 6 hours per order
        duration = timedelta(hours=6)
        end_time = start_time + duration
        machine_times[machine] = end_time  # next order on this machine starts after this

        # Determine wheel_type (product label)
        if product_col:
            wheel_type = str(row[product_col])
        else:
            wheel_type = "VRAC_PRODUCT"

        rows.append(
            {
                "order_id": order_id,
                "operation": "VRAC_BATCH",
                "sequence": 10,  # same simple sequence for now
                "machine": machine,
                "start": start_time,
                "end": end_time,
                "due_date": due_date if pd.notna(due_date) else base_start + timedelta(days=1),
                "wheel_type": wheel_type,
            }
        )

    sched = pd.DataFrame(rows)
    return sched


@st.cache_data
def load_data():
    orders_path = DATA_DIR / "orders.csv"
    if not orders_path.exists():
        st.stop()  # will raise a clear Streamlit error

    # Try to parse due_date if there is such a column
    raw_orders = pd.read_csv(orders_path)
    # Let _build_schedule_from_orders handle parsing, so no parse_dates here
    sched = _build_schedule_from_orders(raw_orders)
    return raw_orders, sched


orders, base_schedule = load_data()

# Working schedule in session (so edits persist)
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = base_schedule.copy()


# ============================ FILTER & LOG STATE =======================
if "filters_open" not in st.session_state:
    st.session_state.filters_open = True
if "filt_max_orders" not in st.session_state:
    st.session_state.filt_max_orders = 40
if "filt_wheels" not in st.session_state:
    st.session_state.filt_wheels = sorted(base_schedule["wheel_type"].unique().tolist())
if "filt_machines" not in st.session_state:
    st.session_state.filt_machines = sorted(base_schedule["machine"].unique().tolist())
if "cmd_log" not in st.session_state:
    st.session_state.cmd_log = []  # rolling debug log


# ============================ CSS / LAYOUT =============================
sidebar_display = "block" if st.session_state.filters_open else "none"
st.markdown(
    f"""
<style>
[data-testid="stSidebar"] {{ display: {sidebar_display}; }}

/* Tighten spacing and keep everything on one screen as much as possible */
.block-container {{
    padding-top: 4px;
    padding-bottom: 36px;
    padding-left: 8px;
    padding-right: 8px;
    max-width: 100%;
}}

/* Hide Streamlit default footer/menu */
#MainMenu, footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# ============================ TOP BAR =============================
# Minimal header to save vertical space (no big title, no filter toggle)
st.write("")


# ============================ SIDEBAR FILTERS =========================
if st.session_state.filters_open:
    with st.sidebar:
        st.header("Filters ⚙️")
        st.session_state.filt_max_orders = st.number_input(
            "Orders",
            1,
            100,
            value=st.session_state.filt_max_orders,
            step=1,
            key="num_orders",
        )

        wheels_all = sorted(base_schedule["wheel_type"].unique().tolist())
        st.session_state.filt_wheels = st.multiselect(
            "VRAC product",
            wheels_all,
            default=st.session_state.filt_wheels or wheels_all,
            key="wheel_ms",
        )

        machines_all = sorted(base_schedule["machine"].unique().tolist())
        st.session_state.filt_machines = st.multiselect(
            "Machine",
            machines_all,
            default=st.session_state.filt_machines or machines_all,
            key="machine_ms",
        )

        if st.button("Reset filters", key="reset_filters"):
            st.session_state.filt_max_orders = 40
            st.session_state.filt_wheels = wheels_all
            st.session_state.filt_machines = machines_all
            st.rerun()

        # ---- Debug panel in sidebar ----
        with st.expander("🔎 Debug (last commands)"):
            if st.session_state.cmd_log:
                last = st.session_state.cmd_log[-1]
                st.markdown("**Last payload:**")
                st.json(last["payload"])
                st.markdown(
                    f"- **OK:** {last['ok']}   \n"
                    f"- **Message:** {last['msg']}   \n"
                    f"- **Source:** {last.get('source','?')}   \n"
                    f"- **Raw:** `{last['raw']}`"
                )
                mini = [
                    {
                        "raw": e["raw"],
                        "intent": e["payload"].get("intent", "?"),
                        "ok": e["ok"],
                        "msg": e["msg"],
                    }
                    for e in st.session_state.cmd_log[-5:]
                ]
                st.markdown("**Last 5 commands:**")
                st.dataframe(pd.DataFrame(mini))
            else:
                st.caption("No commands yet.")


# ============================ CURRENT FILTERED VIEW ====================
max_orders = int(st.session_state.filt_max_orders)
wheel_choice = (
    st.session_state.filt_wheels
    or sorted(base_schedule["wheel_type"].unique().tolist())
)
machine_choice = (
    st.session_state.filt_machines
    or sorted(base_schedule["machine"].unique().tolist())
)

sched = st.session_state.schedule_df.copy()
sched = sched[sched["wheel_type"].isin(wheel_choice)]
sched = sched[sched["machine"].isin(machine_choice)]

# Limit number of orders (by earliest due date)
if "due_date" in sched.columns:
    order_order = (
        sched[["order_id", "due_date"]]
        .drop_duplicates()
        .sort_values("due_date")
        .head(max_orders)["order_id"]
        .tolist()
    )
else:
    order_order = (
        sched[["order_id"]]
        .drop_duplicates()
        .head(max_orders)["order_id"]
        .tolist()
    )

sched = sched[sched["order_id"].isin(order_order)]
sched = sched.sort_values(["machine", "start"])


# ============================ GANTT CHART ==============================
if sched.empty:
    st.info("No operations match the current filters.")
else:
    select_order = alt.selection_point(fields=["order_id"], on="click", clear="dblclick")
    y_machines_sorted = sorted(sched["machine"].unique().tolist())

    base_enc = {
        "y": alt.Y("machine:N", sort=y_machines_sorted, title=None),
        "x": alt.X("start:T", title=None, axis=alt.Axis(format="%a %b %d")),
        "x2": "end:T",
    }

    bars = (
        alt.Chart(sched)
        .mark_bar(cornerRadius=3)
        .encode(
            color=alt.condition(
                select_order,
                alt.Color("order_id:N", legend=None),
                alt.value("#d3d3d3"),
            ),
            opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.25)),
            tooltip=[
                alt.Tooltip("order_id:N", title="Order"),
                alt.Tooltip("operation:N", title="Operation"),
                alt.Tooltip("sequence:Q", title="Seq"),
                alt.Tooltip("machine:N", title="Machine"),
                alt.Tooltip("start:T", title="Start"),
                alt.Tooltip("end:T", title="End"),
                alt.Tooltip("due_date:T", title="Due"),
                alt.Tooltip("wheel_type:N", title="VRAC"),
            ],
        )
    )

    labels = (
        alt.Chart(sched)
        .mark_text(
            align="left",
            dx=6,
            baseline="middle",
            fontSize=10,
            color="white",
        )
        .encode(
            text="order_id:N",
            opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.7)),
        )
    )

    gantt = (
        alt.layer(bars, labels, data=sched)
        .encode(**base_enc)
        .add_params(select_order)
        .properties(width="container", height=420)
        .configure_view(stroke=None)
    )
    st.altair_chart(gantt, use_container_width=True)


# ============================ INTELLIGENCE INPUT =======================
user_cmd = st.chat_input("Type a command (Delay/Advance/Swap)…", key="cmd_input")


# ============================ NLP / INTENT UTILITIES ===================
WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def _num_token_to_float(tok: str):
    tok = tok.strip().lower()
    word_map = {
        "zero": 0,
        "one": 1,
        "a": 1,
        "an": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "half": 0.5,
    }
    if tok in word_map:
        return float(word_map[tok])
    tok = tok.replace(",", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def _parse_time_delta(text: str) -> dict:
    d = {"days": 0.0, "hours": 0.0, "minutes": 0.0}
    for num, unit in re.findall(
        r"([\d\.,]+|\b\w+\b)\s*(days?|d|hours?|h|minutes?|mins?|m)\b", text, flags=re.I
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


def _extract_with_openai(user_text: str):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """
You are an assistant that translates natural-language scheduling commands into a JSON payload.

You are working on a factory scheduling board with operations on different machines.
User can ask to delay, advance, or swap orders.

Return ONLY a JSON object with the following keys, depending on intent:

- intent: one of ["delay_order", "move_order", "swap_orders"]
- order_id: string (required for all intents)
- order_id_2: string (required only for swap_orders)
- machine: string (optional, if user mentions a specific machine)
- delta_days: number (can be negative for advance)
- delta_hours: number
- delta_minutes: number
- target_start: string ISO timestamp or null
- target_end: string ISO timestamp or null
"""
    msg = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = msg.choices[0].message.content
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {}
    return payload, raw


def extract_intent(user_text: str) -> dict:
    txt = user_text.strip()
    low = txt.lower()

    # Try to detect swap
    m_swap = re.search(r"swap\s+order\s+(\S+)\s+with\s+(\S+)", low)
    if m_swap:
        return {
            "intent": "swap_orders",
            "order_id": m_swap.group(1).upper(),
            "order_id_2": m_swap.group(2).upper(),
            "machine": None,
            "delta_days": 0,
            "delta_hours": 0,
            "delta_minutes": 0,
            "target_start": None,
            "target_end": None,
            "source": "rules",
        }

    # Detect delay/advance keywords
    intent = None
    sign = +1
    if any(w in low for w in ["delay", "postpone", "push back", "pushback", "push it"]):
        intent = "delay_order"
        sign = +1
    if any(w in low for w in ["advance", "bring forward", "pull in", "earlier"]):
        intent = "delay_order"
        sign = -1

    delta = _parse_time_delta(low)
    has_delta = any(v != 0 for v in delta.values())

    # Extract order_id pattern like O-1005 or 1005
    m_order = re.search(r"(o[-_ ]?\d+)", low, flags=re.I)
    order_id = (
        m_order.group(1).upper().replace(" ", "").replace("_", "-") if m_order else None
    )

    if intent and has_delta and order_id:
        return {
            "intent": intent,
            "order_id": order_id,
            "order_id_2": None,
            "machine": None,
            "delta_days": sign * delta["days"],
            "delta_hours": sign * delta["hours"],
            "delta_minutes": sign * delta["minutes"],
            "target_start": None,
            "target_end": None,
            "source": "rules",
        }

    # Fallback to OpenAI if available
    if OPENAI_API_KEY:
        payload, raw = _extract_with_openai(txt)
        payload["source"] = "openai"
        payload["_raw"] = raw
        return payload

    return {
        "intent": "unknown",
        "order_id": None,
        "order_id_2": None,
        "machine": None,
        "delta_days": 0,
        "delta_hours": 0,
        "delta_minutes": 0,
        "target_start": None,
        "target_end": None,
        "source": "none",
        "_raw": txt,
    }


# ============================ VALIDATION ================================
def validate_intent(payload: dict, orders_df: pd.DataFrame, schedule_df: pd.DataFrame):
    intent = payload.get("intent")
    if intent not in ["delay_order", "move_order", "swap_orders"]:
        return False, f"Unsupported or missing intent: {intent}"

    order_id = payload.get("order_id")
    if not order_id:
        return False, "Missing order_id."

    if order_id not in schedule_df["order_id"].unique():
        return False, f"Order {order_id} not found in current schedule."

    if intent == "swap_orders":
        order_id_2 = payload.get("order_id_2")
        if not order_id_2:
            return False, "Missing order_id_2 for swap."
        if order_id_2 not in schedule_df["order_id"].unique():
            return False, f"Order {order_id_2} not found in current schedule."

    if intent == "delay_order":
        if (
            payload.get("delta_days") is None
            and payload.get("delta_hours") is None
            and payload.get("delta_minutes") is None
        ):
            return False, "Need a delta (days/hours/minutes) for delay/advance."

    return True, "OK"


# ============================ APPLY OPERATIONS =========================
def apply_delay(schedule_df: pd.DataFrame, order_id: str, delta: timedelta) -> pd.DataFrame:
    df = schedule_df.copy()
    mask = df["order_id"] == order_id
    df.loc[mask, "start"] = df.loc[mask, "start"] + delta
    df.loc[mask, "end"] = df.loc[mask, "end"] + delta
    return df


def apply_swap(schedule_df: pd.DataFrame, order_id_1: str, order_id_2: str) -> pd.DataFrame:
    df = schedule_df.copy()
    df1 = df[df["order_id"] == order_id_1].copy()
    df2 = df[df["order_id"] == order_id_2].copy()

    if df1.empty or df2.empty:
        return df

    df.loc[df["order_id"] == order_id_1, ["start", "end"]] = df2[["start", "end"]].values
    df.loc[df["order_id"] == order_id_2, ["start", "end"]] = df1[["start", "end"]].values
    return df


# ============================ HANDLE USER COMMAND ======================
if user_cmd:
    try:
        payload = extract_intent(user_cmd)
        ok, msg = validate_intent(payload, orders, st.session_state.schedule_df)

        # log it (json-safe)
        log_payload = dict(payload)
        raw = log_payload.pop("_raw", user_cmd)
        st.session_state.cmd_log.append(
            {
                "raw": raw,
                "payload": log_payload,
                "ok": ok,
                "msg": msg,
                "source": payload.get("source", "?"),
            }
        )

        if ok:
            if payload["intent"] == "delay_order":
                delta = timedelta(
                    days=payload.get("delta_days", 0) or 0,
                    hours=payload.get("delta_hours", 0) or 0,
                    minutes=payload.get("delta_minutes", 0) or 0,
                )
                st.session_state.schedule_df = apply_delay(
                    st.session_state.schedule_df, payload["order_id"], delta
                )
                st.success(
                    f"Applied delay/advance on order {payload['order_id']} ({delta})."
                )

            elif payload["intent"] == "swap_orders":
                st.session_state.schedule_df = apply_swap(
                    st.session_state.schedule_df,
                    payload["order_id"],
                    payload["order_id_2"],
                )
                st.success(
                    f"Swapped orders {payload['order_id']} and {payload['order_id_2']}."
                )

        else:
            st.warning(f"Command rejected: {msg}")

    except Exception as e:
        st.error(f"Error while processing command: {e}")
