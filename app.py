
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
from pathlib import Path
import plotly.graph_objects as go

from utils.scheduling import schedule_orders
from utils.nlp import parse_command

st.set_page_config(page_title="FMCG Vrac Scheduler (4 Ops, No-Wait)", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_products():
    return pd.read_csv(DATA_DIR / "vrac_products.csv")

@st.cache_data
def load_lines():
    return pd.read_csv(DATA_DIR / "lines.csv")

@st.cache_data
def load_orders():
    df = pd.read_csv(DATA_DIR / "orders.csv")
    df["due_date"] = pd.to_datetime(df["due_date"]).dt.tz_localize(None)
    return df

def save_orders(df: pd.DataFrame):
    df.to_csv(DATA_DIR / "orders.csv", index=False)
    load_orders.clear()

def to_records(df):
    return df.to_dict(orient="records")

def build_schedule(products_df, orders_df):
    prods = to_records(products_df)
    ords = to_records(orders_df)
    for o in ords:
        o["due_date"] = pd.to_datetime(o["due_date"]).to_pydatetime()
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    ops = schedule_orders(ords, prods, start_time=start_time)
    rows = []
    for op in ops:
        rows.append({
            "order_id": op.order_id,
            "sku_id": op.sku_id,
            "operation": op.op,
            "line_id": op.line_id,
            "start": op.start,
            "end": op.end,
            "duration_h": (op.end - op.start).total_seconds()/3600.0
        })
    return pd.DataFrame(rows)

def gantt(df: pd.DataFrame):
    if df.empty:
        st.info("No schedule to show."); return
    df = df.sort_values("start")
    fig = go.Figure()
    color_map = {"MIX_1":"#1f77b4","TRANS_1":"#9467bd","FILL_1":"#ff7f0e","FIN_1":"#2ca02c"}
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["end"] - row["start"]],
            y=[row["line_id"]],
            base=[row["start"]],
            orientation='h',
            name=f'{row["order_id"]}-{row["operation"]}',
            hovertemplate=(
                f"Order: {row['order_id']}<br>"
                f"SKU: {row['sku_id']}<br>"
                f"Op: {row['operation']} on {row['line_id']}<br>"
                f"Start: {row['start']}<br>"
                f"End: {row['end']}<br>"
                f"Duration: {row['duration_h']:.2f} h"
            ),
            marker_color=color_map.get(row["line_id"])
        ))
    fig.update_layout(
        barmode='stack',
        title="Gantt – 4-Op No-Wait Chain (Mix → Transfer → Fill → Finish)",
        xaxis_title="Time",
        yaxis_title="Line",
        showlegend=False,
        bargap=0.15,
        height=560
    )
    fig.update_yaxes(categoryorder='array', categoryarray=["MIX_1","TRANS_1","FILL_1","FIN_1"])
    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI ----------------
st.title("FMCG Vrac Scheduler (4 Ops, No-Wait)")

with st.container():
    st.subheader("Command Prompt")
    cmd = st.text_input("Type a command (e.g., 'add order VRAC_SHAMPOO_BASE 8000kg due 2025-11-06')")
    if st.button("Apply Command", type="primary"):
        parsed = parse_command(cmd or "")
        st.write("Parsed:", parsed)
        orders_df = load_orders()

        if parsed.get("intent") == "add_order":
            new_id = f"ORD-{len(orders_df)+1:03d}"
            new_row = {
                "order_id": new_id,
                "sku_id": parsed["sku_id"],
                "qty_kg": int(parsed["qty_kg"]),
                "due_date": parsed["due_date"],
            }
            orders_df = pd.concat([orders_df, pd.DataFrame([new_row])], ignore_index=True)
            save_orders(orders_df)
            st.success(f"Order {new_id} added.")
        elif parsed.get("intent") == "delete_order":
            oid = parsed["order_id"]
            before = len(orders_df)
            orders_df = orders_df[orders_df["order_id"] != oid].copy()
            if len(orders_df) < before:
                save_orders(orders_df)
                st.success(f"Order {oid} deleted.")
            else:
                st.warning(f"Order {oid} not found.")
        elif parsed.get("intent") == "move_order":
            oid = parsed["order_id"]; due = parsed["due_date"]
            if oid in set(orders_df["order_id"]):
                orders_df.loc[orders_df["order_id"]==oid, "due_date"] = due
                save_orders(orders_df)
                st.success(f"Order {oid} moved to {due}.")
            else:
                st.warning(f"Order {oid} not found.")
        else:
            st.info("I could not understand that. Try: add order / delete order / move order")

with st.sidebar:
    st.subheader("Data Inputs")
    if st.button("Reload CSVs"):
        load_products.clear(); load_lines.clear(); load_orders.clear()
        st.experimental_rerun()
    st.divider()
    st.subheader("Quick Add")
    products_df = load_products()
    sku = st.selectbox("SKU", products_df["sku_id"])
    qty = st.number_input("Quantity (kg)", value=8000, min_value=1000, step=500)
    due = st.date_input("Due date", date.today() + timedelta(days=3))
    if st.button("Add Order"):
        orders_df = load_orders()
        new_id = f"ORD-{len(orders_df)+1:03d}"
        new_row = {"order_id": new_id, "sku_id": sku, "qty_kg": qty, "due_date": str(due)}
        orders_df = pd.concat([orders_df, pd.DataFrame([new_row])], ignore_index=True)
        save_orders(orders_df)
        st.success(f"Order {new_id} added.")
        st.experimental_rerun()

tabs = st.tabs(["📦 Orders", "🧪 Vrac Products", "🛠️ Lines", "🗓️ Schedule"])

with tabs[0]:
    st.dataframe(load_orders(), use_container_width=True)

with tabs[1]:
    st.dataframe(load_products(), use_container_width=True)

with tabs[2]:
    st.dataframe(load_lines(), use_container_width=True)

with tabs[3]:
    orders_df = load_orders()
    products_df = load_products()
    sched_df = build_schedule(products_df, orders_df)

    last_ops = sched_df.loc[sched_df["operation"]=="FIN"].copy()
    orders = load_orders().copy()
    last_ops = last_ops.merge(orders[["order_id","due_date"]], on="order_id", how="left")
    last_ops["is_on_time"] = pd.to_datetime(last_ops["end"]) <= pd.to_datetime(last_ops["due_date"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Orders", len(orders))
    ontime = (100*last_ops["is_on_time"].mean()) if len(last_ops) else 0.0
    c2.metric("On-time %", f"{ontime:.0f}%")
    horizon_days = max(0, (sched_df["end"].max() - datetime.now()).days) if not sched_df.empty else 0
    c3.metric("Horizon (days)", f"{horizon_days}")

    st.caption("No-wait enforced: MIX ends → TRANS starts → FILL starts → FIN starts with zero waiting.")
    gantt(sched_df)

    csv_bytes = sched_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download schedule as CSV", data=csv_bytes, file_name="fmcg_schedule.csv", mime="text/csv")
