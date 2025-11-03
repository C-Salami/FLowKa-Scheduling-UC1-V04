import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

random.seed(7)

# FMCG vrac products (stored in wheel_type column for compatibility)
PRODUCTS = ["VRAC_SHAMPOO_BASE", "VRAC_CONDITIONER_BASE", "VRAC_HAIR_MASK"]
QUANTITIES = [4000, 6000, 8000, 10000, 12000]
TIMES = ["08:00:00", "12:00:00", "16:00:00"]

# Rates per product and operation (kg/h)
RATES = {
    "VRAC_SHAMPOO_BASE": {
        "mix": 6000,
        "trans": 15000,
        "fill": 5000,
        "fin": 20000,
    },
    "VRAC_CONDITIONER_BASE": {
        "mix": 5000,
        "trans": 15000,
        "fill": 4500,
        "fin": 18000,
    },
    "VRAC_HAIR_MASK": {
        "mix": 4000,
        "trans": 12000,
        "fill": 3500,
        "fin": 16000,
    },
}


def build_orders(num_orders: int = 50) -> pd.DataFrame:
    """
    Create 50 orders starting from 2025-11-03.

    - order_id: O001..O050
    - wheel_type: VRAC_SHAMPOO_BASE / VRAC_CONDITIONER_BASE / VRAC_HAIR_MASK
    - quantity: cycles over [4000, 6000, 8000, 10000, 12000]
    - due_date: from 2025-11-03 to 2025-11-12, times 08:00 / 12:00 / 16:00
    """
    base_date = datetime(2025, 11, 3).date()
    rows = []

    for i in range(1, num_orders + 1):
        idx = i - 1
        order_id = f"O{i:03d}"

        sku = PRODUCTS[idx % len(PRODUCTS)]
        qty = QUANTITIES[idx % len(QUANTITIES)]

        # every 5 orders, push due date by 1 day
        day_offset = idx // 5
        date = base_date + timedelta(days=day_offset)

        time_str = TIMES[idx % len(TIMES)]
        due_date = datetime.fromisoformat(f"{date.isoformat()} {time_str}")

        rows.append(
            {
                "order_id": order_id,
                # keep column name 'wheel_type' to avoid breaking the app
                "wheel_type": sku,
                "quantity": qty,
                "due_date": due_date,
            }
        )

    return pd.DataFrame(rows)


def build_schedule(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 4-operation chain per order:
      1. MIXING/PROCESSING  on 'Mixing/Processing'
      2. TRANSFER/HOLDING   on 'Transfer/Holding'
      3. FILLING/CAPPING    on 'Filling/Capping'
      4. FINISHING/QC       on 'Finishing/QC'

    Hard constraint INSIDE each order:
    op(n+1).start == op(n).end  (strict end-start, no waiting within the chain)

    For simplicity, we schedule orders one full chain after another,
    so resources never overlap between orders.
    """
    rows = []

    # Chain starting point (before first due dates)
    current_start = datetime(2025, 11, 3, 6, 0, 0)

    for _, o in orders_df.sort_values("order_id").iterrows():
        order_id = o["order_id"]
        sku = o["wheel_type"]
        qty = float(o["quantity"])
        due = o["due_date"]

        rates = RATES[sku]

        def dur(rate_kg_per_hour: float) -> timedelta:
            # duration in hours = quantity / rate
            return timedelta(hours=qty / float(rate_kg_per_hour))

        # Op 1: Mixing/Processing
        mix_start = current_start
        mix_end = mix_start + dur(rates["mix"])

        # Op 2: Transfer/Holding
        trans_start = mix_end
        trans_end = trans_start + dur(rates["trans"])

        # Op 3: Filling/Capping
        fill_start = trans_end
        fill_end = fill_start + dur(rates["fill"])

        # Op 4: Finishing/QC
        fin_start = fill_end
        fin_end = fin_start + dur(rates["fin"])

        rows.append(
            {
                "order_id": order_id,
                "wheel_type": sku,
                "operation": "MIXING/PROCESSING",
                "sequence": 1,
                "machine": "Mixing/Processing",
                "start": mix_start,
                "end": mix_end,
                "due_date": due,
            }
        )
        rows.append(
            {
                "order_id": order_id,
                "wheel_type": sku,
                "operation": "TRANSFER/HOLDING",
                "sequence": 2,
                "machine": "Transfer/Holding",
                "start": trans_start,
                "end": trans_end,
                "due_date": due,
            }
        )
        rows.append(
            {
                "order_id": order_id,
                "wheel_type": sku,
                "operation": "FILLING/CAPPING",
                "sequence": 3,
                "machine": "Filling/Capping",
                "start": fill_start,
                "end": fill_end,
                "due_date": due,
            }
        )
        rows.append(
            {
                "order_id": order_id,
                "wheel_type": sku,
                "operation": "FINISHING/QC",
                "sequence": 4,
                "machine": "Finishing/QC",
                "start": fin_start,
                "end": fin_end,
                "due_date": due,
            }
        )

        # Next order starts only after this whole chain finishes
        current_start = fin_end

    return pd.DataFrame(rows)


def main():
    orders_df = build_orders(num_orders=50)
    schedule_df = build_schedule(orders_df)

    Path("data").mkdir(exist_ok=True, parents=True)
    orders_df.to_csv("data/scooter_orders.csv", index=False)
    schedule_df.to_csv("data/scooter_schedule.csv", index=False)
    print("Wrote FMCG vrac data to data/scooter_orders.csv and data/scooter_schedule.csv")


if __name__ == "__main__":
    main()
