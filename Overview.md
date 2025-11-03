
# FLowKa Scheduling UC1 — “Scooter Wheels Scheduler”
**An AI-assisted production scheduling prototype that lets planners adjust a live Gantt chart using natural-language commands in a lightweight Streamlit app.**

**Repository:** [C-Salami / FLowKa-Scheduling-UC1-V01](https://github.com/C-Salami/FLowKa-Scheduling-UC1-V01)

---

## Why We Built This

Manufacturing planners frequently recalibrate schedules because of machine downtime, expedited orders, or supply delays.  
Traditional tools are cumbersome and slow to reflect “what-if” changes.  
This project demonstrates a **human-in-the-loop** approach where planners simply type commands (for example: `delay O076 by 2 hours`) and the schedule updates instantly.

**Key goals:**
- **Speed:** Reduce friction when applying common schedule edits.  
- **Explainability:** Keep changes transparent and reversible.  
- **Adoptability:** Use a familiar web UI (Streamlit) and CSV data for quick pilots.

---

## Architecture at a Glance

### Core Components
- **Frontend:** Streamlit (single-page web app)  
- **Visualization:** Altair (Gantt chart of operations per machine)  
- **Data:** CSV files (orders and operation-level schedule) processed with Pandas  
- **NLP Parser:** Rule-based (Regex) with optional OpenAI API for advanced intent extraction  
- **Scheduling Logic:** Apply *delay*, *move*, *swap* operations and repack machines to avoid overlaps  

### Data Flow
1. Load orders and schedule CSVs  
2. Render Gantt by machine and time  
3. Parse the user’s natural-language command into structured intent  
4. Validate order IDs and parameters  
5. Apply changes and repack machines to keep operations non-overlapping  
6. Re-render the chart with updated bars  

### System Diagram
```
+------------------------+
|   Streamlit Frontend   |
|  (User, Command Input) |
+-----------+------------+
            |
            v
+------------------------+
|   NLP Parser Layer     |
|  (Regex / OpenAI GPT)  |
+-----------+------------+
            |
            v
+------------------------+
|   Validation Engine    |
|   (Order & Data Check) |
+-----------+------------+
            |
            v
+------------------------+
|   Scheduler Logic      |
| (Delay, Move, Swap)    |
+-----------+------------+
            |
            v
+------------------------+
|   Altair Visualization |
+------------------------+
```

---

## Sample Data Included

Deterministic scripts (in the `Scripts/` folder) generate example orders and a machine-routed operation schedule.  
You can replace these CSVs with your own data as long as the columns remain consistent.

### Orders (`scooter_orders.csv`)

| order_id | wheel_type | quantity | due_date |
|-----------|-------------|-----------|-----------|
| O001 | Eco-160 | 50 | 2025-08-20 |
| O002 | Sport-180 | 40 | 2025-08-21 |
| O003 | City-200 | 60 | 2025-08-22 |

### Schedule (`scooter_schedule.csv`)

| order_id | operation | machine | start | end | duration_hours |
|-----------|------------|----------|--------|------|----------------|
| O001 | Lathe | M1 | 2025-08-18 08:00 | 2025-08-18 10:00 | 2.00 |
| O001 | CNC | M2 | 2025-08-18 10:15 | 2025-08-18 12:00 | 1.75 |
| O001 | Drill | M3 | 2025-08-18 12:15 | 2025-08-18 13:30 | 1.25 |

### Production Flow
```
Lathe → CNC → Drill → Paint → Assembly → QA
```
Each order flows through the full routing; durations are preserved when the schedule is shifted, and machine repacking ensures no overlaps on the same machine.

---

## Natural-Language Commands and OpenAI Integration

Out of the box, the app uses a simple rule-based parser.  
To enable more flexible phrasing and better recognition, you can connect an OpenAI API key.  
The app will then send commands to a GPT model for semantic parsing before applying scheduling changes.

### Supported Command Types

| Type | Example | Effect |
|-------|----------|--------|
| Delay | `delay O076 by 2 hours` | Shifts all O076 operations forward by 2 hours |
| Move | `move O045 to 2025-08-22 14:30` | Anchors O045 at the target timestamp |
| Swap | `swap O045 and O076` | Exchanges start anchors of both orders |

### Enable OpenAI (Optional)
1. Create an account at [platform.openai.com](https://platform.openai.com) and generate an API key.  
2. In the project folder, create `.streamlit/secrets.toml`.  
3. Add the following:
   ```
   OPENAI_API_KEY="your_api_key_here"
   ```
4. Restart the Streamlit app.  
   The NLP layer will use the key automatically if configured.

---

## Step-by-Step: Run and Test Locally

### 1. Prerequisites
- Python 3.10 or newer  
- Git  
- Internet connection (required only for OpenAI API usage)

### 2. Clone the Repository
```bash
git clone https://github.com/C-Salami/FLowKa-Scheduling-UC1-V01.git
cd FLowKa-Scheduling-UC1-V01
```

### 3. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate Sample Data (if needed)
```bash
python Scripts/generate_sample_data.py
```
This writes `data/scooter_orders.csv` and `data/scooter_schedule.csv`.

### 6. (Optional) Add OpenAI API Key
Create the file `.streamlit/secrets.toml` and add:
```
OPENAI_API_KEY="sk-..."
```

### 7. Launch the Streamlit App
```bash
streamlit run app.py
```
Then open your browser at [http://localhost:8501](http://localhost:8501).

### 8. Try Commands in the App
- `delay O076 by 2 hours`  
- `move O045 to 2025-08-22 14:30`  
- `swap O045 and O076`  
- `delay O076 by -1 day`

### Validation Examples
- Unknown order: `delay O999 by 1 hour` → shows an error  
- Missing duration: `delay O076` → validation error  
- Invalid swap: `swap O076 and O076` → error message  

---

## Technology Stack

| Layer | Technology | Purpose |
|--------|-------------|----------|
| UI | Streamlit | Interactive, reactive web interface |
| Charts | Altair | Gantt visualization |
| Data | Pandas + CSV | Load, filter, and transform schedule data |
| NLP | Regex / OpenAI GPT | Intent extraction from natural-language commands |
| Logic | Custom Python | Apply edits and repack to avoid overlaps |

---

## Conclusion

FLowKa Scheduling UC1 shows how a simple, open-source stack — Streamlit, Altair, and optional GPT integration — can power an **AI-assisted scheduling assistant** for real manufacturing use cases.  
It enables faster human decisions, keeps planners in control, and provides a foundation for integrating AI within larger ERP or MES environments.

**Project:** [C-Salami / FLowKa-Scheduling-UC1-V01](https://github.com/C-Salami/FLowKa-Scheduling-UC1-V01)  
**Author:** IT Architect — FLowKa Scheduling Project (Use Case 1)
