# Restaurant Operations Analytics Dashboard

A data analytics dashboard built with Streamlit that provides an exploratory and operational view of restaurant transaction data. The dashboard covers data quality inspection, sales performance, customer traffic patterns, staffing levels, and kitchen efficiency metrics — all derived from a single structured dataset of restaurant orders.

---

## Background

This project was developed to answer operational questions that restaurant managers commonly face:

- Which menu items and categories generate the most revenue and volume?
- At what hours and days of the week does customer traffic peak?
- How does staffing level vary across time, and is it aligned with demand?
- How long does it take to fulfill each menu item, and does the answer differ between food and drink orders?

The dataset (`restaurant_orders.csv`) contains one row per order transaction and includes fields for order time, serve time, menu item, category, price, and staffing numbers for kitchen and drinks stations.

---

## Project Structure

```
.
├── app.py                    # Streamlit dashboard application
├── restaurant_orders.csv     # Order transaction dataset
├── requirements.txt          # Python package dependencies
├── packages.txt              # Linux system packages (used for cloud deployment)
└── cctv_jetson_cv/           # Companion module: real-time CCTV vehicle tracking pipeline
```

The `cctv_jetson_cv` folder is a separate, independently deployable module. It runs a computer vision pipeline on Jetson Nano hardware (or locally) that detects and counts vehicles crossing a configurable virtual line. It is maintained in its own repository at [github.com/Rock-Atikhom/cctv-jetson-cv](https://github.com/Rock-Atikhom/cctv-jetson-cv).

---

## Dataset

**File:** `restaurant_orders.csv`

The dataset is loaded locally when the file is present; otherwise the app falls back to the public raw URL hosted on GitHub.

| Column | Type | Description |
|---|---|---|
| `date` | Date | Calendar date of the transaction |
| `order_time` | Datetime | Timestamp when the order was placed |
| `serve_time` | Datetime | Timestamp when the order was fulfilled |
| `menu` | String | Name of the menu item ordered |
| `category` | String | Either `food` or `drink` |
| `price` | Float | Price of the menu item in local currency |
| `kitchen_staff` | Integer | Number of kitchen staff on shift at time of order |
| `drinks_staff` | Integer | Number of drinks staff on shift at time of order |

The dashboard computes a derived `cooking_time` column as the difference between `serve_time` and `order_time` for each transaction.

---

## Dashboard Sections

### Data Profiling

A set of expandable panels that walk through the raw dataset before any analysis:

- **Dataset Preview** — displays the full dataframe so the user can visually inspect rows and column names.
- **Missing Value Check** — counts null values per column to surface data quality gaps before analysis.
- **Column Normalisation** — renames all columns to lowercase with underscores, making downstream groupby operations unambiguous.
- **Outlier Detection** — shows summary statistics (`min`, `max`, `mean`, `std`, percentiles) for `price`, `kitchen_staff`, and `drinks_staff` to flag any anomalous values.
- **Datetime Parsing** — converts `date`, `order_time`, and `serve_time` from string to proper datetime types, enabling time-based aggregations throughout the rest of the dashboard.
- **Menu Catalogue** — prints the full list of unique menu item names in the dataset.

### Sales Analytics

Four charts that break down order volume and revenue:

1. **Overall Quantity by Menu** — a horizontal bar chart ranking every menu item by total order count, making it immediately clear which items drive the most volume.
2. **Proportion by Category** — a pie chart showing the split of total orders between food and drink categories.
3. **Overall Sales by Food Category** — a horizontal bar chart of cumulative revenue for food items, ordered by total revenue descending.
4. **Overall Sales by Drink Category** — the same chart for drink items.

### Consumer Behavior

Two line charts that visualise when customers arrive:

5. **Consumer Behavior by Hour** — aggregates total orders by hour of day (00–23) to identify peak service windows. Each data point is annotated with its count.
6. **Consumer Behavior by Day of Week** — aggregates total orders by day of week (0 = Sunday through 6 = Saturday) to identify which days carry the highest load.

### Staffing Analysis

Two grouped bar charts that compare kitchen and drinks staffing levels over time:

7. **Kitchen and Drinks Staff by Month** — shows the average number of kitchen staff and drinks staff on shift for each calendar month, grouped side by side.
8. **Kitchen and Drinks Staff by Day of Week** — the same comparison broken down by day of week.

### Kitchen Efficiency

Two dataframes that connect staffing and preparation time to individual menu items:

9. **Food by Kitchen Staff** — for each food item, shows average kitchen staff count, average cooking time, and total order count. Ordered by most-ordered item descending.
10. **Drink by Drinks Staff** — the same view for drink items, using drinks staff and cooking time.

---

## Getting Started

### Requirements

- Python 3.8 or higher

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Rock-Atikhom/restaurant-analytics-dashboard.git
cd restaurant-analytics-dashboard
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

If `restaurant_orders.csv` is present in the working directory it will be read directly. If the file is absent, the app will attempt to load it from the public GitHub URL.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework for the dashboard interface |
| `pandas` | Data loading, transformation, and aggregation |
| `plotly-express` | Interactive chart rendering (bar, line, pie) |
| `plost` | Additional Streamlit-native charting utilities |
| `numpy` | Numerical operations used alongside pandas |

---

## Deployment

The `packages.txt` file lists any Linux system-level dependencies needed for cloud deployment platforms such as Streamlit Community Cloud. Currently this file is empty, meaning no additional system packages are required beyond the Python dependencies listed above.

To deploy on Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
3. Set the main file path to `app.py`.
4. The platform will install packages from `requirements.txt` and `packages.txt` automatically.

---

## Related Repository

The CCTV vehicle tracking module that was originally developed alongside this project has been separated into its own repository:

[github.com/Rock-Atikhom/cctv-jetson-cv](https://github.com/Rock-Atikhom/cctv-jetson-cv)

That module runs a real-time object detection and line-crossing count pipeline on RTSP camera streams. It writes crossing events to a structured CSV log (`crossing_log.csv`) using a `TrafficMetricsLog` interface that can be connected to this dashboard for integrated restaurant operations and traffic intelligence reporting.
