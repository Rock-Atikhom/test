# 🍔 Restaurant Analytics Dashboard

An interactive Streamlit dashboard built to analyze restaurant operational data, customer behavior, and menu sales performance.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Rock-Atikhom/test.git
   cd test
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the Streamlit app locally:
```bash
streamlit run app.py
```

---

## 📊 Key Features

- **Data Profiling**: Preview the dataset, audit missing values, normalize columns, and detect outlier values.
- **Sales Analytics**: Deep-dive sales performance breakdown by category (Food vs. Drinks) and individual menu items.
- **Consumer Behavior**: Visual timelines tracking store traffic flow trends by hour of the day and day of the week.
- **Operations Analysis**: Monitor kitchen and drinks staff metrics by month and day, alongside average cooking/preparation times.

---

## 📂 Project Structure

* **[app.py](file:///Users/atikhommuhammadaree/Documents/test/app.py)**: The main Streamlit dashboard application file containing the visualizations and interface.
* **[restaurant_orders.csv](file:///Users/atikhommuhammadaree/Documents/test/restaurant_orders.csv)**: The dataset containing transaction history, order times, serve times, menu items, and staffing metrics.
* **[requirements.txt](file:///Users/atikhommuhammadaree/Documents/test/requirements.txt)**: List of Python packages required to run the dashboard.
* **[packages.txt](file:///Users/atikhommuhammadaree/Documents/test/packages.txt)**: Linux system packages (optional deployment dependency).

---

## 🛠️ Built With

* [Streamlit](https://streamlit.io/) - The fastest way to build and share data apps.
* [Pandas](https://pandas.pydata.org/) - Powerful data manipulation & analysis tool.
* [Plotly Express](https://plotly.com/python/plotly-express/) - Interactive chart visualization.
