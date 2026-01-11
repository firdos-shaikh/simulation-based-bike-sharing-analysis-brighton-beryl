# 🚲 Simulation-Based Cost–Benefit Analysis of Bike-Sharing in Brighton & Hove

This project is a Python-based **agent-based simulation** developed as part of my **MSc Artificial Intelligence & Adaptive Systems dissertation (First Class)** at the **University of Sussex**.

The system models beryl bike-sharing usage across the **Brighton & Hove urban road network** to evaluate operational performance, costs, incidents, and overall system efficiency, supporting data-driven transport planning and decision-making.

---

## 📖 Project Overview

The simulation represents a real-world bike-sharing system using:
- Agent-based modelling for riders and bikes
- A road network graph built from OpenStreetMap data
- Realistic station locations and movement behaviour

The model simulates a full day of operations and captures:
- Trip demand and system utilisation
- Accident and maintenance events
- Operational and infrastructure costs
- System-level performance metrics

An interactive web-based dashboard allows users to run, monitor, and analyse simulations in real time.

---

## 🏙️ Simulation Environment

The model is built around the **Brighton & Hove** road network and includes:
- Real bike station coordinates
- Urban road topology and routing
- Pedal bikes and e-bikes
- Stochastic rider behaviour and event generation

Each simulation run produces unique outcomes due to probabilistic agent decisions and randomised events.

---

## 🎯 Objectives

The simulation aims to:
1. Evaluate the cost–benefit performance of bike-sharing systems  
2. Analyse operational risks such as accidents and maintenance  
3. Support policy-level decisions on sustainable urban mobility  
4. Demonstrate the value of simulation-based analysis in transport planning  

---

## 🧠 Methods & Techniques

- Agent-Based Modelling  
- Simulation-Based Cost–Benefit Analysis  
- Network & Graph Modelling  
- Event-Driven Simulation  
- Data Analytics & Visualisation  

---

## 🛠️ Technologies Used

- Python 3  
- Dash & Dash-Leaflet (interactive dashboard)  
- Pandas (data processing)  
- NetworkX & OSMnx (urban road network modelling)  
- XlsxWriter (simulation output reporting)  

---

## 🎛️ Simulation Controls

| Control | Function |
|------|------|
| Start Simulation | Runs a full-day simulation |
| Stop Simulation | Stops and resets the model |
| Speed Control | Adjusts simulation speed (1×, 30×, 60×) |
| Shutdown Server | Gracefully terminates the application |

---

## 📊 Outputs

Each simulation run generates an Excel file containing:
- **Trips** – Completed journeys and durations  
- **Accidents** – Incident events with timestamps  
- **Maintenance** – Bike maintenance events and costs  
- **Metrics** – Daily system performance summaries  

These outputs enable quantitative evaluation and scenario comparison.

---

## ▶️ How to Run the Simulation

**Install Dependencies**
pip install dash dash-leaflet pandas osmnx networkx xlsxwriter

**Run the Application**
python agent_based_sim_bikes_brighton.py

**Open your browser at:**
http://127.0.0.1:8050/

---

## 📂 Project Structure

```text
agent_based_sim_bikes_brighton.py   # Main simulation engine
stations.csv                        # Bike station locations
brighton_graph.graphml              # Urban road network graph
assets/                             # Map icons and visual assets
README.md
