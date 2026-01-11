import dash
from dash import html, dcc, no_update
from dash.dependencies import Output, Input, State
import dash_leaflet as dl
import pandas as pd
import random
from datetime import datetime, timedelta
import math
import os
import osmnx as ox
import networkx as nx
import xlsxwriter
import time
import threading
from collections import defaultdict

#Initialize Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load road network for Brighton
# To speed up subsequent runs, it loads from a local file if it exists.
print("Loading road network...")
place = "Brighton and Hove, UK"
gdf = ox.geocode_to_gdf(place)
polygon = gdf.geometry.unary_union
if os.path.exists("brighton_graph.graphml"):
    G = ox.load_graphml("brighton_graph.graphml")
else:
    G = ox.graph_from_polygon(polygon, network_type='drive')
    ox.save_graphml(G, "brighton_graph.graphml")
print("Road network loaded.")


def get_road_route(start_lat, start_lon, end_lat, end_lon):
    """
        Calculates the shortest road path between two geographic points using the OSMnx graph.
        Returns:
            list: A list of (lat, lon) tuples representing the route.
        """
    try:
        orig = ox.nearest_nodes(G, start_lon, start_lat)
        dest = ox.nearest_nodes(G, end_lon, end_lat)
        route = nx.shortest_path(G, orig, dest, weight='length')
        return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
    except Exception:
        return [(start_lat, start_lon), (end_lat, end_lon)]


#Load Data Files
stations_df = pd.read_csv("stations.csv")

# Real-world data from FOI response (adjusted for simulation scale)
REAL_WORLD_DATA = {
    # Based on July 2025 data
    'monthly_trips': {
        'e_bike': {
            'car': 2602,
            'motorcycle': 0,
            'taxi': 3826
        },
        'pedal_bike': {
            'car': 828,
            'motorcycle': 0,
            'taxi': 1218
        }
    },
    'trip_lengths': {
        'e_bike': {
            'car': 2.8,  # 7233 / 2602 trips
            'motorcycle': 0,
            'taxi': 2.8
        },
        'pedal_bike': {
            'car': 2.4,  # 1994 / 828 trips
            'motorcycle': 0,
            'taxi': 2.4
        }
    },
    'time_distribution': {
        'weekday': {
            'morning_peak': (7, 9, 0.35),
            'evening_peak': (16, 18, 0.35),
            'off_peak': (9, 16, 0.2),
            'night': (18, 7, 0.1)
        },
        'weekend': {
            'daytime': (10, 18, 0.7),
            'evening': (18, 22, 0.2),
            'night': (22, 10, 0.1)
        }
    }
}


def distribute_bikes_realistically(stations_df):
    """Distributes an initial number of bikes to each station at the start of a simulation day."""
    central_stations = stations_df[
        (stations_df['lat'].between(50.82, 50.83)) &
        (stations_df['lon'].between(-0.15, -0.13))
        ].copy()

    # Assign more bikes to central stations
    central_stations['pedal_bikes'] = [min(6, random.randint(3, 8)) for _ in range(len(central_stations))]
    central_stations['e_bikes'] = [min(6, random.randint(3, 8)) for _ in range(len(central_stations))]

    # Remaining bikes to other stations
    other_stations = stations_df[~stations_df.index.isin(central_stations.index)].copy()
    other_stations['pedal_bikes'] = [min(3, random.randint(0, 4)) for _ in range(len(other_stations))]
    other_stations['e_bikes'] = [min(3, random.randint(0, 4)) for _ in range(len(other_stations))]

    return pd.concat([central_stations, other_stations]).sample(frac=1).reset_index(drop=True)


stations_df = distribute_bikes_realistically(stations_df)

#Simulation State
class SimulationState:
    """A simple class to hold the global state of the simulation."""
    def __init__(self):
        self.running = False
        self.current_date = datetime.now().date()
        self.model = None
        self.speed_factor = 1
        self.last_update_time = datetime.now()


sim_state = SimulationState()


def generate_realistic_trips(stations_df, date_str):
    """
    Generate trips for a single day by converting monthly totals from FOI data
    into a weighted daily estimate.
    """
    # Weighted Daily Calculation Parameters
    # Based on July 2025, which has 23 weekdays and 8 weekend days.
    WEEKEND_WEIGHTING = 1.5
    WEEKDAYS_IN_MONTH = 23
    WEEKEND_DAYS_IN_MONTH = 8
    EFFECTIVE_DAYS_IN_MONTH = (WEEKDAYS_IN_MONTH) + (WEEKEND_DAYS_IN_MONTH * WEEKEND_WEIGHTING)

    date = datetime.strptime(date_str, '%Y-%m-%d')
    is_weekend = date.weekday() >= 5

    # Determine the number of trips for THIS specific day
    daily_trips_to_generate = defaultdict(int)

    for bike_type in ['e_bike', 'pedal_bike']:
        for replaced_mode in ['car', 'motorcycle', 'taxi']:
            monthly_total = REAL_WORLD_DATA['monthly_trips'][bike_type][replaced_mode]
            if monthly_total == 0:
                continue
            daily_base_rate = monthly_total / EFFECTIVE_DAYS_IN_MONTH
            if is_weekend:
                num_trips_today = round(daily_base_rate * WEEKEND_WEIGHTING)
            else:
                num_trips_today = round(daily_base_rate)
            daily_trips_to_generate[(bike_type, replaced_mode)] = num_trips_today

    # Generate the Trip Data
    trips = []
    station_coords = stations_df[['lat', 'lon']].values
    station_ids = stations_df['station_id'].values
    bike_counts = stations_df[['station_id', 'pedal_bikes', 'e_bikes']].copy()

    time_dist = REAL_WORLD_DATA['time_distribution']['weekend' if is_weekend else 'weekday']
    time_slots, time_weights = zip(*[(k, v[2]) for k, v in time_dist.items()])

    for (bike_type, replaced_mode), num_trips in daily_trips_to_generate.items():
        for _ in range(num_trips):
            if bike_type == 'e_bike':
                available_stations = bike_counts[bike_counts['e_bikes'] > 0]
                if len(available_stations) < 2: continue
                start_idx = random.choice(available_stations.index)
                bike_counts.loc[start_idx, 'e_bikes'] -= 1
                avg_speed_kmh = 20
            else:
                available_stations = bike_counts[bike_counts['pedal_bikes'] > 0]
                if len(available_stations) < 2: continue
                start_idx = random.choice(available_stations.index)
                bike_counts.loc[start_idx, 'pedal_bikes'] -= 1
                avg_speed_kmh = 15

            end_idx = random.choice(available_stations[available_stations.index != start_idx].index)
            slot_name = random.choices(time_slots, weights=time_weights)[0]


            start_hour_range = time_dist[slot_name][:2]
            start, end = start_hour_range

            if start > end:
                # Handle time ranges that cross midnight (e.g., 18:00 to 07:00)
                possible_hours = list(range(start, 24)) + list(range(0, end))
                start_hour = random.choice(possible_hours)
            else:
                # Handle normal time ranges
                start_hour = random.randint(start, end - 1 if end > start else start)


            start_minute = random.randint(0, 59)
            start_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=start_hour, minutes=start_minute)
            distance_km = random.gauss(REAL_WORLD_DATA['trip_lengths'][bike_type][replaced_mode], 0.3)
            duration_min = (distance_km / avg_speed_kmh) * 60
            end_time = start_time + timedelta(minutes=duration_min)

            trips.append({
                'trip_id': len(trips) + 1,
                'vehicle_type': 'e-bike' if bike_type == 'e_bike' else 'pedal-bike',
                'start_time': start_time,
                'end_time': end_time,
                'start_lat': stations_df.loc[start_idx, 'lat'],
                'start_lon': stations_df.loc[start_idx, 'lon'],
                'end_lat': stations_df.loc[end_idx, 'lat'],
                'end_lon': stations_df.loc[end_idx, 'lon'],
                'duration_min': duration_min,
                'distance_km': distance_km,
                'is_weekend': is_weekend,
                'returned_to_bay': random.random() < 0.98,
                'in_zone': random.random() < 0.99,
                'properly_locked': random.random() < 0.99,
                'start_station_id': stations_df.loc[start_idx, 'station_id'],
                'end_station_id': stations_df.loc[end_idx, 'station_id'],
                'replaced_mode': replaced_mode
            })

    return pd.DataFrame(trips)

def save_simulation_data(model, date_str):
    """
        Compiles the results from a completed simulation run and saves them to a multi-sheet Excel file.
        Falls back to saving individual CSV files if there's an error with Excel.
        """
    filename = f"simulation_{date_str}.xlsx"
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Prepare DataFrames for each sheet
    trips_df = pd.DataFrame([{
        'trip_id': trip.trip_id,
        'vehicle_type': trip.vehicle_type,
        'start_time': trip.start_time,
        'end_time': trip.end_time,
        'start_station': trip.start_station_id,
        'end_station': trip.end_station_id,
        'duration_min': (trip.end_time - trip.start_time).total_seconds() / 60,
        'distance_km': trip.distance_km,
        'co2_saved_kg': trip.co2_saved_kg,
        'is_weekend': trip.is_weekend
    } for trip in model.completed_agents])

    accidents_df = pd.DataFrame([{
        'latitude': acc.latitude,
        'longitude': acc.longitude,
        'severity': acc.severity,
        'time': acc.time,
        'vehicle_type': acc.vehicle_type
    } for acc in model.accident_events])

    maintenance_df = pd.DataFrame([{
        'bike_id': me.bike_id,
        'vehicle_type': me.vehicle_type,
        'start_time': me.start_time,
        'end_time': me.end_time,
        'downtime_hours': me.downtime.total_seconds() / 3600,
        'cost': me.cost
    } for me in model.maintenance_events])

    metrics_df = pd.DataFrame([model.get_daily_summary()])

    try:
        trips_df.to_excel(writer, sheet_name='Trips', index=False)
        accidents_df.to_excel(writer, sheet_name='Accidents', index=False)
        maintenance_df.to_excel(writer, sheet_name='Maintenance', index=False)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        writer.close()
        print(f"Simulation data saved to {filename}")
    except Exception as e:
        print(f"Error saving Excel file: {str(e)}")
        trips_df.to_csv(f"simulation_{date_str}_trips.csv", index=False)
        accidents_df.to_csv(f"simulation_{date_str}_accidents.csv", index=False)
        maintenance_df.to_csv(f"simulation_{date_str}_maintenance.csv", index=False)
        metrics_df.to_csv(f"simulation_{date_str}_metrics.csv", index=False)



#BikeAgent class
class BikeAgent:
    """Represents a single bike trip from a start to an end station."""
    def __init__(self, trip):
        self.trip_id = trip['trip_id']
        self.vehicle_type = trip['vehicle_type']
        self.start_time = trip['start_time']
        self.end_time = trip['end_time']
        self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.elapsed_time = 0
        self.in_maintenance = False
        self.route = get_road_route(trip['start_lat'], trip['start_lon'], trip['end_lat'], trip['end_lon'])
        self.route_index = 0
        self.lat, self.lon = self.route[0]
        self.accident_prone = random.random() < (0.07 if self.vehicle_type == 'e-bike' else 0.04)
        self.is_weekend = trip['is_weekend']
        self.returned_to_bay = trip['returned_to_bay']
        self.in_zone = trip['in_zone']
        self.properly_locked = trip['properly_locked']
        self.penalties_applied = False
        self.start_station_id = trip['start_station_id']
        self.end_station_id = trip['end_station_id']
        self.completed = False
        self.distance_km = trip['distance_km']
        self.last_update_time = trip['start_time']
        self.speed_factor = 1.0
        self.in_accident = False
        self.replaced_mode = trip['replaced_mode']

        # CO2 savings based on replaced mode
        co2_factors = {
            'car': 0.21,  # kg CO2 per km
            'motorcycle': 0.12,
            'taxi': 0.25
        }
        self.base_co2_saved_kg = self.distance_km * co2_factors[self.replaced_mode]

        if self.vehicle_type == 'pedal-bike':
            self.co2_saved_kg = self.base_co2_saved_kg * 0.8373  # 83.73% replacement
        else:  # e-bike
            self.co2_saved_kg = self.base_co2_saved_kg * 0.5573  # 55.73% replacement

    def step(self, current_time, speed_factor=1.0):
        if self.in_maintenance or self.in_accident or self.route_index >= len(self.route) - 1:
            return

        # Calculate progress based on time elapsed vs total duration
        elapsed = (current_time - self.start_time).total_seconds()
        progress = min(elapsed / self.total_duration, 1.0)

        # Find the two nearest points in the route
        exact_index = progress * (len(self.route) - 1)
        idx1 = int(math.floor(exact_index))
        idx2 = min(idx1 + 1, len(self.route) - 1)

        # Linear interpolation between points
        fraction = exact_index - idx1
        lat1, lon1 = self.route[idx1]
        lat2, lon2 = self.route[idx2]

        self.lat = lat1 + fraction * (lat2 - lat1)
        self.lon = lon1 + fraction * (lon2 - lon1)

        # Update route index for visualization purposes
        self.route_index = idx1

    def get_distance_in_step(self, sim_elapsed):
        """Calculates the distance traveled in a single time step."""
        if self.total_duration > 0:
            fraction_of_trip = sim_elapsed.total_seconds() / self.total_duration
            return self.distance_km * fraction_of_trip
        return 0

class MaintenanceEvent:
    """Represents a period of time when a bike is out of service for maintenance."""
    def __init__(self, bike_agent, start_time, downtime_hours):
        self.bike_id = bike_agent.trip_id
        self.vehicle_type = bike_agent.vehicle_type
        self.start_time = start_time
        self.downtime = timedelta(hours=downtime_hours)
        self.end_time = start_time + self.downtime
        self.active = True
        self.agent = bike_agent
        self.agent.in_maintenance = True
        self.cost = downtime_hours * (20 if bike_agent.vehicle_type == 'e-bike' else 10)

    def step(self, current_time):
        if current_time >= self.end_time and self.active:
            self.agent.in_maintenance = False
            self.agent.in_accident = False
            self.active = False
            return True
        return False


class AccidentEvent:
    """Represents an accident event at a specific location and time."""
    def __init__(self, location_lat, location_lon, severity, time, vehicle_type):
        self.latitude = location_lat
        self.longitude = location_lon
        self.severity = severity
        self.time = time
        self.vehicle_type = vehicle_type
        self.time_since_creation = 0
        self.pulse_duration = 60
        self.active = True
        self.occurred = True

    def step(self):
        self.time_since_creation += 1
        return False


#Simulation Model
class BikeSharingModel:
    def __init__(self, trips_df):
        today = datetime.now().date()
        self.current_time = datetime.combine(today, datetime.min.time())
        self.end_time = datetime.combine(today, datetime.max.time()) - timedelta(minutes=1)
        self.trips_df = trips_df
        self.simulation_date = today
        self.active_agents = []
        self.completed_agents = []
        self.maintenance_events = []
        self.accident_events = []

        #Financial Parameters
        self.capital_investment_total = 13_300_000  # £13.3 million total contract
        self.initial_capex = 8_000_000  # £8 million initial investment (assumption)
        self.contract_years = 4  # 4 year contract
        self.daily_capital_cost = (((self.capital_investment_total - self.initial_capex)/4)/365)
        self.city_surplus_share_ratio = 0.3  # 30% of operator surplus goes to city

        # Operator metrics
        self.co2_saved_kg = 0
        self.co2_price_kg = 0.260  # DEFRA 2025 (£/kg)
        self.co2_monetary_value = 0
        self.accident_costs = 0
        self.maintenance_costs = 0
        self.revenue = 0
        self.penalty_revenue = 0
        self.trips_completed = 0
        self.total_maintenance_events = 0
        self.active_maintenance_count = 0
        self.completed_maintenance_count = 0

        # City perspective metrics
        self.health_benefit_pedal_km = 0.15  # GBP/km
        self.health_benefit_ebike_km = 0.12  # GBP/km
        self.car_replacement_rate = 0.3468  # 34.68% of trips replace car trips (beryl report 2023)
        self.car_congestion_value = 0.15  # GBP/km saved
        self.minor_accidents = 0
        self.serious_accidents = 0
        self.nhs_minor_cost = 500  # GBP per minor accident
        self.nhs_serious_cost = 8000  # GBP per serious accident

        # Tracking variables
        self.pedal_km = 0
        self.ebike_km = 0
        self.car_trips_replaced = 0

        self.last_update_time = datetime.now()
        self.speed_factor = 1
        self.notifications = []
        self.notifications_to_remove = []

    def step(self, speed_factor=1.0):
        now = datetime.now()
        real_elapsed = (now - self.last_update_time).total_seconds()
        self.last_update_time = now
        speed_multipliers = {1: 1, 2: 30, 3: 60}
        sim_elapsed = timedelta(seconds=real_elapsed * speed_multipliers.get(speed_factor, 1))
        self.current_time += sim_elapsed

        # Process starting trips
        starting_trips = self.trips_df[
            (self.trips_df['start_time'] <= self.current_time) &
            (self.trips_df['end_time'] > self.current_time)
            ]

        for _, trip in starting_trips.iterrows():
            if trip['trip_id'] not in [a.trip_id for a in self.active_agents + self.completed_agents]:
                in_maint = any(m.agent.trip_id == trip['trip_id'] for m in self.maintenance_events if m.active)
                if not in_maint:
                    self.active_agents.append(BikeAgent(trip))

        # Process maintenance events
        for me in self.maintenance_events[:]:
            if me.step(self.current_time):  # If maintenance is over...
                # The agent completed its trip THEN went into maintenance.
                if me.agent.completed:
                    me.agent.in_maintenance = False
                    # Return bike to its intended destination station
                    end_station_idx = stations_df[stations_df['station_id'] == me.agent.end_station_id].index[0]
                    if me.agent.vehicle_type == 'pedal-bike':
                        stations_df.loc[end_station_idx, 'pedal_bikes'] += 1
                    else:
                        stations_df.loc[end_station_idx, 'e_bikes'] += 1
                #The agent had an accident and did NOT complete its trip.
                else:
                    me.agent.in_maintenance = False
                    me.agent.in_accident = False  # Reset the accident flag
                    # Return bike to the station where its trip started
                    start_station_idx = stations_df[stations_df['station_id'] == me.agent.start_station_id].index[0]
                    if me.agent.vehicle_type == 'pedal-bike':
                        stations_df.loc[start_station_idx, 'pedal_bikes'] += 1
                    else:
                        stations_df.loc[start_station_idx, 'e_bikes'] += 1


                self.active_maintenance_count -= 1
                self.completed_maintenance_count += 1


        # Process active agents
        for agent in self.active_agents[:]:
            if agent.in_maintenance:
                continue

            # Track distance by bike type
            if agent.vehicle_type == 'pedal-bike':
                self.pedal_km += agent.distance_km * (sim_elapsed.total_seconds() / agent.total_duration)
            else:
                self.ebike_km += agent.distance_km * (sim_elapsed.total_seconds() / agent.total_duration)

            agent.step(self.current_time, speed_factor)

            if agent.route_index >= len(agent.route) - 1 and not agent.completed:
                agent.completed = True
                self.trips_completed += 1
                self.car_trips_replaced += agent.distance_km * self.car_replacement_rate

                # Calculate trip revenue
                unlock_fee = 1.00
                trip_duration = (agent.end_time - agent.start_time).total_seconds() / 60
                if agent.vehicle_type == 'e-bike':
                    minute_rate = 0.14 if not agent.is_weekend else 0.18
                else:
                    minute_rate = 0.06 if not agent.is_weekend else 0.08

                trip_cost = unlock_fee + (minute_rate * trip_duration)
                self.revenue += trip_cost

                # Apply penalties if not already applied
                if not agent.penalties_applied:
                    if not agent.returned_to_bay:
                        self.penalty_revenue += 10
                    if not agent.in_zone:
                        self.penalty_revenue += 25
                    if not agent.properly_locked:
                        self.penalty_revenue += 10
                    agent.penalties_applied = True

                # Handle maintenance after trip completion
                if random.random() < 0.02:
                    downtime = random.uniform(0.5, 3)
                    maintenance = MaintenanceEvent(agent, self.current_time, downtime)
                    self.maintenance_events.append(maintenance)
                    self.maintenance_costs += maintenance.cost
                    self.total_maintenance_events += 1
                    self.active_maintenance_count += 1
                else:
                    # Return bike to station if no maintenance needed
                    end_station_idx = stations_df[stations_df['station_id'] == agent.end_station_id].index[0]
                    if agent.vehicle_type == 'pedal-bike':
                        stations_df.loc[end_station_idx, 'pedal_bikes'] += 1
                    else:
                        stations_df.loc[end_station_idx, 'e_bikes'] += 1

                self.co2_saved_kg += agent.co2_saved_kg
                self.completed_agents.append(agent)
                self.active_agents.remove(agent)

        # Check for accidents
        ACCIDENT_PROB_PER_KM = 1 / 9753

        for agent in self.active_agents[:]:
            if agent.in_maintenance or agent.in_accident:
                continue

            # Calculate distance traveled in this specific step
            distance_this_step = agent.get_distance_in_step(sim_elapsed)

            # The probability of an accident is now based on distance covered
            accident_probability = distance_this_step * ACCIDENT_PROB_PER_KM

            # E-bikes are still slightly more prone
            if agent.vehicle_type == 'e-bike':
                accident_probability *= 1.5

            if random.random() < accident_probability:
                severity = random.choices(['serious', 'minor'], weights=[0.1, 0.9])[0]
                accident = AccidentEvent(agent.lat, agent.lon, severity, self.current_time, agent.vehicle_type)
                self.accident_events.append(accident)

                if severity == 'serious':
                    self.accident_costs += 500
                    self.serious_accidents += 1
                else:
                    self.accident_costs += 100
                    self.minor_accidents += 1

                agent.in_accident = True
                downtime = random.uniform(1.0, 6.0)
                maintenance = MaintenanceEvent(agent, self.current_time, downtime)
                self.maintenance_events.append(maintenance)
                self.maintenance_costs += maintenance.cost
                self.total_maintenance_events += 1
                self.active_maintenance_count += 1

                self.notifications.append({
                    'id': accident.time.timestamp(),
                    'message': f"New {severity} accident with {agent.vehicle_type}!",
                    'time_since_creation': 0,
                    'severity': severity,
                    'vehicle_type': agent.vehicle_type,
                    'time': accident.time.strftime('%H:%M:%S')
                })

        # Update notifications
        active_notifications = []
        for note in self.notifications:
            note['time_since_creation'] += 1
            if note['time_since_creation'] <= 60:
                active_notifications.append(note)
            else:
                self.notifications_to_remove.append(note['id'])
        self.notifications = active_notifications

        # Update CO₂ monetary value
        self.co2_monetary_value = self.co2_saved_kg * self.co2_price_kg

    def get_daily_summary(self):
        # Operator calculations
        operator_profit = (self.revenue + self.penalty_revenue + self.co2_monetary_value) - \
                          (self.maintenance_costs + self.accident_costs)

        # City's share of operator surplus (30% only if profitable)
        city_surplus_share = operator_profit * self.city_surplus_share_ratio if operator_profit > 0 else 0

        operator_profit = operator_profit-city_surplus_share

        # City perspective calculations
        city_co2_value = self.co2_saved_kg * self.co2_price_kg
        health_benefit = (self.pedal_km * self.health_benefit_pedal_km) + (self.ebike_km * self.health_benefit_ebike_km)
        congestion_reduction = self.car_trips_replaced * self.car_congestion_value
        accident_health_costs = (self.minor_accidents * self.nhs_minor_cost +
                                 self.serious_accidents * self.nhs_serious_cost)

        total_city_benefits = (city_surplus_share +
                               city_co2_value +
                               health_benefit +
                               congestion_reduction)

        net_city_benefit = total_city_benefits - self.daily_capital_cost

        return {
            # Operator metrics
            'date': self.simulation_date.strftime('%Y-%m-%d'),
            'trips_completed': self.trips_completed,
            'revenue': self.revenue,
            'penalty_revenue': self.penalty_revenue,
            'co2_saved_kg': self.co2_saved_kg,
            'co2_monetary_value': self.co2_monetary_value,
            'total_revenue': self.revenue + self.penalty_revenue + self.co2_monetary_value,
            'operator_profit': operator_profit,
            'active_maintenance': self.active_maintenance_count,
            'completed_maintenance': self.completed_maintenance_count,
            'total_maintenance': self.total_maintenance_events,
            'maintenance_costs': self.maintenance_costs,
            'accident_costs': self.accident_costs,
            'minor_accidents': self.minor_accidents,
            'serious_accidents': self.serious_accidents,

            # City perspective metrics
            'city_surplus_share': city_surplus_share,
            'city_co2_value': city_co2_value,
            'health_benefit': health_benefit,
            'congestion_reduction': congestion_reduction,
            'total_city_benefits': total_city_benefits,
            'accident_health_costs': accident_health_costs,
            'daily_capital_cost': self.daily_capital_cost,
            'net_city_benefit': net_city_benefit,
            'pedal_km': self.pedal_km,
            'ebike_km': self.ebike_km,
            'car_trips_replaced': self.car_trips_replaced,

            # Contract information
            'capital_investment_total': self.capital_investment_total,
            'contract_years': self.contract_years,
            'city_surplus_share_ratio': self.city_surplus_share_ratio
        }

    def is_day_complete(self):
        return self.current_time >= self.end_time and len(self.active_agents) == 0


# App Layout
app.layout = html.Div([
    html.H1("Brighton & Hove Pedal-Bike & E-Bike Sharing Simulation",
            style={'textAlign': 'center', 'margin': '10px 0 20px 0'}),

    html.Div([
        html.Button('Start Simulation', id='start-button', n_clicks=0,
                    style={'marginRight': '10px', 'padding': '10px 20px'}),
        html.Button('Stop Simulation', id='stop-button', n_clicks=0,
                    style={'marginRight': '10px', 'padding': '10px 20px', 'backgroundColor': '#ff4444', 'color': 'white'}),
        html.Button('Shutdown Server', id='shutdown-button', n_clicks=0,
                    style={'padding': '10px 20px', 'backgroundColor': '#666666', 'color': 'white'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div([
        html.Label("Simulation Speed:", style={'marginRight': '10px'}),
        dcc.Slider(
            id='speed-slider',
            min=1,
            max=3,
            step=1,
            value=1,
            marks={
                1: '1x (real-time)',
                2: '2x (30×)',
                3: '3x (60×)'
            },
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '300px', 'margin': '20px auto'}),

    html.Div(id='simulation-date', style={
        'textAlign': 'center',
        'fontSize': '18px',
        'fontWeight': 'bold',
        'marginBottom': '10px'
    }),

    html.Div(id='accident-notification', style={
        'position': 'fixed',
        'top': '100px',
        'right': '20px',
        'zIndex': '1000',
        'maxWidth': '300px'
    }),

    dcc.Interval(id='interval', interval=100, n_intervals=0, disabled=True),

    html.Div([
        html.Div([
            html.H3("Daily Metrics", style={'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div(id="time-div", style={'padding': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),

            html.H4("Operator Perspective", style={'marginTop': '20px'}),
            html.Div(id="financial-div", style={'padding': '10px', 'fontSize': '16px'}),
            html.Div(id="co2-div", style={'padding': '10px', 'fontSize': '16px'}),
            html.Div(id="accident-div", style={'padding': '10px', 'fontSize': '16px', 'color': 'darkred'}),
            html.Div(id="maintenance-div", style={'padding': '10px', 'fontSize': '16px', 'color': 'darkblue'}),
            html.Div(id="operator_total_text-div", style={'padding': '10px', 'fontSize': '16px'}),

            html.H4("City Perspective", style={'marginTop': '20px'}),
            html.Div(id="city-benefits-div", style={'padding': '10px', 'fontSize': '16px', 'color': 'darkgreen'}),
            html.Div(id="city-costs-div", style={'padding': '10px', 'fontSize': '16px', 'color': 'darkred'}),

            html.H4("Annual Projection", style={'textAlign': 'center', 'margin': '15px 0 10px 0'}),
            html.Div(id="annual-div", style={'padding': '10px', 'fontSize': '14px'})
        ], style={
            'width': '30%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '15px',
            'height': '80vh',
            'overflowY': 'auto',
            'boxShadow': '0 0 10px rgba(0,0,0,0.1)',
            'borderRadius': '5px',
            'marginRight': '10px'
        }),

        html.Div([
            dl.Map(
                id="map",
                center=(50.8225, -0.1372),
                zoom=14,
                style={
                    'height': '80vh',
                    'width': '100%',
                    'margin': '0',
                    'padding': '0',
                    'borderRadius': '5px'
                },
                children=[
                    dl.TileLayer(),
                    dl.FeatureGroup(id="stations-layer"),
                    dl.FeatureGroup(id="bikes-layer"),
                    dl.FeatureGroup(id="accidents-layer")
                ]
            )
        ], style={
            'width': '68%',
            'display': 'inline-block',
            'padding': '0',
            'margin': '0'
        })
    ], style={
        'display': 'flex',
        'margin': '0 auto',
        'maxWidth': '95%',
        'padding': '10px'
    }),

    html.Div(id="daily-summary", style={
        'width': '95%',
        'margin': '20px auto',
        'padding': '15px',
        'borderTop': '1px solid #ddd'
    })
])


# Update control_simulation callback
@app.callback(
    [Output('interval', 'disabled'),
     Output('simulation-date', 'children')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    prevent_initial_call=True
)
def control_simulation(start_clicks, stop_clicks):
    global stations_df, sim_state

    # Ensure we have a valid SimulationState instance
    if not hasattr(sim_state, 'running'):
        sim_state = SimulationState()

    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if button_id == 'start-button':
            # Initialize simulation
            sim_state.running = True
            sim_state.current_date = datetime.now().date()
            sim_state.last_update_time = datetime.now()

            # Ensure stations_df exists and is properly formatted
            if not isinstance(stations_df, pd.DataFrame):
                stations_df = pd.read_csv("stations.csv")

            stations_df = distribute_bikes_realistically(stations_df)
            trips_df = generate_realistic_trips(stations_df, sim_state.current_date.strftime('%Y-%m-%d'))
            sim_state.model = BikeSharingModel(trips_df)

            return False, f"Simulation Date: {sim_state.current_date.strftime('%Y-%m-%d')}"

        elif button_id == 'stop-button':
            # Stop simulation
            sim_state.running = False
            sim_state.model = None

            # Reset stations
            if not isinstance(stations_df, pd.DataFrame):
                stations_df = pd.read_csv("stations.csv")
            stations_df = distribute_bikes_realistically(stations_df)

            return True, f"Simulation Reset (Date: {datetime.now().date().strftime('%Y-%m-%d')})"

        return no_update, no_update

    except Exception as e:
        print(f"Error in control_simulation: {str(e)}")
        return True, f"Error: {str(e)}"

@app.callback(
    Output('interval', 'interval'),
    Input('speed-slider', 'value')
)
def update_interval(speed):
    return 100


@app.callback(
    Output('simulation-date', 'children', allow_duplicate=True),
    Input('shutdown-button', 'n_clicks'),
    prevent_initial_call=True
)
def shutdown_server(n_clicks):
    """
    Shuts down the server gracefully after a 1-second delay
    to allow a response to be sent to the browser.
    """
    print("Server shutdown initiated...")

    def do_exit():
        time.sleep(1)  # Wait one second before shutting down
        os._exit(0)    # Terminate the process

    # Start the shutdown function in a new thread
    threading.Thread(target=do_exit).start()

    # Immediately return a success message to the browser
    return "Server shutdown"

@app.callback(
    [Output("stations-layer", "children"),
     Output("bikes-layer", "children"),
     Output("accidents-layer", "children"),
     Output("time-div", "children"),
     Output("financial-div", "children"),
     Output("co2-div", "children"),
     Output("accident-div", "children"),
     Output("maintenance-div", "children"),
     Output("operator_total_text-div", "children"),
     Output("city-benefits-div", "children"),
     Output("city-costs-div", "children"),
     Output("annual-div", "children"),
     Output("daily-summary", "children"),
     Output('simulation-date', 'children', allow_duplicate=True),
     Output('interval', 'disabled', allow_duplicate=True),
     Output('accident-notification', 'children')],
    [Input("interval", "n_intervals")],
    [State('start-button', 'n_clicks'),
     State('stop-button', 'n_clicks'),
     State('speed-slider', 'value')],
    prevent_initial_call=True
)
def update_map(n, start_clicks, stop_clicks, speed_factor):
    default_return = [no_update] * 16

    if not sim_state.running or not sim_state.model:
        return default_return

    try:
        model = sim_state.model

        if not model.is_day_complete():
            model.step(speed_factor)

            # Stations markers
            stations_markers = []
            for _, row in stations_df.iterrows():
                station_popup = html.Div([
                    html.H4(f"Station {row['station_id']}"),
                    html.P(f"Pedal Bikes: {row['pedal_bikes']}"),
                    html.P(f"E-Bikes: {row['e_bikes']}")
                ])
                stations_markers.append(dl.Marker(
                    position=(row['lat'], row['lon']),
                    children=[dl.Tooltip(station_popup)],
                    icon={
                        "iconUrl": app.get_asset_url("parking.png"),
                        "iconSize": [30, 30],
                        "iconAnchor": [15, 15]
                    }
                ))

            # Bike markers
            bike_markers = []
            for agent in model.active_agents:
                if agent.in_accident:
                    icon_url = app.get_asset_url("explosion.png")
                elif agent.vehicle_type == 'pedal-bike':
                    icon_url = app.get_asset_url("pedal_new.png")
                else:
                    icon_url = app.get_asset_url("ebike_new.png")

                bike_markers.append(dl.Marker(
                    position=[agent.lat, agent.lon],
                    children=dl.Tooltip(f"{agent.vehicle_type} #{agent.trip_id}"),
                    icon={
                        "iconUrl": icon_url,
                        "iconSize": [20, 20],
                        "iconAnchor": [15, 15]
                    }
                ))

            # Accident markers
            accident_markers = []
            for acc in model.accident_events:
                if acc.active:
                    if acc.time_since_creation <= acc.pulse_duration:
                        pulse_radius = 15 + 5 * math.sin(acc.time_since_creation * 0.5)
                        accident_markers.append(dl.CircleMarker(
                            center=[acc.latitude, acc.longitude],
                            radius=pulse_radius,
                            color='red' if acc.severity == 'serious' else 'orange',
                            fill=True,
                            fillOpacity=0.3,
                            stroke=False
                        ))

                    accident_markers.append(dl.CircleMarker(
                        center=[acc.latitude, acc.longitude],
                        radius=10,
                        color='red' if acc.severity == 'serious' else 'orange',
                        fill=True,
                        fillOpacity=0.7,
                        children=dl.Tooltip(
                            f"{acc.vehicle_type} accident ({acc.severity}) at {acc.time.strftime('%H:%M')}")
                    ))

                    accident_markers.append(dl.Marker(
                        position=[acc.latitude, acc.longitude],
                        icon={
                            "iconUrl": app.get_asset_url("explosion.png"),
                            "iconSize": [30, 30],
                            "iconAnchor": [15, 15]
                        }
                    ))

            # Metrics calculations
            daily_summary = model.get_daily_summary()

            time_text = f"Current Time: {model.current_time.strftime('%H:%M:%S')}"

            financial_text = [
                html.P(f"Trip Revenue: £{daily_summary['revenue']:.2f}"),
                html.P(f"Penalty Revenue: £{daily_summary['penalty_revenue']:.2f}")
            ]

            co2_text = [
                html.P(f"Total CO₂ Saved Today: {daily_summary['co2_saved_kg']:.2f} kg"),
                html.P(f"CO₂ Savings Value: £{daily_summary['co2_monetary_value']:.2f}")
            ]

            accident_text = [
                html.P(f"Accidents Today: {daily_summary['minor_accidents'] + daily_summary['serious_accidents']}"),
                html.P(f"Operator Accident Costs: £{daily_summary['accident_costs']:.2f}")
            ]

            maintenance_text = [
                html.P(f"Active Maintenance: {daily_summary['active_maintenance']}"),
                html.P(f"Completed Maintenance: {daily_summary['completed_maintenance']}"),
                html.P(f"Total Maintenance Events: {daily_summary['total_maintenance']}"),
                html.P(f"Maintenance Costs: £{daily_summary['maintenance_costs']:.2f}")
            ]

            operator_total_text = [
                html.P(f"Total Revenue: £{daily_summary['total_revenue']:.2f}"),
                html.P(f"Operator Estimated Profit: £{daily_summary['operator_profit']:.2f}")
            ]

            city_benefits_text = [
                html.P(f"City Share of Surplus: £{daily_summary['city_surplus_share']:.2f}"),
                html.P(f"CO₂ Value: £{daily_summary['city_co2_value']:.2f}"),
                html.P(f"Health Benefits: £{daily_summary['health_benefit']:.2f}"),
                html.P(f"Congestion Reduction: £{daily_summary['congestion_reduction']:.2f}"),
                html.P(f"Total City Benefits: £{daily_summary['total_city_benefits']:.2f}")
            ]

            city_costs_text = [
                html.P(f"Daily Capital Cost: £{daily_summary['daily_capital_cost']:.2f}"),
                html.P(f"Net City Benefit: £{daily_summary['net_city_benefit']:.2f}")
            ]

            annual_text = [
                html.H5("Operator Annual:"),
                html.P(f"Revenue: £{daily_summary['revenue'] * 365:,.2f}"),
                html.P(f"Profit: £{daily_summary['operator_profit'] * 365:,.2f}"),

                html.H5("City Annual:"),
                html.P(f"Surplus Share: £{daily_summary['city_surplus_share'] * 365:,.2f}"),
                html.P(f"CO₂ Value: £{daily_summary['city_co2_value'] * 365:,.2f}"),
                html.P(f"Net City Benefit: £{daily_summary['net_city_benefit'] * 365:,.2f}"),

                html.H5("Contract Information:"),
                html.P(f"Total Capital Investment: £{daily_summary['capital_investment_total']:,.2f}"),
                html.P(f"Contract Years: {daily_summary['contract_years']}"),
                html.P(f"City Surplus Share Ratio: {daily_summary['city_surplus_share_ratio'] * 100:.0f}%")
            ]

            summary_table = html.Div([
                html.H4("Daily Summary"),
                html.Table([
                    html.Tbody([
                        html.Tr([html.Th("Metric"), html.Th("Value")]),
                        html.Tr([html.Td("Trips Completed"), html.Td(daily_summary['trips_completed'])]),
                        html.Tr([html.Td("Pedal Bike KM"), html.Td(f"{daily_summary['pedal_km']:.2f}")]),
                        html.Tr([html.Td("E-Bike KM"), html.Td(f"{daily_summary['ebike_km']:.2f}")]),
                        html.Tr([html.Td("CO₂ Saved (kg)"), html.Td(f"{daily_summary['co2_saved_kg']:.2f}")]),
                        html.Tr([html.Td("Operator Profit"), html.Td(f"£{daily_summary['operator_profit']:.2f}")]),
                        html.Tr(
                            [html.Td("City Surplus Share"), html.Td(f"£{daily_summary['city_surplus_share']:.2f}")]),
                        html.Tr([html.Td("Net City Benefit"), html.Td(f"£{daily_summary['net_city_benefit']:.2f}")])
                    ], style={'width': '100%', 'borderCollapse': 'collapse'})
                ])
            ])

            notification_components = []
            for note in model.notifications:
                if note['id'] not in model.notifications_to_remove:
                    notification_components.append(html.Div(
                        f"{note['message']} ({note['time']})",
                        style={
                            'color': 'red' if note['severity'] == 'serious' else 'orange',
                            'fontWeight': 'bold',
                            'margin': '5px',
                            'padding': '5px',
                            'border': '1px solid red' if note['severity'] == 'serious' else '1px solid orange',
                            'borderRadius': '5px',
                            'backgroundColor': '#ffeeee' if note['severity'] == 'serious' else '#fff7e6',
                            'transition': 'opacity 1s ease-out'
                        },
                        id={'type': 'accident-notification', 'index': note['id']}
                    ))

            model.notifications_to_remove = []

            return (
                stations_markers,
                bike_markers,
                accident_markers,
                time_text,
                financial_text,
                co2_text,
                accident_text,
                maintenance_text,
                operator_total_text,
                city_benefits_text,
                city_costs_text,
                annual_text,
                summary_table,
                f"Simulation Date: {sim_state.current_date.strftime('%Y-%m-%d')}",
                False,
                notification_components if notification_components else []
            )
        else:
            if not sim_state.running:
                return [no_update] * 16  # Exit immediately if already handled

            sim_state.running = False

            # Now, safely save the data
            save_simulation_data(model, sim_state.current_date.strftime('%Y-%m-%d'))

            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                html.Div("Simulation completed for today!",
                         style={'color': 'green', 'fontWeight': 'bold'}),
                f"Simulation Completed (Date: {sim_state.current_date.strftime('%Y-%m-%d')})",
                True,
                []
            )

    except Exception as e:
        print(f"Error in callback: {str(e)}")
        return default_return


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)