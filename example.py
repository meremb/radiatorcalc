import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import math
from typing import List
from dataclasses import dataclass, field

@dataclass
class Radiator:
    radiator_nr: int
    collector: str
    power: float
    heat_loss: float
    length_circuit: float
    space_temperature: float
    delta_T: float
    mass_flow_rate: float = field(init=False, default=0.0)
    supply_temperature: float = field(init=False, default=0.0)
    return_temperature: float = field(init=False, default=0.0)
    diameter: float = field(init=False, default=0.0)
    pressure_loss: float = field(init=False, default=0.0)

    def calculate_mass_flow_rate(self):
        self.mass_flow_rate = self.heat_loss / 4180 / (self.supply_temperature - self.return_temperature) * 3600

    def calculate_supply_temperature(self, constant_c):
        self.supply_temperature = self.space_temperature + (constant_c / (constant_c - 1)) * self.delta_T

    def calculate_return_temperature(self, Q_ratio, max_supply_temperature):
        T_factor = 49.83
        exponent_radiator = 1.34
        self.return_temperature = ((Q_ratio**(1/exponent_radiator) * T_factor)**2) / (max_supply_temperature - self.space_temperature) + self.space_temperature

    def calculate_diameter(self, possible_diameters: List[int]):
        if math.isnan(self.mass_flow_rate):
            raise ValueError("The mass flow rate cannot be NaN. Check the configuration of the number of collectors.")
        diameter = 1.4641 * self.mass_flow_rate**0.4217
        acceptable_diameters = [d for d in possible_diameters if d >= diameter]
        if not acceptable_diameters:
            raise ValueError(f"Calculated diameter exceeds the maximum allowable diameter for mass flow rate: {self.mass_flow_rate}")
        self.diameter = min(acceptable_diameters, key=lambda x: abs(x - diameter))

    def calculate_pressure_loss(self):
        self.pressure_loss = self.calculate_pressure_radiator_kv(self.length_circuit, self.diameter, self.mass_flow_rate)

    @staticmethod
    def calculate_pressure_radiator_kv(length_circuit: float, diameter: float, mass_flow_rate: float) -> float:
        pressure_loss_piping = Radiator.calculate_pressure_loss_piping(diameter, length_circuit, mass_flow_rate)
        kv_radiator = 2
        pressure_loss_radiator = 97180 * (mass_flow_rate / 1000 / kv_radiator)**2
        return pressure_loss_piping + pressure_loss_radiator

    @staticmethod
    def calculate_pressure_loss_piping(diameter: float, length_circuit: float, mass_flow_rate: float) -> float:
        kv_piping = 51626 * (diameter / 1000)**2 - 417.39 * (diameter / 1000) + 1.5541
        resistance_meter = 97180 * (mass_flow_rate / 1000 / kv_piping)**2
        coefficient_local_losses = 1.3
        return resistance_meter * length_circuit * coefficient_local_losses

@dataclass
class Collector:
    collector_name: str
    circuit_length: float
    mass_flow_rate: float = field(init=False, default=0.0)
    diameter: float = field(init=False, default=0.0)
    pressure_loss: float = field(init=False, default=0.0)

    def calculate_pressure_loss(self, possible_diameters: List[int]):
        self.calculate_diameter(possible_diameters)
        self.pressure_loss = self.calculate_pressure_collector_kv(self.circuit_length, self.diameter, self.mass_flow_rate)

    def calculate_diameter(self, possible_diameters: List[int]):
        if math.isnan(self.mass_flow_rate):
            raise ValueError("The mass flow rate cannot be NaN.")
        diameter = 1.4641 * self.mass_flow_rate**0.4217
        acceptable_diameters = [d for d in possible_diameters if d >= diameter]
        if not acceptable_diameters:
            raise ValueError(f"Calculated diameter exceeds the maximum allowable diameter for mass flow rate: {self.mass_flow_rate}")
        self.diameter = min(acceptable_diameters, key=lambda x: abs(x - diameter))

    @staticmethod
    def calculate_pressure_collector_kv(length_circuit: float, diameter: float, mass_flow_rate: float) -> float:
        pressure_loss_piping = Radiator.calculate_pressure_loss_piping(diameter, length_circuit, mass_flow_rate)
        kv_collector = 14.66
        pressure_loss_boiler = 200
        pressure_loss_collector = 97180 * (mass_flow_rate / 1000 / kv_collector)**2
        return pressure_loss_piping + pressure_loss_collector + pressure_loss_boiler

def calculate_c(Q_ratio: float, delta_T: float) -> float:
    T_factor = 49.83
    exponent_radiator = 1.34
    return math.exp(delta_T / T_factor / Q_ratio**(1 / exponent_radiator))

def validate_data(df: pd.DataFrame) -> bool:
    required_columns = ['Radiator power', 'Length circuit', 'Space Temperature']
    for col in required_columns:
        if df[col].isnull().any() or (df[col] <= 0).any():
            return False
    return True

def main() -> None:
    st.title('Radiator Distribution Calculator')
    st.write('Enter the details for each radiator to calculate the total pressure loss and supply/return temperatures.')

    st.sidebar.header('Configuration')
    num_radiators = st.sidebar.number_input('Number of Radiators', min_value=1, value=3, step=1)
    num_collectors = st.sidebar.number_input('Number of Collectors', min_value=1, value=1, step=1)
    delta_T = st.sidebar.slider('Delta T (°C)', min_value=3, max_value=20, value=5, step=1)

    radiator_data = {'Radiator nr': list(range(1, num_radiators + 1)), 'Collector': [f'Collector {i + 1}' for i in range(num_collectors)], 'Radiator power': [0.0] * num_radiators, 'Calculated heat loss': [0.0] * num_radiators, 'Length circuit': [0.0] * num_radiators, 'Space Temperature': [20.0] * num_radiators}
    radiator_df = pd.DataFrame(radiator_data)

    collector_data = {'Collector': [f'Collector {i + 1}' for i in range(num_collectors)], 'Collector circuit length': [0.0] * num_collectors}
    collector_df = pd.DataFrame(collector_data)

    edited_radiator_df = st.data_editor(radiator_df, use_container_width=True, num_rows="dynamic", height=min(600, 50 + 35 * num_radiators))
    edited_collector_df = st.data_editor(collector_df, use_container_width=True, num_rows="dynamic", height=min(600, 50 + 35 * num_collectors))

    if st.button('Calculate Pressure Loss and Supply/Return Temperatures'):
        edited_radiator_df[['Radiator power', 'Calculated heat loss', 'Length circuit', 'Space Temperature']] = edited_radiator_df[['Radiator power', 'Calculated heat loss', 'Length circuit', 'Space Temperature']].apply(pd.to_numeric, errors='coerce')
        edited_collector_df[['Collector circuit length']] = edited_collector_df[['Collector circuit length']].apply(pd.to_numeric, errors='coerce')

        if validate_data(edited_radiator_df):
            radiators = [Radiator(row['Radiator nr'], row['Collector'], row['Radiator power'], row['Calculated heat loss'], row['Length circuit'], row['Space Temperature'], delta_T) for index, row in edited_radiator_df.iterrows()]
            collectors = [Collector(row['Collector'], row['Collector circuit length']) for index, row in edited_collector_df.iterrows()]

            for radiator in radiators:
                radiator.calculate_mass_flow_rate()
                radiator.calculate_supply_temperature(calculate_c(radiator.heat_loss / radiator.power, delta_T))
                radiator.calculate_return_temperature(radiator.heat_loss / radiator.power, radiator.supply_temperature)
                radiator.calculate_diameter([15, 20, 25])
                radiator.calculate_pressure_loss()

            for collector in collectors:
                collector.mass_flow_rate = sum(r.mass_flow_rate for r in radiators if r.collector == collector.collector_name)
                collector.calculate_pressure_loss([20, 25, 32])

            results_data = {
                'Radiator nr': [r.radiator_nr for r in radiators],
                'Supply Temperature (°C)': [r.supply_temperature for r in radiators],
                'Return Temperature (°C)': [r.return_temperature for r in radiators],
                'Diameter (mm)': [r.diameter for r in radiators],
                'Pressure Loss (Pa)': [r.pressure_loss for r in radiators],
            }

            results_df = pd.DataFrame(results_data)
            st.write(results_df)

            total_pressure_loss = sum([c.pressure_loss for c in collectors])
            st.write(f'Total Pressure Loss: {total_pressure_loss:.2f} Pa')

            fig = px.bar(results_df, x='Radiator nr', y='Pressure Loss (Pa)', title='Pressure Loss per Radiator')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error('Please ensure all input fields are filled in correctly with valid data.')

if __name__ == '__main__':
    main()
