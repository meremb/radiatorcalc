import math
import numpy as np
import pandas as pd

from typing import List


POSSIBLE_DIAMETERS = [8, 10, 12, 13, 14, 16, 20, 22, 28, 36]
T_FACTOR = 49.83
EXPONENT_RADIATOR = 1.34


def calculate_c(q_ratio: float, delta_t: float) -> float:
    """Calculate the constant 'c' based on Q_ratio and delta_T."""
    c = math.exp(delta_t / T_FACTOR / q_ratio ** (1 / EXPONENT_RADIATOR))
    return c


def calculate_tsupply(space_temperature: float, constant_c: float, delta_t: float) -> float:
    """Calculate the supply temperature based on space temperature, constant_c, and delta_T."""
    return space_temperature + (constant_c / (constant_c - 1)) * delta_t


def calculate_treturn(q_ratio: float, space_temperature: float, max_supply_temperature: float) -> float:
    return (((q_ratio ** (1 / EXPONENT_RADIATOR) * T_FACTOR) ** 2)/(max_supply_temperature - space_temperature) +
            space_temperature)


def calculate_mass_flow_rate(supply_temperature: float, return_temperature:float, heat_loss:float) -> float:
    return heat_loss/4180/(supply_temperature-return_temperature)*3600


def calculate_diameter(mass_flow_rate: float, possible_diameters: List[int]) -> float:
    if math.isnan(mass_flow_rate):
        raise ValueError("The mass flow rate cannot be NaN check the configuration of the number of collectors.")
    diameter = 1.4641*mass_flow_rate**0.4217
    acceptable_diameters = [d for d in possible_diameters if d >= diameter]
    if not acceptable_diameters:
        raise ValueError(f"Calculated diameter exceeds the maximum allowable diameter for mass flow rate:{mass_flow_rate}")
    nearest_diameter = min(acceptable_diameters, key=lambda x: abs(x - diameter))

    return nearest_diameter


def merge_and_calculate_total_pressure_loss(edited_radiator_df: pd.DataFrame, edited_collector_df: pd.DataFrame) -> (
        pd.DataFrame):
    """
    Merge radiator DataFrame with collector DataFrame on 'Collector' column and calculate total pressure loss.
    """
    merged_df = pd.merge(edited_radiator_df, edited_collector_df[['Collector', 'Collector pressure loss']],
                         on='Collector',
                         how='left')
    # Calculate total pressure loss by adding existing Pressure Loss and Collector Pressure Loss
    merged_df['Total Pressure Loss'] = merged_df['Pressure loss'] + merged_df['Collector pressure loss']
    return merged_df


def calculate_pressure_radiator_kv(length_circuit: float, diameter: float, mass_flow_rate: float) -> float:
    """Using simplified functions for the kv of a component the pressure loss for the circuit is calculated. """
    pressure_loss_piping = calculate_pressure_loss_piping(diameter, length_circuit, mass_flow_rate)
    kv_radiator = 2
    pressure_loss_radiator = 97180*(mass_flow_rate/1000/kv_radiator)**2
    return pressure_loss_piping + pressure_loss_radiator


def calculate_pressure_collector_kv(length_circuit: float, diameter: float, mass_flow_rate: float) -> float:
    """Using simplified functions for the kv of a component the pressure loss for the head circuit is calculated. """
    pressure_loss_piping = calculate_pressure_loss_piping(diameter, length_circuit, mass_flow_rate)
    kv_collector = 14.66
    pressure_loss_boiler = 200
    pressure_loss_collector = 97180*(mass_flow_rate/1000/kv_collector)**2
    return pressure_loss_piping + pressure_loss_collector + pressure_loss_boiler


def calculate_pressure_loss_piping(diameter: float, length_circuit: float, mass_flow_rate: float) -> float:
    """Using simplified functions for the kv the pressure loss for the piping is calculated. """
    # formula piping is specific for pex tubes we should make this an optional selection
    kv_piping = 51626 * (diameter / 1000) ** 2 - 417.39 * (diameter / 1000) + 1.5541
    resistance_meter = 97180 * (mass_flow_rate / 1000 / kv_piping) ** 2
    coefficient_local_losses = 1.3
    pressure_loss_piping = resistance_meter * length_circuit * coefficient_local_losses
    return pressure_loss_piping


def update_collector_mass_flow_rate(edited_radiator_df: pd.DataFrame, edited_collector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the total mass flow rate for each collector based on the radiator data,
    and update the collector DataFrame with these values.
    """
    # Calculate total mass flow rate for each collector
    collector_mass_flow_rate = (
        edited_radiator_df.groupby('Collector')['Mass flow rate']
        .sum().reset_index()
    )

    # Merge the mass flow rates into the collector DataFrame
    updated_collector_df = pd.merge(edited_collector_df, collector_mass_flow_rate, on='Collector', how='left')
    return updated_collector_df


def calculate_pressure_valve_kv(mass_flow_rate: float) -> float:
    """Calculate pressure loss for thermostatic valve at position N. """
    kv_max_valve_n = 0.7
    pressure_loss_valve = 97180*(mass_flow_rate/1000/kv_max_valve_n)**2
    return pressure_loss_valve


def calculate_kv_position_valve(merged_df, custom_kv_max=None, n=None):
    merged_df = calculate_kv_needed(merged_df)
    #kv formula polynomials fitted using data sheets
    a = 0.0114
    b = - 0.0086
    c = 0.0446
    initial_positions = calculate_valve_position(a, b, c, merged_df['kv_needed'])

    if custom_kv_max is not None and n is not None:
        kv_needed_array = merged_df['kv_needed'].to_numpy()
        adjusted_positions = adjust_position_with_custom_values(custom_kv_max, n, kv_needed_array)
        merged_df['Valve position'] = adjusted_positions.flatten()  # Ensure this is a single-dimensional array
    else:
        initial_positions = np.ceil(initial_positions)
        merged_df['Valve position'] = initial_positions.flatten()  # Ensure this is a single-dimensional array

    return merged_df


def calculate_kv_needed(merged_df):
    merged_df = merged_df.copy()
    merged_df['Total pressure valve circuit'] = merged_df['Total Pressure Loss'] + merged_df[
        'Thermostatic valve pressure loss N']
    maximum_pressure = max(merged_df['Total pressure valve circuit'])
    merged_df['Pressure difference valve'] = maximum_pressure - merged_df['Total Pressure Loss']
    merged_df['kv_needed'] = (merged_df['Mass flow rate'] / 1000) / (
                merged_df['Pressure difference valve'] / 100000) ** 0.5
    return merged_df


def calculate_valve_position(a, b, c, kv_needed):
    discriminant = b ** 2 - 4 * a * (c - kv_needed)
    discriminant = np.where(discriminant < 0, 0, discriminant)
    root = -b + np.sqrt(discriminant) / (2 * a)
    root = np.where(discriminant <= 0, 0.1, root)
    return root


def adjust_position_with_custom_values(kv_max, n, kv_needed):
    ratio_kv = kv_needed / 0.7054
    adjusted_ratio_kv = (ratio_kv * kv_max) / kv_max
    ratio_position = np.clip(np.sqrt(adjusted_ratio_kv), 0,1)
    adjusted_position = np.ceil(ratio_position * n)
    return adjusted_position


def calculate_position_valve_with_ratio(kv_max, n, kv_needed):
    ratio_kv = kv_needed / kv_max
    a = 0.8053
    b = 0.1269
    c = 0.0468
    ratio_position = calculate_valve_position(a, b, c, ratio_kv)
    final_position = np.ceil(ratio_position * n)
    return final_position


def validate_data(df: pd.DataFrame) -> bool:
    """Validate the input data to ensure all required fields are correctly filled."""
    required_columns = ['Radiator power', 'Length circuit', 'Space Temperature']
    for col in required_columns:
        if df[col].isnull().any() or (df[col] <= 0).any():
            return False
    return True


def calculate_water_volume(diameter: float, length_circuit: float) -> float:
    water_volume_pipe = (np.pi * (diameter/2)**2)/1000000 * length_circuit * 1000
    return water_volume_pipe


