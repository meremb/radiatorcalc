import pandas as pd
import numpy as np
from SALib.analyze import sobol
import random
from SALib.sample.sobol import sample

from utils.helpers import POSSIBLE_DIAMETERS, calculate_c, calculate_Tsupply, calculate_Treturn, calculate_mass_flow_rate, \
    calculate_diameter, merge_and_calculate_total_pressure_loss, calculate_pressure_radiator_kv, \
    calculate_pressure_collector_kv, calculate_pressure_valve_kv, update_collector_mass_flow_rate, \
    calculate_kv_position_valve, validate_data

# Define the parameter ranges
num_samples = 1024  # Number of samples for sensitivity analysis
param_ranges = {
    'num_radiators': (3, 7),
    'num_collectors': (1, 2),
    'heat_loss_ratio': (0.2, 0.8),
    'circuit_length': (5, 20),
    'space_temperature': (16, 24),
}
positions = 8  # Fixed value for valve positions
kv_max = 0.7  # Fixed value for kv max
delta_T = 5  # Fixed delta T value

# Define the problem for SALib
problem = {
    'num_vars': len(param_ranges),
    'names': list(param_ranges.keys()),
    'bounds': [[low, high] for (low, high) in param_ranges.values()]
}

# Generate Sobol samples
param_values = sample(problem, num_samples)

# Convert Sobol samples to specific discrete values if necessary
param_values_df = pd.DataFrame(param_values, columns=problem['names'])
param_values_df['num_radiators'] = np.round(param_values_df['num_radiators'] * (7 - 3) + 3).astype(int)
param_values_df['num_collectors'] = np.round(param_values_df['num_collectors'] * (2 - 1) + 1).astype(int)
param_values_df['circuit_length'] = np.round(param_values_df['circuit_length'] * (20 - 5) + 5).astype(int)

# Initialize results list
results = []

# Run the parameter analysis for each sample
for index, sample in param_values_df.iterrows():
    num_radiators = int(sample['num_radiators'])  # Convert to integer
    num_collectors = int(sample['num_collectors'])  # Convert to integer
    ratio = sample['heat_loss_ratio']
    length = int(sample['circuit_length'])  # Convert to integer
    space_temp_range = [16, 20, 24]  # Space temperatures

    # Generate collector options based on num_collectors
    collector_options = [f'Collector {i + 1}' for i in range(num_collectors)]

    # Ensure each collector is assigned at least once
    collectors_assigned = random.sample(collector_options, num_collectors)

    # Assign remaining radiators to collectors randomly
    additional_collectors_needed = num_radiators - num_collectors
    if additional_collectors_needed > 0:
        collectors_assigned += [random.choice(collector_options) for _ in range(additional_collectors_needed)]

    # Shuffle the final list to randomize the order of assignment
    random.shuffle(collectors_assigned)

    # Randomly assign space temperatures and heat loss ratios to radiators
    space_temperatures_assigned = np.random.choice(space_temp_range, num_radiators, replace=True)
    heat_loss_ratios_assigned = np.random.uniform(0.2, 0.8, num_radiators)

    # Initialize DataFrame columns and create rows based on number of radiators
    radiator_initial_data = {
        'Radiator nr': list(range(1, num_radiators + 1)),
        'Collector': collectors_assigned,
        'Radiator power': [1000.0] * num_radiators,  # Example fixed power
        'Calculated heat loss': heat_loss_ratios_assigned * 1000,
        'Length circuit': [length] * num_radiators,
        'Space Temperature': space_temperatures_assigned,
    }

    radiator_data = pd.DataFrame(radiator_initial_data)

    collector_initial_data = {
        'Collector': [f'Collector {i + 1}' for i in range(num_collectors)],
        'Collector circuit length': [length] * num_collectors,
    }

    collector_data = pd.DataFrame(collector_initial_data)

    # Validate data
    if validate_data(radiator_data):
        # Calculate Q_ratio and constant_c for each radiator
        radiator_data['Q_ratio'] = radiator_data['Calculated heat loss'] / radiator_data['Radiator power']
        radiator_data['Constant_c'] = radiator_data.apply(lambda row: calculate_c(row['Q_ratio'], delta_T), axis=1)

        # Calculate supply temperature for each radiator
        radiator_data['Supply Temperature'] = radiator_data.apply(
            lambda row: calculate_Tsupply(row['Space Temperature'], row['Constant_c'], delta_T),
            axis=1
        )

        # Determine the maximum supply temperature
        max_supply_temperature = radiator_data['Supply Temperature'].max()

        radiator_data['Return Temperature'] = radiator_data.apply(
            lambda row: calculate_Treturn(row['Q_ratio'], row['Space Temperature'], max_supply_temperature),
            axis=1
        )

        # Update all radiators with the maximum supply temperature
        radiator_data['Supply Temperature'] = max_supply_temperature
        radiator_data['Mass flow rate'] = radiator_data.apply(
            lambda row: calculate_mass_flow_rate(row['Supply Temperature'], row['Return Temperature'],
                                                 row['Calculated heat loss']),
            axis=1
        )

        radiator_data['Diameter'] = radiator_data.apply(
            lambda row: calculate_diameter(row['Mass flow rate'], POSSIBLE_DIAMETERS),
            axis=1
        )
        radiator_data['Diameter'] = radiator_data['Diameter'].max()

        # Calculate pressure loss for each row
        radiator_data['Pressure loss'] = radiator_data.apply(
            lambda row: calculate_pressure_radiator_kv(row['Length circuit'], row['Diameter'],
                                                       row['Mass flow rate']),
            axis=1
        )

        # Now same for the collector
        collector_data = update_collector_mass_flow_rate(radiator_data, collector_data)

        collector_data['Diameter'] = collector_data.apply(
            lambda row: calculate_diameter(row['Mass flow rate'], POSSIBLE_DIAMETERS),
            axis=1
        )
        collector_data['Diameter'] = collector_data['Diameter'].max()

        collector_data['Collector pressure loss'] = collector_data.apply(
            lambda row: calculate_pressure_collector_kv(
                row['Collector circuit length'],
                row['Diameter'],
                row['Mass flow rate']
            ),
            axis=1
        )

        merged_df = merge_and_calculate_total_pressure_loss(edited_radiator_df=radiator_data,
                                                            edited_collector_df=collector_data)

        merged_df['Thermostatic valve pressure loss N'] = merged_df.apply(
            lambda row: calculate_pressure_valve_kv(
                row['Mass flow rate']
            ),
            axis=1
        )

        # Calculate the thermostatic valve position
        merged_df = calculate_kv_position_valve(merged_df=merged_df, n=positions, custom_kv_max=kv_max)

        # Store the detailed result for each radiator
        for index, row in merged_df.iterrows():
            results.append({
                'Radiator nr': row['Radiator nr'],
                'Collector': row['Collector'],
                'Number of Radiators': num_radiators,
                'Number of Collectors': num_collectors,
                'Heat Loss Ratio': ratio,
                'Circuit Length': length,
                'Space Temperature': row['Space Temperature'],
                'Diameter': row['Diameter'],
                'Pressure Loss': row['Total Pressure Loss'],
                'Supply Temperature': row['Supply Temperature'],
                'Return Temperature': row['Return Temperature']
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Ensure results_df has enough rows to match the number of samples
if len(results_df) < len(param_values):
    raise ValueError("Not enough samples collected for the given number of parameters.")

# Gather the output values for each sample
# Ensure the output length matches the number of samples
Y = np.array([results_df.iloc[i % len(results_df)]['Pressure Loss'] for i in range(len(param_values))])

# Perform Sobol sensitivity analysis
Si = sobol.analyze(problem, Y)

# Print results
print("First-order Sobol indices:", Si['S1'])  # First-order indices
print("Total-order Sobol indices:", Si['ST'])  # Total-order indices