import pandas as pd
import numpy as np
from SALib.analyze import sobol
import random
from SALib.sample.saltelli import sample

from utils.helpers import POSSIBLE_DIAMETERS, validate_data
from refactorclass import Radiator, Circuit, Collector, Valve

num_samples = 1024
param_ranges = {
    'num_radiators': (1, 3),
    'num_collectors': (1, 2),
    'heat_loss_ratio': (0.2, 0.8),
    'circuit_length': (5, 20),
    'space_temperature': (16, 24),
}
positions = 8
kv_max = 0.7
delta_T = 5

problem = {
    'num_vars': len(param_ranges),
    'names': list(param_ranges.keys()),
    'bounds': [[low, high] for (low, high) in param_ranges.values()]
}
param_values = sample(problem, num_samples)
param_values_df = pd.DataFrame(param_values, columns=problem['names'])

results = []
for index, sample in param_values_df.iterrows():
    num_radiators = int(sample['num_radiators'])
    num_collectors = int(sample['num_collectors'])
    ratio = sample['heat_loss_ratio']
    length = int(sample['circuit_length'])
    space_temp_range = [16, 20, 24]
    collector_options = [f'Collector {i + 1}' for i in range(num_collectors)]
    collectors_assigned = random.sample(collector_options, num_collectors)
    additional_collectors_needed = num_radiators - num_collectors
    if additional_collectors_needed > 0:
        collectors_assigned += [random.choice(collector_options) for _ in range(additional_collectors_needed)]
    random.shuffle(collectors_assigned)
    space_temperatures_assigned = np.random.choice(space_temp_range, num_radiators, replace=True)
    heat_loss_ratios_assigned = np.random.uniform(0.2, 0.8, num_radiators)

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

    if validate_data(radiator_data):
        radiators = []
        for _, row in radiator_data.iterrows():
            radiator = Radiator(
                q_ratio=row['Calculated heat loss'] / row['Radiator power'],
                delta_t=delta_T,
                space_temperature=row['Space Temperature'],
                heat_loss=row['Calculated heat loss']
            )
            radiators.append(radiator)
        radiator_data['Supply Temperature'] = [r.supply_temperature for r in radiators]
        max_supply_temperature = max(r.supply_temperature for r in radiators)

        for r in radiators:
            r.supply_temperature = max_supply_temperature
            r.return_temperature = r.calculate_treturn(max_supply_temperature)
            r.mass_flow_rate = r.calculate_mass_flow_rate()
        radiator_data['Supply Temperature'] = max_supply_temperature
        radiator_data['Return Temperature'] = [r.return_temperature for r in radiators]
        radiator_data['Mass flow rate'] = [r.mass_flow_rate for r in radiators]
        radiator_data['Diameter'] = [
            r.calculate_diameter(POSSIBLE_DIAMETERS) for r in radiators
        ]
        radiator_data['Diameter'] = radiator_data['Diameter'].max()
        radiator_data['Pressure loss'] = [
            Circuit(
                length_circuit=row['Length circuit'],
                diameter=row['Diameter'],
                mass_flow_rate=row['Mass flow rate']
            ).calculate_pressure_radiator_kv() for _, row in radiator_data.iterrows()
        ]

        collectors = [Collector(name=name) for name in collector_options]
        for collector in collectors:
            collector.update_mass_flow_rate(radiator_data)

        collector_data_updated = collector_data.copy()
        for collector in collectors:
            collector_data_updated.loc[
                collector_data_updated['Collector'] == collector.name, 'Mass flow rate'] = collector.mass_flow_rate

        collector_data_updated['Diameter'] = [
            collector.calculate_diameter(POSSIBLE_DIAMETERS) for collector in collectors
        ]
        collector_data_updated['Diameter'] = collector_data_updated['Diameter'].max()

        collector_data_updated['Collector pressure loss'] = [
            Circuit(
                length_circuit=row['Collector circuit length'],
                diameter=row['Diameter'],
                mass_flow_rate=row['Mass flow rate']
            ).calculate_pressure_collector_kv() for _, row in collector_data_updated.iterrows()
        ]

        merged_df = Collector(name='').calculate_total_pressure_loss(
            radiator_df=radiator_data,
            collector_df=collector_data_updated
        )

        valve = Valve(kv_max=kv_max, n=positions)
        merged_df['Thermostatic valve pressure loss N'] = merged_df['Mass flow rate'].apply(
            valve.calculate_pressure_valve_kv)

        merged_df = valve.calculate_kv_position_valve(merged_df, custom_kv_max=kv_max, n=positions)

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

results_df = pd.DataFrame(results)
if len(results_df) < len(param_values):
    raise ValueError("Not enough samples collected for the given number of parameters.")
Y = np.array([results_df.iloc[i % len(results_df)]['Pressure Loss'] for i in range(len(param_values))])
Si = sobol.analyze(problem, Y)

print("First-order Sobol indices:", Si['S1'])  # First-order indices
print("Total-order Sobol indices:", Si['ST'])  # Total-order indices