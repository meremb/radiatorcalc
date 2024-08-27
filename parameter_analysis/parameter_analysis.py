import random

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import validate_data
from refactorclass import Radiator, Circuit, Collector, Valve


# for this sensitivity we set the max possible to 40 as we have some large flows for some configurations
POSSIBLE_DIAMETERS = [8, 10, 12, 13, 14, 16, 20, 22, 28, 36, 50]
# Define the parameter ranges
num_radiators_list = [3, 5, 7, 10]  # Example values
num_collectors_list = [1, 2, 3]  # Example values
heat_loss_ratios = np.linspace(0.2, 0.8, 13)  # Varying the ratio heat_loss to radiator power
circuit_lengths = [5, 10, 20]  # Example circuit lengths
space_temperatures = [16, 20, 24]  # Space Temperatures to be randomly assigned
positions = 8  # Fixed value for valve positions
kv_max = 0.7  # Fixed value for kv max
delta_T = [5, 10, 20]  # Fixed delta T value

results = []

for num_radiators in num_radiators_list:
    for num_collectors in num_collectors_list:
        for ratio in heat_loss_ratios:
            for length in circuit_lengths:
                for T in delta_T:
                    collector_options = [f'Collector {i + 1}' for i in range(num_collectors)]
                    collectors_assigned = random.sample(collector_options, num_collectors)

                    additional_collectors_needed = num_radiators - num_collectors
                    if additional_collectors_needed > 0:
                        collectors_assigned += [random.choice(collector_options) for _ in range(additional_collectors_needed)]
                    random.shuffle(collectors_assigned)
                    initial_space_temps = random.sample(space_temperatures, min(num_radiators, len(space_temperatures)))

                    additional_space_temps_needed = num_radiators - len(initial_space_temps)
                    if additional_space_temps_needed > 0:
                        initial_space_temps += random.choices(space_temperatures, k=additional_space_temps_needed)
                    random.shuffle(initial_space_temps)
                    random_heat_loss_ratios = random.choices(heat_loss_ratios, k=num_radiators)

                    radiator_initial_data = {
                        'Radiator nr': list(range(1, num_radiators + 1)),
                        'Collector': collectors_assigned,
                        'Radiator power': [2500.0] * num_radiators,
                        'Calculated heat loss': [2500.0 * ratio] * num_radiators,
                        'Length circuit': [length] * num_radiators,
                        'Space Temperature': initial_space_temps,  # Assigned space temperatures
                        'Delta T': [T] * num_radiators,
                    }

                    assert all(len(lst) == num_radiators for lst in radiator_initial_data.values()), "Data lengths are not consistent."

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
                                delta_t=T,
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
                                collector_data_updated[
                                    'Collector'] == collector.name, 'Mass flow rate'] = collector.mass_flow_rate

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
                                'Heat Loss Ratio': row['Calculated heat loss'] / 2500.0,
                                'Circuit Length': length,
                                'Space Temperature': row['Space Temperature'],
                                'Supply Temperature': row['Supply Temperature'],
                                'Diameter': row['Diameter'],
                                'Pressure Loss': row['Total Pressure Loss'],
                                'Delta T': row['Delta T']
                            })

results_df = pd.DataFrame(results)


def plot_parallel_coordinates_overview(results):
    overview_data = []
    for result in results:
        overview_data.append({
            'Number of Radiators': result['Number of Radiators'],
            'Number of Collectors': result['Number of Collectors'],
            'Heat Loss Ratio': result['Heat Loss Ratio'],
            'Circuit Length (m)': result['Circuit Length'],
            'Average Total Pressure Loss': result['Pressure Loss']
        })

    overview_df = pd.DataFrame(overview_data)

    fig = px.parallel_coordinates(
        overview_df,
        dimensions=['Number of Radiators', 'Number of Collectors', 'Heat Loss Ratio', 'Circuit Length (m)', 'Average Total Pressure Loss'],
        color='Average Total Pressure Loss',
        color_continuous_scale=px.colors.diverging.Tealrose,
        labels={
            'Number of Radiators': 'Num Radiators',
            'Number of Collectors': 'Num Collectors',
            'Heat Loss Ratio': 'Heat Loss Ratio',
            'Circuit Length (m)': 'Circuit Length',
            'Average Total Pressure Loss': 'Avg Pressure Loss (Pa)'
        },
        title='Global Overview of Parameter Analysis'
    )
    fig.show()


def sensitivity_analysis(results, parameter_name, dependent_var='Total Pressure Loss'):
    sensitivity_results = []

    for result in results:
        param_value = result[parameter_name]
        avg_pressure_loss = result['total_pressure_loss_per_circuit'][dependent_var].mean()
        sensitivity_results.append({
            parameter_name: param_value,
            'Average Pressure Loss': avg_pressure_loss,
        })

    return pd.DataFrame(sensitivity_results)


def plot_heatmaps(results_df):
    result = results_df.copy()
    for config in result['Number of Radiators'].unique():
        for collectors in result['Number of Collectors'].unique():
            filtered_df = result[(result['Number of Radiators'] == config) &
                                     (result['Number of Collectors'] == collectors)]

            fig_diameter = px.density_heatmap(filtered_df, x='Heat Loss Ratio', y='Circuit Length',
                                              z='Diameter', color_continuous_scale='Viridis',
                                              title=f'Diameter Heatmap - Radiators: {config}, Collectors: {collectors}',
                                              labels={'Diameter': 'Diameter (mm)'})
            fig_diameter.show()

            fig_pressure = px.density_heatmap(filtered_df, x='Heat Loss Ratio', y='Circuit Length',
                                              z='Pressure Loss', color_continuous_scale='Plasma',
                                              title=f'Pressure Loss Heatmap - Radiators: {config}, Collectors: {collectors}',
                                              labels={'Pressure Loss': 'Pressure Loss (Pa)'})
            fig_pressure.show()


def plot_surface_plots(results_df):
    result = results_df.copy()
    result = result.dropna(subset=['Heat Loss Ratio', 'Circuit Length', 'Diameter', 'Pressure Loss'])

    if result.empty:
        print("No valid data available for plotting.")
        return

    diameter_grid = result.pivot_table(index='Circuit Length', columns='Heat Loss Ratio', values='Diameter')
    pressure_loss_grid = result.pivot_table(index='Circuit Length', columns='Heat Loss Ratio',
                                                values='Pressure Loss')

    diameter_grid = diameter_grid.interpolate().fillna(0)
    pressure_loss_grid = pressure_loss_grid.interpolate().fillna(0)

    x = diameter_grid.columns.values
    y = diameter_grid.index.values
    z_diameter = diameter_grid.values
    z_pressure_loss = pressure_loss_grid.values
    fig_diameter = go.Figure(data=[go.Surface(
        z=z_diameter,
        x=x,
        y=y,
        colorscale='Viridis'
    )])
    fig_diameter.update_layout(
        title='Surface Plot of Diameter',
        scene=dict(
            xaxis_title='Heat Loss Ratio',
            yaxis_title='Circuit Length (m)',
            zaxis_title='Diameter (mm)'
        )
    )
    fig_diameter.show()
    fig_pressure = go.Figure(data=[go.Surface(
        z=z_pressure_loss,
        x=x,
        y=y,
        colorscale='Plasma'
    )])
    fig_pressure.update_layout(
        title='Surface Plot of Pressure Loss',
        scene=dict(
            xaxis_title='Heat Loss Ratio',
            yaxis_title='Circuit Length (m)',
            zaxis_title='Pressure Loss (Pa)'
        )
    )
    fig_pressure.show()


def plot_facet_grids(results_df):
    result = results_df.copy()
    result['Circuit Length'] = result['Circuit Length'].astype(str)
    result['Delta T'] = result['Delta T'].astype(str)
    fig1 = px.scatter(result, x='Heat Loss Ratio', y='Diameter', color='Delta T',
                     title='Diameter vs Heat Loss Ratio (Qneeded/Qradiator)',
                     labels={'Diameter': 'Diameter (mm)'})
    fig1.show()
    fig2 = px.scatter(result, x='Heat Loss Ratio', y='Pressure Loss', color='Circuit Length',
                     facet_col='Number of Radiators', facet_row='Number of Collectors',
                     title='Pressure Loss vs Heat Loss Ratio (Faceted by Radiators and Collectors)',
                     labels={'Pressure Loss': 'Pressure Loss (Pa)'})
    fig2.show()


def main():
    # Plot overview of results
    plot_parallel_coordinates_overview(results)
    plot_surface_plots(results_df)
    plot_facet_grids(results_df)


if __name__ == "__main__":
    main()

