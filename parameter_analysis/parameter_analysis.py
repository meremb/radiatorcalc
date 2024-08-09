import random

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import POSSIBLE_DIAMETERS, calculate_c, calculate_Tsupply, calculate_Treturn, calculate_mass_flow_rate, \
    calculate_diameter, merge_and_calculate_total_pressure_loss, calculate_pressure_radiator_kv, \
    calculate_pressure_collector_kv, calculate_pressure_valve_kv, update_collector_mass_flow_rate, \
    calculate_kv_position_valve, validate_data

# Define the parameter ranges
num_radiators_list = [3, 5, 7]  # Example values
num_collectors_list = [1, 2]  # Example values
heat_loss_ratios = np.linspace(0.2, 0.8, 13)  # Varying the ratio heat_loss to radiator power
circuit_lengths = [5, 10, 20]  # Example circuit lengths
space_temperatures = [16, 20, 24]  # Space Temperatures to be randomly assigned
positions = 8  # Fixed value for valve positions
kv_max = 0.7  # Fixed value for kv max
delta_T = 5  # Fixed delta T value

# Initialize results list
results = []

for num_radiators in num_radiators_list:
    for num_collectors in num_collectors_list:
        for ratio in heat_loss_ratios:
            for length in circuit_lengths:
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

                # Ensure each space temperature is assigned at least once
                initial_space_temps = random.sample(space_temperatures, min(num_radiators, len(space_temperatures)))

                # Assign remaining space temperatures randomly
                additional_space_temps_needed = num_radiators - len(initial_space_temps)
                if additional_space_temps_needed > 0:
                    initial_space_temps += random.choices(space_temperatures, k=additional_space_temps_needed)

                # Shuffle the final list of space temperatures to randomize the order of assignment
                random.shuffle(initial_space_temps)

                # Assign heat loss ratios completely randomly
                random_heat_loss_ratios = random.choices(heat_loss_ratios, k=num_radiators)

                # Initialize DataFrame columns and create rows based on number of radiators
                radiator_initial_data = {
                    'Radiator nr': list(range(1, num_radiators + 1)),
                    'Collector': collectors_assigned,
                    'Radiator power': [2000.0] * num_radiators,  # Example fixed power
                    'Calculated heat loss': [2000.0 * ratio] * num_radiators,  # Assigned heat loss ratios
                    'Length circuit': [length] * num_radiators,
                    'Space Temperature': initial_space_temps,  # Assigned space temperatures
                }

                # Ensure all lists have the same length before creating the DataFrame
                assert all(len(lst) == num_radiators for lst in radiator_initial_data.values()), "Data lengths are not consistent."

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
                            'Heat Loss Ratio': row['Calculated heat loss'] / 2000.0,  # Inverse calculation for storage
                            'Circuit Length': length,
                            'Space Temperature': row['Space Temperature'],  # Include Space Temperature in results
                            'Supply Temperature': row['Supply Temperature'],  # Include Supply Temperature in results
                            'Diameter': row['Diameter'],
                            'Pressure Loss': row['Total Pressure Loss'],
                        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)


# Now 'results' contains all the data for analysis. You can further analyze or export this data.
# Example: Plotting Total Pressure Loss per Circuit for each result
def plot_diameter_distribution(results_df):
    fig = px.scatter(
        results_df,
        x='Radiator nr',
        y='Diameter',
        color='Heat Loss Ratio',
        size='Pressure Loss',
        hover_data=['Collector', 'Circuit Length', 'Number of Radiators', 'Number of Collectors'],
        facet_col='Number of Radiators',
        facet_row='Number of Collectors',
        title='Diameter Distribution per Radiator across Different Configurations',
        labels={'Diameter': 'Diameter (mm)', 'Radiator nr': 'Radiator Number'}
    )
    fig.update_traces(marker=dict(opacity=0.7))
    fig.show()

# Function to plot the pressure loss for each radiator
def plot_pressure_loss_distribution(results_df):
    fig = px.scatter(
        results_df,
        x='Radiator nr',
        y='Pressure Loss',
        color='Heat Loss Ratio',
        size='Diameter',
        hover_data=['Collector', 'Circuit Length', 'Number of Radiators', 'Number of Collectors'],
        facet_col='Number of Radiators',
        facet_row='Number of Collectors',
        title='Pressure Loss per Radiator across Different Configurations',
        labels={'Pressure Loss': 'Pressure Loss (Pa)', 'Radiator nr': 'Radiator Number'}
    )
    fig.update_traces(marker=dict(opacity=0.7))
    fig.show()


# Function to plot the global overview using parallel coordinates
# Function to plot the global overview using parallel coordinates
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
    for config in results_df['Number of Radiators'].unique():
        for collectors in results_df['Number of Collectors'].unique():
            filtered_df = results_df[(results_df['Number of Radiators'] == config) &
                                     (results_df['Number of Collectors'] == collectors)]

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
    # Filter out any rows where data might be missing or invalid
    results_df = results_df.dropna(subset=['Heat Loss Ratio', 'Circuit Length', 'Diameter', 'Pressure Loss'])

    if results_df.empty:
        print("No valid data available for plotting.")
        return

    # Create pivot tables to ensure the data is in a grid format
    diameter_grid = results_df.pivot_table(index='Circuit Length', columns='Heat Loss Ratio', values='Diameter')
    pressure_loss_grid = results_df.pivot_table(index='Circuit Length', columns='Heat Loss Ratio',
                                                values='Pressure Loss')

    # Ensure the grids are properly filled (using interpolation or filling missing values)
    diameter_grid = diameter_grid.interpolate().fillna(0)
    pressure_loss_grid = pressure_loss_grid.interpolate().fillna(0)

    # Extract x, y, z values from the grids
    x = diameter_grid.columns.values
    y = diameter_grid.index.values
    z_diameter = diameter_grid.values
    z_pressure_loss = pressure_loss_grid.values

    # Create 3D Surface plot for Diameter
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

    # Create 3D Surface plot for Pressure Loss
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
    fig = px.scatter(results_df, x='Heat Loss Ratio', y='Diameter', color='Circuit Length',
                     facet_col='Number of Radiators', facet_row='Number of Collectors',
                     title='Diameter vs Heat Loss Ratio (Faceted by Radiators and Collectors)',
                     labels={'Diameter': 'Diameter (mm)'})
    fig.show()

    fig = px.scatter(results_df, x='Heat Loss Ratio', y='Pressure Loss', color='Circuit Length',
                     facet_col='Number of Radiators', facet_row='Number of Collectors',
                     title='Pressure Loss vs Heat Loss Ratio (Faceted by Radiators and Collectors)',
                     labels={'Pressure Loss': 'Pressure Loss (Pa)'})
    fig.show()


def main():
    # Plot overview of results
    plot_parallel_coordinates_overview(results)
    plot_surface_plots(results_df)
    plot_facet_grids(results_df)

if __name__ == "__main__":
    main()

