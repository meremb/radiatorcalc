import streamlit as st
import pandas as pd
from typing import List, Dict

from utils.helpers import POSSIBLE_DIAMETERS, calculate_c, calculate_Tsupply, calculate_Treturn, calculate_mass_flow_rate, \
    calculate_diameter, merge_and_calculate_total_pressure_loss, calculate_pressure_radiator_kv, \
    calculate_pressure_collector_kv, calculate_pressure_valve_kv, update_collector_mass_flow_rate, \
    calculate_kv_position_valve, validate_data
from utils.plotting import plot_pressure_loss, plot_thermostatic_valve_position, plot_mass_flow_distribution, \
    plot_temperature_heatmap


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title('Radiator Distribution Calculator')
    st.write('Enter the details for each radiator to calculate the total pressure loss and supply/return temperatures.')

    # Sidebar Configuration
    st.sidebar.header('Configuration')
    num_radiators = st.sidebar.number_input('Number of Radiators', min_value=1, value=3, step=1)
    num_collectors = st.sidebar.number_input('Number of Collectors', min_value=1, value=1, step=1)
    positions = st.sidebar.number_input('Number of positions for valve', min_value=1, value=8, step=1)
    kv_max = st.sidebar.number_input('kv max for the valve', min_value=0.50, value=0.70, step=0.01)

    # Add slider for delta T input
    delta_T = st.sidebar.slider('Delta T (°C)', min_value=3, max_value=20, value=5, step=1)


    # Initialize DataFrame columns and create rows based on number of radiators
    radiator_columns: List[str] = [
        'Radiator nr', 'Collector', 'Radiator power', 'Calculated heat loss',
        'Length circuit', 'Space Temperature'
    ]

    collector_options = [f'Collector {i + 1}' for i in range(num_collectors)]

    radiator_initial_data: Dict[str, List] = {
        'Radiator nr': list(range(1, num_radiators + 1)),
        'Collector': [collector_options[0]] * num_radiators,
        'Radiator power': [0.0] * num_radiators,
        'Calculated heat loss': [0.0] * num_radiators,
        'Length circuit': [0.0] * num_radiators,
        'Space Temperature': [20.0] * num_radiators,  # Default space temperature
    }

    radiator_data: pd.DataFrame = pd.DataFrame(radiator_initial_data, columns=radiator_columns)

    # Initialize DataFrame columns and create rows based on number of collectors
    collector_columns: List[str] = [
        'Collector', 'Collector circuit length'
    ]
    collector_initial_data: Dict[str, List] = {
        'Collector': [f'Collector {i + 1}' for i in range(num_collectors)],
        'Collector circuit length': [0.0] * num_collectors,
    }

    collector_data: pd.DataFrame = pd.DataFrame(collector_initial_data, columns=collector_columns)


    # Display editable data editor with container width and controlled height
    edited_radiator_df: pd.DataFrame = st.data_editor(
        radiator_data,
        key='editable_table',
        num_rows="dynamic",
        use_container_width=True,  # Automatically use the container width
        height=min(600, 50 + 35 * num_radiators),
        column_config={
            'Radiator nr': st.column_config.NumberColumn(
                "Radiator nr", format="%d", width=100),
            'Collector': st.column_config.SelectboxColumn(
                "Collector", options=collector_options, width=150),
            'Radiator power': st.column_config.NumberColumn(
                "Radiator power (W)", format="%.2f", width=150),
            'Calculated heat loss': st.column_config.NumberColumn(
                "Heat loss (W)", format="%.2f", width=150),
            'Length circuit': st.column_config.NumberColumn(
                "Circuit Length (m)", format="%.2f", width=150),
            'Space Temperature': st.column_config.NumberColumn(
                "Space Temperature (°C)", format="%.1f", width=150),
        }
    )

    # Display editable data editor for collectors
    edited_collector_df: pd.DataFrame = st.data_editor(
        collector_data,
        key='collector_table',
        num_rows="dynamic",
        use_container_width=True,  # Automatically use the container width
        height=min(600, 50 + 35 * num_collectors),
        column_config={
            'Circuit': st.column_config.TextColumn("Circuit", width=150),
            'Collector circuit length': st.column_config.NumberColumn(
                "Collector circuit Length (m)", format="%.2f", width=150),
        }
    )

    # When Calculate button is clicked
    if st.button('Calculate Pressure Loss and Supply/Return Temperatures'):
        # Convert numeric columns to appropriate types
        numeric_columns: List[str] = [
            'Radiator power', 'Calculated heat loss','Length circuit',
            'Space Temperature'
        ]
        edited_radiator_df[numeric_columns] = edited_radiator_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        collector_numeric_columns: List[str] = [
            'Collector circuit length',
        ]
        edited_collector_df[collector_numeric_columns] = edited_collector_df[collector_numeric_columns].apply(
            pd.to_numeric, errors='coerce')

        # Validate data
        if validate_data(edited_radiator_df):
            # Calculate Q_ratio and constant_c for each radiator
            edited_radiator_df['Q_ratio'] = edited_radiator_df['Calculated heat loss'] / edited_radiator_df['Radiator power']
            edited_radiator_df['Constant_c'] = edited_radiator_df.apply(lambda row: calculate_c(row['Q_ratio'], delta_T), axis=1)

            # Calculate supply temperature for each radiator
            edited_radiator_df['Supply Temperature'] = edited_radiator_df.apply(
                lambda row: calculate_Tsupply(row['Space Temperature'], row['Constant_c'], delta_T),
                axis=1
            )

            # Determine the maximum supply temperature
            max_supply_temperature = edited_radiator_df['Supply Temperature'].max()

            edited_radiator_df['Return Temperature'] = edited_radiator_df.apply(
                lambda row: calculate_Treturn(row['Q_ratio'], row['Space Temperature'], max_supply_temperature),
                axis=1
            )

            # Update all radiators with the maximum supply temperature
            edited_radiator_df['Supply Temperature'] = max_supply_temperature
            edited_radiator_df['Mass flow rate'] = edited_radiator_df.apply(
                lambda row: calculate_mass_flow_rate(row['Supply Temperature'], row['Return Temperature'],
                                                     row['Calculated heat loss']),
                axis=1
            )

            edited_radiator_df['Diameter'] = edited_radiator_df.apply(
                lambda row: calculate_diameter(row['Mass flow rate'], POSSIBLE_DIAMETERS),
                axis=1
            )
            edited_radiator_df['Diameter'] = edited_radiator_df['Diameter'].max()

            # Calculate pressure loss for each row
            edited_radiator_df['Pressure loss'] = edited_radiator_df.apply(
                lambda row: calculate_pressure_radiator_kv(
                    row['Length circuit'],
                    row['Diameter'],
                    row['Mass flow rate']
                ),
                axis=1
            )

            # now same for the collector
            edited_collector_df = update_collector_mass_flow_rate(edited_radiator_df, edited_collector_df)

            edited_collector_df['Diameter'] = edited_collector_df.apply(
                lambda row: calculate_diameter(row['Mass flow rate'], POSSIBLE_DIAMETERS),
                axis=1
            )
            edited_collector_df['Diameter'] = edited_collector_df['Diameter'].max()

            edited_collector_df['Collector pressure loss'] = edited_collector_df.apply(
                lambda row: calculate_pressure_collector_kv(
                    row['Collector circuit length'],
                    row['Diameter'],
                    row['Mass flow rate']
                ),
                axis=1
            )

            merged_df = merge_and_calculate_total_pressure_loss(edited_radiator_df=edited_radiator_df,
                                                                edited_collector_df=edited_collector_df)

            merged_df['Thermostatic valve pressure loss N'] = merged_df.apply(
                lambda row: calculate_pressure_valve_kv(
                    row['Mass flow rate']
                ),
                axis=1
            )
            # calculate the thermostatic valve position
            merged_df = calculate_kv_position_valve(merged_df=merged_df, n=positions, custom_kv_max=kv_max)

            # Group by 'Radiator' and calculate total pressure loss per circuit
            total_pressure_loss_per_circuit: pd.DataFrame = (
                merged_df.groupby('Radiator nr')['Total Pressure Loss']
                .sum().reset_index()
            )

            # Display results
            st.write('### Results')
            st.write('**Individual Radiator Pressure Loss, Supply Temperature, and Return Temperature**')
            st.dataframe(
                merged_df[['Radiator nr', 'Collector', 'Pressure loss', 'Total Pressure Loss',
                           'Thermostatic valve pressure loss N', 'kv_needed', 'Supply Temperature',
                           'Return Temperature', 'Mass flow rate', 'Diameter']],
                use_container_width=True
            )

            # Display results
            st.write('### Results')
            st.write('**Individual Collector results**')
            st.dataframe(
                edited_collector_df[['Collector', 'Collector pressure loss', 'Mass flow rate', 'Diameter']],
                use_container_width=True
            )

            st.write('**Total Pressure Loss per Circuit**')
            st.dataframe(total_pressure_loss_per_circuit, use_container_width=True)

            # Display visualizations
            plot_pressure_loss(total_pressure_loss_per_circuit)
            plot_thermostatic_valve_position(merged_df)
            plot_mass_flow_distribution(merged_df)
            plot_temperature_heatmap(merged_df)
        else:
            st.error("Invalid input data. Please check your inputs.")


if __name__ == "__main__":
    main()

