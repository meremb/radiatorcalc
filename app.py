import streamlit as st
import pandas as pd
from typing import List, Dict

from utils.helpers import POSSIBLE_DIAMETERS, Radiator, Circuit, Collector, Valve, validate_data
from utils.plotting import plot_pressure_loss, plot_thermostatic_valve_position, plot_mass_flow_distribution, \
    plot_temperature_heatmap

def main() -> None:
    st.title('Radiator Distribution Calculator')
    st.write('Enter the details for each radiator to calculate the total pressure loss and supply/return temperatures.')

    st.sidebar.header('Configuration')
    num_radiators = st.sidebar.number_input('Number of Radiators', min_value=1, value=3, step=1)
    num_collectors = st.sidebar.number_input('Number of Collectors', min_value=1, value=1, step=1)
    positions = st.sidebar.number_input('Number of positions for valve', min_value=1, value=8, step=1)
    kv_max = st.sidebar.number_input('kv max for the valve', min_value=0.50, value=0.70, step=0.01)
    delta_T = st.sidebar.slider('Delta T (°C)', min_value=3, max_value=20, value=5, step=1)
    supply_temp_input = st.sidebar.number_input('Supply Temperature (°C)', min_value=45.0, max_value=70.0, step=1.0, format="%.1f")

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
        'Space Temperature': [20.0] * num_radiators,
    }

    radiator_data: pd.DataFrame = pd.DataFrame(radiator_initial_data, columns=radiator_columns)

    collector_columns: List[str] = [
        'Collector', 'Collector circuit length'
    ]
    collector_initial_data: Dict[str, List] = {
        'Collector': [f'Collector {i + 1}' for i in range(num_collectors)],
        'Collector circuit length': [0.0] * num_collectors,
    }

    collector_data: pd.DataFrame = pd.DataFrame(collector_initial_data, columns=collector_columns)

    edited_radiator_df: pd.DataFrame = st.data_editor(
        radiator_data,
        key='editable_table',
        hide_index=True,
        use_container_width=True,
        height=min(600, 50 + 35 * num_radiators),
        column_config={
            'Radiator nr': st.column_config.NumberColumn(
                "Radiator", format="%d"),
            'Collector': st.column_config.SelectboxColumn(
                "Collector", options=collector_options),
            'Radiator power': st.column_config.NumberColumn(
                "Radiator (W)", format="%.2f"),
            'Calculated heat loss': st.column_config.NumberColumn(
                "Heat loss (W)", format="%.2f"),
            'Length circuit': st.column_config.NumberColumn(
                "Circuit length (m)", format="%.2f"),
            'Space Temperature': st.column_config.NumberColumn(
                "T indoor(°C)", format="%.1f"),
        }
    )

    edited_collector_df: pd.DataFrame = st.data_editor(
        collector_data,
        key='collector_table',
        hide_index=True,
        use_container_width=True,
        height=min(600, 50 + 35 * num_collectors),
        column_config={
            'Collector': st.column_config.TextColumn("Collector", width=150),
            'Collector circuit length': st.column_config.NumberColumn(
                "Collector circuit Length (m)", format="%.2f", width=150),
        }
    )

    if st.button('Calculate Pressure Loss and Supply/Return Temperatures'):
        try:
            numeric_columns = [
                'Radiator power', 'Calculated heat loss', 'Length circuit', 'Space Temperature'
            ]
            edited_radiator_df[numeric_columns] = edited_radiator_df[numeric_columns].apply(pd.to_numeric,
                                                                                            errors='coerce')
            collector_numeric_columns = [
                'Collector circuit length',
            ]
            edited_collector_df[collector_numeric_columns] = edited_collector_df[collector_numeric_columns].apply(
                pd.to_numeric, errors='coerce')

            if not validate_data(edited_radiator_df):
                st.error("Invalid input data. Please check your inputs.")
                return

            radiators = []
            for _, row in edited_radiator_df.iterrows():
                radiator = Radiator(
                    q_ratio=row['Calculated heat loss'] / row['Radiator power'],
                    delta_t=delta_T,
                    space_temperature=row['Space Temperature'],
                    heat_loss=row['Calculated heat loss']
                )
                if row['Radiator power'] < row['Calculated heat loss']:
                    radiator.warn_radiator_power()
                radiators.append(radiator)

            edited_radiator_df['Supply Temperature'] = [r.supply_temperature for r in radiators]
            max_supply_temperature = supply_temp_input if supply_temp_input else max(
                r.supply_temperature for r in radiators)

            for r in radiators:
                r.supply_temperature = max_supply_temperature
                r.return_temperature = r.calculate_treturn(max_supply_temperature)
                r.mass_flow_rate = r.calculate_mass_flow_rate()

            edited_radiator_df['Supply Temperature'] = max_supply_temperature
            edited_radiator_df['Return Temperature'] = [r.return_temperature for r in radiators]
            edited_radiator_df['Mass flow rate'] = [r.mass_flow_rate for r in radiators]

            edited_radiator_df['Diameter'] = [
                r.calculate_diameter(POSSIBLE_DIAMETERS) for r in radiators
            ]
            edited_radiator_df['Diameter'] = edited_radiator_df['Diameter'].max()
            edited_radiator_df['Pressure loss'] = [
                Circuit(
                    length_circuit=row['Length circuit'],
                    diameter=row['Diameter'],
                    mass_flow_rate=row['Mass flow rate']
                ).calculate_pressure_radiator_kv() for _, row in edited_radiator_df.iterrows()
            ]

            collectors = [Collector(name=name) for name in collector_options]
            for collector in collectors:
                collector.update_mass_flow_rate(edited_radiator_df)

            collector_data_updated = edited_collector_df.copy()
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
                radiator_df=edited_radiator_df,
                collector_df=collector_data_updated
            )

            valve = Valve(kv_max=kv_max, n=positions)
            merged_df['Thermostatic valve pressure loss N'] = merged_df['Mass flow rate'].apply(
                valve.calculate_pressure_valve_kv)
            merged_df = valve.calculate_kv_position_valve(merged_df, custom_kv_max=kv_max, n=positions)

            total_pressure_loss_per_circuit = merged_df.groupby('Radiator nr')[
                'Total Pressure Loss'].sum().reset_index()

            total_water_volume = 0
            for _, row in edited_radiator_df.iterrows():
                circuit = Circuit(
                    length_circuit=row['Length circuit'],
                    diameter=row['Diameter'],
                    mass_flow_rate=row['Mass flow rate']
                )
                total_water_volume += circuit.calculate_water_volume() + 7
            for _, row in collector_data_updated.iterrows():
                circuit = Circuit(
                    length_circuit=row['Collector circuit length'],
                    diameter=row['Diameter'],
                    mass_flow_rate=row['Mass flow rate']
                )
                total_water_volume += circuit.calculate_water_volume()

            st.write('### Results')
            st.write(f"**Total Water Volume across heating system:** {total_water_volume:.2f} liters")
            st.write('**Individual Radiator Pressure Loss, Supply Temperature, and Return Temperature**')
            st.dataframe(
                merged_df[['Radiator nr', 'Collector', 'Pressure loss', 'Total Pressure Loss',
                           'Thermostatic valve pressure loss N', 'kv_needed', 'Supply Temperature',
                           'Return Temperature', 'Mass flow rate', 'Diameter']],
                use_container_width=True,
                hide_index=True
            )

            st.write('**Individual Collector results**')
            st.dataframe(
                collector_data_updated[['Collector', 'Collector pressure loss', 'Mass flow rate', 'Diameter']],
                use_container_width=True,
                hide_index=True
            )

            st.write('**Total Pressure Loss per Circuit**')
            st.dataframe(total_pressure_loss_per_circuit, use_container_width=True, hide_index=True)

            plot_pressure_loss(total_pressure_loss_per_circuit)
            plot_thermostatic_valve_position(merged_df)
            plot_mass_flow_distribution(merged_df)
            plot_temperature_heatmap(merged_df)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
