import streamlit as st
import pandas as pd
from typing import List, Dict
from utils.helpers import POSSIBLE_DIAMETERS, Radiator, Circuit, Collector, Valve, validate_data, calculate_weighted_delta_t
from utils.plotting import plot_pressure_loss, plot_thermostatic_valve_position, plot_mass_flow_distribution, plot_temperature_heatmap

def get_user_configuration():
    st.sidebar.header("Configuratie")
    config = {
        "num_radiators": st.sidebar.number_input("Aantal radiatoren", min_value=1, value=3, step=1),
        "num_collectors": st.sidebar.number_input("Aantal collectors", min_value=1, value=1, step=1),
        "positions": st.sidebar.number_input("Aantal klepposities", min_value=1, value=8, step=1),
        "kv_max": st.sidebar.number_input("Valve kv max", min_value=0.5, value=0.7, step=0.01),
        "delta_T": st.sidebar.slider("Delta T (°C)", min_value=3, max_value=20, value=5, step=1),
        "supply_temp_input": st.sidebar.number_input("Aanvoertemperatuur (°C)", value=None, format="%.1f")
    }

    # Nieuwe optie diameter vastzetten
    st.sidebar.markdown("---")
    fix_diameter = st.sidebar.checkbox("Diameter vastzetten voor alle radiatoren")
    if fix_diameter:
        fixed_diameter = st.sidebar.selectbox("Kies diameter (mm)", [12, 14, 16, 18, 20], index=2)
    else:
        fixed_diameter = None

    config["fix_diameter"] = fix_diameter
    config["fixed_diameter"] = fixed_diameter
    return config

def initialize_radiator_data(num_radiators, collector_options):
    data = {
        'Radiator nr': list(range(1, num_radiators + 1)),
        'Collector': [collector_options[0]] * num_radiators,
        'Radiator power 75/65/20': [0.0] * num_radiators,
        'Calculated heat loss': [0.0] * num_radiators,
        'Length circuit': [0.0] * num_radiators,
        'Space Temperature': [20.0] * num_radiators,
        'Extra power': [0.0] * num_radiators,
    }
    return pd.DataFrame(data)

def initialize_collector_data(num_collectors):
    data = {
        'Collector': [f'Collector {i + 1}' for i in range(num_collectors)],
        'Collector circuit length': [0.0] * num_collectors
    }
    return pd.DataFrame(data)

def calculate_radiator_results(df, delta_T, supply_temp_input, fix_diameter=False, fixed_diameter=None):
    radiators = []
    for _, row in df.iterrows():
        radiator = Radiator(
            q_ratio=(row['Calculated heat loss'] - row['Extra power']) / row['Radiator power 75/65/20'] if row['Radiator power 75/65/20'] != 0 else 0,
            delta_t=delta_T,
            space_temperature=row['Space Temperature'],
            heat_loss=row['Calculated heat loss']
        )
        radiators.append(radiator)

    max_supply_temperature = supply_temp_input or max(r.supply_temperature for r in radiators if r.supply_temperature is not None)

    for r in radiators:
        r.supply_temperature = max_supply_temperature
        r.return_temperature = r.calculate_treturn(max_supply_temperature)
        r.mass_flow_rate = r.calculate_mass_flow_rate()

    df['Supply Temperature'] = max_supply_temperature
    df['Return Temperature'] = [r.return_temperature for r in radiators]
    df['Mass flow rate'] = [r.mass_flow_rate for r in radiators]

    if fix_diameter and fixed_diameter is not None:
        df['Diameter'] = fixed_diameter
    else:
        df['Diameter'] = [
            r.calculate_diameter(POSSIBLE_DIAMETERS) for r in radiators
        ]
        df['Diameter'] = df['Diameter'].max()

    df['Pressure loss'] = [
        Circuit(length_circuit=row['Length circuit'], diameter=row['Diameter'], mass_flow_rate=row['Mass flow rate']).calculate_pressure_radiator_kv()
        for _, row in df.iterrows()
    ]

    return df, radiators


def calculate_collector_results(df, collector_options, collector_df):
    collectors = [Collector(name=name) for name in collector_options]
    for collector in collectors:
        collector.update_mass_flow_rate(df)

    collector_df['Mass flow rate'] = [c.mass_flow_rate for c in collectors]
    collector_df['Diameter'] = [c.calculate_diameter(POSSIBLE_DIAMETERS) for c in collectors]
    collector_df['Collector pressure loss'] = [
        Circuit(length_circuit=row['Collector circuit length'], diameter=row['Diameter'], mass_flow_rate=row['Mass flow rate']).calculate_pressure_collector_kv()
        for _, row in collector_df.iterrows()
    ]
    return collector_df

def calculate_full_results(radiator_df, collector_df, kv_max, positions):
    merged_df = Collector(name='').calculate_total_pressure_loss(radiator_df, collector_df)
    valve = Valve(kv_max=kv_max, n=positions)
    merged_df['Valve pressure loss N'] = merged_df['Mass flow rate'].apply(valve.calculate_pressure_valve_kv)
    merged_df = valve.calculate_kv_position_valve(merged_df, custom_kv_max=kv_max, n=positions)
    return merged_df

def display_results(merged_df, collector_df, radiators):
    st.subheader("Resultaten")
    st.dataframe(merged_df, use_container_width=True)
    st.dataframe(collector_df, use_container_width=True)

    plot_pressure_loss(merged_df[['Radiator nr', 'Total Pressure Loss']])
    plot_thermostatic_valve_position(merged_df)
    plot_mass_flow_distribution(merged_df)
    plot_temperature_heatmap(merged_df)

    weighted_delta_t = calculate_weighted_delta_t(radiators, merged_df)
    total_mass_flow_rate = sum(r.mass_flow_rate for r in radiators)

    st.write(f"Weighted Delta T: {weighted_delta_t:.2f} °C")
    st.write(f"Total Mass Flow Rate: {total_mass_flow_rate:.2f} kg/h")

def main():
    config = get_user_configuration()
    collector_options = [f'Collector {i + 1}' for i in range(config["num_collectors"])]

    if 'radiator_data' not in st.session_state:
        st.session_state['radiator_data'] = initialize_radiator_data(config["num_radiators"], collector_options)

    if 'collector_data' not in st.session_state:
        st.session_state['collector_data'] = initialize_collector_data(config["num_collectors"])

    radiator_df = st.data_editor(st.session_state['radiator_data'], key='editable_table', hide_index=True, use_container_width=True)
    collector_df = st.data_editor(st.session_state['collector_data'], key='collector_table', hide_index=True, use_container_width=True)

    if st.button("Bereken drukverlies en temperaturen"):
        try:
            numeric_columns = ['Radiator power 75/65/20', 'Calculated heat loss', 'Length circuit', 'Space Temperature']
            radiator_df[numeric_columns] = radiator_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            if not validate_data(radiator_df):
                st.error("Ongeldige invoer. Controleer de data aub.")
                return

            radiator_df, radiators = calculate_radiator_results(
                radiator_df,
                config["delta_T"],
                config["supply_temp_input"],
                fix_diameter=config["fix_diameter"],
                fixed_diameter=config["fixed_diameter"]
            )
            collector_df = calculate_collector_results(radiator_df, collector_options, collector_df)
            merged_df = calculate_full_results(radiator_df, collector_df, config["kv_max"], config["positions"])

            st.session_state['radiator_df'] = merged_df.copy()
            st.session_state['collector_df'] = collector_df.copy()
            st.session_state['length_circuits'] = radiator_df.set_index('Radiator nr')['Length circuit'].to_dict()
            st.session_state['kv_max'] = config["kv_max"]
            st.session_state['positions'] = config["positions"]

            display_results(merged_df, collector_df, radiators)

        except Exception as e:
            st.error(f"Fout tijdens berekening: {e}")

if __name__ == "__main__":
    main()
