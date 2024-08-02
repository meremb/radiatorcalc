import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import math
from typing import List, Dict


possible_diameters = [8, 10, 12, 13, 14, 16, 20, 22, 28, 36]
def calculate_pressure_loss(
        power: float, mass_flow_rate: float, diameter: float,
        length_supply: float, length_return: float) -> float:
    """Calculate the pressure loss based on the given parameters."""
    if diameter > 0 and (length_supply + length_return) > 0:
        pressure_loss: float = (
            power * mass_flow_rate / (diameter * (length_supply + length_return)))
    else:
        pressure_loss = 0
    return pressure_loss


def calculate_c(Q_ratio: float, delta_T: float) -> float:
    """Calculate the constant 'c' based on Q_ratio and delta_T."""
    T_factor = 49.83
    exponent_radiator = 1.34
    c = math.exp(delta_T / T_factor / Q_ratio**(1 / exponent_radiator))
    return c


def calculate_Tsupply(space_temperature: float, constant_c: float, delta_T: float) -> float:
    """Calculate the supply temperature based on space temperature, constant_c, and delta_T."""
    return space_temperature + (constant_c / (constant_c - 1)) * delta_T


def calculate_Treturn(Q_ratio: float, space_temperature: float, max_supply_temperature: float) -> float:
    T_factor = 49.83
    exponent_radiator = 1.34
    return ((Q_ratio**(1/exponent_radiator)*T_factor)**2)/(max_supply_temperature-space_temperature)+space_temperature


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


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title('Radiator Distribution Calculator')
    st.write('Enter the details for each radiator to calculate the total pressure loss and supply/return temperatures.')

    # Sidebar Configuration
    st.sidebar.header('Configuration')
    num_radiators = st.sidebar.number_input('Number of Radiators', min_value=1, value=3, step=1)
    num_collectors = st.sidebar.number_input('Number of Collectors', min_value=1, value=1, step=1)

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
                lambda row: calculate_diameter(row['Mass flow rate'], possible_diameters),
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
                lambda row: calculate_diameter(row['Mass flow rate'], possible_diameters),
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
            merged_df = calculate_kv_position_valve(merged_df=merged_df)

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


def merge_and_calculate_total_pressure_loss(edited_radiator_df: pd.DataFrame, edited_collector_df: pd.DataFrame) -> (
        pd.DataFrame):
    """
    Merge radiator DataFrame with collector DataFrame on 'Collector' column and calculate total pressure loss.

    Parameters:
    - radiator_df (pd.DataFrame): DataFrame containing radiator data with a 'Collector' column and 'Pressure Loss'.
    - collector_df (pd.DataFrame): DataFrame containing collector data with a 'Collector' column and 'Collector Pressure Loss'.

    Returns:
    - pd.DataFrame: Updated DataFrame with total pressure loss.
    """
    merged_df = pd.merge(edited_radiator_df, edited_collector_df[['Collector', 'Collector pressure loss']],
                         on='Collector',
                         how='left')
    # Calculate total pressure loss by adding existing Pressure Loss and Collector Pressure Loss
    merged_df['Total Pressure Loss'] = merged_df['Pressure loss'] + merged_df['Collector pressure loss']
    return merged_df


def calculate_pressure_loss_friction(
        length_supply: float, diameter: float, mass_flow_rate: float, rho=977.7, mu=0.414) -> float:
    """
    Calculate pressure loss in a tube based on friction (Darcy-Weisbach equation).

    Parameters:
    - length: Length of the tube (m)
    - diameter: Inner diameter of the installed tube (mm)
    - mass_flow_rate: Mass flow rate of the fluid (kg/h)
    - rho: Density of the fluid (kg/m³), default: 977.7 (for water at 25°C)
    - mu: Dynamic viscosity of the fluid (Pa·s or N·s/m²), default: 0.414 (for water at 25°C)

    Returns:
    - Pressure loss (Pa)
    """
    mass_flow_rate_seconds = mass_flow_rate / rho / 3600
    area_tube = diameter**2 * np.pi / 4 / 1000000
    velocity = mass_flow_rate_seconds / area_tube

    # Calculate Reynolds number
    Re = rho * velocity * diameter / mu

    # Calculate friction factor (f)
    if Re < 2000:
        f = 64 / Re
    else:
        # For turbulent flow, use empirical correlation or data
        # Example: f = 0.316 / Re**0.25 (for smooth pipes)
        f = 0.316 / Re**0.25

    # Calculate pressure loss using Darcy-Weisbach equation
    delta_p = f * (length_supply / diameter) * (rho * velocity**2 / 2) * 1000

    return delta_p


def calculate_pressure_radiator_kv(length_cicuit: float, diameter: float, mass_flow_rate: float) -> float:
    """Using simplified functions for the kv of a component the pressure loss for the circuit is calculated. """
    pressure_loss_piping = calculate_pressure_loss_piping(diameter, length_cicuit, mass_flow_rate)
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


def calculate_pressure_valve_kv(mass_flow_rate: float) -> float:
    """Calculate pressure loss for thermostatic valve at position N. """
    kv_max_valve_n = 0.7
    pressure_loss_valve = 97180*(mass_flow_rate/1000/kv_max_valve_n)**2
    return pressure_loss_valve


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


def calculate_kv_position_valve(merged_df, custom_kv_max=None, n=None):
    merged_df = merged_df.copy()
    merged_df['Total pressure valve circuit'] = merged_df['Total Pressure Loss'] + merged_df['Thermostatic valve pressure loss N']
    maximum_pressure = max(merged_df['Total pressure valve circuit'])
    merged_df['Pressure difference valve'] = maximum_pressure - merged_df['Total Pressure Loss']
    merged_df['kv_needed'] = (merged_df['Mass flow rate']/1000)/(merged_df['Pressure difference valve']/100000)**0.5
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
        merged_df['Valve position'] = initial_positions.flatten()  # Ensure this is a single-dimensional array

    return merged_df


def calculate_valve_position(a, b, c, kv_needed):
    discriminant = b ** 2 - 4 * a * (c - kv_needed)
    discriminant = np.where(discriminant < 0, 0, discriminant)
    root = -b + np.sqrt(discriminant) / (2 * a)
    root = np.where(discriminant < 0, 1, root)
    result = np.ceil(root)
    return result


def adjust_position_with_custom_values(kv_max, n, kv_needed):
    ratio_kv = kv_needed / 0.7054
    adjusted_ratio_kv = (ratio_kv * kv_max) / kv_max
    ratio_position = np.clip(np.sqrt(adjusted_ratio_kv), 0,1)
    adjusted_position = np.ceil(ratio_position * n)
    return adjusted_position



def validate_data(df: pd.DataFrame) -> bool:
    """Validate the input data to ensure all required fields are correctly filled."""
    required_columns = ['Radiator power', 'Length circuit', 'Space Temperature']
    for col in required_columns:
        if df[col].isnull().any() or (df[col] <= 0).any():
            return False
    return True


def plot_pressure_loss(df: pd.DataFrame) -> None:
    """Plot the total pressure loss per collector using Plotly."""
    fig = px.bar(df, x='Radiator nr', y='Total Pressure Loss',
                 labels={'Radiator nr': 'Radiator nr', 'Total Pressure Loss': 'Total Pressure Loss'})
    fig.update_layout(title='Total Pressure Loss per radiator circuit',
                      xaxis_title='Radiator', yaxis_title='Total Pressure Loss')
    st.plotly_chart(fig)


def plot_thermostatic_valve_position(df: pd.DataFrame) -> None:
    """Plot the valve position for each radiator circuit with improved visualization."""
    # Convert valve positions to discrete values if needed
    df['Valve position'] = df['Valve position'].astype(int)

    # Create the bar plot
    fig = px.bar(
        df,
        x='Radiator nr',
        y='Valve position',
        color='Valve position',  # Color by valve position
        color_continuous_scale=px.colors.sequential.Greens,  # Use a sequential color scale
        labels={'Radiator nr': 'Radiator', 'Valve position': 'Valve Position'},
        title='Valve Position Required to Balance the Circuit',
        text='Valve position'  # Show the valve position on each bar
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Radiator',
        yaxis_title='Valve Position',
        coloraxis_showscale=False  # Hide the color scale since it's redundant
    )

    # Update text display on the bars for better readability
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker_line_color='black',  # Add black borders to the bars
        marker_line_width=1.5
    )

    # Display the plot
    st.plotly_chart(fig)


def plot_mass_flow_distribution(df: pd.DataFrame) -> None:
    """Plot a pie chart of mass flow rate distribution among radiators."""
    fig = px.pie(df, names='Radiator nr', values='Mass flow rate',
                 title='Mass Flow Rate Distribution Among Radiators',
                 labels={'Radiator nr': 'Radiator', 'Mass flow rate': 'Mass Flow Rate'})
    st.plotly_chart(fig)


def plot_temperature_heatmap(df: pd.DataFrame) -> None:
    """Plot a heatmap of supply and return temperatures across radiators."""
    # Prepare the data for the heatmap
    heatmap_data = df[['Radiator nr', 'Supply Temperature', 'Return Temperature']].set_index('Radiator nr')
    heatmap_data = heatmap_data.transpose()  # Transpose to have Temperature Type as rows

    # Create the heatmap using imshow
    fig = px.imshow(heatmap_data,
                    labels={'x': 'Radiator nr', 'y': 'Temperature Type', 'color': 'Temperature (°C)'},
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Bluered')

    fig.update_layout(title='Heatmap of Supply and Return Temperatures')
    st.plotly_chart(fig)



if __name__ == "__main__":
    main()

