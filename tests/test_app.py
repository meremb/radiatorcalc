import numpy as np
import pandas as pd
import pytest

from utils.helpers import calculate_c, calculate_treturn, calculate_mass_flow_rate, \
    calculate_diameter, merge_and_calculate_total_pressure_loss, \
    calculate_pressure_radiator_kv, calculate_pressure_collector_kv, calculate_pressure_valve_kv, \
    update_collector_mass_flow_rate, calculate_kv_position_valve, calculate_valve_position, validate_data, \
    calculate_position_valve_with_ratio, POSSIBLE_DIAMETERS, calculate_water_volume, calculate_tsupply, \
    calculate_pressure_loss_piping


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        'Radiator power': [100.0, 200.0, 150.0],
        'Length circuit': [10.0, 12.0, 11.0],
        'Space Temperature': [20.0, 16.0, 24.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_merged_df():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        'Radiator nr': [1, 2, 3],
        'Length circuit': [10.0, 12.0, 11.0],
        'Mass flow rate': [62.3346, 278.6284, 161.3589],
        'Thermostatic valve pressure loss N': [770.617, 15396.8443, 5163.7682],
        'Total Pressure Loss': [621.4229, 3416.6006, 1197.1213],
    }
    return pd.DataFrame(data)


def test_validate_data_valid(sample_dataframe):
    """Test validate_data with a valid DataFrame."""
    assert validate_data(sample_dataframe) is True


def test_validate_data_missing_values():
    """Test validate_data with missing values in DataFrame."""
    data = {
        'Radiator power': [100.0, None, 100.0],
        'Mass flow rate': [0.1, 0.1, None],
        'Diameter': [0.02, 0.02, 0.02],
        'Length circuit': [10.0, 10.0, 0.0],
    }
    df = pd.DataFrame(data)
    assert validate_data(df) is False


def test_validate_data_non_positive_values():
    """Test validate_data with non-positive values in DataFrame."""
    data = {
        'Radiator power': [100.0, 0.0, 100.0],
        'Mass flow rate': [0.1, 0.1, 0.0],
        'Diameter': [0.0, 0.02, 0.02],
        'Length circuit': [10.0, 10.0, 0.0],
    }
    df = pd.DataFrame(data)
    assert validate_data(df) is False


def test_supply():
    Q_ratio = 0.65
    delta_T = 5
    space_temperature = 24
    constant_c = calculate_c(Q_ratio,delta_T)
    T_supply_expected = 62.68
    T_supply_calculated = calculate_tsupply(space_temperature, constant_c, delta_T)
    assert pytest.approx(T_supply_expected, rel=1e-3) == T_supply_calculated


def test_return():
    Q_ratio = 0.46
    space_temperature = 20
    max_supply = 63
    T_return_expected = 38.12
    T_return_calculated = calculate_treturn(Q_ratio, space_temperature, max_supply)
    assert pytest.approx(T_return_expected, rel=1e-3) == T_return_calculated


def test_calculate_mass_flow_rate():
    supply_temperature = 63
    return_temperature = 37
    heat_loss = 1156
    mass_flow_expected = 38.29
    mass_flow_calculated = calculate_mass_flow_rate(supply_temperature, return_temperature, heat_loss)
    assert pytest.approx(mass_flow_expected, rel=1e-3) == mass_flow_calculated

def test_calculate_diameter():
    mass_flow_rate = 331
    diameter_expected = 20
    diameter_calculated = calculate_diameter(mass_flow_rate, POSSIBLE_DIAMETERS)
    assert pytest.approx(diameter_expected, rel=1e-2) == diameter_calculated
    mass_flow_rate = 5000
    with pytest.raises(ValueError, match="Calculated diameter exceeds the maximum allowable diameter for mass flow rate:5000"):
        calculate_diameter(mass_flow_rate, POSSIBLE_DIAMETERS)
    mass_flow_rate = np.nan
    with pytest.raises(ValueError, match="The mass flow rate cannot be NaN check the configuration of the number of collectors."):
        calculate_diameter(mass_flow_rate, POSSIBLE_DIAMETERS)


def test_calculate_pressure_loss_piping():
    length_circuit = 8
    mass_flow_rate = 248
    diameter = 14
    pressure_loss_calculated = calculate_pressure_loss_piping(
        length_circuit=length_circuit, mass_flow_rate=mass_flow_rate, diameter=diameter
                                                              )
    assert pytest.approx(pressure_loss_calculated, rel=1e-2) == 1829.26



def test_calculate_pressure_loss_radiator_kv():
    length_circuit = 8
    mass_flow_rate = 248
    diameter = 14
    pressure_loss_expected = 1829+1494
    pressure_loss_calculated = calculate_pressure_radiator_kv(length_circuit, diameter, mass_flow_rate)
    assert pytest.approx(pressure_loss_expected, rel=1e-2) == pressure_loss_calculated


def test_calculate_pressure_loss_collector_kv():
    length_circuit = 6
    mass_flow_rate = 230
    diameter = 20
    pressure_loss_expected = 208 + 26
    pressure_loss_calculated = calculate_pressure_collector_kv(length_circuit, diameter,mass_flow_rate)
    assert pytest.approx(pressure_loss_expected, rel=1e-2) == pressure_loss_calculated


def test_calculate_pressure_valve_kv():
    mass_flow_rate = 248
    pressure_loss_expected = 12197
    pressure_loss_calculated = calculate_pressure_valve_kv(mass_flow_rate)
    assert pytest.approx(pressure_loss_expected, rel=1e-2) == pressure_loss_calculated

def test_collector_mass_flow_rate():
    radiator_data = {
        'Radiator nr': [1, 2, 3, 4],
        'Collector': ['Collector 1', 'Collector 1', 'Collector 2', 'Collector 2'],
        'Mass flow rate': [10.0, 15.0, 20.0, 25.0],
    }
    edited_radiator_df = pd.DataFrame(radiator_data)
    collector_data = {
        'Collector': ['Collector 1', 'Collector 2'],
        'Circuit length': [1, 2]
    }
    edited_collector_df = pd.DataFrame(collector_data)

    edited_collector_df = update_collector_mass_flow_rate(edited_radiator_df, edited_collector_df)
    expected_collector_data = {
        'Collector': ['Collector 1', 'Collector 2'],
        'Circuit length': [1, 2],
        'Mass flow rate': [25.0, 45.0],
    }
    expected_df = pd.DataFrame(expected_collector_data)
    pd.testing.assert_frame_equal(edited_collector_df, expected_df)


def test_merge_and_calculate_total_pressure_loss():
    radiator_data = {
        'Radiator nr': [1, 2, 3],
        'Collector': ['Collector 1', 'Collector 1', 'Collector 2'],
        'Pressure loss': [100.0, 150.0, 20.0]
    }
    radiator_df = pd.DataFrame(radiator_data)
    collector_data = {
        'Collector': ['Collector 1', 'Collector 2'],
        'Collector pressure loss': [220.689, 2.484]
    }
    collector_df = pd.DataFrame(collector_data)
    expected_data = {
        'Radiator nr': [1, 2, 3],
        'Collector': ['Collector 1', 'Collector 1', 'Collector 2'],
        'Pressure loss': [100.0, 150.0, 20.0],
        'Collector pressure loss': [220.689, 220.689, 2.484],
        'Total Pressure Loss': [320.689+2.484 + 350, 370.689+2.484 + 350, 22.484 + 350]
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = merge_and_calculate_total_pressure_loss(radiator_df, collector_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_calculate_kv_position_valve(sample_merged_df):
    merged_df = calculate_kv_position_valve(sample_merged_df)
    position_valve_expected = np.array([4, 8, 6])
    position_valve_calculated = merged_df['Valve position']
    assert np.array_equal(position_valve_calculated, position_valve_expected)


def test_calculate_valve_position(sample_merged_df):
    """Test if we get a valve position of 1 when discriminant is negative"""
    sample_merged_df['kv needed'] = np.array([0.13042, 0.013042, 0.7])
    a = 0.0114
    b = - 0.0086
    c = 0.0446
    position_valve_calculated = calculate_valve_position(a, b, c, sample_merged_df['kv needed'])
    # we still need to ceil the results
    position_valve_calculated = np.ceil(position_valve_calculated)
    position_valve_expected = np.array([3, 1, 8])
    assert np.array_equal(position_valve_calculated, position_valve_expected)


def test_with_custom_thermostatic_valve(sample_merged_df):
    custom_kv_max = 0.67
    n = 9
    merged_df_custom = calculate_kv_position_valve(sample_merged_df, custom_kv_max=custom_kv_max, n=n)
    position_valve_expected_custom = np.array([5, 9, 7])  # Replace with expected values for custom tests case
    position_valve_calculated_custom = merged_df_custom['Valve position']
    assert np.array_equal(position_valve_calculated_custom, position_valve_expected_custom), \
        f"Expected {position_valve_expected_custom}, but got {position_valve_calculated_custom}"


def test_calculate_position_valve_with_ratio(sample_merged_df):
    # first with original kv_max and n positions
    kv_max = 0.7054
    n = 8
    kv_needed = np.array([0.13042, 0.013042, 0.7])
    position_valve_calculated = calculate_position_valve_with_ratio(kv_max=kv_max, n=n, kv_needed=kv_needed)
    position_valve_expected = np.array([3, 1, 8])
    assert np.array_equal(position_valve_calculated, position_valve_expected)
    # now we use same kv as in previous test and we find better approximation for the positions of the valve
    kv_max = 0.67
    n = 9
    kv_needed = np.array([0.14615, 0.71008, 0.38445])
    position_valve_calculated = calculate_position_valve_with_ratio(kv_max=kv_max, n=n, kv_needed=kv_needed)
    position_valve_expected = np.array([4, 9, 7])
    assert np.array_equal(position_valve_calculated, position_valve_expected), \
        f"Expected {position_valve_expected}, but got {position_valve_calculated}"


def test_calculate_water_volume():
    diameter = 14
    circuit_length = 8
    water_pipe_volume_calculated = calculate_water_volume(diameter=diameter, length_circuit=circuit_length)
    water_pipe_volume_measured = 1.23
    assert pytest.approx(water_pipe_volume_measured, rel=1e-2) == water_pipe_volume_calculated



# Sample data for testing
radiator_data = pd.DataFrame({
    'Radiator nr': [1, 2, 3],
    'Collector': ['Collector 1', 'Collector 1', 'Collector 2'],
    'Radiator power': [1500, 1500, 1500],
    'Calculated heat loss': [1000, 1000, 1000],
    'Length circuit': [10, 5, 7],
    'Space Temperature': [20, 20, 20]
})

collector_data = pd.DataFrame({
    'Collector': ['Collector 1', 'Collector 2'],
    'Collector circuit length': [5, 5],
})