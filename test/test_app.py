import numpy as np
import pandas as pd
import pytest
from app import validate_data, calculate_pressure_loss_friction, calculate_c, calculate_Treturn, \
    calculate_pressure_loss, calculate_mass_flow_rate, calculate_diameter, calculate_pressure_radiator_kv, \
    update_collector_mass_flow_rate, merge_and_calculate_total_pressure_loss, calculate_pressure_collector_kv, \
    calculate_pressure_valve_kv, calculate_valve_position, calculate_kv_position_valve


@pytest.mark.parametrize(
    "power, mass_flow_rate, diameter, length_supply, length_return, expected",
    [
        (100.0, 0.1, 0.02, 10.0, 10.0, 25.0),  # Simple case
        (0.0, 0.1, 0.02, 10.0, 10.0, 0.0),     # Zero power
        (100.0, 0.0, 0.02, 10.0, 10.0, 0.0),   # Zero mass flow rate
        (100.0, 0.1, 0.0, 10.0, 10.0, 0.0),    # Zero diameter
        (100.0, 0.1, 0.02, 0.0, 0.0, 0.0),     # Zero length supply and return
        (100.0, 0.1, 0.02, 10.0, 0.0, 50.0),   # Zero length return
        (100.0, 0.1, 0.02, 0.0, 10.0, 50.0),   # Zero length supply
    ]
)
def test_calculate_pressure_loss(
    power: float, mass_flow_rate: float, diameter: float,
    length_supply: float, length_return: float, expected: float
) -> None:
    """Test the calculate_pressure_loss function."""
    result = calculate_pressure_loss(power, mass_flow_rate, diameter, length_supply, length_return)
    assert result == pytest.approx(expected, rel=1e-2), (
        f"Expected {expected}, got {result}"
    )

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
        'kv needed': [0.13042, 0.013042, 0.7],
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


@pytest.mark.parametrize('length_supply, diameter, mass_flow_rate, rho, mu, expected_loss', [
    (10.0, 8, 50.0, 1000.0, 0.4, 1750),   # Expected loss for first row
    (15.0, 12, 120.5, 1100.0, 0.5, 1713),  # Expected loss for second row
    (20.0, 16, 300.0, 1200.0, 0.414, 2513)   # Expected loss for third row
])
def test_calculate_pressure_loss_from_dataframe(length_supply, diameter, mass_flow_rate, rho, mu, expected_loss):
    """Test calculate_pressure_loss_friction using values from sample_dataframe."""
    pressure_loss = calculate_pressure_loss_friction(length_supply, diameter, mass_flow_rate, rho, mu)
    assert pytest.approx(pressure_loss, rel=1e-3) == expected_loss

def test_supply():
    Q_ratio = 0.65
    delta_T = 5
    space_temperature = 24
    constant_c = calculate_c(Q_ratio,delta_T)
    T_supply_expected = 62.68
    T_supply_calculated = space_temperature + (constant_c/(constant_c-1))*delta_T
    assert pytest.approx(T_supply_expected, rel=1e-3) == T_supply_calculated


def test_return():
    Q_ratio = 0.46
    space_temperature = 20
    max_supply = 63
    T_return_expected = 38.12
    T_return_calculated = calculate_Treturn(Q_ratio,space_temperature, max_supply)
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
    possible_diameters = [8, 10, 12, 13, 14, 16, 20, 26]
    diameter_calculated = calculate_diameter(mass_flow_rate, possible_diameters)
    assert pytest.approx(diameter_expected, rel=1e-2) == diameter_calculated
    mass_flow_rate = 5000
    with pytest.raises(ValueError, match="Calculated diameter exceeds the maximum allowable diameter for mass flow rate:5000"):
        calculate_diameter(mass_flow_rate, possible_diameters)
    mass_flow_rate = np.nan
    with pytest.raises(ValueError, match="The mass flow rate cannot be NaN check the configuration of the number of collectors."):
        calculate_diameter(mass_flow_rate, possible_diameters)




def test_calculate_pressure_loss_radiator_kv():
    length_circuit = 8
    mass_flow_rate = 248
    diameter = 14
    pressure_loss_expected = 1829+1494
    pressure_loss_calculated = calculate_pressure_radiator_kv(length_circuit, diameter, mass_flow_rate)
    assert pytest.approx(pressure_loss_expected, rel=1e-2) == pressure_loss_calculated


def test_calculate_pressure_loss_collector_kv():
    length_circuit = 6
    mass_flow_rate = 287
    diameter = 16
    pressure_loss_expected = 953+40+200
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
        'Total Pressure Loss': [320.689, 370.689, 22.484]
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
    a = 0.0114
    b = - 0.0086
    c = 0.0446
    position_valve_calculated = calculate_valve_position(a, b, c, sample_merged_df['kv needed'])
    position_valve_expected = np.array([3, 1, 8])
    assert np.array_equal(position_valve_calculated, position_valve_expected)

