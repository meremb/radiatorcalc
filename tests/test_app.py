import numpy as np
import pytest
from utils.helpers import Radiator, POSSIBLE_DIAMETERS, Circuit, Collector, Valve

import pandas as pd
import math


@pytest.fixture
def sample_radiator():
    return Radiator(q_ratio=0.65, delta_t=5, space_temperature=24, heat_loss=1156)

@pytest.fixture
def sample_radiator2():
    return Radiator(q_ratio=0.46, delta_t=5, space_temperature=20, heat_loss=1000)

@pytest.fixture
def sample_circuit():
    return Circuit(length_circuit=8, diameter=14, mass_flow_rate=248)

@pytest.fixture
def sample_collector():
    return Collector(name='Collector 1', pressure_loss=10, mass_flow_rate=1.5)

@pytest.fixture
def sample_valve():
    return Valve(kv_max=0.7, n=9)

@pytest.fixture
def radiator_df():
    data = {
        'Radiator nr': [1, 2, 3],
        'Collector': ['Collector 1', 'Collector 1', 'Collector 2'],
        'Length circuit': [10.0, 12.0, 11.0],
        'Mass flow rate': [62.3346, 278.6284, 161.3589],
        'Thermostatic valve pressure loss N': [770.617, 15396.8443, 5163.7682],
        'Total Pressure Loss': [621.4229, 3416.6006, 1197.1213],
    }
    return pd.DataFrame(data)

@pytest.fixture
def collector_df():
    data = {
        'Collector': ['Collector 1', 'Collector 2'],
        'Collector pressure loss': [220.689, 2.484]
    }
    return pd.DataFrame(data)


def test_calculate_c(sample_radiator):
    constant_c = 1.148
    assert math.isclose(sample_radiator.calculate_c(), constant_c, rel_tol=1e-3)


def test_calculate_tsupply(sample_radiator):
    tsupply = 62.28
    assert math.isclose(sample_radiator.calculate_tsupply(), tsupply, rel_tol=1e-2)


def test_calculate_treturn(sample_radiator2):
    treturn = 38.12
    assert math.isclose(sample_radiator2.calculate_treturn(max_supply_temperature=63), treturn, rel_tol=1e-2)


def test_calculate_pressure_loss_piping(sample_circuit):
    pressure_loss = 1829.26
    assert math.isclose(sample_circuit.calculate_pressure_loss_piping(), pressure_loss, rel_tol=1e-2)


def test_calculate_pressure_radiator_kv(sample_circuit):
    pressure_loss = 1829 + 1494
    assert math.isclose(sample_circuit.calculate_pressure_radiator_kv(), pressure_loss, rel_tol=1e-2)


def test_calculate_water_volume(sample_circuit):
    water_volume = 1.23
    assert math.isclose(sample_circuit.calculate_water_volume(), water_volume, rel_tol=1e-2)


def test_calculate_pressure_collector_kv(sample_circuit):
    pressure_collector = 1857.07
    assert math.isclose(sample_circuit.calculate_pressure_collector_kv(), pressure_collector, rel_tol=1e-2)


def test_update_mass_flow_rate(sample_collector, radiator_df):
    sample_collector.update_mass_flow_rate(radiator_df)
    mass_flow_calculated = 340.96
    assert math.isclose(sample_collector.mass_flow_rate, mass_flow_calculated, rel_tol=1e-2)


def test_calculate_total_pressure_loss(sample_collector):
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
        'Total Pressure Loss': [320.689 + 2.484 + 350, 370.689 + 2.484 + 350, 22.484 + 350]
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = sample_collector.calculate_total_pressure_loss(radiator_df, collector_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_calculate_mass_flow_rate(sample_radiator):
    mass_flow_rate = 38.12
    sample_radiator.supply_temperature = 63
    sample_radiator.return_temperature = 37
    assert math.isclose(sample_radiator.calculate_mass_flow_rate(), mass_flow_rate, rel_tol=1e-2)


def test_calculate_diameter(sample_collector):
    sample_collector.mass_flow_rate = 331
    assert sample_collector.calculate_diameter(possible_diameters=POSSIBLE_DIAMETERS) == 20


def test_calculate_diameter_value_error(sample_radiator):
    sample_radiator.mass_flow_rate = math.nan
    with pytest.raises(ValueError):
        sample_radiator.calculate_diameter(possible_diameters=POSSIBLE_DIAMETERS)


def test_calculate_pressure_valve_kv(sample_valve):
    pressure_valve = 12197
    assert math.isclose(sample_valve.calculate_pressure_valve_kv(mass_flow_rate=248), pressure_valve, rel_tol=1e-2)


@pytest.mark.skip('not_needed_test')
def test_calculate_kv_needed(sample_valve, radiator_df):
    sample_valve.calculate_kv_needed(radiator_df)
    expected_data = {
        'Radiator nr': [1, 2, 3],
        'Collector': ['Collector 1', 'Collector 1', 'Collector 2'],
        'Mass flow rate': [1.5, 2.0],
        'Pressure loss': [15, 20],
        'Total Pressure Loss': [15, 20],
        'Thermostatic valve pressure loss N': [5, 10],
        'Total pressure valve circuit': [20, 30],
        'Pressure difference valve': [5, 10],
        'kv_needed': [0.208, 0.316]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(sample_valve.calculate_kv_needed(radiator_df), expected_df)


def test_calculate_valve_position(sample_valve):
    kv_needed = np.array([0.208, 0.316])
    positions = sample_valve.calculate_valve_position(0.0114, -0.0086, 0.0446, kv_needed)
    expected_positions = np.array([3.81, 4.90])
    assert np.allclose(positions, expected_positions, rtol=1e-2)


def test_adjust_position_with_custom_values(sample_valve):
    kv_needed = np.array([0.208, 0.316])
    adjusted_positions = sample_valve.adjust_position_with_custom_values(kv_needed)
    expected_positions = np.array([5, 7])
    assert np.allclose(adjusted_positions, expected_positions, rtol=1e-2)


def test_calculate_kv_position_valve(sample_valve, radiator_df):
    result_df = sample_valve.calculate_kv_position_valve(radiator_df, custom_kv_max=0.7054, n=8)
    position_valve_calculated = result_df['Valve position']
    position_valve_expected = np.array([4, 8, 6])
    assert np.array_equal(position_valve_calculated, position_valve_expected)


def test_calculate_position_valve_with_ratio(sample_valve):
    kv_needed = np.array([0.13042, 0.013042, 0.7])
    final_positions = sample_valve.calculate_position_valve_with_ratio(kv_max=0.7054, n=8, kv_needed=kv_needed)
    expected_positions = np.array([3, 1, 8])
    assert np.allclose(final_positions, expected_positions, rtol=1e-2)
    kv_needed = np.array([0.14615, 0.71008, 0.38445])
    final_positions = sample_valve.calculate_position_valve_with_ratio(kv_max=0.67, n=9, kv_needed=kv_needed)
    expected_positions = np.array([4, 9, 7])
    assert np.allclose(final_positions, expected_positions, rtol=1e-2)