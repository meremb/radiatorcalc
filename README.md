# Radiator Distribution Calculator

This project is a Streamlit application that calculates and visualizes the performance of radiators in a heating system. It evaluates pressure losses, supply and return temperatures, mass flow rates, and pipe diameters based on user inputs such as radiator power, calculated heat loss, circuit length, and space temperature. 

The tool also provides the option to manually set the supply temperature or calculate it automatically to optimize radiator performance and energy efficiency.

---

## Project Structure

```
radiatorcalc/
├── app.py                     # Main Streamlit application file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── utils/                     # Utility modules for calculations and plotting
    ├── helpers.py             # Contains classes and helper functions for calculations
    └── plotting.py            # Contains functions for generating plots
```

---

## Key Modules

### `app.py`
The main Streamlit application that:
- Collects user inputs for radiators, collectors, and valve configurations.
- Validates input data for correctness.
- Calculates supply and return temperatures, mass flow rates, pipe diameters, and pressure losses.
- Visualizes the results with interactive plots and heatmaps using Plotly.
- Displays warnings and suggestions if the supply temperature is too low for the selected radiators.

### `utils/helpers.py`
Contains the following classes and methods:
- `Radiator`: Calculates supply and return temperatures, mass flow rate, pipe diameter, and pressure loss for each radiator.
- `Circuit`: Computes pressure loss and water volume in each heating circuit.
- `Collector`: Aggregates mass flow rate and calculates pressure losses for collectors.
- `Valve`: Determines thermostatic valve positions and calculates pressure losses across the valve.
- `validate_data()`: Checks input data for completeness and correctness.

### `utils/plotting.py`
- Utilizes Plotly to create the following visualizations:
  - Pressure loss per circuit
  - Thermostatic valve positions
  - Mass flow distribution
  - Temperature heatmap

---

## How to Use

### Prerequisites

Ensure you have Python installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

---

## Features

1. **User Inputs**: Collects inputs for:
   - Number of radiators and collectors
   - Radiator power and calculated heat loss
   - Circuit lengths and space temperatures
   - Valve positions and maximum Kv values
   - Optional supply temperature for boundary conditions

2. **Calculations**:
   - Supply and return temperatures
   - Mass flow rates and pipe diameters
   - Pressure losses across radiators, circuits, and collectors
   - Thermostatic valve positions

3. **Visualization**:
   - Interactive Plotly charts to display pressure losses, valve positions, mass flow distribution, and temperature heatmaps.

4. **Error Handling**:
   - Input validation to catch and display errors.
   - Suggestions for adjustments if supply temperature constraints are not met.

---

## Example Usage

Here's an example of how a `Radiator` object is created and used for calculations:

```python
from utils.helpers import Radiator

radiator = Radiator(
    q_ratio=0.8, 
    delta_t=5, 
    space_temperature=20, 
    heat_loss=800
)

supply_temp = radiator.supply_temperature
return_temp = radiator.calculate_treturn(supply_temp)
mass_flow_rate = radiator.calculate_mass_flow_rate()
diameter = radiator.calculate_diameter([10, 12, 15, 18, 22])
```

---

## Customization

- Modify classes in `utils/helpers.py` to adjust calculation methods.
- Adjust visualizations in `utils/plotting.py` for different styles or additional plots.

---

## Testing

To test the application:
- Add unit tests in a `tests/` directory using `pytest`.
- Run tests with:

```bash
pytest tests/
```

---

## Contributing

Contributions are welcome! Feel free to submit a Pull Request or open an Issue for bugs or feature requests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

