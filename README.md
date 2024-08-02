Sure! Below is a sample `README.md` for your project, based on the information you provided and the refactoring steps discussed.

---

# Radiator Performance Analysis

This project is a Streamlit application that calculates and visualizes the performance of radiators based on user inputs such as heat loss, power, space temperature, and circuit length. The app also includes a method to calculate the optimal valve position for a specific thermostatic valve.

## Project Structure

```
your_project/
├── app.py                     # Main Streamlit application file
├── main.py                    # (Optional) Entry point for the application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── utils/                     # Utility modules
│   ├── calculations.py        # Contains functions for radiator performance calculations
│   ├── plotting.py            # Contains functions for generating plots and heatmaps
│   └── valve_position.py      # Contains the function to calculate valve position
└── data/                      # (Optional) Directory for storing data or configuration files
```

### Key Modules
- **`calculations.py`**: Contains functions to calculate various parameters such as constant `c`, supply and return temperatures, mass flow rate, pipe diameter, pressure loss, and required Kv value.
  
- **`plotting.py`**: Contains functions to generate line plots and heatmaps using Plotly.

- **`valve_position.py`**: Contains the `bereken_positie` function to calculate the optimal valve position based on Kv value, maximum Kv of the valve, and the number of valve positions.

## How to Use

### Prerequisites

Ensure you have Python installed, and install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Application

To start the Streamlit application, run the following command:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

### Project Overview

1. **Inputs**: The application takes various user inputs such as heat loss, power, space temperature, circuit length, and others to compute radiator performance metrics.

2. **Calculations**: The calculations for the radiator's performance include determining the supply and return temperatures, mass flow rate, diameter of the pipes, and pressure loss.

3. **Valve Position Calculation**: Based on the calculated Kv value and user-provided maximum Kv and number of valve positions, the optimal valve position is computed using the formula:
   ```python
   def bereken_positie(Kv_nodig, Kv_max, n):
       r_Kv_nodig = Kv_nodig / Kv_max
       r_p = math.sqrt(r_Kv_nodig)
       return math.ceil(r_p * n)
   ```

4. **Visualization**: The application generates interactive plots and heatmaps to visualize the calculated data.

## Example Usage

Here's an example of how to use the `process_radiator` function in `app.py`:

```python
power = 1000
heat_loss = 800
length_circuit = 20
space_temperature = 20
delta_T = 10
Kv_max = 1.5
n = 5

radiator_data = process_radiator(power, heat_loss, length_circuit, space_temperature, delta_T, Kv_max, n)
```

### Customization

You can modify the functions in the `utils/` directory to adjust the calculations or visualizations to suit your specific needs.

## Testing

You can add unit tests for the functions in the `utils/` directory under a `tests/` directory. Use `pytest` for running the tests.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you find a bug or have a feature request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

This `README.md` provides an overview of the project, how to set it up, and how to use it. You can expand it further based on additional details or requirements specific to your project.