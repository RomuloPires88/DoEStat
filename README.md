# doe

# Taguchi DOE Class

A Python implementation of the **Taguchi Method** for Design of Experiments (DOE), providing tools to optimize processes, analyze factor effects, and evaluate quality metrics in robust engineering applications.

## Overview
The `Taguchi` class offers a streamlined way to perform DOE using the Taguchi method, including support for single and multi-response problems. It provides robust statistical techniques such as **Signal-to-Noise (S/N) Ratios**, **Multi-Response Performance Index (MRPI)**, effect analysis, interaction checks, response prediction, and ANOVA.

## Features
### Key Functionalities
- **Signal-to-Noise (S/N) Ratios**: Evaluate quality characteristics with:
  - "Bigger is better"
  - "Smaller is better"
  - "Nominal is better"
- **Multi-Response Performance Index (MRPI)**: Handle multiple responses with:
  - Weighted methods (`"wgt"`)
  - Envelopment methods (`"env"`)
- **Effect Analysis**: Analyze and visualize the influence of factors and interactions.
- **Interaction Checks**: Display interaction graphs to identify dependencies.
- **Prediction**: Predict responses for specific factor combinations.
- **ANOVA**: Perform Analysis of Variance to understand variance contributions.

## Installation
Ensure you have Python and the necessary dependencies installed. This class requires standard scientific libraries like `numpy` and `matplotlib`.
```bash
pip install numpy matplotlib
```

## Class Reference

### Class: `Taguchi`
#### Parameters:
- **X** *(matrix)*: Experimental design matrix containing factors (effects/interactions).
- **y** *(array-like)*: Response vector or matrix. Supports single or multiple responses.
- **T** *(float, optional)*: Target value for "Nominal is better" S/N ratio (default: 0).
- **sn** *(str, optional)*: Quality characteristic for S/N ratio:
  - `"max"` → Bigger is better
  - `"min"` → Smaller is better
  - `"target"` → Nominal is better
- **mrpi** *(str, optional)*: Method for multi-response evaluation:
  - `"wgt"` → Weighted method
  - `"env"` → Envelopment method
- **r1, r2** *(str, optional)*: Quality characteristics for multiple responses:
  - `"max"` → Bigger is better
  - `"min"` → Smaller is better
  - `"target"` → Nominal is better

#### Attributes:
- **y** *(array-like)*: Processed response vector/matrix after applying S/N ratios or MRPI.
- **vector_y** *(array-like)*: Final response vector/matrix ready for analysis.

### Methods:
#### `vector_y()`
- Returns the processed response vector/matrix.
- **Usage**:
  ```python
  doe = Taguchi(X, y)
  response = doe.vector_y()
  ```

#### `effect_analysis()`
- Calculates the total response for each factor level and visualizes their effects.
- **Usage**:
  ```python
  doe.effect_analysis()
  ```

#### `check_interactions()`
- Displays interaction graphs between selected factors.
- **Usage**:
  ```python
  doe.check_interactions()
  ```

#### `prev()`
- Predicts the response values for selected factors or interactions.
- **Usage**:
  ```python
  predicted_response = doe.prev()
  ```

#### `anova()`
- Performs Analysis of Variance (ANOVA) and returns a summary table of variance contributions.
- **Usage**:
  ```python
  anova_table = doe.anova()
  ```

## Notes
- The input data **X** and **y** must be formatted correctly:
  - **X**: Matrix representing the experimental design.
  - **y**: Response vector or matrix with dimensions consistent with **X**.
- For MRPI with multiple responses, ensure quality characteristics (**r1, r2**) and methods (`"wgt"` or `"env"`) are appropriately defined.

## Example Usage
Here is an example to demonstrate the functionality of the `Taguchi` class:

```python
import numpy as np
from your_module import Taguchi  # Replace 'your_module' with the file name

# Experimental Design Matrix (X)
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2]
])

# Response Vector (y)
y = np.array([15, 20, 30, 35])

# Initialize Taguchi Class
doe = Taguchi(X, y, sn="max")

# Processed Response Vector
y_processed = doe.vector_y()
print("Processed Response:", y_processed)

# Effect Analysis
doe.effect_analysis()

# Interaction Check
doe.check_interactions()

# Response Prediction
predicted = doe.prev()
print("Predicted Response:", predicted)

# ANOVA Analysis
anova_table = doe.anova()
print("ANOVA Table:\n", anova_table)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This class is inspired by the robust statistical methods of the **Taguchi Method**, widely used in quality engineering to optimize processes and improve product quality.
