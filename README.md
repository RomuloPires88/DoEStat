# DoE Statistc (doestat.py)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

A Python implementation for Design of Experiments (DOE), providing tools to optimize processes, analyze factor effects, and evaluate quality metrics in robust engineering applications.

## 📌 Overview
Design of Experiments (DOE) is a powerful approach for systematically planning, conducting, analyzing, and interpreting controlled tests to evaluate factor effects on responses. This package offers two distinct yet complementary methods to support experimental design and analysis:

1 - Taguchi Method:
The `Taguchi` approach provides a robust framework for process optimization and quality improvement. By minimizing the variability in responses and optimizing the mean performance, this method is especially effective in manufacturing, engineering, and other applied sciences. Key features include:
- Evaluation of quality metrics using **Signal-to-Noise Ratios (S/N)** for "Bigger is Better," "Smaller is Better," and "Nominal is Better" objectives.
- Handling of multiple responses via techniques like the **Multi-Response Performance Index (MRPI)**.
- Visualization of factor effects, interaction graphs, and predictive capabilities for optimizing responses.
- Statistical tools such as Analysis of Variance **(ANOVA)** for variance decomposition and factor impact assessment.

2 - Factorial Design Analysis:
`Factorial designs` are fundamental for understanding the interactions and contributions of multiple factors in experimental studies. This method focuses on:
- Calculation and visualization of factorial effects, including main effects and interactions.
- Representation of effects using Gaussian-based probability plots and percentage contribution charts to reveal the most influential factors.
- Support for error estimation using t-Student distribution, enhancing the reliability of conclusions.
- Compatibility with both simple and interaction-inclusive experimental designs.

## ✨ Features
### Key Functionalities
#### Taguchi Method
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

#### Factorial Design Analysis
- **Factorial Effect Calculation**: Quantify the contributions of main effects and interactions in factorial designs.
- **Probability Plot Visualization**: Represent effects in Gaussian-based probability plots to identify significant factors.
- **Percentage Contribution Analysis**: Display and quantify the relative influence of each factor or interaction on overall variability.
- **Error Estimation**: Compute effect errors and confidence intervals using the t-Student distribution for robustness.
- **Flexible Design Support**: Analyze coded factorial matrices with or without interactions.
- **Regression Plot Visualization**: Displays the regression plot using the provided matrix data.
- **Residual Plot Visualization**: Shows the residual plot for the experimental data.
- **Histogram Plot Visualization**: Displays a histogram plot for the distribution of residuals.
- **Coefficient Error Plot Visualization**: Displays the confidence intervals for the calculated regression coefficients, providing a visual representation of their variability and reliability.
- **Response Surface**: Generate s surface response and contour plot with coded data or real values.
- **Model Equation**: Displays the model equation.

## 🔧 Installation
Ensure you have Python and the necessary dependencies installed. This class requires standard scientific libraries like `numpy` and `matplotlib`.
```bash
pip install pandas numpy matplotlib.pyplot plotly.graph_objs scipy.stats seaborn IPython.display sys itertools re
```
Recommended Stable Versions
```bash
pandas:     2.3.1
numpy:      1.26.4
matplotlib: 3.10.5
plotly:     5.24.1
scipy:      1.13.1
seaborn:    0.13.2
IPython:    8.27.0
Python:     3.12.7
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
#### `vector_y`
- Returns the processed response vector/matrix.
- **Usage**:
  ```python
  doe.Taguchi(X, y).vector_y()
  ```

#### `effect_analysis`
- Calculates the total response for each factor level and visualizes their effects.
- **Usage**:
  ```python
  doe.Taguchi(X, y).effect_analysis
  ```

#### `check_interactions()`
- Displays interaction graphs between selected factors.
- **Usage**:
  ```python
  doe.Taguchi(X, y).check_interactions()
  ```

#### `prev()`
- Predicts the response values for selected factors or interactions.
- **Usage**:
  ```python
   doe.Taguchi(X, y).prev()
  ```

#### `anova()`
- Performs Analysis of Variance (ANOVA) and returns a summary table of variance contributions.
- **Usage**:
  ```python
  doe.Taguchi(X, y).anova()
  ```

## 📎 Notes
- The input data **X** and **y** must be formatted correctly:
  - **X**: Matrix representing the experimental design.
  - **y**: Response vector or matrix with dimensions consistent with **X**.
- For MRPI with multiple responses, ensure quality characteristics (**r1, r2**) and methods (`"wgt"` or `"env"`) are appropriately defined.

## 🚀 Example Usage
Here is an example to demonstrate the functionality of the `Taguchi` class:

```python
df = pd.read_excel('test.xlsx', sheet_name = 'teste1')
X = df.iloc[:,1:8]
y = df.iloc[:,8:]

print('DataFrame')
print(df)
print('Matrix X')
print(X)
print('Vector y')
print(y)
```
```python
DataFrame 
      Exp   A  B  C  D  E  F  G  R1  R2
0        1  1  1  1  1  1  1  1  11  11
1        2  1  1  1  2  2  2  2   4   4
2        3  1  2  2  1  1  2  2   4  10
3        4  1  2  2  2  2  1  1   4   8
4        5  2  1  2  1  2  1  2   9   4
5        6  2  1  2  2  1  2  1   4   3
6        7  2  2  1  1  2  2  1   1   4
7        8  2  2  1  2  1  1  2  10   8 
 Matrix X 
    A  B  C  D  E  F  G
0  1  1  1  1  1  1  1
1  1  1  1  2  2  2  2
2  1  2  2  1  1  2  2
3  1  2  2  2  2  1  1
4  2  1  2  1  2  1  2
5  2  1  2  2  1  2  1
6  2  2  1  1  2  2  1
7  2  2  1  2  1  1  2 
 Vector y 
    R1  R2
0  11  11
1   4   4
2   4  10
3   4   8
4   9   4
5   4   3
6   1   4
7  10   8
```

```python
doe.Taguchi(X,y).effect_analysis
```

![imagem](https://github.com/user-attachments/assets/4c6c51bb-4253-4764-8256-afa375229ebd)

![imagem](https://github.com/user-attachments/assets/31d81a18-79fe-4343-9b54-bca1cf69724c)

![imagem](https://github.com/user-attachments/assets/8373f369-2a03-471a-adc9-ac570bad62af)

```python
doe.Taguchi(X,y).prev('F-1,E-1,A-1')
```
```python
The predict value is:
10.38
```

```python
doe.Taguchi(X,y).anova(method='Replica')
```

![imagem](https://github.com/user-attachments/assets/544606c3-1d3b-4259-9a03-4cc5b3af3c66)

### Class: `Auxvalues`
`A auxiliar class that is used by others Classes`

### Class: `Analysis`
#### Parameters:
- **X** *(matrix)*: Matrix representing the factors (effects/interactions) to be analyzed.
- **y** *(array-like)*: Vector or matrix containing the response variable(s).
- **yc** *(array-like, optional)*: Vector of central points (default: None).
- **type_matrix** *(str, optional)*: Specifies if the design includes interactions to be calculated (default: None):
  - `"interaction"`
- **effect_error** *(str, optional)*: Specifies the type of effect error to be considered (default: None):
  - `"cp"` → Central Points
  - `"replica"` → Replica

#### Attributes:
- **matrix_x** *(array-like)*: Experimental design matrix, including factors and interactions.
- **vector_y** *(array-like)*: Response vector or the mean of replicates.
- **error** *(float)*: Calculated effect error.

### Methods:
`matrix_x`
- Generates design matrix, including factors and interactions
- **Usage**:
  ```python
   doe.Analysis(X, y).matrix_x
  ```
`vector_y`
- Show the response vector or the mean of replicates
- **Usage**:
  ```python
   doe.Analysis(X, y).vector_y
  ```
`error`
- Show the error effect
- **Usage**:
  ```python
   doe.Analysis(X, y).error
  ```

`effect_analysis()`
- Analyzes the effects of factors and optionally excludes specified variables.
- Generates:
  - Probability Effects Plot: Visualizes the Gaussian distribution of effects, with confidence intervals.
  - Percentage Effects Plot: Displays the contribution of each effect to the overall variability as a horizontal bar plot.
- **Usage**:
  ```python
   doe.Analysis(X, y).effect_analysis()
  ```
  
# 📎Notes:
- This class is ideal for factorial designs and provides tools to interpret experimental results. Input Requirements:
  - X: Matrix of coded factors (interactions can be generated within the class).
  - y: Response vector or matrix (should be formatted appropriately).
- Effect Error and Confidence Intervals: Calculated using central points or replicates, leveraging the t-Student distribution for robustness.
- Graphs:
  - Automatically saved as image files.

## 🚀 Examples Usage
Here is an example to demonstrate the functionality of the `Analysis` class:
```python
doe.Analysis(X,y,yc,type_matrix='interaction', effect_error='cp').effect_analysis()
```
![imagem](https://github.com/user-attachments/assets/ecc1d17a-f2d4-4c92-b2c4-be5a614897bd)

![imagem](https://github.com/user-attachments/assets/2eca9cf8-341f-47a2-ab0e-fee4bf973b25)

### Class: `Regression`
#### Parameters:
- **X** *(matrix)*: Matrix representing the factors (effects/interactions) to be analyzed.
- **y** *(array-like)*: Vector or matrix containing the response variable(s).
- **yc** *(array-like, optional)*: Vector of central points (default: None).
- **type_matrix** *(str, optional)*: Specifies if the design includes interactions to be calculated (default: None):
  - `"interaction"`
- **effect_error** *(str, optional)*: Specifies the type of effect error to be considered (default: None):
  - `"cp"` → Central Points
  - `"replica"` → Replica
- **selected_factors** *(str, optional (default=None))*: Specifies the factors (effects/interactions) to be analyzed
- **regression** *(str, optional (default = 'quadratic'))*: Specifies the type of regression:
  - `"quadratic"` → Includes quadratic coefficients and, if the type_matrix='interaction', interaction coefficients.
  - `"linear"` → Includes linear coefficients and, if the type_matrix='interaction', interaction coefficients.

#### Attributes:
- **Xb** *(array-like)*: Reorganized array for the analyzed factors.

- Generates design matrix, including factors and interactions
- **Usage**:
  ```python
   doe.Analysis(X, y).matrix_x
  ```

#### Methods:
`anova`
- Generates and displays a DataFrame summarizing the analysis of variance, includes results for the F-test, p-value, and evaluation of null and alternative hypotheses.
- **Usage**:
  ```python
    doe.Regression(X, y).effect_analysis()
  ```

`regression_plot`
- Generates and displays the regression plot using the provided matrix data.
- **Usage**:
  ```python
   doe.Regression(X, y).regression_plot
  ```
`residual_plot`
- Generates and displays the residual plot for the experimental data.
- **Usage**:
  ```python
   doe.Regression(X, y).residual_plot
  ```

`histogram_plot`
- Generates and displays the histogram plot for the distribution of residuals.
- **Usage**:
  ```python
   doe.Regression(X, y).histogram_plot
  ```

`coefficient_error_plot`
- Generates and displays the confidence intervals for the calculated regression coefficients.
- **Usage**:
  ```python
   doe.Regression(X, y).coefficient_error_plot
  ```
  
`analysis`
- Displays Regression plot, Residual plot, Histogram plot, Coefficient error plot.
- **Usage**:
  ```python
   doe.Regression(X, y).analysis
  ```

`coefficient`
- Generates and displays the table with the model coefficients and if their are significants.
- **Usage**:
  ```python
   doe.Regression(X, y).coefficient
  ```

`curve()`
- Generates and displays a 1D graph model for better visualization.
- **Usage**:
  ```python
   doe.Regression(X, y).curve()
  ```

`surface()`
- Generates and displays the response surface model and corresponding contour plot or a 3D surface model for better visualization.
- **Usage**:
  ```python
   doe.Regression(X, y).surface()
  ```

`show_equation()`
- Displays the model equation.
- **Usage**:
  ```python
   doe.Regression(X, y).show_equation()
  ```

`find_xy()`
- Finds possible (x, y) values that satisfy the response surface equation for a given z.
- **Usage**:
  ```python
   doe.Regression(X, y).find_xy()
  ```

## 🚀 Examples Usage
Here is an example to demonstrate the functionality of the `Regression` class:
  ```python
   doe.Regression(X,y,yc, type_matrix='interaction', order=2, effect_error='cp').analysis
  ```
![imagem](https://github.com/user-attachments/assets/6ac5f2bf-b322-4121-92f0-81fa966991aa)

![imagem](https://github.com/user-attachments/assets/6b5c3238-e40a-4fd7-b0e6-a5e7c049f287)

![imagem](https://github.com/user-attachments/assets/9f6e0cff-5cd2-4bc0-876f-0b0f6c66b325)

![imagem](https://github.com/user-attachments/assets/7e3209ad-b9aa-4852-af70-bc0654d58a57)

  ```python
    v1=[70,85]
    v2=[6,13.83]
    doe.Regression(X,y,yc, type_matrix='interaction', order=2, effect_error='cp', selected_factors=['Temperature','Alcohol/oil']).surface(v1=v1,v2=v2,plot3D=True, significant_coeff=True)
  ```
![imagem](https://github.com/user-attachments/assets/4bba62a7-6df0-4aea-978b-b1560fa126b4)

  ```python
    v=[16,30]
    doe.Regression(X4,yzn,yczn,type_matrix='interaction',effect_error='cp',regression='quadratic',selected_factors=['time/min']).curve(v=v)
  ```
<img width="799" height="583" alt="imagem" src="https://github.com/user-attachments/assets/02612441-4eb1-4c67-8a3c-737e0e7c9e57" />


# 📎Notes:
- This class is suitable for regression calculations in factorial designs and provides visual and numerical tools for interpreting experimental results. Input data should be formatted appropriately:
  - X: should represent the coded matrix of factors. Interactions can be calculated within this class.
  - y: should be the corresponding response vector or matrix.
- Graphs:
  - Automatically saved as image files.
  
## License
Copyright (C) 2025 Romulo Pires

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
