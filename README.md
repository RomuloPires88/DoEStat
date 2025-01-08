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

## Notes
- The input data **X** and **y** must be formatted correctly:
  - **X**: Matrix representing the experimental design.
  - **y**: Response vector or matrix with dimensions consistent with **X**.
- For MRPI with multiple responses, ensure quality characteristics (**r1, r2**) and methods (`"wgt"` or `"env"`) are appropriately defined.

## Example Usage
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
`effect_analysis(exclude_variables=None)`
- Analyzes the effects of factors and optionally excludes specified variables.
- Generates:
  - Probability Effects Plot: Visualizes the Gaussian distribution of effects, with confidence intervals.
  - Percentage Effects Plot: Displays the contribution of each effect to the overall variability as a horizontal bar plot.
- **Usage**:
  ```python
   doe.Analysis(X, y).effect_analysis(exclude_variables=['A', 'B'])
  ```
# Notes:
- This class is ideal for factorial designs and provides tools to interpret experimental results. Input Requirements:
  - X: Matrix of coded factors (interactions can be generated within the class).
  - y: Response vector or matrix (should be formatted appropriately).
- Effect Error and Confidence Intervals: Calculated using central points or replicates, leveraging the t-Student distribution for robustness. Graphs:
  - Automatically saved as image files.
 
## Examples
```python
doe.Analysis(X,y,yc,type_matrix='interaction', effect_error='cp').effect_analysis()
```
![imagem](https://github.com/user-attachments/assets/ecc1d17a-f2d4-4c92-b2c4-be5a614897bd)

![imagem](https://github.com/user-attachments/assets/7142cbf4-825b-416c-a9d0-311973faaaae)



## License

## Acknowledgments
This class is inspired by the robust statistical methods of the **Taguchi Method**, widely used in quality engineering to optimize processes and improve product quality.



