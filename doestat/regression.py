# Copyright (C) 2025 Romulo Pires
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
from scipy.stats import norm, t, f, gaussian_kde #library for regression
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from IPython.display import display, HTML
import sys #library for surface graph
import itertools # Calculate matrices
import re # Provides regular expression support
from .auxvalues import Auxvalues


class Regression:
    """
    Class -> Regression(X, y)
    
    A class designed to perform regression analysis, build ANOVA tables, calculate regression coefficients, and create response surface models.
    
    Features:
    ---------
    - ANOVA: Calculates the Analysis of Variance and generates a DataFrame summarizing the results.
    - Regression Plot: Displays the regression plot using the provided matrix data.
    - Residual Plot: Shows the residual plot for the experimental data.
    - Histogram Plot: Displays a histogram plot for the distribution of residuals.
    - Coefficient error plot: Displays the confidence intervals for the calculated regression coefficients, providing a visual representation of their variability and reliability.
    - Response Surface Plot: Displays the response surface for the selected variables along with the corresponding contour plot, providing a visual understanding of the relationship between factors and the response.
    
    Parameters:
    -----------
    X  : matrix
        A matrix representing the factors (effects/interactions) to be analyzed.
    y  : array-like
        A vector or matrix containing the response variable(s).
    yc : array-like, optional (default=None)
        A vector of central points.
    type_matrix : str, optional (default=None)
        Specifies if the design includes interactions to be calculated.
        - "interaction"
    effect_error : str, optional (default=None)
        Specifies the type of effect error to be considered:
        - "cp" -> "Central Points"
        - "replica" -> "Replica"
    selected_factors: list[str], optional (default=None)
        Specifies the factors (effects/interactions) to be analyzed
    regression: str, optional (default = 'quadratic')
        Specifies the type of regression:
        - "quadratic" -> Includes quadratic coefficients and, if the type_matrix='interaction', interaction coefficients.
        - "linear" -> Includes linear coefficients and, if the type_matrix='interaction', interaction coefficients.
   
    Attributes:
    -----------
    Xb : array-like
        Reorganized array for the analyzed factors.
    
    Methods:
    --------
    1. anova:
        Generates and displays:
        - A DataFrame summarizing the analysis of variance.
        - Includes results for the F-test, p-value, and evaluation of null and alternative hypotheses.
        Usage: `doe.Regression(X, y).effect_analysis()`
    2. regression_plot:
        Generates and displays:
        - The regression plot using the provided matrix data.
        Usage: `doe.Regression(X, y).regression_plot`
    3. residual_plot:
        Generates and displays:
        - The residual plot for the experimental data.
        Usage: `doe.Regression(X, y).residual_plot`
    4. histogram_plot:
        Generates and displays:
        - The histogram plot for the distribution of residuals.
        Usage: `doe.Regression(X, y).histogram_plot`   
    5. coefficient_error_plot:
        Generates and displays:
        - The confidence intervals for the calculated regression coefficients.
        Usage: `doe.Regression(X, y).coefficient_error_plot` 
    6. analysis:
        Displays:
        - Regression plot, Residual plot, Histogram plot, Coefficient error plot.
        Usage: `doe.Regression(X, y).analysis` 
    7. coefficient:
        Generates and displays:
        - The table with the model coefficients and if their are significants.  
        Usage: `doe.Regression(X, y).coefficient`      
    8. curve:
        Generates and displays:
        - The curve plot for one variable.
        - A 1D curve model for better visualization.
        Usage: `doe.Regression(X, y).curve()`
    9. surface:
        Generates and displays:
        - The response surface model and corresponding contour plot.  
        - A 3D surface model for better visualization.
        Usage: `doe.Regression(X, y).surface()`      
    10. show_equation:
        Displays:
        - The model equation.  
        Usage: `doe.Regression(X, y).show_equation()`   
    11. find_xy:
        - Finds possible (x, y) values that satisfy the response surface equation for a given z.
        Usage: `doe.Regression(X, y).find_xy()`  

    Notes:
    ------
    - This class is suitable for regression calculations in factorial designs and provides visual and numerical tools for interpreting experimental results.
    - Input data should be formatted appropriately:
      - `X` should represent the coded matrix of factors. Interactions can be calculated within this class.
      - `y` should be the corresponding response vector or matrix.
    """
    
    def __init__(self, x, y, yc=[], type_matrix='no_interaction', order=None, effect_error=None,selected_factors=None,regression='quadratic'):
        self.aux = Auxvalues()
        self.type_matrix = type_matrix
        self.order = order
        self.effect_error = effect_error
        self.regression = regression
        self.X = self.aux.amatrix(x, type_matrix=self.type_matrix, order=self.order)
        self.yc = yc
        self.selected_factors = selected_factors or self.X.columns.tolist()
        self.Xb = self.create_Xb
        self.y_array, self.y,self.exp_error,_,_ = self.aux.avector(x, y, yc, effect_error=self.effect_error)
        

    @property
    def create_Xb(self):
        Xb = self.X[self.selected_factors].copy()
        Xb.insert(0, 'Intercept', 1)

        if self.regression == 'quadratic':
            if self.selected_factors != self.X.columns.tolist(): # Calculate interaction if selected_factor is set
                if self.type_matrix == 'interaction':
                    max_order = len(self.selected_factors) if self.order is None else self.order
                    for r in range(2, max_order + 1):
                        for combination in itertools.combinations(self.selected_factors, r):
                            name_interaction = ' × '.join(combination)
                            Xb[name_interaction] = Xb[list(combination)].prod(axis=1)
        
                for factor in self.selected_factors: # Add the square term with interaction calculated before
                    if ' × ' not in factor:  
                        Xb[f'{factor}²'] = Xb[factor] ** 2
            else:
                for factor in self.selected_factors: # Add square coefficient with interactions calculated in auxiliar class
                    if ' × ' not in factor:
                        Xb.insert(
                        loc=len(Xb.columns),
                        column=f'{factor}²',
                        value=Xb[factor] ** 2
                    )
                    
            return Xb
           
        elif self.regression =='linear':
            if self.selected_factors != self.X.columns.tolist(): # Calculate interaction if selected_factor is set
                if self.type_matrix == 'interaction':
                    max_order = len(self.selected_factors) if self.order is None else self.order
                    for r in range(2, max_order + 1):
                        for combination in itertools.combinations(self.selected_factors, r):
                            name_interaction = ' × '.join(combination)
                            Xb[name_interaction] = Xb[list(combination)].prod(axis=1)    
            else:
                pass
            
            return Xb

    @property
    def __var_coeffs(self): 
        # Matrixes calculations
        self.XbTXinv = np.linalg.inv(np.matmul(self.Xb.T,self.Xb)) # inv(X'*X)
        self.XbTy = np.matmul(self.Xb.T, self.y) # (X'*Y)
        self.b = np.matmul(self.XbTXinv, self.XbTy) # b = inv(X'*X)*(X'*Y)

        # Prediction of the model
        self.y_pred = np.matmul(self.Xb,self.b) # y predict
        self.y_resid = np.array(self.y_pred)[:, np.newaxis] - np.array(self.y_array) # y residual

        # ANOVA variables
        self.rss = np.sum(self.y_resid**2) # Residual Sum of Square
        self.dof_rss = np.array(self.y_array).size - self.Xb.shape[1] # Degree of freedom of RSS
        self.ess = np.sum(((np.array(self.y_pred) - np.mean(self.y_array))**2)*np.array(self.y_array).shape[1]) # Regression Sum of Square or Explained Sum of Square
        self.dof_ess = self.Xb.shape[1] - 1 # Degree of freedom of ESS
        self.tss = np.sum((np.array(self.y_array) - np.mean(self.y_array))**2) # Total Sum of Square
        self.dof_tss = self.dof_ess + self.dof_rss # Degree of freedom of TSS
        if self.effect_error == 'cp':
            self.sspe = np.sum((np.array(self.yc) - np.mean(self.yc))**2)
            self.dof_sspe = len(self.yc) - 1
        else:
            self.sspe = np.sum((np.array(self.y_array) - np.array(self.y)[:, np.newaxis])**2) # Sum of Square of Pure Error
            self.dof_sspe = (np.array(self.y_array).shape[1] - 1) * np.array(self.y_array).shape[0] # Degree of freedom of SSPE
        self.lof = self.rss - self.sspe # Lack of Fit
        self.dof_lof = self.dof_rss - self.dof_sspe # Degree of freedom of LOF
        self.r2 = self.ess/self.tss #  Coefficient of Determination R² (R-squared)
        self.r2max = (self.tss-self.sspe)/self.tss # R² max
        self.r = self.r2**0.5 # R
        self.n = np.array(self.y_array).size # Number of experiments
        self.k = self.Xb.shape[1] - 1 # Number of factors
        self.r_ajust = 1-(((self.n - 1)/(self.n - self.k -1))*(1 - self.r2)) # R² ajust
        self.mse_exp = self.rss/self.dof_rss # Mean Square Error (by residual) Experimental
        self.msr_exp = self.ess/self.dof_ess # Mean Square Regression Experimental
        self.mst_exp = self.tss/self.dof_tss # Mean Total Square Experimental
        self.mse_pe = self.lof/self.dof_lof # Mean Square Error (by residual) Pure Error
        self.msr_pe = self.sspe/self.dof_sspe # Mean Square Regression Pure Error  
        self.ftest_exp = self.msr_exp/self.mse_exp # F-test for Experimental Data
        self.ftab_exp = f.ppf(0.95, self.dof_ess, self.dof_rss) # F-tabulated for Experimental Data
        self.fratio_exp = self.ftest_exp/self.ftab_exp # F ratio for Experimental Data
        self.pvalue_exp = f.sf(self.ftest_exp, self.dof_ess, self.dof_rss) # p-value for Experimental Data
        self.ftest_pe = self.mse_pe/self.msr_pe # F-test for Pure Error Data
        self.ftab_pe = f.ppf(0.95, self.dof_lof, self.dof_sspe) # F-tabulated for Pure Error Data
        self.fratio_pe = self.ftest_pe/self.ftab_pe # F ratio for Pure Error Data
        self.pvalue_pe = f.sf(self.ftest_pe, self.dof_lof, self.dof_sspe) # p-value for Pure Error Data
        
        # Calculation for regression and residual
        if self.pvalue_exp < 0.05 and self.pvalue_pe > 0.05:
            self.t = t.ppf(1 - 0.05/2, self.dof_rss)  # Test t with 95% - Residual --To change for 99% use 0.01
        elif self.pvalue_exp > 0.05 and self.pvalue_pe < 0.05:
            self.t = t.ppf(1 - 0.05/2, self.dof_sspe) # Test t with 95% - Pure Error
        elif self.pvalue_exp < 0.05 and self.pvalue_pe < 0.05:
            self.t = t.ppf(1 - 0.05/2, self.dof_rss)  # Pass in both tests - Residual
        else:
            # Fails in boht tests
            self.t = t.ppf(1 - 0.05/2, self.dof_rss)
            print("Warning: Both p-value tests failed. The model may be poorly adjusted.")
            
        self.variance =  np.diag(self.XbTXinv/len(self.y_array)) * self.mse_exp # NOTE! the matrix was divide by number of replica
        self.error = self.variance**0.5
        self.ci = self.error * self.t # Confidence interval
         
        return self.__dict__
    
    @property
    def anova(self): # ANOVA table
        data = self.__var_coeffs
        # DataFrame ANOVA
        df = pd.DataFrame({
            "Source": ["Factors", "Residual", "Total", "Pure Error","Lack of Fit"],
            "Sum of Squares": [round(data["ess"],3), round(data["rss"],3), round(data["tss"],3), round(data["sspe"],3),
                               round(data["lof"],3)],
            "Df": [data["dof_ess"], data["dof_rss"], data["dof_tss"],data["dof_sspe"] ,data["dof_lof"]],
            "Mean of Squares": [round(data["msr_exp"],3), round(data["mse_exp"],3), round(data["mst_exp"],3),
                                round(data["msr_pe"],3),round(data["mse_pe"],3)],
            "F-Test": [round(data["ftest_exp"],3),"","",round(data["ftest_pe"],3),""],
            "F-Tabulated": [f'F(0.05;{data["dof_ess"]};{data["dof_rss"]}) = {round(data["ftab_exp"], 3)}',
                          "","", 
                          f'F(0.05;{data["dof_lof"]};{data["dof_sspe"]}) = {round(data["ftab_pe"], 3)}',
                          ""],
            "F-Ratio": [round(data["fratio_exp"],3),"","", round(data["fratio_pe"],3),""],
            "p-Value": [f"{data['pvalue_exp']:.4e}","", "", f"{data['pvalue_pe']:.4e}", "" ]
        })
        
        return df

    @property
    def regression_plot(self):
        data = self.__var_coeffs
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y, data["y_pred"], color='blue', s=15)
        plt.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)], color='red', linewidth=1, linestyle='--')
        plt.xlabel('Experimental')
        plt.ylabel('Predict')
        # plt.title('Regression Plot')
        info_text = f"R² = {round(data['r2'],4)}\n"
        info_text += f"R²ₘₐₓ = {round(data['r2max'],4)}\n"
        info_text += f"R²ₐₗᵤₛₜ = {round(data['r_ajust'],4)}"
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('Regression Plot.png',transparent=True)
        plt.show()

    @property
    def residual_plot(self):
        data = self.__var_coeffs
        plt.figure(figsize=(10, 6))
        plt.scatter(data["y_pred"], np.mean(data["y_resid"], axis = 1), color='blue', s=15)
        plt.axhline(0,color='red', linewidth=1, linestyle='--')
        plt.xlabel('Predict')
        plt.ylabel('Residual')
        # plt.title('Residual Plot')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('Residual Plot.png',transparent=True)
        plt.show()

    @property
    def histogram_plot(self):
        data = self.__var_coeffs
        plt.figure(figsize=(10, 6))
        plt.hist(np.mean(data["y_resid"],axis=1), color ='indigo',bins='fd',edgecolor='black')
        plt.ylabel('Density')
        plt.xlabel('Residual')
        # plt.title('Histogram of the Residual')
        plt.tight_layout()
        plt.savefig('Histogram Plot.png',transparent=True)
        plt.show()

    @property
    def coefficient_error_plot(self):
        data = self.__var_coeffs
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.Xb.columns,data["b"] ,data["ci"], fmt='o', linewidth=2, capsize=6, color='blue')
        plt.axhline(0,color='red', linewidth=1, linestyle='--')
        plt.ylabel('Coefficient Values')
        plt.xlabel('Coefficient')
        # plt.title('Regression of Coefficients',fontweight='black')
        for i, label in enumerate(self.Xb.columns):
            plt.annotate(f'{data["b"][i]:.2f}', 
                         (i, data["b"][i]),  
                         textcoords="offset points",  
                         xytext=(10, 0),  
                         ha='left',  
                         fontsize=9,
                         color='black')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid()
        plt.savefig('Coefficient Error Plot.png',transparent=True)
        if self.pvalue_exp > 0.05 and self.pvalue_pe < 0.05:
            print('The model exhibits a lack of fit, and the confidence interval was calculated using the mean square of pure error, based on its degrees of freedom.')
        plt.show()

    @property
    def coefficient(self):
        """
        Displays the table with coefficient
        
        """
        data = self.__var_coeffs
        df = pd.DataFrame({
            "Term":self.Xb.columns,
            "Coefficient":data['b'].round(4),
            "Error": [f"±{error:.4f}" for error in data['ci'].round(4)],
            "Significant": [abs(coef) > error for coef, error in zip(data['b'], data['ci'])]
        })
        return df

    def show_equation(self, significant_coeff=False, exclude_variables=None):
        """
        Displays the model Equation
    
        Parameters:
        -----------
        significant_coeff: bool, optional(default=False)
            If True, equation will be calculated using only the significant coefficients that fall within the confidence interval.
        exclude_variables: list, optional (default=None)
            List of variable names to exclude (set their coefficients to zero).
        """
    
        b_values = self.__var_coeffs['b']
        terms = self.Xb.columns
        coefficients_raw = b_values.round(4)
        errors = self.__var_coeffs['ci']
                
        # Convert to named series for easier manipulation
        b_named = pd.Series(coefficients_raw, index=terms)
        
        # Apply significant coefficient filtering
        if significant_coeff:
            for i, term in enumerate(terms):
                if abs(b_named[term]) <= errors[i]:
                    b_named[term] = 0
                    
        # Exclude variables by zeroing their coefficients
        if exclude_variables:
            valid_variables = [var for var in exclude_variables if var in b_named.index]
            invalid_variables = [var for var in exclude_variables if var not in b_named.index]
    
            if not valid_variables:
                raise ValueError("None of the variables to exclude were found in the model.")
            if invalid_variables:
                raise ValueError(f"The following variables were not found in the model: {', '.join(invalid_variables)}")
    
            b_named[valid_variables] = 0
            display(HTML("<div style='text-align: left; font-weight: bold; font-size: 12px;'>Factors excluded</div>"))
            print(valid_variables)
            print('\n')
    
        equation_parts = []
        intercept = b_named.iloc[0]

        for term, coef in b_named.items():
            if term.lower() == "intercept" or coef == 0: 
                continue
            sign = "+" if coef >= 0 else "-"
            equation_parts.append(f"{sign} {abs(coef)} × {term}")
        
        equation = f"Z = {intercept:.4f} " #+ " ".join(equation_parts)
        if equation_parts:
            equation +=" " + " " .join(equation_parts)
    
        display(HTML(f"<p style='text-align: center; font-weight: bold;font-size:14px;'>{equation}</p>"))
    

    @property
    def ftest_plot(self): 
        data = self.__var_coeffs
        fig = plt.figure(constrained_layout=True,figsize=(10,8))      
        subfigs = fig.subfigures(2,2, wspace=0.07, width_ratios=[1,1])
        #F test Experimantal data
        axs2 = subfigs[0,0].subplots(1, 3)
        axs2[0].bar('MSR/MSE',data["ftest_exp"],color='navy' ,)
        axs2[0].set_title('F-Test \n (Regression)',fontweight='black')
        axs2[1].bar(f'F(0.05;{data["dof_ess"]};{data["dof_rss"]})',data["ftab_exp"],color='navy')
        axs2[1].set_title('F-tabulated',fontweight='black')
        axs2[2].bar('F-Test/ F-tabulated',data["fratio_exp"], color= 'navy')
        axs2[2].set_title('Ratio',fontweight='black',fontsize=12,y=1.031)
        axs2[2].axhline(1,color='w')
                
        #F test Pure error data
        axs1 = subfigs[0,1].subplots(1, 3)
        axs1[0].bar('MSLoF/MSPE',data["ftest_pe"],color='darkred' ,)
        axs1[0].set_title('F-Test \n (Lack of Fit)',fontweight='black')
        axs1[1].bar(f'F(0.05;{data["dof_lof"]};{data["dof_sspe"]})',data["ftab_pe"],color='darkred')
        axs1[1].set_title('F-tabulated',fontweight='black')
        axs1[2].bar('F-Test/ F-tabulated',data["fratio_pe"], color= 'darkred')
        axs1[2].set_title('Ratio',fontweight='black',fontsize=12,y=1.031)
        axs1[2].axhline(1,color='black')

    @property
    def analysis(self):
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Regression Plot</div>")) 
        self.regression_plot
        print('\n')
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Residual Plot</div>")) 
        self.residual_plot
        print('\n')
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Histogram of the Residual</div>")) 
        self.histogram_plot
        print('\n')
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Regression of Coefficients</div>")) 
        self.coefficient_error_plot
        # self.ftest_plot
    
    def curve(self,v=[], resolution = 101, significant_coeff = False, exclude_variables = None):
        """
        Generates and displays the 1D Curve Model.
        
        Parameters:
        -----------
        v : list, optional (default=[])
            A list containing the real minimum and maximum values for the selected variable.  
            Example: `[min_value, max_value]`.
        resolution: int, optional (default=101)
            Controls the precision of the x calculations by defining the number of points in the grid.
            The default value results in a grid of 101 points for calculation.
        significant_coeff: bool, optional(default=False)
            If `True`, the curve will be calculated using only the significant coefficients that fall within the confidence interval.
        exclude_variables : list, optional (default=None)
            List of variable names to exclude (set their coefficients to zero).
    
        Returns:
        --------
        - A graphical representation of the 1D Curve Model.
    
        Example:
        --------
        Generate the curve plot:
        regression.curve(v=[35, 45])
    
        """
        data = self.__var_coeffs
        coefficients = data['b']
        errors = data['ci']
        
        # Check if only significant coefficients should be used
        if significant_coeff:
            # Set coefficients to zero if they are not significant
            coefficients = [coef if abs(coef) > error else 0 for coef, error in zip(coefficients, errors)]

        # Convert to named series for easier manipulation
        b_named = pd.Series(coefficients, index=self.Xb.columns)
        # Exclude variables by zeroing their coefficients
        if exclude_variables:
            valid_variables = [var for var in exclude_variables if var in b_named.index]
            invalid_variables = [var for var in exclude_variables if var not in b_named.index]
    
            if not valid_variables:
                raise ValueError("None of the variables to exclude were found in the model.")
            if invalid_variables:
                raise ValueError(f"The following variables were not found in the model: {', '.join(invalid_variables)}")
    
            b_named[valid_variables] = 0

        # Assign the processed coefficients to the final variable
        b0, b1, b11 = b_named.values
        
        cod_X = {}
        for col in self.selected_factors:
            min_value = self.X[col].min()
            max_value = self.X[col].max()
            cod_X[col] = {"min": min_value, "max": max_value}
        array_cod1 = np.linspace(cod_X[self.selected_factors[0]]['min'] ,cod_X[self.selected_factors[0]]['max'] ,num=resolution)
        x = np.linspace(cod_X[self.selected_factors[0]]['min'] ,cod_X[self.selected_factors[0]]['max'] ,num=resolution)
        self.z = (b0 + b1*x + b11*x**2).round(4)
        
        if v:
            if len(v) == 2:
                array_real1 = np.linspace(v[0],v[1], num=resolution)
                x = np.linspace(v[0],v[1],num=resolution)
            elif len(v) != 2:
                raise ValueError('It is necessary to provide exactly two values for both variable 1 and variable 2.')
        else:
            pass 

        z = self.z
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=z, mode = 'lines', line=dict(color='black', width=1)))

        fig.update_layout(
            xaxis_title=self.selected_factors[0],
            yaxis_title='Z',
            width=800,
            height=600,

            plot_bgcolor='white',
            paper_bgcolor='white',

            font=dict(
                family="Arial",
                size=12,
                color="Black"
            ),

            margin=dict(l=80, r=40, b=80, t=80)
        )

        axis_style = dict(
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
            tickwidth=1,
            tickcolor='black',
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5
        )

        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Response Plot</div>")) 
        
        self.show_equation(significant_coeff=significant_coeff,exclude_variables=exclude_variables)
        # iplot(fig)
        fig.show()


    def surface(self, v1=[], v2=[], resolution = 101, significant_coeff = False, plot3D = False, exclude_variables = None):
        """
        Generates and displays the Response Surface Model.
        
        Parameters:
        -----------
        v1 : list, optional (default=[])
            A list containing the real minimum and maximum values for the first selected variable.  
            Example: `[min_value, max_value]`.
        v2 : list, optional (default=[])
            A list containing the real minimum and maximum values for the second selected variable.  
            Example: `[min_value, max_value]`.
        resolution: int, optional (default=101)
            Controls the precision of the x and y calculations by defining the number of points in the grid.
            The default value results in a grid of 10201 points (101x101) for calculation.
        significant_coeff: bool, optional(default=False)
            If `True`, the surface will be calculated using only the significant coefficients that fall within the confidence interval.
        plot3D : bool, optional (default=False)
            If `True`, generates an interactive 3D Response Surface plot.  
            If `False`, displays the Response Surface and Contour plot.
        exclude_variables : list, optional (default=None)
            List of variable names to exclude (set their coefficients to zero).
    
        Returns:
        --------
        - A graphical representation of the Response Surface.
    
        Example:
        --------
        Generate the surface and contour plot:
        regression.surface(v1=[35, 45], v2=[58, 78])
    
        Generate an interactive 3D surface plot:
        regression.surface(v1=[35, 45], v2=[58, 78], plot3D=True)
    
        """
        data = self.__var_coeffs
        coefficients = data['b']
        errors = data['ci']
        
        # Check if only significant coefficients should be used
        if significant_coeff:
            # Set coefficients to zero if they are not significant
            coefficients = [coef if abs(coef) > error else 0 for coef, error in zip(coefficients, errors)]

        # Convert to named series for easier manipulation
        b_named = pd.Series(coefficients, index=self.Xb.columns)
        # Exclude variables by zeroing their coefficients
        if exclude_variables:
            valid_variables = [var for var in exclude_variables if var in b_named.index]
            invalid_variables = [var for var in exclude_variables if var not in b_named.index]
    
            if not valid_variables:
                raise ValueError("None of the variables to exclude were found in the model.")
            if invalid_variables:
                raise ValueError(f"The following variables were not found in the model: {', '.join(invalid_variables)}")
    
            b_named[valid_variables] = 0

        # Assign the processed coefficients to the final variable
        b0, b1, b2, b12, b11, b22 = b_named.values
        
        cod_X = {}
        for col in self.selected_factors:
            min_value = self.X[col].min()
            max_value = self.X[col].max()
            cod_X[col] = {"min": min_value, "max": max_value}
        array_cod1 = np.linspace(cod_X[self.selected_factors[0]]['min'] ,cod_X[self.selected_factors[0]]['max'] ,num=resolution)
        array_cod2 = np.linspace(cod_X[self.selected_factors[1]]['min'] ,cod_X[self.selected_factors[1]]['max'] ,num=resolution)
        x,y = np.meshgrid(array_cod1,array_cod2)
        self.z = (b0 + b1*x + b2*y + b12*x*y + b11*x**2 + b22*y**2).round(4)
        
        if len(v1) == 2 and len(v2) == 2:
            array_real1 = np.linspace(v1[0],v1[1], num=resolution)
            array_real2 = np.linspace(v2[0],v2[1], num=resolution)
            x,y = np.meshgrid(array_real1,array_real2)
        elif (len(v1) == 1 or len(v2) == 1) or (len(v1) > 2 or len(v2) > 2):
            raise ValueError('It is necessary to provide exactly two values for both variable 1 and variable 2.')

        if plot3D == False:
            z = self.z
            
            # Surface
            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_subplot(1,2,1, projection='3d')
            ax1.plot_surface(x, y, z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
            # ax.set_title('Model Surface', fontweight='black')
            ax1.set_xlabel(self.selected_factors[0], fontsize = 12)
            ax1.set_ylabel(self.selected_factors[1], fontsize = 12)
            ax1.set_zlabel('Z', fontsize=12)
    
            # Countor
            ax2 = fig.add_subplot(1,2,2)
            contours = ax2.contour(x, y, z, 3,colors='black', levels=6)
            ax2.clabel(contours, inline=True, fontsize=12)
            ax2.set_xlabel(self.selected_factors[0], fontsize = 12)
            ax2.set_ylabel(self.selected_factors[1], fontsize = 12)
            # ax2.scatter(x.max(), y.max(), color='darkred',marker=(5, 1),s=100)
            max_index = np.unravel_index(np.argmax(z, axis=None), z.shape)
            ax2.annotate(r'$z_{max}= %.2f$' % z.max().round(0), 
                         (x[max_index], y[max_index]),color='k')
            plt.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()],  cmap='viridis', alpha=1)
            plt.colorbar(aspect=6, pad=.15)
    
            display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Response Surface and Contour Plot</div>")) 
            self.show_equation(significant_coeff=significant_coeff,exclude_variables=exclude_variables)
            plt.tight_layout(w_pad=5)
            plt.savefig('Response Surface and Contour Plot.png',transparent=True)
            plt.show()
            
        else:
            z = self.z
            surface = go.Surface(x=x,y=y,z=z,colorscale='Viridis')
            layout = go.Layout(
                scene = dict(
                    xaxis = dict(title = self.selected_factors[0]),
                    yaxis = dict(title = self.selected_factors[1]),
                    zaxis = dict(title = 'Z')
            ),
            width=1000,
            height=600,
        )
            fig = go.Figure(data = [surface],layout = layout)
            display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Response Surface Plot</div>")) 
            self.show_equation(significant_coeff=significant_coeff)
            # iplot(fig)
            fig.show()

    def find_xy(self, v1=[], v2=[], z = None, tolerance = 0.01, resolution = 101, significant_coeff = False):
        """
        Finds possible (x, y) values that satisfy the response surface equation for a given z.
        
        Parameters:
        -----------
        v1 : list, optional (default=[])
            A list containing the real minimum and maximum values for the first selected variable.
        v2 : list, optional (default=[])
            A list containing the real minimum and maximum values for the second selected variable.
        z : float
            Target z value to find corresponding (x, y).
        tolerance: float, optional (default=0.01)
            Maximum allowable difference between the calculated z value and the target z value.
        resolution: int, optional (default=101)
            Controls the precision of the x and y calculations by defining the number of points in the grid.
            The default value results in a grid of 10201 points (101x101) for calculation.
        significant_coeff : bool, optional (default=False)
            If True, only uses significant coefficients that fall within the confidence    
            
        Returns:
        --------
        DataFrame of x and y values for z target.
    """
        data = self.__var_coeffs
        coefficients = data['b']
        errors = data['ci']
        
        # Check if only significant coefficients should be used
        if significant_coeff:
            # Set coefficients to zero if they are not significant
            coefficients = [coef if abs(coef) > error else 0 for coef, error in zip(coefficients, errors)]
        
        # Assign the processed coefficients to the final variable
        b0, b1, b2, b12, b11, b22 = coefficients

        cod_X = {}
        for col in self.selected_factors:
            min_value = self.X[col].min()
            max_value = self.X[col].max()
            cod_X[col] = {"min": min_value, "max": max_value}
        array_cod1 = np.linspace(cod_X[self.selected_factors[0]]['min'] ,cod_X[self.selected_factors[0]]['max'] ,num=resolution)
        array_cod2 = np.linspace(cod_X[self.selected_factors[1]]['min'] ,cod_X[self.selected_factors[1]]['max'] ,num=resolution)
        x,y = np.meshgrid(array_cod1,array_cod2)
        self.z = (b0 + b1*x + b2*y  + b12*x*y + b11*x**2 + b22*y**2).round(4)
        
        if len(v1) == 2 and len(v2) == 2:
            array_real1 = np.linspace(v1[0],v1[1], num=resolution)
            array_real2 = np.linspace(v2[0],v2[1], num=resolution)
            x,y = np.meshgrid(array_real1,array_real2)
        elif (len(v1) == 1 or len(v2) == 1) or (len(v1) > 2 or len(v2) > 2):
            raise ValueError('It is necessary to provide exactly two values for both variable 1 and variable 2.')

        indices = np.isclose(self.z, z, atol=tolerance)
        
        xy = pd.DataFrame({self.selected_factors[0]: x[indices], self.selected_factors[1]: y[indices]})

        return xy
