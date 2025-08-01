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
from IPython.display import display, HTML
import sys #library for surface graph

class Taguchi:
    """
    Class -> Taguchi(X, y)
    
    A class designed to perform Design of Experiments (DOE) using the Taguchi method, 
    a robust statistical approach widely used in quality engineering.
    
    Features:
    - Signal-to-Noise (S/N) Ratios: Supports evaluation of quality characteristics using three types of S/N ratios:
      - "Bigger is better"
      - "Smaller is better"
      - "Nominal is better"
    - Multi-Response Performance Index (MRPI): Capable of handling multiple responses with different units, using:
      - Weighted methods ("wgt")
      - Envelopment methods ("env")
    - Flexibility in defining response objectives and quality characteristics for multiple responses.
    
    Parameters:
    -----------
    X : matrix
        Matrix containing the factors (effects/interactions) to be analyzed in the DOE.
    y : array-like
        Vector or matrix containing the response(s). Can handle single or multiple responses.
    T : float, optional (default=0)
        Target value for the "Nominal is better" S/N ratio.
    sn : str, optional (default=None)
        Specifies the quality characteristic for S/N ratio:
        - "max"    -> "Bigger is better"
        - "min"    -> "Smaller is better"
        - "target" -> "Nominal is better"
    mrpi : str, optional (default=None)
        Specifies the method for multi-response evaluation:
        - "wgt" -> Weighted method
        - "env" -> Envelopment method
    r1, r2 : str, optional (default=None)
        Quality characteristics for multiple responses:
        - "max"    -> "Bigger is better"
        - "min"    -> "Smaller is better"
        - "target" -> "Nominal is better"
    
    Attributes:
    -----------
    y : array-like
        The processed response vector/matrix after applying S/N ratios or MRPI.
    vector_y : array-like
        Returns the final response vector or matrix ready for analysis.
    
    Methods:
    --------
    1. vector_y : 
        Returns the processed response vector or matrix.
        Usage: doe.Taguchi(X, y).vector_y
    
    2. effect_analysis : 
        Calculates the total response for each factor level and visualizes their effects in a graph.
        Usage: doe.Taguchi(X, y).effect_analysis
    
    3. check_interactions : 
        Displays interaction graphs between selected factors, highlighting dependencies and interactions.
        Usage: doe.Taguchi(X, y).check_interactions()
    
    4. pred : 
        Predicts the response values for selected factors or interactions based on the DOE design.
        Usage: doe.Taguchi(X, y).pred()
    
    5. anova : 
        Performs Analysis of Variance (ANOVA) and returns a table summarizing the variance contributions of each factor.
        Usage: doe.Taguchi(X, y).anova()
    
    Notes:
    ------
    - This class is tailored for applications of the Taguchi method in quality engineering,
        providing tools to optimize processes and improve quality metrics.
    - Ensure the input data (X and y) is formatted correctly:
      - X should be a matrix representing the experimental design.
      - y should be a vector or matrix of responses with consistent dimensions relative to X.
    - When using MRPI with multiple responses, ensure appropriate quality characteristics 
      (r1, r2) and methods ("wgt" or "env") are defined.
    """
    
    def __init__(self, x, y, T = 0, sn = None, mrpi = None, r1 = None, r2 = None):
        self.X = x # Matrix X
        self.T = T # Target value
        self.sn = sn # Quality caracteristic
        self.mrpi = mrpi # Multi-Response Performance Index
        self.r1 = r1 # Vector 1
        self.r2 = r2 # Vector 2
        self.y_raw = pd.DataFrame(y) # Raw values for vector y
        
        # Vector y and s/n ratio
        if self.sn == "min":
            self.y = pd.DataFrame(-10 * np.log10(np.mean(y**2, axis=1)))
        elif self.sn == "target":
            self.y = pd.DataFrame(-10 * np.log10(np.mean((y - self.T)**2, axis=1)))
        elif self.sn == "max":
            self.y = pd.DataFrame(-10 * np.log10(np.mean((1/y)**2, axis=1)))
        elif self.sn == None and self.mrpi == None:
            self.y = y

        # MRPI
        if self.mrpi in ["wgt", "env"]:
            y1 = (pd.DataFrame(y)).iloc[:,0]
            y2 = (pd.DataFrame(y)).iloc[:,1]
        elif self.sn == None and self.mrpi == None:
            self.y = y

        # Default weight function
        def get_weight(y, mode):
            if mode == "max":
                return y / sum(y)
            elif mode == "min":
                return (1 / y) / sum(1 / y)
            elif mode == "target":
                return (1 / (y - self.T)) / sum(1 / (y - self.T))


        # Define weight combinations based on r1 and r2
        if self.mrpi in ["wgt", "env"]:
            weight_combinations = {
                ("max", "max"): (get_weight(y1, "max"), get_weight(y2, "max")),
                ("max", "min"): (get_weight(y1, "max"), get_weight(y2, "min")),
                ("min", "max"): (get_weight(y1, "min"), get_weight(y2, "max")),
                ("min", "min"): (get_weight(y1, "min"), get_weight(y2, "min")),
                ("max", "target"): (get_weight(y1, "max"), get_weight(y2, "target")),
                ("min", "target"): (get_weight(y1, "min"), get_weight(y2, "target"))
            }

        # Process MRPI calculation for weight or envelopment
        if self.mrpi == "wgt":
            if (r1, r2) in weight_combinations:
                w1, w2 = weight_combinations[(r1, r2)]
                if w1 is not None and w2 is not None:
                    MRPI = w1 * y1 + w2 * y2
                    self.y = pd.DataFrame(MRPI)
            else:
                print(f"\nChoose a valid quality characteristic for r1 and r2")
    
        elif self.mrpi == "env":
            if (r1, r2) in weight_combinations:
                w1, w2 = weight_combinations[(r1, r2)]
                if w1 is not None and w2 is not None:
                    MRPI = (w1 * y1) / (w2 * y2)
                    self.y = pd.DataFrame(MRPI)
            else:
                print(f"\nChoose a valid quality characteristic for r1 and r2")

        if (self.r1 or self.r2) == "target" and self.T == 0:
            print(f"\nThe nominal value T was not choosen")

    @property 
    def matrix_x(self): # Matrix with factors and interactions
        """
        Returns the design matrix (X) including factors and interactions.
        This matrix represents the selected experimental design and is used for effect calculations and analysis.
        """
        return self.X

    @property 
    def vector_y(self): # Array for calculated response(s)
        """
        Returns the response vector (y).

        This array contains the calculated responses used in the analysis.
        """
        return self.y

    @property 
    def __total_mean(self): # Mean of vector y
        return (self.y.mean().mean())

    @property
    def __sum_by_level(self): # Sum of y by level
        results = {} # Empty dictionary
        combined_responses = self.y.sum(axis=1) # Sum of responses
        
        # For each collunm in X, calculate sum by level
        for column in self.X.columns:
            grouped = self.X.groupby(column).apply(lambda group: combined_responses[group.index].sum(),
                                                  include_groups=False)
            results[column] = pd.DataFrame({
                'Level': grouped.index,
                'Sum': grouped.values,
                'Number of experiments': self.X.groupby(column).size()*self.y.shape[1]
            })     
        return results
    
    @property
    def __mean_by_level(self): # Mean of y by level
        results = {} # Empty dictionary
        combined_responses = self.y.mean(axis=1) # Mean of responses
        
        # For each collunm in X, calculate mean by level
        for column in self.X.columns:
            grouped = self.X.groupby(column).apply(lambda group: combined_responses[group.index].mean(),
                                                  include_groups=False)
            diff_abs = grouped.max() - grouped.min()
            results[column] = pd.DataFrame({
                'Level': grouped.index,
                'Mean': grouped.values,
                'Difference':[diff_abs] * len(grouped)
            })

        #Order by "Difference"
        ranges = {factor: df["Difference"].iloc[0] for factor, df in results.items()}
        sorted_ranges = pd.Series(ranges).rank(ascending=False).astype(int)

        #Add order to Dataframe
        for factor, rank in sorted_ranges.items():
            results[factor]["Order"] = [rank] * len(results[factor])
            
        return results

    def __sum_table(self): # Creates a summary table with levels as rows and factors as columns for sum

        # Build a dictionary for the final DataFrame
        summary_data = {"Factor/Interactions": []}
        levels = None
    
        # Iterate for the factors and write the sum values for each level
        for factor, df in self.__sum_by_level.items():
            # Get levels
            if levels is None:
                levels = df["Level"].tolist()
            
            # Add the factor name and the sum
            summary_data[factor] = df["Sum"].tolist()
    
        # Add the levels as the first column
        summary_data["Factor/Interactions"] = [f"Level {level}" for level in levels]
    
        # "Factor/Interactions" in the first column
        columns = ["Factor/Interactions"] + [col for col in summary_data if col != "Factor/Interactions"]
        summary_data = {col: summary_data[col] for col in columns}

        # Convert summary_data to dataframe
        df = pd.DataFrame(summary_data)
        
        return df
    
    def __mean_table(self): # Creates a summary table with levels as rows and factors as columns for mean

        # Build a dictionary a the final DataFrame
        summary_data = {"Factor/Interactions": []}
        levels = None
    
        # Iterate for the factors and write the mean values for each level
        for factor, df in self.__mean_by_level.items():
            # Get levels
            if levels is None:
                levels = df["Level"].tolist()
            
            # Add the factor name and the means
            summary_data[factor] = df["Mean"].tolist()
    
            # Add the range in the end of each factor
            range_value = df["Difference"].iloc[0]  # Is a constant for the factor
            summary_data[factor].append(range_value)

            # Add rank in the end of each factor
            rank = df["Order"].iloc[0]
            summary_data[factor].append(rank.astype(int))
    
        # Add the levels as the first column and "Difference" in the end
        summary_data["Factor/Interactions"] = [f"Level {level}" for level in levels] + ["Difference"] + ["Order"]
    
        # "Factor/Interactions" in the first column
        columns = ["Factor/Interactions"] + [col for col in summary_data if col != "Factor/Interactions"]
        summary_data = {col: summary_data[col] for col in columns}

        # Convert summary_data to dataframe
        df = pd.DataFrame(summary_data)
    
        return df
        
    def __effect_graph(self): # Create the effect graph
        mean_data = self.__mean_by_level # Import the mean and difference for factors
        __total_mean = self.__total_mean # Import the total mean
        plt.figure(figsize=(10, 6))

        # Iteration
        for factor, df in mean_data.items():
            levels = df['Level'].apply(lambda x: f'{factor}{x}')  # Create the name for levels
            means = df['Mean']  # Get the mean for each level
            
            # Plot the graph
            plt.scatter(levels, means, label=f'Factor {factor}', color = 'blue', s=25) # Add scatter
            plt.plot(levels, means, label=f'Factor {factor}', color='blue', linewidth=1, linestyle='-') # Add line for factors
            plt.axhline(y=__total_mean, color='red', linestyle='--', linewidth=1, label='total_mean') # Add line for total mean
    
        # Graph parameters
        # plt.title('Effect Graph')
        plt.xlabel('Factors')
        plt.ylabel('Mean Values')
        plt.xticks(rotation=45) 
        plt.grid(True)
        
        #  Show graph
        plt.tight_layout()
        plt.savefig('Effect Graph.png',transparent=True) 
        plt.show()
        
    @property
    def effect_analysis(self): 
        """
        Show the sum table, mean table and effect graph
        """
        # Diplay the sum table
        display(HTML("<div style='text-align: center; font-weight: bold;'>Sum of Results</div>"))
        display(HTML("<div style='display: flex; justify-content: center;'>" + self.__sum_table().to_html() + "</div>"))
        print('\n')
        # Diplay the mean table
        display(HTML("<div style='text-align: center; font-weight: bold;'>Main Effects Table</div>"))
        display(HTML("<div style='display: flex; justify-content: center;'>" + self.__mean_table().to_html() + "</div>"))
        print('\n')
        # Diplay the effect graph
        print('\n')
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Effect Graph</div>"))
        self.__effect_graph()
    
    def pred(self, factors=None): 
        """
            Predict the experimental results by chosen Effect/Interaction factors.
    
        Parameters:
        factors (str): A comma-separated string of factors and levels in the format 'Factor-Level'.
    
        Example:
        .pred(factors='A-1,B-3') will calculate results for Factor A at Level 1 and Factor B at Level 3.
        """
        mean_data = self.__mean_by_level  # Access the precomputed means
        selected_factors = factors.strip().split(',')
        results = {}
    
        # Parse and validate selected factors and levels
        for factor in selected_factors:
            if '-' not in factor:
                print(f"Error: '{factor}' is not in the correct 'Factor-Level' format.")
            
            # Split factor and level
            factor_name, level = factor.split('-', 1)
            factor_name = factor_name.strip()
            level = level.strip()
            
            # Ensure level is the correct type (e.g., int or float)
            try:
                level = int(level)  # Attempt conversion to integer
            except ValueError:
                print(f"Error: Level '{level}' is not a valid number.")
                continue  # Skip invalid levels
    
            # Validate factor name
            if factor_name not in mean_data:
                print(f"Error: Factor '{factor_name}' is not recognized.")
                continue
    
            # Validate level
            level_values = mean_data[factor_name]["Level"].astype(type(level))
            if level not in level_values.values:
                print(f"Error: Level '{level}' is not valid for Factor '{factor_name}'.")
                continue
             
            # Retrieve the means for the valid levels
            mean = mean_data[factor_name].loc[
                mean_data[factor_name]["Level"] == level, ["Level", "Mean"]
            ]
            results[factor_name] = mean
    
        # Calculate the prediction
        total_sum = 0  # Initialize the total sum
        total_items = 0
        for factor, data in results.items():
            total_sum += data['Mean'].sum()
            total_items += len(data['Mean'])
    
        predict = total_sum - (total_items - 1) * self.__total_mean
    
        # Display the results
        display(HTML("<div style='text-align: left; font-weight: bold;'>The predict value is:</div>"))
        display(HTML("<div style='display: flex; justify-content: left;'>" + str(predict.round(2)) + "</div>"))
    
        # return predict
    
    def check_interaction(self,factors=None): 
        """
        Method to calculate the mean of responses for combinations of two selected factors and their levels.
        
        Parameters:
        factors (str): A comma-separated string of factors.

        Example:
        .check_interaction(factors='A,B') will calculate the mean of responses for combinations of two selected factors and their levels.
        """
        mean_data = self.__mean_by_level  # Access the precomputed means
        selected_factors = factors.strip().split(',')
        results = {}
              
        # Check if two factors was selected
        if len(selected_factors) != 2:
            print("You must select exactly two factors.")
            return None
            
        # Validate factor name
        for factor in selected_factors:
            if factor not in mean_data:
                print(f"Error: Factor '{selected_factors}' is not recognized. The avaliable Factors are '{list(mean_data.keys())}'")
                return None
            
        factor1, factor2 = selected_factors
    
        # Calculate the mean for factors by levels
        combinations = (
            self.X.groupby([factor1, factor2])
            .apply(lambda group: self.y.iloc[group.index].mean(axis=1).mean(),
                  include_groups=False)  # Mean of experimental results
            .reset_index(name="Mean")
        )
        
        # Create the column for combinations
        combinations["Combination"] = factor1 + combinations[factor1].astype(str) + factor2 + combinations[factor2].astype(str)
        
        # Reorganize the columns 
        result_table = combinations[["Combination", "Mean"]]
        
        # Show the table result
        display(HTML(result_table.to_html(index=False)))
        
        # Plot graph
        x1 = result_table['Combination'][0] + '\n' + result_table['Combination'][2], result_table['Combination'][1] + '\n' + result_table['Combination'][3]
        y1 = result_table['Mean'][0], result_table['Mean'][1]
        
        x2 = result_table['Combination'][0] + '\n' + result_table['Combination'][2], result_table['Combination'][1] + '\n' + result_table['Combination'][3]   
        y2 = result_table['Mean'][2], result_table['Mean'][3]
        
        plt.figure(figsize=(4, 3))
        plt.scatter(x1, y1, color = 'blue', s=25)
        plt.plot(x1, y1, color='blue', linewidth=1, linestyle='-')
        plt.scatter(x2, y2, color = 'blue', s=25)
        plt.plot(x2, y2, color='blue', linewidth=1, linestyle='-')
        plt.xlabel('Combination')
        plt.ylabel('Mean Values')
        
        plt.xticks([])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Interaction Graph.png',transparent=True) # Save figure
        plt.show()
    
        # Severity index
        max_mean = result_table['Mean'].max()
        min_mean = result_table['Mean'].min()
        si = ((abs((result_table['Mean'][1] - result_table['Mean'][0]) - (result_table['Mean'][3] - result_table['Mean'][2])))/(2*(max_mean - min_mean)))*100
    
        display(HTML("<div style='text-align: left; font-weight: bold;'>The Severity Index (SI) is:</div>"))
        display(HTML("<div style='display: flex; justify-content: left;'>" + str(si.round(2)) +'%' + "</div>"))
    
    
    def anova(self, method = None, column_error = None):
        """
        Returns the ANOVA table
        
        Parameters:
        method (str): 'Replica'. Use the replica of experiments to calculate the data in ANOVA
        column_error(list): List of factors/interactions to used for residual calculation

        Example:
        .anova(method='Replica')
        .anova(column_error=['B','D','E'])
        """       
        # CF: Correction factor
        cf = (((self.y.sum()).sum()) ** 2) / self.y.size
        
        # SSTotal: Total Sum of Squares
        sstotal = (((self.y ** 2).sum()).sum()) - cf
        
        # SSA: Sum of Squares for each factor
        sum_by_level = self.__sum_by_level
        ssa = {}
        for column, df in sum_by_level.items():
            sum_of_squares = (df['Sum'] ** 2)
            counts = df['Number of experiments']
            ssa[column] = (((sum_of_squares / counts)).sum()) - cf
 
        # dof: Degrees of Freedom
        dof = {}
        combined_responses = self.y.sum(axis=1)
        for column in self.X.columns:
            grouped = self.X.groupby(column).apply(lambda group: combined_responses[group.index].sum(),
                                                  include_groups=False)
            dof_value = len(grouped.index) - 1 
            dof[column] = dof_value
        # MSA: Mean Squares for each factor
        msa = {key: ssa[key] / dof[key] for key in ssa if key in dof}
        
        # RSS: Residual Sum of Squares
        if method == 'Replica':
            rss_replica = sstotal - sum(ssa.values())
            # dof_replica = (self.y.shape[1] - 1) * self.y.shape[0]
            dof_replica = (self.y_raw.shape[1] - 1) * self.y_raw.shape[0]
        else:
            rss_replica = 0
            dof_replica = 0

        # RSS: Residual by error
        if column_error:
            rss_error = sum(ssa[col] for col in column_error if col in ssa)
            dof_error = sum(dof[col] for col in column_error if col in dof)
        else:
            rss_error = 0
            dof_error = 0

        # Total of residual
        rss = rss_replica + rss_error
        dof_rss = dof_replica + dof_error

        # F Tabulated
        if dof_rss is not None:
            dof1 = list(dof.values())[0]  # Numerator degrees of freedom
            dof2 = dof_rss  # Denominator degrees of freedom
            alpha = 0.05  # 95% confidence
            f_tab = f.ppf(1 - alpha, dof1, dof2)
        else:
            f_tab = None
        # F Test
        if dof_rss > 0:
            msa_rss = rss/dof_rss
        else:
            msa_rss = 1 # Avoid error

        # C(%)
        percent_contrib = {key: (value / sstotal) * 100 for key, value in list(ssa.items())}
      
        # Cache results for reuse
        self._cache = {
            "cf": cf,
            "sstotal": sstotal,
            "ssa": ssa,
            "df": dof,
            "msa": msa,
            "rss": (rss, dof_rss),
            "f_tab": f_tab
        }
        
        # DataFrame ANOVA
        df = pd.DataFrame({
            "Variables": list(ssa.keys()),
            "Sum of Squares": list(ssa.values()),
            "Df": list(dof.values()),
            "Mean of Squares": list(msa.values()),
            "F-test": [(x / msa_rss) if column_error or method else "" for x in list(msa.values())],
            "C(%)": list(percent_contrib.values()),
            "Order": None
        })
        
        # Add order by C(%)
        order_series = df["C(%)"].rank(ascending=False, method="dense").astype(int)
        df["Order"] = order_series

        # Remove column used for error
        if column_error:
            df = df[~df["Variables"].isin(column_error)]

        # Add lines RSS and Total
        rss_line = {
            "Variables": "Residual",
            "Sum of Squares":rss,
            "Df": dof_rss,
            "Mean of Squares": msa_rss if column_error or method else "",
            "F-test": "",
            "C(%)": (rss/sstotal)*100,
            "Order": "",
        }
        total_line = {
            "Variables": "Total",
            "Sum of Squares": sstotal,
            "Df": sum(dof.values())+dof_replica, # sum all factors and replica if exist
            "Mean of Squares": "",
            "F-test": f"F(0.05;1;{dof_rss}) = {f_tab:.2f}" if column_error or method else "",
            "C(%)": sum(value for key, value in percent_contrib.items() if not column_error or key not in column_error) + (rss/sstotal)*100,
            "Order": "",
        }
        
        # Concatante all results to the DataFrame
        df = pd.concat([df, pd.DataFrame([rss_line, total_line])], ignore_index=True)

        return df
        # return self._cache

    def ri(self,sn0=0,sn1=0): 
        """
        Return the relative improvement of optimization for a selected essay.

        Parameters:
        sn0 (float): S/N ratio of selected essay
        sn1 (float): S/N ratio of optimization value

        Example:
        .ri(sn0=32,sn1=34)
        """
        if sn0 and sn1 !=0:
            ri = (1-10**(-(sn1-sn0)/10))*100
            display(HTML("<div style='text-align: left; font-weight: bold;'>The predict value is:</div>"))
            display(HTML("<div style='display: flex; justify-content: left;'>" + str(round(ri,3)) + " %</div>"))
        else:
            print('Chose correctly which S/N ratio you need to compare')
