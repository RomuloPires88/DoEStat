import pandas as pd
import numpy as np
from scipy.stats import norm, linregress, t, f #library for regression
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns #library for analysis
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import iplot
from IPython.display import display, Latex, Markdown, Math, HTML
import sys #library for surface graph
from sympy import symbols #sympy library for symbols, diff, solve and subs
import plotly.io as pio
import itertools # Calculate matrices


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
    
    4. prev : 
        Predicts the response values for selected factors or interactions based on the DOE design.
        Usage: doe.Taguchi(X, y).prev()
    
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
        # self.y = y # removed to include s/n ratio
        self.T = T # Target value !!! Maybe split in two values, one for r1 and other for r2 or create another variable
        self.sn = sn # Quality caracteristic
        self.mrpi = mrpi # Multi-Response Performance Index
        self.r1 = r1 # Vector 1
        self.r2 = r2 # Vector 2
        
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
                ("target", "max"): (get_weight(y1, "target"), get_weight(y2, "max")), # Maybe doesnt exist
                ("target", "min"): (get_weight(y1, "target"), get_weight(y2, "min")), # Maybe doesnt exist
                ("max", "target"): (get_weight(y1, "max"), get_weight(y2, "target")),
                ("min", "target"): (get_weight(y1, "min"), get_weight(y2, "target")),
                ("target", "target"): (get_weight(y1, "target"), get_weight(y2, "target")) # Maybe doesnt exist
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
            grouped = self.X.groupby(column).apply(lambda group: combined_responses[group.index].sum())
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
            grouped = self.X.groupby(column).apply(lambda group: combined_responses[group.index].mean())
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
        plt.savefig('Effect Graph.png',transparent=True) # Save figure 
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
    
    def prev(self, factors=None): 
        """
        Predict the experimental results by chosen Effect/Interaction factors.
    
        Parameters:
        factors (str): A comma-separated string of factors and levels in the format 'Factor-Level'.
    
        Example:
        .prev(factors='A-1,B-3') will calculate results for Factor A at Level 1 and Factor B at Level 3.
        """
        mean_data = self.__mean_by_level  # Access the precomputed means
        selected_factors = factors.strip().split(',')
        results = {}
        # print(mean_data) # check Data
    
        # Parse and validate selected factors and levels
        for factor in selected_factors:
            if '-' not in factor:
                print(f"Error: '{factor}' is not in the correct 'Factor-Level' format.")
            
            # Split factor and level
            factor_name, level = factor.split('-', 1)
            factor_name = factor_name.strip()
            level = level.strip()
            # print(factor_name,level) # check DataFrame
            
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
    
    def check_interaction(self,factors=None): # Check this for factors with more than 2 levels
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
        # print(mean_data) # check Data
        
        
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
            .apply(lambda group: self.y.iloc[group.index].mean(axis=1).mean())  # Mean of experimental results
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
        #plt.xlabel('Combination')
        plt.ylabel('Mean Values')
        plt.xticks()
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
            grouped = self.X.groupby(column).apply(lambda group: combined_responses[group.index].sum())
            dof_value = len(grouped.index) - 1 
            dof[column] = dof_value
        # MSA: Mean Squares for each factor
        msa = {key: ssa[key] / dof[key] for key in ssa if key in dof}
        
        # RSS: Residual Sum of Squares
        if method == 'Replica':
            rss_replica = sstotal - sum(ssa.values())
            dof_replica = (self.y.shape[1] - 1) * self.y.shape[0]
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


    def __get_cached_property(self, name): # Delete this, maybe
        if name not in self._cache:
            self._calculate()
        return self._cache[name]

    def ri(self,sn0=0,sn1=0): #Relative improvement
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


class Auxvalues:
    """
    Auxiliar class to calculate:
    matrix with interactions
    vector y and possible mean values
    number of experiments (n)
    number of factors (k)
    experimental error
    effect error
    t-Student    
    """

    def amatrix(self, x, type_matrix=None):
        # Define Matrix X
        if type_matrix not in ["interaction"]: # Calculate the interactions
            self.X = x
            return self.X
        else:
            matrix = x.copy()
            factors = matrix.columns # Get the name of factors
           
            # Iterate above combinations 
            for r in range(2, len(factors) + 1):
                for combination in itertools.combinations(factors, r):
                    # Name of interaction
                    name_interaction = ' × '.join(combination)
                    # Multiply factors
                    matrix[name_interaction] = x[list(combination)].prod(axis=1)
            self.X = matrix
            return self.X

    def avector(self, x, y, yc=[], effect_error=None):
        self.k = x.shape[1] # Number of factors 
        
        # Define the response vector `y` and calculate the effect error
        if effect_error == "cp": 
            if yc is not None and len(yc) > 0:
                self.yc = yc # Central point
                self.n_yc = len(self.yc) # Number of center points
                self.dof_yc = self.n_yc - 1 # Degree of freedom of central points
                self.y_array = np.array(y) # Keep the original y vector
                self.y = np.mean(np.array(y), axis=1)  # Compute the mean and convert to an array
                self.exp_error = np.array(self.yc).std(ddof=1)  # Calculate the experimental error
                self.eff_error = 2 * self.exp_error / (self.n_yc * 2**self.k)**0.5  # Calculate the effect error
                self.t = t.ppf(1 - 0.05 / 2, self.dof_yc)  # Compute the t-value (two-tailed, 95% confidence)
                # self.sspe = np.sum((self.yc.values - np.mean(self.yc.values))**2) # Compute the Square Sum of Pure Error
                # self.dof_sspe = self.dof_yc # Dof SSPE = Dof Central points
                return self.y_array, self.y, self.exp_error, self.eff_error, self.t
            else:
                raise ValueError("Central points were not provided.")
        elif effect_error == "replica":
            if y.shape[1] > 1:
                self.y_array = np.array(y) # Keep the original y vector
                self.y = np.mean(np.array(y), axis=1)  # Compute the mean and convert to an array
                self.dof_y = y.shape[0]*(y.shape[1] - 1) # Degree of freedom of replicates
                self.exp_error = ((y.var(ddof=1, axis=1)).mean())**0.5 # Calculate the experimental error
                self.n_y = y.shape[1] # Number of replicates by experiment
                self.eff_error = 2 * self.exp_error / (self.n_y * 2**self.k)**0.5  # Calculate the effect error
                self.t = t.ppf(1 - 0.05 / 2, self.dof_y)  # Compute the t-value
                return self.y_array, self.y, self.exp_error, self.eff_error, self.t
            else:
               raise ValueError("Ensure that the response vector `y` includes replicates.")           
        else:
            self.y_array = np.array(y) # Keep the original y vector
            self.y = y.squeeze().to_numpy()  # Convert to a 1D array if possible
            self.exp_error = 0 # Default experimental error value
            self.eff_error = 0  # Default effect error value
            self.t = 0  # Default t-value
            self.sspe = 2 # Compute the Square Sum of Pure Error
            self.dof_sspe = 22 # Degree of freedom by default
            return self.y_array, self.y, self.exp_error, self.eff_error, self.t
        
        
class Analysis: 
    
    """
    Class -> Analysis(X, y)
    
    A class designed for evaluating factorial planning effects, offering insights into the contributions 
    of individual factors and their interactions in experimental designs.
    
    Features:
    ---------
    - Factorial Effect Analysis: Calculates and visualizes the effects of factors in a factorial design.
    - Probability Graph: Displays the distribution of effects using a Gaussian-based probability plot.
    - Percentage Contributions: Quantifies the relative contributions of each effect to the overall variability.
    - Error Handling: Supports effect error calculation and confidence intervals using the t-Student distribution.
    
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
   
    Attributes:
    -----------
    matrix_x : array-like
        The selected or calculated factorial design matrix.
    vector_y : array-like
        The response vector provided during instantiation, or the mean of replicates.
    error : float
        The calculated effect error
    
    Methods:
    --------
    1. effect_analysis:
        Generates and displays:
        - A probability graph to visualize the Gaussian distribution of effects.
        - A bar plot illustrating the percentage contributions of effects.
        Saves the generated plots as an image file.
        Usage: `doe.Analysis(X, y).effect_analysis()`
    
    Notes:
    ------
    - This class is suitable for factorial designs and provides visual and numerical tools for interpreting experimental results.
    - Input data should be formatted appropriately:
      - `X` should represent the coded matrix of factors. Interactions can be calculated within this class.
      - `y` should be the corresponding response vector or matrix.
    - The Effect Error and t-Student values are calculated within the class to enable confidence interval estimation, improving the robustness of the analysis.
    - Graphs generated by this class are inspired by the factorial effect routines in Octave and adapted for Python.
    
    Acknowledgments:
    ----------------
    - Prof. Dr. Edenir Pereira Filho
    - B.S. André Simão
    """

    def __init__(self, x, y, yc=[], type_matrix=None, effect_error=None):
        self.aux = Auxvalues()
        self.type_matrix = type_matrix
        self.effect_error = effect_error
        self.X = self.aux.amatrix(x, type_matrix=self.type_matrix)
        self.y_array, self.y, self.exp_error, self.eff_error, self.t = self.aux.avector(x, y, yc, effect_error=self.effect_error)
        
    @property
    def error(self):
        """
        Returns the calculated error effect
        """
        return self.eff_error
                
    @property 
    def matrix_x(self): # Matrix with factors and interactions
        """
        Returns the design matrix (X) including factors and interactions.
        
        This matrix represents the selected experimental design 
        and is used for effect calculations and analysis.
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
    def __effect(self):  # Returns product values between effects and response
        return (self.X.T * self.y).T # Multiply the transposed matrix by the response vector, then transpose the result back

    @property
    def __n_effect(self):  # Returns dimensions of the matrix with effects (coded_value * response)
        return self.X.shape

    @property
    def __effect_indices(self):  # Returns list with respective interactions
        return self.X.T.index

    @property
    def __generate_start_center_end_gauss(self):  # Returns the values of the Gaussian
        start = [0]
        center = []
        end = []
        gauss = []
        for i in range(self.__n_effect[1]):
            end.append(start[i] + (1 / self.__n_effect[1]))
            start.append(end[i])
            center.append((start[i] + end[i]) / 2)
            gauss.append(norm.ppf(center))
        return gauss

    @property
    def __define_gaussian(self):  # Returns the values of the Gaussian
        return self.__generate_start_center_end_gauss[self.__n_effect[1] - 1]
        
    @property
    def __calculate_effects(self):  # Returns vector with effects
        effects = (np.einsum('ij->j', self.__effect)) / (self.__n_effect[0] / 2) # np.einsum -> function that sums columns of a matrix
        if effects.size == 0 or np.all(np.isnan(effects)):
            return "Check if the Matrix or Vector was selected correctly"
        return effects

    @property
    def __calculate_percentage_effects(self):  # Returns vector with probability
        return (self.__calculate_effects ** 2 / np.sum(self.__calculate_effects ** 2)) * 100

    @property
    def __sort_effects_probabilities(self):  # Returns dataframe sorted in ascending order with effect values
        data = pd.DataFrame({'Effects': self.__calculate_effects}, index=self.__effect_indices)
        data = data.sort_values('Effects', ascending=True)
        return data

    @property
    def __define_ci(self):  # Returns set of Confidence Interval points
        return np.full(len(self.__define_gaussian), self.eff_error * self.t)

    @property
    def __probability_effect(self):
        plt.figure(figsize=(10, 6))
        # Confidence Interval
        plt.plot(-1 * self.__define_ci, self.__define_gaussian, color='red') # Left
        plt.plot(0 * self.__define_ci, self.__define_gaussian, color='blue') # Center
        plt.plot(self.__define_ci, self.__define_gaussian, color='red')      # Right
        # Scatter plot
        plt.scatter(self.__sort_effects_probabilities['Effects'], self.__define_gaussian, s=25, color='darkred')
        #plt.title('Probability Effects Plot', fontsize=12, fontweight='black', loc='center')
        plt.ylabel('z')
        plt.xlabel('Effects')
        plt.grid(True)
        # Mark points
        for i, label in enumerate(self.__sort_effects_probabilities.index):
            plt.annotate(label, (self.__sort_effects_probabilities['Effects'].values[i],
                                    self.__define_gaussian[i]),
                                    textcoords="offset points",  # Use offset points for text positioning
                                    xytext=(10, 0),  # This adds an offset of 5 points on the y-axis (you can adjust this)
                                    ha='left',  # Horizontal alignment of the text (centered on the point)
                                    fontsize=9,  # Optional: set the font size of the annotation
                                    color='black'  # Optional: set the color of the annotation text
                           )

        plt.tight_layout()
        plt.savefig('Probability Effects Plot.png',transparent=True) # Save figure 
        plt.show()

    @property
    def __percentage_effect(self):
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        sns.barplot(
            x='%', 
            y='Effects', 
            data=pd.DataFrame({
                'Effects': self.__effect_indices, 
                '%': self.__calculate_percentage_effects  
            }),
            color='purple',
            orient='h'
        )
        plt.tight_layout()
        plt.savefig('Percentage Effects Plot.png',transparent=True) # Save figure 
        plt.show()
        
    def effect_analysis(self, exclude_variables=None):
        """
        Analyzes the effect of factors and optionally excludes specified variables from the analysis.

        Parameters:
        exclude_variables (list of str, optional): 
            A list of variable names to exclude from the analysis. 
            If provided, the method recalculates the probability and percentage effects 
            after removing the specified variables.
    
        Example:
        .effect_analysis(exclude_variables=['A', 'B']) 
            Recalculates the probability and percentage effects excluding variables 'A' and 'B'.
        """
        # Check the variables in matrix X
        if exclude_variables:
            valid_variables = [var for var in exclude_variables if var in self.X.columns]
            invalid_variables = [var for var in exclude_variables if var not in self.X.columns]
            
            if not valid_variables:
                raise ValueError("None of the variables to exclude were found in the matrix.")     
            
            if invalid_variables:
                raise ValueError(f"The following variables were not found in the matrix: {', '.join(invalid_variables)}")
            
            # Remove the variables for new analysis
            self.X = self.X.drop(columns=valid_variables)
            display(HTML("<div style='text-align: left; font-weight: bold; font-size: 12px;'>Factors excluded</div>"))
            print(valid_variables)
            print('\n')
            
        # Display the variables used    
        display(HTML("<div style='text-align: left; font-weight: bold; font-size: 12px;'>Factors used</div>"))
        print(list(self.X.columns))
        
        # Display the Probability Effects Plot
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Probability Effects Plot</div>"))
        self.__probability_effect
        print('\n')
        
        # Display the Percentage Effects Plot
        display(HTML("<div style='text-align: center; font-weight: bold; font-size: 18px;'>Percentage Effects Plot</div>"))
        self.__percentage_effect

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
        - "quadratic" -> Includes quadratic and interaction coefficients.
        - "linear" -> Includes only linear coefficients.
        - "interaction_only" ->  Includes linear coefficients and interactions only.
   
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
        - Regression plot, Residual plot, Histogram plot, Coefficiente error plot.
        Usage: `doe.Regression(X, y).analysis` 
    7.surface:
        Generates and displays:
        - The response surface model and corresponding contour plot.  
        - A 3D surface model for better visualization.
        Usage: `doe.Regression(X, y).surface()`      

    Notes:
    ------
    - This class is suitable for regression calculations in factorial designs and provides visual and numerical tools for interpreting experimental results.
    - Input data should be formatted appropriately:
      - `X` should represent the coded matrix of factors. Interactions can be calculated within this class.
      - `y` should be the corresponding response vector or matrix.
    """
    
    def __init__(self, x, y, yc=[], type_matrix='no_interaction', effect_error=None,selected_factors=None,regression='quadratic'):
        self.aux = Auxvalues()
        self.type_matrix = type_matrix
        self.effect_error = effect_error
        self.regression = regression
        self.X = self.aux.amatrix(x, type_matrix=self.type_matrix)
        self.yc = yc
        self.selected_factors = selected_factors or self.X.columns.tolist()
        self.Xb = self.create_Xb
        self.y_array, self.y,self.exp_error,_,_ = self.aux.avector(x, y, yc, effect_error=self.effect_error)
        

    @property
    def create_Xb(self):
        if len(self.selected_factors) > 2 :
            raise ValueError("More than two factors can not be calculated by regression.")
        else:
            if self.regression == 'quadratic':
                Xb = self.X[self.selected_factors]
                Xb['{}²'.format(self.selected_factors[0])] = Xb[self.selected_factors[0]] ** 2
                if len(self.selected_factors) > 1:
                    Xb['{}²'.format(self.selected_factors[1])] = Xb[self.selected_factors[1]] ** 2
                    Xb['{} × {}'.format(self.selected_factors[0], self.selected_factors[1])] = Xb[self.selected_factors[0]] * Xb[self.selected_factors[1]]
                Xb.insert(0, 'Intercept', 1)        
                return Xb
            elif self.regression == 'interaction_only':
                Xb = self.X[self.selected_factors]
                if len(self.selected_factors) > 1:
                    Xb['{} × {}'.format(self.selected_factors[0], self.selected_factors[1])] = Xb[self.selected_factors[0]] * Xb[self.selected_factors[1]]
                Xb.insert(0, 'Intercept', 1) 
                return Xb
            elif self.regression =='linear':
                Xb = self.X[self.selected_factors]
                Xb.insert(0, 'Intercept', 1) 
                return Xb

    # @property
    # def cov_matrix(self): # Covariance Matrix of the Coefficients
    #     return np.linalg.inv(np.matmul(self.Xb.values.T,self.Xb.values)).round(3) #  inv(X'*X)

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
            self.t = t.ppf(1 - 0.05/2, self.dof_rss) # Test t with 95% Acrescentar o t para quando há falta de ajuste
        elif self.pvalue_exp > 0.05 and self.pvalue_pe < 0.05:
            self.t = t.ppf(1 - 0.05/2, self.dof_sspe)
        else:
            raise ValueError("The model cannot be adjusted as it failed both p-value tests for significance.")
        self.variance =  np.diag(self.XbTXinv/len(self.y_array)) * self.mse_exp # NOTE! the matrix was divide by number of replica
        self.error = self.variance**0.5
        self.ci = self.error * self.t # Confidence interval
         
        return self.__dict__
    
    @property
    def anova(self): # ANOVA table
        data = self.__var_coeffs
        # DataFrame ANOVA
        df = pd.DataFrame({
            "Source": ["Factors", "Residual", "Total", "Pure Error","Lack o Fit"],
            "Sum of Squares": [round(data["ess"],3), round(data["rss"],3), round(data["tss"],3), round(data["sspe"],3),
                               round(data["lof"],3)],
            "Df": [data["dof_ess"], data["dof_rss"], data["dof_tss"],data["dof_sspe"] ,data["dof_lof"]],
            "Mean fo Squares": [round(data["msr_exp"],3), round(data["mse_exp"],3), round(data["mst_exp"],3),
                                round(data["msr_pe"],3),round(data["mse_pe"],3)],
            "F-Test": [round(data["ftest_exp"],3),"","",round(data["ftest_pe"],3),""],
            "F-Tabuled": [f'F(0.05;{data["dof_ess"]};{data["dof_rss"]}) = {round(data["ftab_exp"], 3)}',
                          "","", 
                          f'F(0.05;{data["dof_lof"]};{data["dof_sspe"]}) = {round(data["ftab_pe"], 3)}',
                          ""],
            "F-Ratio": [round(data["fratio_exp"],3),"","", round(data["fratio_pe"],3),""],
            "p-Value": [f"{data['pvalue_exp']:.4e}","", "", f"{data['pvalue_pe']:.4e}", "" ],
            "Evaluation": ["Reject H₀ and accept H₁" if data['pvalue_exp'] < 0.05 else "Accept H₀ and reject H₁",
                           "", "",
                           "Reject H₀ and accept H₁" if data['pvalue_pe'] < 0.05 else "Accept H₀ and reject H₁",
                           ""]
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
        plt.tight_layout()
        plt.grid()
        plt.show()
        
    @property
    def __ftest_plot(self): # Maybe this is not important!!!!
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
        # self.__ftest_plot

    def surface(self, v1=[], v2=[], plot3D = False):
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
        plot3D : bool, optional (default=False)
            If `True`, generates an interactive 3D Response Surface plot.  
            If `False`, displays the Response Surface and Contour plot.
    
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
        b0,b1,b2,b11,b22,b12 = data["b"]
        
        cod_X = {}
        for col in self.selected_factors:
            min_value = self.X[col].min()
            max_value = self.X[col].max()
            cod_X[col] = {"min": min_value, "max": max_value}
        array_cod1 = np.linspace(cod_X[self.selected_factors[0]]['min'] ,cod_X[self.selected_factors[0]]['max'] ,num=101)
        array_cod2 = np.linspace(cod_X[self.selected_factors[1]]['min'] ,cod_X[self.selected_factors[1]]['max'] ,num=101)
        x,y = np.meshgrid(array_cod1,array_cod2)
        self.z = (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
        
        
        if len(v1) == 2 and len(v2) == 2:
            array_real1 = np.linspace(v1[0],v1[1], num=101)
            array_real2 = np.linspace(v2[0],v2[1], num=101)
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
            plt.tight_layout(w_pad=5)
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
            # iplot(fig)
            fig.show()
