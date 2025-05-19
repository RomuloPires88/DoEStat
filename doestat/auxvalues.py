# Copyright (C) 2025 Romulo Pires
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.stats import norm, t, f, gaussian_kde #library for regression
import itertools # Calculate matrices


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

    def amatrix(self, x, type_matrix=None, order=None):
        # Define Matrix X
        if type_matrix not in ["interaction"]: # Calculate the interactions
            self.X = x
            return self.X
        else:
            matrix = x.copy()
            factors = matrix.columns # Get factor names

            max_order = len(factors) if order is None else order
            
            # Iterate above combinations 
            for r in range(2, max_order + 1):
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
            return self.y_array, self.y, self.exp_error, self.eff_error, self.t