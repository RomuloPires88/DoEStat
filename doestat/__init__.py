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


from .taguchi import Taguchi
from .auxvalues import Auxvalues
from .analysis import Analysis
from .regression import Regression

__all__ = ["Taguchi", "Auxvalues", "Analysis", "Regression"]
