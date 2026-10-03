#!/usr/bin/env python3
"""Refresh the original Bessel-plot tables without changing their figure design.
Requires numpy and scipy. Run from any directory; tables are beside the figure sources.
"""
from pathlib import Path
import numpy as np
from scipy.special import jv, yv

def main() -> None:
    figures = Path(__file__).resolve().parents[1] / 'figs'
    for family, function in [('J', jv), ('Y', yv)]:
        for order in range(4):
            path = figures / f'bessel-{family}{order}.dat'
            if not path.is_file():
                raise FileNotFoundError(path)
            table = np.loadtxt(path)
            x = table[:, 0]
            if family == 'Y' and np.any(x <= 0):
                raise ValueError(f'{path.name}: Y requires positive x in this plot.')
            np.savetxt(path, np.column_stack((x, function(order, x))), fmt='%.16e')
            print(path.name)

if __name__ == '__main__':
    main()
