import numpy as np

class SumPricesCV:
    """
    Receive an array of paths, return the sum the prices.
    Mainly created to implement Control Variates variance reduction method, 
    although it works with any path passed.
    """
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Function that receives 2D array (sotck trajectories along rows)
        and returns 1D array with sum of prices

        Args:
            paths (np.ndarray): 2D array shape: (n_paths, n_steps+1)

        Returns:
            np.ndarray: 1D array shape: (n_paths, )
        """
        stockpath_sum = np.sum(paths, axis=1)

        return stockpath_sum