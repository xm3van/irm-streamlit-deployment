# src/interest_rate_models/base_irm.py

class BaseIRM:
    """
    Abstract base class for Interest Rate Models (IRMs).
    """
    def calculate_rate(self, utilization: float, current_rate: float = None, time_elapsed: float = None, prev_derivative: float = None) -> float:
        """
        Calculate the interest rate based on utilization.
        
        Parameters:
        - utilization: Utilization ratio (U_t)
        - current_rate: Current interest rate (r_t) [Optional for certain IRMs]
        - prev_derivative: Previous utilization derivative (U'_t) [Optional for certain IRMs]
        
        Returns:
        - Interest rate (r_t')
        """
        raise NotImplementedError("calculate_rate must be implemented by subclasses.")
