from .base_irm import BaseIRM

class QuadraticIRM(BaseIRM):
    """
    Quadratic Interest Rate Model (IRM).
    
    This model calculates interest rates as a quadratic function of the utilization ratio.
    
    Attributes:
        rate_min (float): The minimum interest rate (as a decimal, e.g., 0.0001 for 0.01%).
        rate_max (float): The maximum interest rate (as a decimal, e.g., 10 for 1000%).
    """
    
    def __init__(self, rate_min: float = 0.0001, rate_max: float = 10):
        """
        Initialize the QuadraticIRM model.
        
        Parameters:
            rate_min (float): The minimum interest rate.
            rate_max (float): The maximum interest rate.
        """
        if rate_min <= 0 or rate_max <= 0:
            raise ValueError("rate_min and rate_max must be positive.")
        if rate_min >= rate_max:
            raise ValueError("rate_max must be greater than rate_min.")
        
        self.rate_min = rate_min
        self.rate_max = rate_max
    
    def calculate_rate(self, utilization: float, current_rate: float) -> float:
        """
        Calculate the interest rate based on the utilization ratio.
        
        Parameters:
            utilization (float): Current utilization ratio (0 <= utilization <= 1).
        
        Returns:
            float: Interest rate.
        """
        if not (0 <= utilization <= 1):
            raise ValueError("Utilization must be between 0 and 1.")
        
        quadratic_utilization = utilization ** 2
        rate = self.rate_min * (self.rate_max / self.rate_min) ** quadratic_utilization
        
        return max(self.rate_min, min(self.rate_max, rate))  # Ensure within min/max bounds

    @staticmethod
    def param_validator(p: dict) -> bool:
        """
        Feasibility rule for sensitivity scan:
            • positive rates
            • max > min
        """
        return (p["rate_min"] > 0) and (p["rate_max"] > 0) and (p["rate_max"] > p["rate_min"])

    
    