from base_irm import BaseIRM

class SemiLogIRM(BaseIRM):

    """
    Semi-Log Interest Rate Model (IRM).
    
    This model calculates interest rates as an exponential function of the utilization ratio.
    
    Attributes:
        rate_min (float): The minimum interest rate (as a decimal, e.g., 0.0001 for 0.01%).
        rate_max (float): The maximum interest rate (as a decimal, e.g., 10 for 1000%).
    """
    
    def __init__(self, rate_min: float = 0.0001, rate_max: float = 10):
  
        """
        Initialize the SemiLogIRM model.
        
        Parameters:
            rate_min (float): The minimum interest rate (in %)
            rate_max (float): The maximum interest rate (in %) - 1000 = 1000%
        """
        if rate_min <= 0 or rate_max <= 0:
            raise ValueError("rate_min and rate_max must be positive.")
        if rate_min >= rate_max:
            raise ValueError("rate_max must be greater than rate_min.")
        
        self.rate_min = rate_min
        self.rate_max = rate_max

    def calculate_rate(self, utilization, current_rate=None):
        """
        Calculate the interest rate based on the utilization ratio.
        
        Parameters:
            utilization (float): Current utilization ratio (0 <= utilization <= 1).
        
        Returns:
            float: Interest rate.
        """
        if not (0 <= utilization <= 1):
            raise ValueError("Utilization must be between 0 and 1.")
        
        rate = self.rate_min * (self.rate_max / self.rate_min) ** utilization

        # sanity check that rate is between min rate & max rate 
        rate = max(self.rate_min, min(self.rate_max, rate))

        return rate #returns borrow APR
    
    @staticmethod
    def param_validator(p: dict) -> bool:
        """
        Feasibility rule for sensitivity scan:
            • positive rates
            • max > min
        """
        return (p["rate_min"] > 0) and (p["rate_max"] > 0) and (p["rate_max"] > p["rate_min"])




