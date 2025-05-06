from .base_irm import BaseIRM

class PieceWiseLinearIRM(BaseIRM):
    def __init__(self, r0: float = 0.02, r1: float = 0.1, r2: float = 0.3, u_opt: float = 0.8):
        """
        Initialize the Piece-Wise Linear IRM.
        
        Parameters:
        - r0: Base rate (intercept).
        - r1: Slope for u ≤ u_opt.
        - r2: Slope for u > u_opt.
        - u_opt: Optimal utilization rate.
        """
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.u_opt = u_opt
    
    def calculate_rate(self, utilization: float, current_rate: float = None) -> float:
        """
        Calculate the interest rate based on utilization using a piece-wise linear function.
        
        Parameters:
        - utilization: Current utilization ratio (U_t)
        - current_rate: Current interest rate (r_t) [Not used in piece-wise linear IRM]
        - prev_derivative: Previous utilization derivative (U'_t) [Not used]
        
        Returns:
        - Interest rate (r_t)
        """
        if not (0 <= utilization <= 1):
            raise ValueError("Utilization must be between 0 and 1.")
        
        if utilization <= self.u_opt:
            rate = self.r0 + self.r1 * utilization
        else:
            rate = self.r0 + self.r1 * self.u_opt + self.r2 * (utilization - self.u_opt)
        
        return rate

    @staticmethod
    def param_validator(p: dict) -> bool:
        """Class‑specific feasibility check."""
        return (p["r0"] >= 0) and (p["r2"] >= p["r1"])
