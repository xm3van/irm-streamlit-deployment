from .base_irm import BaseIRM
from typing import Dict, Tuple, Optional

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
    def param_validator(p: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter dict. Returns (is_valid, message).

        Accepts dicts like: {"r0": 0.02, "r1": 0.10, "r2": 0.30, "u_opt": 0.80}
        """
        # Presence & type checks
        required = ("r0", "r1", "r2", "u_opt")
        for k in required:
            if k not in p:
                return False, f"Missing parameter: {k}"
            try:
                float(p[k])
            except (TypeError, ValueError):
                return False, f"Parameter {k} must be a number."

        r0, r1, r2, u_opt = float(p["r0"]), float(p["r1"]), float(p["r2"]), float(p["u_opt"])

        # Bounds / logical checks
        if r0 < 0:
            return False, "r0 must be ≥ 0."
        if r1 < 0 or r2 < 0:
            return False, "r1 and r2 must be ≥ 0."
        if not (0.0 <= u_opt <= 1.0):
            return False, "u_opt must be within [0, 1]."
        if r2 < r1:
            return False, "Require r2 ≥ r1 to avoid weaker post-kink response."

        return True, None