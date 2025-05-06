from .base_irm import BaseIRM

class SecondaryMonetaryPolicy(BaseIRM):
    """
    Secondary Monetary Policy model with stateful preservation of r_0.
    """

    def __init__(self, u_opt=0.85, alpha=0.35, beta=3.0, shift=0, external_rate=None):
    # def __init__(self, u_opt=None, alpha=None, beta=None, shift=None):

        """
        Initialize the SMP parameters.

        Parameters:
        - u_0: Target utilization rate (e.g., 0.8 for 80%).
        - alpha: Low ratio (e.g., 0.075).
        - beta: High ratio (e.g., 5.0).
        - r0: Base AMM rate (e.g., 0.05 for 5%).
        - shift: Adjustment factor (e.g., 0.04 for 4%).
        """
        self.u_opt = u_opt
        self.alpha = alpha
        self.beta = beta
        self.shift = shift
        self.external_rate = external_rate

        # Derived parameters
        self._calculate_derived_parameters()

    def _calculate_derived_parameters(self):
        """
        Calculate derived parameters (u_inf, A, r_minf).
        """
        denominator = (self.beta - 1) * self.u_opt - (1 - self.u_opt) * (1 - self.alpha)
        if abs(denominator) < 1e-6:
            raise ValueError("Invalid parameter combination leading to division by zero.")

        self.u_inf = ((self.beta - 1) * self.u_opt) / denominator
        # self.u_inf = ((self.beta - 1) * self.u_opt) / (
        #     ((self.beta - 1) * self.u_opt - (1 - self.u_opt) * (1 - self.alpha))
        # )
        self.A = (1 - self.alpha) * self.u_inf * (self.u_inf - self.u_opt) / self.u_opt
        self.r_minf = self.alpha - (self.A / self.u_inf)

        # print(self.u_inf, self.r_minf, self.A, self.u_opt)

    def calculate_rate(self, utilization: float, rate: float) -> float:
        """
        Calculate the interest rate based on utilization and preserve r0.

        Parameters:
        - utilization: Current utilization rate (0 ≤ utilization ≤ 1).

        Returns:
        - Interest rate (r_t).
        """

        # external_rate = random.uniform(0.12, 0.15) #simulate external market rate between 0.12 to 0.15
        # Compute the current rate
        r_t = (self.external_rate * (self.r_minf + (self.A / (self.u_inf - utilization)))) + self.shift


        # Update r0 for the next step
        return r_t
    
