import math

SECONDS_PER_YEAR = 365.24 * 24 * 60 * 60


def apr_to_per_second(apr_decimal: float) -> float:
    return apr_decimal / SECONDS_PER_YEAR


def per_second_to_apr(rate_per_second: float) -> float:
    return rate_per_second * SECONDS_PER_YEAR


class FraxVariableRateV3:
    """
    Frax Variable IRM V3

    Two components:
    1) Adaptive full-utilization rate
    2) Piecewise-linear utilization curve
    """

    def __init__(
        self,
        min_target_util: float,
        max_target_util: float,
        vertex_utilization: float,
        zero_util_rate: float,
        min_full_util_rate: float,
        max_full_util_rate: float,
        rate_half_life: float,
        vertex_rate_percent: float,
    ):
        self.min_target_util = min_target_util
        self.max_target_util = max_target_util
        self.vertex_utilization = vertex_utilization
        self.zero_util_rate = zero_util_rate
        self.min_full_util_rate = min_full_util_rate
        self.max_full_util_rate = max_full_util_rate
        self.rate_half_life = rate_half_life
        self.vertex_rate_percent = vertex_rate_percent

    # ---------- Core mechanics ----------

    def get_full_utilization_interest(self, delta_time: float, utilization: float, old_full_util_rate: float) -> float:
        u = utilization
        old = old_full_util_rate

        if u < self.min_target_util:
            delta_u = (self.min_target_util - u) / self.min_target_util
            decay_growth = self.rate_half_life + (delta_u**2 * delta_time)
            new_full = old * self.rate_half_life / decay_growth

        elif u > self.max_target_util:
            delta_u = (u - self.max_target_util) / (1.0 - self.max_target_util)
            decay_growth = self.rate_half_life + (delta_u**2 * delta_time)
            new_full = old * decay_growth / self.rate_half_life

        else:
            new_full = old

        return min(max(new_full, self.min_full_util_rate), self.max_full_util_rate)

    def get_new_rate(self, delta_time: float, utilization: float, old_full_util_rate: float):
        new_full = self.get_full_utilization_interest(delta_time, utilization, old_full_util_rate)

        vertex_interest = (
            (new_full - self.zero_util_rate) * self.vertex_rate_percent
            + self.zero_util_rate
        )

        if utilization < self.vertex_utilization:
            new_rate = self.zero_util_rate + (
                utilization * (vertex_interest - self.zero_util_rate)
                / self.vertex_utilization
            )
        else:
            new_rate = vertex_interest + (
                (utilization - self.vertex_utilization)
                * (new_full - vertex_interest)
                / (1.0 - self.vertex_utilization)
            )

        return new_rate, new_full, vertex_interest