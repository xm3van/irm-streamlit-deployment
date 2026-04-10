from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
RCOMP_CAP_PER_SECOND = 3170979198376  # from _compoundInterestRateCAP
RCUR_CAP = 10**20  # 10,000% APR in 1e18 format
X_MAX = 11090370147631773313 / 1e18  # ln(RCOMP_MAX + 1) expressed as real number


class BaseIRM:
    def calculate_rate(self, utilization: float, current_rate: float = None) -> float:
        raise NotImplementedError


@dataclass
class SiloIRMConfig:
    ulow: float
    uopt: float
    ucrit: float
    ki: float
    kcrit: float
    klow: float
    klin: float
    beta: float
    ri: float
    Tcrit: float


class SiloInterestRateModelV2(BaseIRM):
    def __init__(
        self,
        ulow: float = 0.35,
        uopt: float = 0.70,
        ucrit: float = 0.90,
        ki: float = 2e-9,
        kcrit: float = 5e-9,
        klow: float = 1e-9,
        klin: float = 5e-10,
        beta: float = 1e-6,
        ri: float = 1e-9,
        Tcrit: float = 0.0,
    ):
        self.ulow = ulow
        self.uopt = uopt
        self.ucrit = ucrit
        self.ki = ki
        self.kcrit = kcrit
        self.klow = klow
        self.klin = klin
        self.beta = beta
        self.ri = ri
        self.Tcrit = Tcrit

    def get_config(self) -> SiloIRMConfig:
        return SiloIRMConfig(
            ulow=self.ulow,
            uopt=self.uopt,
            ucrit=self.ucrit,
            ki=self.ki,
            kcrit=self.kcrit,
            klow=self.klow,
            klin=self.klin,
            beta=self.beta,
            ri=self.ri,
            Tcrit=self.Tcrit,
        )

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def calculate_rate(self, utilization: float, current_rate: float = None) -> float:
        out = self.calculate_current_interest_rate(utilization=utilization, dt_seconds=0.0)
        return out["rcur_apr"]

    def calculate_current_interest_rate(self, utilization: float, dt_seconds: float) -> Dict[str, float]:
        u = self.clamp(utilization, 0.0, 1.0)
        T = max(dt_seconds, 0.0)

        if u > self.ucrit:
            rp = self.kcrit * (1.0 + self.Tcrit + self.beta * T) * (u - self.ucrit)
        else:
            rp = min(0.0, self.klow * (u - self.ulow))

        rlin = self.klin * u
        ri_floor = max(self.ri, rlin)
        ri_next = max(ri_floor + self.ki * (u - self.uopt) * T, rlin)
        rcur_per_second = max(ri_next + rp, rlin)
        rcur_annual = min(rcur_per_second * SECONDS_PER_YEAR, RCUR_CAP / 1e18)

        return {
            "u": u,
            "T": T,
            "rp": rp,
            "rlin": rlin,
            "ri_next": ri_next,
            "rcur_per_second": rcur_per_second,
            "rcur_apr": rcur_annual,
        }

    def calculate_compound_interest_rate(self, utilization: float, dt_seconds: float) -> Dict[str, float]:
        u = self.clamp(utilization, 0.0, 1.0)
        T = max(dt_seconds, 0.0)

        ri = self.ri
        Tcrit = self.Tcrit
        slopei = self.ki * (u - self.uopt)

        if u > self.ucrit:
            rp = self.kcrit * (1.0 + Tcrit) * (u - self.ucrit)
            slope = slopei + self.kcrit * self.beta * (u - self.ucrit)
            Tcrit_out = Tcrit + self.beta * T
        else:
            rp = min(0.0, self.klow * (u - self.ulow))
            slope = slopei
            Tcrit_out = max(0.0, Tcrit - self.beta * T)

        rlin = self.klin * u
        ri = max(ri, rlin)
        r0 = ri + rp
        r1 = r0 + slope * T

        if r0 >= rlin and r1 >= rlin:
            x = (r0 + r1) * T / 2.0
            region = "Lower bound inactive"
        elif r0 < rlin and r1 < rlin:
            x = rlin * T
            region = "Lower bound active for whole interval"
        elif r0 >= rlin and r1 < rlin:
            x = rlin * T - ((r0 - rlin) ** 2) / slope / 2.0
            region = "Lower bound active after crossing"
        else:
            x = rlin * T + ((r1 - rlin) ** 2) / slope / 2.0
            region = "Lower bound active before crossing"

        ri_out = max(ri + slopei * T, rlin)

        overflow = False
        if x >= X_MAX:
            rcomp = 65536.0
            overflow = True
        else:
            rcomp = max(math.exp(x) - 1.0, 0.0)

        cap = (RCOMP_CAP_PER_SECOND * T) / 1e18
        cap_applied = rcomp > cap
        if cap_applied:
            rcomp = cap

        if overflow or cap_applied:
            ri_out = 0.0
            Tcrit_out = 0.0

        apr_equivalent = math.log1p(max(rcomp, 0.0)) / T * SECONDS_PER_YEAR if T > 0 and rcomp > 0 else 0.0

        return {
            "u": u,
            "T": T,
            "slopei": slopei,
            "rp": rp,
            "slope": slope,
            "rlin": rlin,
            "r0": r0,
            "r1": r1,
            "x": x,
            "ri_out": ri_out,
            "Tcrit_out": Tcrit_out,
            "rcomp": rcomp,
            "apr_equivalent": apr_equivalent,
            "overflow": overflow,
            "cap_applied": cap_applied,
            "region": region,
        }

    @staticmethod
    def param_validator(p: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        required = ("ulow", "uopt", "ucrit", "ki", "kcrit", "klow", "klin", "beta", "ri", "Tcrit")
        for k in required:
            if k not in p:
                return False, f"Missing parameter: {k}"
            try:
                float(p[k])
            except (TypeError, ValueError):
                return False, f"Parameter {k} must be a number."

        ulow = float(p["ulow"])
        uopt = float(p["uopt"])
        ucrit = float(p["ucrit"])
        ki = float(p["ki"])
        kcrit = float(p["kcrit"])
        klow = float(p["klow"])
        klin = float(p["klin"])
        beta = float(p["beta"])
        Tcrit = float(p["Tcrit"])

        if not (0.0 <= ulow <= 1.0):
            return False, "ulow must be within [0, 1]."
        if not (0.0 <= uopt <= 1.0):
            return False, "uopt must be within [0, 1]."
        if not (0.0 <= ucrit <= 1.0):
            return False, "ucrit must be within [0, 1]."
        if not (ulow <= uopt <= ucrit):
            return False, "Require ulow ≤ uopt ≤ ucrit."
        if min(ki, kcrit, klow, klin, beta) < 0:
            return False, "ki, kcrit, klow, klin, beta must be ≥ 0."
        if Tcrit < 0:
            return False, "Tcrit must be ≥ 0."

        return True, None