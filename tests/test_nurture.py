from matplotlib import pyplot as plt

from flodym.dimensions import Dimension, DimensionSet
from flodym.flodym_arrays import StockArray
from flodym.stocks import (
    InflowDrivenDSM,

)
from flodym.lifetime_models import WeibullLifetime


N_YEARS = 30
N_YEARS_HIST = 10

WEIBULL_SCALE = 10
WEIBULL_SHAPE = 2

LT_EXT_FAC = 1.2


# Create dimensions
years_hist = list(range(0, N_YEARS_HIST + 1))
dims_hist = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=years_hist, dtype=int),
    ]
)
inflow_hist = StockArray(dims=dims_hist, name="inflow")
inflow_hist[0] = 1

# Create cohort dimension (required for initial stock functionality)
cohort_dim_hist = Dimension(name="cohort", letter="c", items=years_hist, dtype=int)

# Create lifetime model
lifetime_hist = WeibullLifetime(dims=dims_hist, time_letter="t", weibull_scale=WEIBULL_SCALE, weibull_shape=WEIBULL_SHAPE)

# Create InflowDrivenDSM with cohort_dim
dsm_hist = InflowDrivenDSM(
    dims=dims_hist,
    inflow=inflow_hist,
    cohort_dim=cohort_dim_hist,
    lifetime_model=lifetime_hist,
    time_letter="t",
    name="test_dsm_initial",
)
dsm_hist.compute()

# -------------------------------------------------------

years_all = list(range(0, N_YEARS + 1))  # 0 to 30 years
dims_all = DimensionSet(
    dim_list=[
        Dimension(name="TIME", letter="T", items=years_all, dtype=int),
    ]
)

# Create cohort dimension (required for initial stock functionality)
cohort_dim_all = Dimension(name="COHORT", letter="C", items=years_all, dtype=int)
initial_stock_dims = DimensionSet(dim_list=[cohort_dim_all])

# Create lifetime model
extended_lifetime = WeibullLifetime(dims=dims_all, time_letter="T", weibull_scale=WEIBULL_SCALE * LT_EXT_FAC, weibull_shape=WEIBULL_SHAPE)


initial_stock = StockArray(dims=initial_stock_dims, name="initial_stock")
initial_stock[{"C": cohort_dim_hist}] = dsm_hist.get_stock_by_cohort()[{"t": N_YEARS_HIST}]
dsm = InflowDrivenDSM(
    dims=dims_all,
    cohort_dim=cohort_dim_all,
    lifetime_model=extended_lifetime,
    time_letter="T",
    name="test_dsm_initial",
)

dsm.set_initial_stock(initial_stock, N_YEARS_HIST)

dsm.compute()

# Verify stock balance
dsm.check_stock_balance()

# Compute
plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(dsm_hist.stock.values)
plt.plot(years_all[N_YEARS_HIST:], dsm.stock.values[N_YEARS_HIST:])

plt.subplot(1, 2, 2)
plt.plot(dsm_hist.outflow.values)
plt.plot(years_all[N_YEARS_HIST:], dsm.outflow.values[N_YEARS_HIST:])

plt.show()

