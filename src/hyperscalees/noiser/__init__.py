from . import base_noiser, open_es, eggroll, alteggroll, sparse, eggroll_baseline_subtraction
from . import adaptive_surrogate_eggroll, predictive_eggroll

all_noisers = {
    "noop": base_noiser.Noiser,
    "open_es": open_es.OpenES,
    "eggroll": eggroll.EggRoll,
    "adaptive_surrogate_eggroll": adaptive_surrogate_eggroll.AdaptiveSurrogateEggRoll,
    "predictive_eggroll": predictive_eggroll.PredictiveEggRoll,
    "eggrollbs": eggroll_baseline_subtraction.EggRollBS,
    "alteggroll": alteggroll.EggRoll,
    "reeggroll": eggroll.EggRoll,
    "sparse": sparse.Sparse
}
