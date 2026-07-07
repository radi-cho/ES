from . import (
    alteggroll,
    base_noiser,
    eggroll,
    eggroll_baseline_subtraction,
    open_es,
    predictive_eggroll,
    sparse,
)

all_noisers = {
    "noop": base_noiser.Noiser,
    "open_es": open_es.OpenES,
    "eggroll": eggroll.EggRoll,
    "predictive_eggroll": predictive_eggroll.PredictiveEggRoll,
    "eggrollbs": eggroll_baseline_subtraction.EggRollBS,
    "alteggroll": alteggroll.EggRoll,
    "reeggroll": eggroll.EggRoll,
    "sparse": sparse.Sparse
}
