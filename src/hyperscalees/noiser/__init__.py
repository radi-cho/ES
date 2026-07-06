from . import (
    alteggroll,
    base_noiser,
    diag_eggroll,
    eggroll,
    eggroll_baseline_subtraction,
    open_es,
    product_space_eggroll,
    sparse,
)

all_noisers = {
    "noop": base_noiser.Noiser,
    "open_es": open_es.OpenES,
    "eggroll": eggroll.EggRoll,
    "diag_eggroll": diag_eggroll.DiagEggRoll,
    "product_space_eggroll": product_space_eggroll.ProductSpaceEggRoll,
    "eggrollbs": eggroll_baseline_subtraction.EggRollBS,
    "alteggroll": alteggroll.EggRoll,
    "reeggroll": eggroll.EggRoll,
    "sparse": sparse.Sparse
}
