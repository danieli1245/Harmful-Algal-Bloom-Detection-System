TIER_CONFIG = {
    "free": {
        "modalities": ['chlor_a'],
        "days": 5,
        "threads": 1,
        "model_path": "models\Free_tier_model.pkl"
    },
    "tier1": {
        "modalities": ['chlor_a', 'Rrs_412', 'Rrs_443'],
        "days": 10,
        "threads": 4,
        "model_path": "models\Tier-1_model.h5"
    },
    "tier2": {
        "modalities": ['chlor_a', 'Rrs_412', 'Rrs_443'],
        "days": 10,
        "threads": 8, # Max threads
        "model_path": "models\Tier-2_model.h5"
    },
    "admin": {
        "modalities": ['chlor_a', 'Rrs_412', 'Rrs_443'],
        "days": 10,
        "threads": 8, # Max threads
        "model_path": "models\Tier-2_model.h5"
    }
}