import os
import numpy as np
from datasets import load_from_disk, concatenate_datasets

# ============================================================
# CONFIG
# ============================================================

RAW_ROOT = "/nfs1/antyklimi/dialects/data/raw_cretan"
OUT_ROOT = "/nfs1/antyklimi/dialects/data/curriculum_cretan"

SAMPLES_PER_STAGE = 8000
SEED = 42

os.makedirs(OUT_ROOT, exist_ok=True)

# ============================================================
# LOAD POOLS
# ============================================================

pools = {
    "greek_asr": load_from_disk(f"{RAW_ROOT}/greek_asr"),
    "greek_trans": load_from_disk(f"{RAW_ROOT}/greek_trans"),
    "cretan_asr": load_from_disk(f"{RAW_ROOT}/cretan_asr"),
    "cretan_trans": load_from_disk(f"{RAW_ROOT}/cretan_trans"),
}

print("\nPool sizes:")
for k, v in pools.items():
    print(f"{k}: {len(v)}")

# ============================================================
# SAFE SAMPLING WITH REPLACEMENT
# ============================================================

def sample(ds, n, seed):
    ds = ds.shuffle(seed=seed)

    if n <= len(ds):
        return ds.select(range(n))

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=n, replace=True)

    return ds.select(indices.tolist())

# ============================================================
# SAVE STAGE
# ============================================================

def save_stage(exp_name, stage_idx, parts):

    ds = concatenate_datasets(parts)
    ds = ds.shuffle(seed=SEED)

    assert len(ds) == SAMPLES_PER_STAGE, (
        f"{exp_name} stage {stage_idx}: "
        f"expected {SAMPLES_PER_STAGE}, got {len(ds)}"
    )

    out = f"{OUT_ROOT}/{exp_name}/stage_{stage_idx}"

    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds.save_to_disk(out)

    print(
        f"{exp_name} | stage_{stage_idx} "
        f"| {len(ds)} samples"
    )

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {

    # Main curriculum (Table 1)
    "exp1_main": [
        {"greek_trans": 1.0},
        {"greek_asr": 0.4, "cretan_trans": 0.6},
        {"cretan_asr": 0.6, "cretan_trans": 0.4},
        {"cretan_asr": 1.0},
    ],

    # Translation-heavy
    "exp2_translation_heavy": [
        {"greek_trans": 1.0},
        {"greek_asr": 0.2, "cretan_trans": 0.8},
        {"cretan_asr": 0.4, "cretan_trans": 0.6},
        {"cretan_asr": 1.0},
    ],

    # ASR-heavy
    "exp3_asr_heavy": [
        {"greek_trans": 1.0},
        {"greek_asr": 0.6, "cretan_trans": 0.4},
        {"cretan_asr": 0.8, "cretan_trans": 0.2},
        {"cretan_asr": 1.0},
    ],

    # Balanced
    "exp4_balanced": [
        {"greek_trans": 1.0},
        {"greek_asr": 0.5, "cretan_trans": 0.5},
        {"cretan_asr": 0.5, "cretan_trans": 0.5},
        {"cretan_asr": 1.0},
    ],

    # No translation (Table 4)
    "exp5_no_translation": [
        {"greek_asr": 1.0},
        {"greek_asr": 0.5, "cretan_asr": 0.5},
        {"cretan_asr": 1.0},
        {"cretan_asr": 1.0},
    ],

    # No donor language (Table 3)
    "exp6_no_greek": [
        {"cretan_trans": 1.0},
        {"cretan_asr": 0.4, "cretan_trans": 0.6},
        {"cretan_asr": 0.6, "cretan_trans": 0.4},
        {"cretan_asr": 1.0},
    ],
}

# ============================================================
# BUILD ALL CURRICULA
# ============================================================

print("\nBuilding all Cretan curricula...\n")

for exp_name, stages in EXPERIMENTS.items():

    print("\n" + "=" * 40)
    print(f"Building {exp_name}")
    print("=" * 40)

    for stage_idx, cfg in enumerate(stages):

        parts = []

        for pool_name, weight in cfg.items():

            n_samples = int(SAMPLES_PER_STAGE * weight)

            ds_part = sample(
                pools[pool_name],
                n_samples,
                seed=SEED + stage_idx
            )

            parts.append(ds_part)

            print(
                f"Stage {stage_idx} | "
                f"{pool_name}: {n_samples}"
            )

        save_stage(
            exp_name=exp_name,
            stage_idx=stage_idx,
            parts=parts
        )

print("\nAll Cretan curricula built successfully.")
print(f"Saved to: {OUT_ROOT}")
