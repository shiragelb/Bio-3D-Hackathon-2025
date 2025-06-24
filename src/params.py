LOAD_PREPARED_MODEL = [True, False][1]
SEED = 42

# Data processing:
EMBED_SIZE = 2560
EMBED_LAYER = 9

# Training params
WEIGHT_DECAY = 0.1
N_EPOCHS = 30
LR = 1e-3

# Nets
MODEL_TYPE = ["FF_classifier", "transformer_classifier"][0]
FF_HIDDEN_DIM = 2056
FF_DROPOUT = 0

TRANSFORMER_DROPOUT= 0




# ESM_EMBED_SIZE_TO_LAYER = {320: 6,  # which transformer layer to take for the embedding
#                            480: 12,
#                            640: 30,
#                            1280: 33,
#                            2560: max 36, 9 was better in ex4
#                            5120: 48}