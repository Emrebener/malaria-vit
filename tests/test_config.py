from malaria_vit import config

def test_models_registry_has_three_vits():
    assert set(config.MODELS) == {"deit_s", "swin_t", "vit_b16"}
    assert config.MODELS["vit_b16"].name == "vit_base_patch16_224"

def test_seed_is_fixed():
    assert config.SEED == 42
