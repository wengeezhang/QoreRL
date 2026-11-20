import pytest
import torch

from qorerl.recipes.smoothquant import make_smoothing_artifact


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def smoothing_artifact():
    def build(cfg, factors):
        return make_smoothing_artifact(
            factors, model=cfg.model, model_revision=cfg.model_revision, recipe_id=cfg.policy_recipe_id
        )

    return build
