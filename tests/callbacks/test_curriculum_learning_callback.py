# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf as om

from llmfoundry.utils.builders import build_callback


def test_curriculum_learning_callback_builds():
    kwargs = {
        'duration': '1ep',
        'schedule': [{
            'duration': '1ep',
            'train_loader': {}
        }]
    }
    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )
    assert callback is not None
