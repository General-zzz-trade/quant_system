# decision/signals/ml
"""Machine learning based signals — lazy re-export to break decision ↔ strategy cycle."""


def __getattr__(name: str):  # noqa: ANN001
    if name == "FeaturesContract":
        from strategy.signals.ml.features_contract import FeaturesContract
        return FeaturesContract
    if name == "ModelRunnerSignal":
        from strategy.signals.ml.model_runner import ModelRunnerSignal
        return ModelRunnerSignal
    if name == "MultiTimeframeSignal":
        from strategy.signals.ml.multi_tf_signal import MultiTimeframeSignal
        return MultiTimeframeSignal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FeaturesContract", "ModelRunnerSignal", "MultiTimeframeSignal"]
