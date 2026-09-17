from src.core.config import config

from typing import List


class ModelManager:
    """Resolve the model id Cline sent into an upstream Anthropic model id."""

    def __init__(self, config):
        self.config = config

    def resolve(self, client_model: str) -> str:
        """Map a client model id to the upstream Anthropic model id.

        Resolution order:
          1. explicit MODEL_MAP entry (exact, then case-insensitive)
          2. already an Anthropic model id (``claude-*``) -> pass through
          3. name-class hints (opus/sonnet/haiku) -> BIG/MIDDLE/SMALL
          4. PASSTHROUGH_UNKNOWN_MODELS -> unchanged, else DEFAULT_MODEL
        """
        if not client_model:
            return self.config.default_model

        mapped = self.config.model_map.get(client_model)
        if mapped:
            return mapped
        lowered = client_model.strip().lower()
        for source, target in self.config.model_map.items():
            if source.strip().lower() == lowered:
                return target

        if lowered.startswith("claude-"):
            return client_model

        if "opus" in lowered:
            return self.config.big_model
        if "sonnet" in lowered:
            return self.config.middle_model
        if "haiku" in lowered:
            return self.config.small_model

        if self.config.passthrough_unknown_models:
            return client_model
        return self.config.default_model

    def list_models(self) -> List[str]:
        """Model ids advertised to clients at GET /v1/models."""
        return self.config.expose_models()


model_manager = ModelManager(config)