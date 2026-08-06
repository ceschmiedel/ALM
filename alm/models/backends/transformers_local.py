"""In-process HuggingFace backend with PEFT/LoRA adapters.

The economically important behaviour lives in :class:`_BaseModelPool`: base
weights are cached per ``base_model``, and each expert attaches only its own
adapter to that shared base.  Ten experts on one 3B base load 3B parameters
once plus ten adapters of a few megabytes each — the same principle S-LoRA
demonstrates at scale, applied in-process for single-node deployments.

For serving many adapters under real concurrency, prefer vLLM
(:class:`~alm.models.backends.openai_compat.VLLMBackend`); this backend is for
embedded and on-premise single-process use, and for evaluating a freshly
trained adapter before it is published.

Requires the ``local`` extra::

    pip install "alm-federation[local]"
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from alm.core.errors import BackendError, ConfigurationError
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)

_IMPORT_HINT = (
    'local inference needs the optional extra: pip install "alm-federation[local]"'
)


class _BaseModelPool:
    """Process-wide cache of base models and their attached adapters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._adapters: dict[str, set[str]] = {}

    def load(self, base_model: str, *, device: str, dtype: str) -> tuple[Any, Any]:
        with self._lock:
            if base_model in self._models:
                return self._models[base_model], self._tokenizers[base_model]
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(_IMPORT_HINT) from exc

            torch_dtype = getattr(torch, dtype, None) if dtype else None
            logger.info("Loading base model %s on %s", base_model, device)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None,
            )
            if device == "cpu":
                model = model.to("cpu")
            model.eval()
            self._models[base_model] = model
            self._tokenizers[base_model] = tokenizer
            self._adapters[base_model] = set()
            return model, tokenizer

    def attach_adapter(self, base_model: str, adapter_name: str, adapter_uri: str) -> None:
        """Attach a LoRA adapter to an already-loaded base, once."""
        with self._lock:
            attached = self._adapters.setdefault(base_model, set())
            if adapter_name in attached:
                return
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover
                raise ConfigurationError(_IMPORT_HINT) from exc

            model = self._models[base_model]
            if isinstance(model, PeftModel):
                model.load_adapter(adapter_uri, adapter_name=adapter_name)
            else:
                model = PeftModel.from_pretrained(
                    model, adapter_uri, adapter_name=adapter_name
                )
                self._models[base_model] = model
            attached.add(adapter_name)
            logger.info(
                "Adapter %s attached to %s (%d adapters share this base)",
                adapter_name,
                base_model,
                len(attached),
            )

    def select_adapter(self, base_model: str, adapter_name: str | None) -> None:
        model = self._models[base_model]
        if adapter_name and hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        elif hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bases_loaded": len(self._models),
                "adapters": {k: sorted(v) for k, v in self._adapters.items()},
            }

    def unload(self) -> None:
        with self._lock:
            self._models.clear()
            self._tokenizers.clear()
            self._adapters.clear()


POOL = _BaseModelPool()


class TransformersBackend(ChatBackend):
    """Generate locally with transformers, optionally through a LoRA adapter."""

    name = "transformers"

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        # Generation is blocking and GPU-bound; keep the event loop free so the
        # DAG's other branches continue to make progress.
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        base_model = self.spec.base_model or self.spec.model_name
        if not base_model:
            raise ConfigurationError(
                f"model {self.spec.model_id!r} must set base_model or model_name",
                model_id=self.spec.model_id,
            )
        device = str(self.spec.params.get("device", "auto"))
        dtype = str(self.spec.params.get("dtype", "bfloat16"))

        model, tokenizer = POOL.load(base_model, device=device, dtype=dtype)

        adapter_name = self.spec.adapter or None
        if adapter_name:
            if not self.spec.adapter_uri:
                raise ConfigurationError(
                    f"model {self.spec.model_id!r} declares adapter {adapter_name!r} "
                    "but no adapter_uri to load it from",
                    model_id=self.spec.model_id,
                )
            POOL.attach_adapter(base_model, adapter_name, self.spec.adapter_uri)
            model, _ = POOL.load(base_model, device=device, dtype=dtype)
        POOL.select_adapter(base_model, adapter_name)

        messages = [m.model_dump() for m in request.messages]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = request.prompt_text() + "\nassistant:"

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._resolved_max_tokens(request),
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generate_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generate_kwargs["top_p"] = request.top_p
        generate_kwargs.update(self._merged_params({}))
        generate_kwargs.pop("device", None)
        generate_kwargs.pop("dtype", None)

        try:
            import torch

            with torch.no_grad():
                output = model.generate(**inputs, **generate_kwargs)
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(_IMPORT_HINT) from exc
        except Exception as exc:
            raise BackendError(
                f"local generation failed for {self.spec.model_id!r}: {exc}",
                backend=self.name,
                model_id=self.spec.model_id,
            ) from exc

        prompt_length = int(inputs["input_ids"].shape[-1])
        completion_ids = output[0][prompt_length:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        for stop in request.stop:
            index = text.find(stop)
            if index >= 0:
                text = text[:index].strip()

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_length,
            completion_tokens=int(completion_ids.shape[-1]),
            finish_reason="stop",
            raw={"pool": POOL.stats()},
        )

    async def health(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False
