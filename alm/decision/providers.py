"""Log-probability providers — the two ways DecLM can see a model's next token.

* :class:`TransformersLogprobs` runs the model in process and reads the full
  next-token distribution.  Exact, and the path a trained decision adapter is
  evaluated on: it shares the base-model pool with the generative backend, so a
  LoRA adapter is attached to an already-loaded base.
* :class:`OpenAICompatLogprobs` asks an OpenAI-compatible server (vLLM, or any
  endpoint that honours ``logprobs``) for the top next-token candidates.  The
  server only reveals the top few, so a label it does not list is given the
  lowest listed value — an upper bound on what it could have been.

Ollama's native API returns text only and is not supported; serve the same
weights through transformers or vLLM instead.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from alm.core.errors import BackendError, ConfigurationError
from alm.decision.readout import LogprobBatch, ReadoutPrompt, ReadoutUnsupportedError
from alm.models.backends import create_backend
from alm.models.spec import ModelSpec


def _messages(prompt: ReadoutPrompt) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]


# -- in-process --------------------------------------------------------------


def render_chat(tokenizer: Any, prompt: ReadoutPrompt) -> tuple[str, bool]:
    """Render a prompt so the next token is the first token of the answer.

    Returns the text and whether the tokenizer's chat template produced it (in
    which case the template already placed any special tokens).
    """
    if not getattr(tokenizer, "chat_template", None):
        return f"{prompt.system}\n\n{prompt.user}\nAnswer:", False
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Reasoning models must answer directly: a thought would occupy the
        # position the label is read from.
        text = tokenizer.apply_chat_template(_messages(prompt), enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(_messages(prompt), **kwargs)
    return text, True


def label_token_ids(tokenizer: Any, labels: Sequence[str], *, leading_space: bool) -> list[int]:
    """Map each label to exactly one token id, or refuse.

    Reading the first piece of a multi-token label would silently read the
    wrong thing, and two labels sharing a token cannot be told apart.
    """
    ids: list[int] = []
    for label in labels:
        candidates = (f" {label}", label) if leading_space else (label, f" {label}")
        found = None
        for candidate in candidates:
            tokens = tokenizer.encode(candidate, add_special_tokens=False)
            if len(tokens) == 1:
                found = int(tokens[0])
                break
        if found is None:
            raise ReadoutUnsupportedError(
                f"label {label!r} is not a single token for this model's tokenizer"
            )
        ids.append(found)
    if len(set(ids)) != len(ids):
        raise ReadoutUnsupportedError(f"labels {list(labels)} collide on the same token")
    return ids


class TransformersLogprobs:
    """Next-token log-probabilities from an in-process transformers model."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.batch_size = int(spec.params.get("readout_batch_size", 8))

    async def next_token_logprobs(self, prompts: Sequence[ReadoutPrompt]) -> LogprobBatch:
        # The forward pass is blocking; keep the event loop free.
        return await asyncio.to_thread(self._run, list(prompts))

    def _load(self) -> tuple[Any, Any]:
        from alm.models.backends.transformers_local import POOL

        base_model = self.spec.base_model or self.spec.model_name
        if not base_model:
            raise ConfigurationError(
                f"model {self.spec.model_id!r} must set base_model or model_name",
                model_id=self.spec.model_id,
            )
        device = str(self.spec.params.get("device", "auto"))
        dtype = str(self.spec.params.get("dtype", "bfloat16"))
        model, tokenizer = POOL.load(base_model, device=device, dtype=dtype)
        adapter = self.spec.adapter or None
        if adapter:
            if not self.spec.adapter_uri:
                raise ConfigurationError(
                    f"model {self.spec.model_id!r} declares adapter {adapter!r} but no adapter_uri",
                    model_id=self.spec.model_id,
                )
            POOL.attach_adapter(base_model, adapter, self.spec.adapter_uri)
            model, tokenizer = POOL.load(base_model, device=device, dtype=dtype)
        POOL.select_adapter(base_model, adapter)
        return model, tokenizer

    def _run(self, prompts: list[ReadoutPrompt]) -> LogprobBatch:
        import torch

        model, tokenizer = self._load()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded: list[list[int]] = []
        label_ids: list[list[int]] = []
        for prompt in prompts:
            text, templated = render_chat(tokenizer, prompt)
            encoded.append(tokenizer.encode(text, add_special_tokens=not templated))
            # After a chat template the answer starts a fresh turn; after the
            # plain "Answer:" fallback it follows a space.
            label_ids.append(label_token_ids(tokenizer, prompt.labels, leading_space=not templated))

        device = model.device
        results: list[list[float]] = [[] for _ in prompts]
        # Similar lengths batch together, so padding stays small.
        order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
        try:
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                width = max(len(encoded[i]) for i in chunk)
                input_ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for row, index in enumerate(chunk):
                    ids = encoded[index]
                    # Left padding keeps every prompt's last token in the last column.
                    input_ids[row, width - len(ids) :] = torch.tensor(ids)
                    mask[row, width - len(ids) :] = 1
                input_ids, mask = input_ids.to(device), mask.to(device)
                positions = (mask.cumsum(-1) - 1).clamp(min=0)
                with torch.no_grad():
                    try:
                        output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions)
                    except TypeError:
                        output = model(input_ids=input_ids, attention_mask=mask)
                logprobs = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
                for row, index in enumerate(chunk):
                    results[index] = [float(logprobs[row, t]) for t in label_ids[index]]
        except ReadoutUnsupportedError:
            raise
        except Exception as exc:
            raise BackendError(
                f"readout failed for {self.spec.model_id!r}: {exc}",
                backend="transformers",
                model_id=self.spec.model_id,
            ) from exc

        return LogprobBatch(logprobs=results, prompt_tokens=sum(len(ids) for ids in encoded))


# -- over HTTP ---------------------------------------------------------------


class OpenAICompatLogprobs:
    """Next-token log-probabilities from ``logprobs`` on a chat completions endpoint."""

    #: The most candidates the OpenAI API shape lets a server reveal.
    top_logprobs = 20

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.backend = create_backend(spec)
        self.concurrency = int(spec.params.get("readout_concurrency", 8))

    async def next_token_logprobs(self, prompts: Sequence[ReadoutPrompt]) -> LogprobBatch:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(prompt: ReadoutPrompt) -> tuple[list[float], int]:
            async with semaphore:
                return await self._one(prompt)

        outcomes = await asyncio.gather(*(one(p) for p in prompts))
        return LogprobBatch(
            logprobs=[lp for lp, _ in outcomes],
            prompt_tokens=sum(tokens for _, tokens in outcomes),
        )

    async def _one(self, prompt: ReadoutPrompt) -> tuple[list[float], int]:
        payload = {
            "model": self.spec.served_name,
            "messages": _messages(prompt),
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        url = f"{self.backend.base_url()}/chat/completions"
        response = await self.backend._http().post(url, json=payload, headers=self.backend._headers())
        if response.status_code >= 400:
            raise BackendError(
                f"{self.spec.backend} returned HTTP {response.status_code}: {response.text[:400]}",
                backend=self.spec.backend,
                status_code=response.status_code,
                model_id=self.spec.model_id,
            )
        data = response.json()
        try:
            candidates = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ReadoutUnsupportedError(
                f"the endpoint of model {self.spec.model_id!r} returned no logprobs"
            ) from exc

        # A label can appear as several tokens ("A", " A"); their mass adds up.
        mass: dict[str, float] = {}
        for candidate in candidates:
            token = str(candidate.get("token", "")).strip()
            mass[token] = mass.get(token, 0.0) + math.exp(float(candidate["logprob"]))
        if not any(label in mass for label in prompt.labels):
            raise BackendError(
                f"model {self.spec.model_id!r} put none of the labels {list(prompt.labels)} "
                "among its top next tokens; it is not answering with the label",
                backend=self.spec.backend,
                model_id=self.spec.model_id,
            )
        floor = min(float(c["logprob"]) for c in candidates)
        logprobs = [math.log(mass[label]) if label in mass else floor for label in prompt.labels]
        usage = data.get("usage") or {}
        return logprobs, int(usage.get("prompt_tokens", 0) or 0)
