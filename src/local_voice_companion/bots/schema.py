"""BotManifest: configuration + persona + runtime intent.

A bot is NOT a trained model, NOT a copied weight file and NOT a cloned voice.
It must therefore never contain host paths, absolute model locations, applied
pipelines or anything else that would tie it to the machine that created it.
Importing the same manifest on another machine re-runs selection there.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.types import SelectionPolicy

AUTO = "auto"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str = "You are a concise and friendly voice assistant."
    description: str = ""
    speaking_style: str = "natural"
    verbosity: Literal["brief", "normal", "detailed"] = "brief"

    @field_validator("system_prompt")
    @classmethod
    def _limit_prompt(cls, value: str) -> str:
        value = value.strip()
        if len(value) > 4000:
            raise ValueError("system_prompt exceeds 4000 characters")
        return value


class Language(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str = "zh"
    alternatives: list[str] = Field(default_factory=list)


class StageBinding(BaseModel):
    """How the bot *wants* a stage to run. `auto` means "let the runtime pick"."""

    model_config = ConfigDict(extra="forbid")

    provider: str = AUTO
    model: str = AUTO
    voice: str | None = None
    device: str | None = None

    @property
    def explicit(self) -> bool:
        return self.provider != AUTO or self.model != AUTO


class RuntimeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: SelectionPolicy = SelectionPolicy.AUTO
    prefer_local: bool = True
    allow_api_llm: bool = True


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    barge_in: bool = True
    history_turns: int = 24
    listen_mode: Literal["continuous", "push_to_talk"] = "continuous"
    max_speech_seconds: float = 12.0
    temperature: float = 0.35
    max_tokens: int = 64


class BotManifest(BaseModel):
    """The portable definition of a bot."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = 2
    id: str
    name: str
    persona: Persona = Field(default_factory=Persona)
    language: Language = Field(default_factory=Language)
    runtime: RuntimeBinding = Field(default_factory=RuntimeBinding)
    asr: StageBinding = Field(default_factory=StageBinding)
    llm: StageBinding = Field(default_factory=StageBinding)
    tts: StageBinding = Field(default_factory=StageBinding)
    conversation: Conversation = Field(default_factory=Conversation)
    template: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not _ID_PATTERN.match(cleaned):
            raise ValueError(
                "id must be lowercase letters, digits, '-' or '_' (2-64 chars), starting alphanumeric"
            )
        return cleaned

    @model_validator(mode="after")
    def _touch(self) -> "BotManifest":
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at
        return self

    @classmethod
    def new(
        cls,
        id: str,
        name: str,
        *,
        system_prompt: str = "You are a concise and friendly voice assistant.",
        language: str = "zh",
        voice: str | None = None,
        policy: str | SelectionPolicy = SelectionPolicy.AUTO,
        template: str = "",
        **extra: Any,
    ) -> "BotManifest":
        return cls(
            id=id,
            name=name,
            persona=Persona(system_prompt=system_prompt),
            language=Language(primary=language),
            runtime=RuntimeBinding(policy=SelectionPolicy(policy)),
            tts=StageBinding(voice=voice) if voice else StageBinding(),
            template=template,
            **extra,
        )

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BotManifest":
        return cls.model_validate(dict(payload))

    def merge(self, patch: Mapping[str, Any]) -> "BotManifest":
        """Partial update for PUT /bots/{id}.

        Nested mappings are merged one level deep so a caller can patch a single
        field (`{"persona": {"system_prompt": ...}}`) without having to resend
        the rest of the block. Note the id is re-validated on the way through,
        which is what stops a manifest import from renaming a bot in place.
        """

        payload = self.to_dict()
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key].update(value)
            else:
                payload[key] = value
        merged = type(self).from_dict(payload)
        merged.touch()
        return merged

    def effective_language(self, fallback: str = "zh") -> str:
        return self.language.primary or fallback
