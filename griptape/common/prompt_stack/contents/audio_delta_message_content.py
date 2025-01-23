from __future__ import annotations

from typing import Optional

from attrs import define, field

from griptape.common import BaseDeltaMessageContent


@define
class AudioDeltaMessageContent(BaseDeltaMessageContent):
    id: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    data: bytes = field(kw_only=True, metadata={"serializable": True})
    transcript: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
