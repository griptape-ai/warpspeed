from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import openai
from attrs import Factory, define, field
from openai import AssistantEventHandler
from typing_extensions import override

from griptape.artifacts import BaseArtifact, TextArtifact
from griptape.drivers.assistant import BaseAssistantDriver
from griptape.events import EventBus, TextChunkEvent
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from openai.types.beta.threads import Text, TextDelta
    from openai.types.beta.threads.runs import ToolCall, ToolCallDelta


@define
class OpenAiAssistantDriver(BaseAssistantDriver):
    class EventHandler(AssistantEventHandler):
        @override
        def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
            if delta.value is not None:
                EventBus.publish_event(TextChunkEvent(token=delta.value))

        @override
        def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall) -> None:
            if delta.type == "code_interpreter" and delta.code_interpreter is not None:
                if delta.code_interpreter.input:
                    EventBus.publish_event(TextChunkEvent(token=delta.code_interpreter.input))
                if delta.code_interpreter.outputs:
                    EventBus.publish_event(TextChunkEvent(token="\n\noutput >"))
                    for output in delta.code_interpreter.outputs:
                        if output.type == "logs" and output.logs:
                            EventBus.publish_event(TextChunkEvent(token=output.logs))

    base_url: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    api_key: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": False})
    organization: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    thread_id: Optional[str] = field(default=None, kw_only=True)
    assistant_id: str = field(kw_only=True)
    event_handler: AssistantEventHandler = field(
        default=Factory(lambda: OpenAiAssistantDriver.EventHandler()), kw_only=True, metadata={"serializable": False}
    )
    auto_create_thread: bool = field(default=True, kw_only=True)

    _client: Optional[openai.OpenAI] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            organization=self.organization,
        )

    def try_run(self, *args: BaseArtifact) -> TextArtifact:
        if self.thread_id is None and self.auto_create_thread:
            self.thread_id = self.client.beta.threads.create().id
        response = self._create_run(*args)

        response.meta.update({"assistant_id": self.assistant_id, "thread_id": self.thread_id})

        return response

    def _create_run(self, *args: BaseArtifact) -> TextArtifact:
        content = "\n".join(arg.value for arg in args)
        message_id = self.client.beta.threads.messages.create(thread_id=self.thread_id, role="user", content=content)
        with self.client.beta.threads.runs.stream(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            event_handler=self.event_handler,
        ) as stream:
            stream.until_done()
            last_messages = stream.get_final_messages()

            message_contents = []
            for message in last_messages:
                message_contents.append("".join(content.text.value for content in message.content))
            message_text = "\n".join(message_contents)

            response = TextArtifact(message_text)

            response.meta.update(
                {"assistant_id": self.assistant_id, "thread_id": self.thread_id, "message_id": message_id}
            )
            return response
