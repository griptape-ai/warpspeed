from marshmallow import fields, post_load
from galaxybrain.schemas import PolymorphicSchema, StepSchema


class ToolkitStepSchema(StepSchema):
    prompt_template = fields.Str(required=True)
    tools = fields.List(fields.Nested(PolymorphicSchema()), required=True)
    context = fields.Dict(keys=fields.Str(), values=fields.Raw())
    driver = fields.Nested(PolymorphicSchema(), allow_none=True)

    @post_load
    def make_step(self, data, **kwargs):
        from galaxybrain.workflows import ToolkitStep

        return ToolkitStep(**data)
