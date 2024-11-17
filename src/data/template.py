from torchtune.data._prompt_templates import PromptTemplate
from functools import partial

YearMakeModelTemplate = partial(
    PromptTemplate,
    template={
        "user": ("input: ", "\n\output: "),
    },
)
