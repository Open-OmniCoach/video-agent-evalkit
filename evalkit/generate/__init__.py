"""Reference generation functions.

A function is selected by an importable ``module:function`` path and follows::

    async def generate(runtime, case, sampling) -> GenerationResult

Provided references:

* ``evalkit.generate.openai_chat:generate`` -- single-turn, OpenAI Chat shape.
* ``evalkit.generate.agent:generate`` -- multi-turn agentic template.
"""
