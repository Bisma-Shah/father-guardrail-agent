import asyncio
import rich
from pydantic import BaseModel
from connection import config
from agents import (
    Agent,
    Runner,
    input_guardrail,
    InputGuardrailTripwireTriggered,
    GuardrailFunctionOutput
)

# -------------------- Output Model --------------------
class OutdoorActivity(BaseModel):
    message: str
    isTooCold: bool


# -------------------- Father Agent --------------------
father_agent = Agent(
    name="Father Agent",
    instructions="If the temperature is below 26C, do not allow the child to go outside.",
    output_type=OutdoorActivity
)


# -------------------- Guardrail Function --------------------
@input_guardrail
async def father_guardrail(ctx, agent, input):
    result = await Runner.run(
        father_agent,
        input,
        run_config=config
    )

    rich.print(result.final_output)

    return GuardrailFunctionOutput(
        output_info=result.final_output.message,
        tripwire_triggered=result.final_output.isTooCold
    )


# -------------------- Child Agent --------------------
child_agent = Agent(
    name="Child Agent",
    instructions="You are an excited child who wants to run outside.",
    input_guardrails=[father_guardrail]
)


# -------------------- Main Function --------------------
async def main():
    try:
        result = await Runner.run(
            child_agent,
            "I want to run outside. Temperature is 28C.",
            run_config=config
        )
        print("Child went outside.")

    except InputGuardrailTripwireTriggered:
        print("Father stopped the child. It's too cold!")


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    asyncio.run(main())
