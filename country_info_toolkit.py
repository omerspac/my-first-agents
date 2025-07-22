import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig

load_dotenv()

set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model_gemini = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model_gemini,
    model_provider=external_client,
    tracing_disabled=True
)

# ALL AGENTS
capital_teller = Agent(
    name="Capital Agent",
    instructions="Tell the capital city of the given country.",
    model=model_gemini,
)

language_teller = Agent(
    name="Language Agent",
    instructions="Tell the major spoken language of the given country.",
    model=model_gemini,
)

population_teller = Agent(
    name="Population Agent",
    instructions="Tell the approximate population of the given country.",
    model=model_gemini,
)

# Orchestrator Agent
orchestrator = Agent(
    name="Orchestrator Agent",
    instructions="""
    You are an orchestrator agent that receives a country name and returns a summary with its capital, major language, and population. You will receive inputs from three agents and format the result nicely.""",
    model=model_gemini,
)

# MAIN FUNCTIONS
async def main(prompt_country_name: str):
    capital_result = await Runner.run(
        capital_teller,
        prompt_country_name,
        run_config=config,
    )

    language_result = await Runner.run(
        language_teller,
        prompt_country_name,
        run_config=config,
    ) 

    population_result = await Runner.run(
        population_teller,
        prompt_country_name,
        run_config=config,
    )

    combined_info = f"""
    Country: {prompt_country_name}
    Capital: {capital_result.final_output}
    Language: {language_result.final_output}
    Population: {population_result.final_output}
    """

    orchestrator_result = await Runner.run(
        orchestrator,
        combined_info,
        run_config=config,
    )

    print("\nCountry Info:\n", orchestrator_result.final_output)

if __name__ == "__main__":
    print("Orchestrator Agent: Hello! I am an orchestrator bot created by Muhammad Omer. How may i help you with any country information?")
    country = input("Enter a country name: ")
    asyncio.run(main(country))