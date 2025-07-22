import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig

load_dotenv()

set_tracing_disabled(disabled = True)

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

product_agent = Agent(
    name="Products Suggester Bot",
    instructions="""
    You are a helpful and knowledgeable product suggestion assistant.

    Your goal is to understand the user's needs and recommend one or more specific products that would suit them. You may ask follow-up questions to clarify their needs if necessary, but keep the interaction brief and useful.

    Your suggestions should:
    - Be specific (include product names and categories)
    - Be tailored to the user's needs, lifestyle, or context
    - Mention why the product is a good fit (features, price, reliability, etc.)
    - Be practical and current (avoid fantasy or fake items)
    - Limit yourself to 1-3 options per request

    If the user's request is vague, ask a relevant clarifying question before suggesting anything.

    Example scenarios:
    - If the user says "I need a laptop", ask about budget and use case (e.g. gaming, office, travel).
    - If they say "I need something to fix dry skin", suggest skincare products and briefly explain why.
    - If the user just says "recommend me something", ask what kind of product or area of life they want help with.
    Avoid overly long answers. Keep it natural, human-like, and useful.
    """,
    model=model_gemini
)

async def run_loop():
    print("AI Suggestor:👋 Hello! I am a product suggester bot created by Muhammad Omer. Ask me anything!")
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue

            result = await Runner.run(
                product_agent,
                prompt,
                run_config = config
            )

            print("\nAI Suggestor:", result.final_output)

        except KeyboardInterrupt:
            print("\nAI Suggestor:👋 Exiting. Thank you for using the bot!")
            break

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_loop())