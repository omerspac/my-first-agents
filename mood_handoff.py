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

async def main(prompt: str):
    mood_agent = Agent(
        name="Mood Classifier Agent",
        instructions="""
        You are a mood detection agent. Your task is to analyze the user's message and classify their emotional state.

        You MUST output only one of the following mood labels: "happy", "sad", "stressed", "neutral", or "angry".

        Respond with only the mood label in lowercase, nothing else.
        """,
        model=model_gemini,
    )

    activity_agent = Agent(
        name="Activity Suggestor Agent",
        instructions="""
        You are an assistant that suggests helpful activities to improve the user's emotional well-being.

        If the user's mood is "sad", suggest relaxing or cheering activities like listening to music, calling a friend, etc.

        If the mood is "stressed", suggest stress-relief actions like meditation, walking, stretching, etc.

        Give 2 or 3 suggestions, keep the tone positive and supportive.
        """,
        model=model_gemini,
    )

    mood_result = await Runner.run(
        mood_agent,
        prompt,
        run_config=config,
    )

    mood = mood_result.final_output.strip().lower()
    print("Detected Mood:", mood)

    if mood in ["sad", "stressed"]:
        activity_result = await Runner.run(
            activity_agent,
            mood,
            run_config=config,
        )
        print("Suggested Activities:", activity_result.final_output)
        
    else:
        print("No suggestions needed. Have a great day!")

if __name__ == "__main__":
    print("AI Mood Analyzer: Hello! I am a mood analyzer bot created by Muhammad Omer. How may I help you?")
    user_prompt = input("Prompt: ")
    asyncio.run(main(user_prompt))
