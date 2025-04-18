#!/usr/bin/env python3
import asyncio
import os
from droidrun.agent.react_agent import ReActAgent
from droidrun.agent.llm_reasoning import LLMReasoner
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if the API key exists
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")

async def main():
    # Create an LLM instance (choose your preferred provider)
    llm = LLMReasoner(
        llm_provider="gemini",  # Can be "openai", "anthropic", or "gemini"
        model_name="gemini-2.0-flash",  # Choose appropriate model for your provider
        api_key=api_key,  # Using the validated API key
        temperature=0.2
    )
    
    # Create and run the agent
    agent = ReActAgent(
        task="Open the Settings app and check the Android version(日本語で表示されます,日本語で会話しよう)",
        llm=llm
    )
    # chromeを開いて, cb-cloud.comのサイトを表示する(日本語で会話しよう)
    steps = await agent.run()
    print(f"Execution completed with {len(steps)} steps")

if __name__ == "__main__":
    asyncio.run(main())
