"""
Sample LangChain ReAct agent traced with Langfuse.

Features
--------
* ReAct-style agent powered by an **Ollama** chat model
* **DuckDuckGo** search tool for live web results
* **Langfuse** callback handler captures nested traces for the LLM, tool, and overall agent run

Prerequisites
-------------
```bash
pip install \
    langchain==0.3.* \
    langchain-community==0.3.* \
    langchain-ollama \
    duckduckgo-search \
    langfuse \
    python-dotenv
```
You also need an Ollama server running locally (`ollama serve`) and a pulled model, e.g. `ollama pull qwen3`.

Environment variables expected (add to a `.env` file or export directly):
```
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or your self‑host URL
OLLAMA_MODEL=qwen3                         # optional, defaults to qwen3
OLLAMA_BASE_URL=http://localhost:11434     # optional, defaults to Ollama local server
```
"""

import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# ---- Langfuse -------------------------------------------------------------
from langfuse import Langfuse

# Initialise a single Langfuse callback handler that we will pass to every
# LangChain component. Re‑using the same handler keeps all sub‑runs (LLM call,
# tools, agent etc.) inside **one** trace tree.
langfuse_host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
print(f"Langfuse host: {langfuse_host}")
print()

if(os.getenv("LANGFUSE_PUBLIC_KEY") is None or os.getenv("LANGFUSE_SECRET_KEY") is None):
    print("Please set your Langfuse credentials in the environment variables.")
    print("You can find them in your Langfuse dashboard under Settings > API Keys.")
    print("See https://docs.langfuse.com/docs/getting-started/quickstart#step-2-get-your-api-keys")
    exit(1)

# Predefine a trace/run ID – recommended for grouping distributed calls.
RUN_ID = uuid.uuid4()

ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen3")
temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0))

print(f"Ollama host: {ollama_host}")
print(f"Ollama model: {ollama_model}")
if temperature != 0:
    print(f"Ollama temperature: {temperature}")
print()


langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=langfuse_host,
)

# ---------------------------------------------------------------------------
# Helper / CLI entry
# ---------------------------------------------------------------------------

def run(question: str) -> None:
    # Start (or fetch) a trace with our predefined id
    trace = langfuse_client.trace(
        id=str(RUN_ID),
        name="DuckDuckGo agent",
        input=question,
        description="DuckDuckGo search agent with Ollama LLM",
        tags=["duckduckgo", "ollama", "agent"],
        metadata={
            "ollama_model": ollama_model,
            "ollama_base_url": ollama_host,
        },   
    )

    # LangChain handler scoped to that trace
    langfuse_handler = trace.get_langchain_handler()

    # ---- LLM & tools ----------------------------------------------------------
    from langchain_ollama import OllamaLLM 
    from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

    llm = OllamaLLM(
        model=ollama_model,
        base_url=ollama_host,
        temperature=temperature,
        callbacks=[langfuse_handler] if langfuse_handler is not None else [],
    )


    search_tool = DuckDuckGoSearchRun(
        callbacks=[langfuse_handler] if langfuse_handler is not None else [],
    )

    # ---- Agent ----------------------------------------------------------------
    from langchain import hub
    from langchain.agents import create_react_agent
    from langchain.schema.runnable import RunnableConfig

    prompt = hub.pull("hwchase17/react")
    # create_react_agent returns a **Runnable** (already supports .invoke/.stream)
    agent_runnable = create_react_agent(llm=llm, tools=[search_tool], prompt=prompt)

    """Query the agent and print the result + Langfuse trace URL."""
    result = agent_runnable.invoke(
        input={"input": question, "intermediate_steps": []},
        config=RunnableConfig(run_id=RUN_ID, run_name="Ollama_agent", 
        callbacks=[cb for cb in [langfuse_handler] if cb is not None]),
    )

    # The runnable returns an AgentFinish dict; extract the final answer
    output = result["output"] if isinstance(result, dict) else result

    print("\nAnswer:", output)
    if langfuse_handler is not None:
        print("Trace URL:", langfuse_handler.get_trace_url())
    else:
        print("Trace URL: Langfuse handler is not initialized.")
    print("Trace ID :", RUN_ID)


if __name__ == "__main__":
    run("Who won the Preakness Stakes in 2025?")

