import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    from pprint import pprint

    return


@app.cell
def _():
    import os
    from langchain.chat_models import init_chat_model
    from langchain.messages import SystemMessage, HumanMessage

    return HumanMessage, SystemMessage, init_chat_model


@app.cell
def _(HumanMessage, SystemMessage, init_chat_model):
    model_gemini = init_chat_model("gemini-2.5-flash")

    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("Explain Langchain in one sentence."),
    ]
    return messages, model_gemini


@app.cell(disabled=True)
def _(messages, model_gemini):
    response1 = model_gemini.invoke(messages)
    print(response1.content)
    return


@app.cell
def _():
    from langchain.agents import create_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent_assistant = create_agent(
        model="ollama:qwen3.5:9b",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent_assistant.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
    )
    print(result["messages"][-1].content_blocks)
    return (create_agent,)


@app.cell
def _():
    SYSTEM_PROMPT = """You are a literary data assistant.

    ## Capabilities

    - `fetch_text_from_url`: loads document text from a URL into the conversation.
    Do not guess line counts or positions—ground them in tool results from the saved file."""
    return (SYSTEM_PROMPT,)


@app.cell
def _():
    import urllib.error
    import urllib.request

    from langchain.tools import tool


    @tool
    def fetch_text_from_url(url: str) -> str:
        """Fetch the document from a URL.
        """
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            return f"Fetch failed: {e}"
        text = raw.decode("utf-8", errors="replace")
        return text

    return (fetch_text_from_url,)


@app.cell
def _(init_chat_model):
    ollama_model = init_chat_model(
        "ollama:qwen3.5:9b",
        temperature=0.5,
        timeout=300,
        max_tokens=25000,
    )
    return


@app.cell
def _():
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    return (checkpointer,)


@app.cell
def _(
    SYSTEM_PROMPT,
    checkpointer,
    create_agent,
    fetch_text_from_url,
    model_gemini,
):
    from deepagents import create_deep_agent

    agent = create_agent(
        model=model_gemini,
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    deep_agent = create_deep_agent(
        model=model_gemini,
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    content = f"""Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
    URL: https://www.gutenberg.org/files/64317/64317-0.txt

    Answer as much as you can:

    1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
    2) The 1-based line number of the first line in the file that contains `Daisy`.
    3) A two-sentence neutral synopsis.

    Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
    your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
    the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-lc"}},
    )
    deep_agent_result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-da"}},
    )
    print(agent_result["messages"][-1].content_blocks)
    print("\n")
    print(deep_agent_result["messages"][-1].content_blocks)
    return agent_result, deep_agent_result


@app.cell
def _(agent_result, mo):
    mo.output.append(agent_result["messages"])
    mo.output.append(type(agent_result["messages"][-1].content[0]['text']))
    mo.output.append(agent_result["messages"][-1].content[0]['text'])
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(deep_agent_result, mo):
    mo.output.append(deep_agent_result["messages"])
    mo.output.append(type(deep_agent_result["messages"][-1].content[0]['text']))
    mo.output.append(deep_agent_result["messages"][-1].content[0]['text'])
    return


@app.cell
def _(deep_agent_result, mo):
    mo.output.append(deep_agent_result["messages"][0].pretty_print())
    mo.output.append(deep_agent_result["messages"][-1].pretty_print())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
