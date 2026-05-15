import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understanding agents in Langchain

    Running the tutorial <https://docs.langchain.com/oss/python/langgraph/workflows-agents>
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Structured Output and Tool Calling

    Choose a model that provides structured outputs
    """)
    return


@app.cell
def _():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", project="rag-sample-proj"
    )  # Using ADC, so we have specify the project name
    return (llm,)


@app.cell
def _(llm, mo):
    from pydantic import BaseModel, Field


    class SearchQuery(BaseModel):
        search_query: str = Field(
            None, description="Query that is optimized for web search."
        )
        justification: str = Field(
            None, description="Why query is relevant to user's request."
        )

        def __repr__(self):
            return f"{self.search_query=} \n{self.justification=}"

    # Augment LLM with structured outputs
    structured_llm = llm.with_structured_output(SearchQuery)

    structured_output = structured_llm.invoke("How does Calcium CT score relate to high cholestrol?")
    mo.output.append(structured_output)

    def multiply(a:int, b: int) -> int:
        return a * b

    llm_with_tools = llm.bind_tools([multiply])

    msg = llm_with_tools.invoke("What is 2 times 3?")

    mo.output.append(msg.tool_calls)
    return (structured_output,)


@app.cell
def _(structured_output):
    structured_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prompt chaining

    When LLM call uses the output of a previous LLM call
    """)
    return


@app.cell
def _(llm, mo):
    # Using Graph API
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END

    # Graph state
    class State(TypedDict):
        topic: str
        joke: str
        improved_joke: str
        final_joke: str

    # Nodes (functions)
    def generate_joke(state: State):
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {'joke': msg.content}

    def check_punchline(state: State):
        if '?' in state['joke'] or '!' in state['joke']:
            return "Pass"
        return "Fail"

    def improve_joke(state: State):
        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
        return {"improved_joke": msg.content}

    def polish_joke(state: State):
        msg = llm.invoke(f"Add a surprizing twist to this joke: {state['improved_joke']}")
        return {"final_joke": msg.content}

    # building graphs
    workflow = StateGraph(State)

    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_node("polish_joke", polish_joke)

    # adding edge to connect nodes
    workflow.add_edge(START, "generate_joke")
    workflow.add_conditional_edges("generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END})
    workflow.add_edge("improve_joke", "polish_joke")
    workflow.add_edge("polish_joke", END)

    # Compile graph
    chain = workflow.compile()

    # Show workflow
    # mo.output.append()

    mo.image(chain.get_graph().draw_mermaid_png())
    return (chain,)


@app.cell
def _(chain):
    # Invoke joke graph
    state = chain.invoke({"topic": "cats"})
    return (state,)


@app.cell
def _(state):
    # with mo.persistent_cache("joke_example"):
    state.items()
    return


@app.cell
def _(state):
    print(f"Initial joke: \n {state['joke'][0]['text']}")
    print("\n --- --- --- \n")

    if "improved_joke" in state:
        print("Improved joke:")
        print(state["improved_joke"][0]['text'])
        print("\n--- --- ---\n")

        print("Final joke:")
        print(state["final_joke"][0]['text'])
    else:
        print("Final joke:"[0]['text'])
        print(state["joke"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
