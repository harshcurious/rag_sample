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
def _():
    # Using Graph API
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END



    return


if __name__ == "__main__":
    app.run()
