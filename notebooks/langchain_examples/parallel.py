import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parallelization

    ![img](https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/parallelization.png?w=1100&fit=max&auto=format&n=dL5Sn6Cmy9pwtY0V&q=85&s=6227d2c39f332eaeda23f7db66871dd7)
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END

    return BaseModel, END, Field, START, StateGraph, TypedDict


@app.cell
def _():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", project="rag-sample-proj"
    )  # Using ADC, so we have specify the project name
    return (llm,)


@app.cell
def _(END, START, StateGraph, TypedDict, llm, mo):
    # Graph state
    class State(TypedDict):
        topic: str
        joke: str
        story: str
        poem: str
        combined_output: str


    # Node functions
    def call_llm_joke(state: State):
        msg = llm.invoke(f"Write a joke about {state['topic']}")
        return {"joke": msg.text}


    def call_llm_story(state: State):
        msg = llm.invoke(f"Write a story about {state['topic']}")
        return {"story": msg.text}


    def call_llm_poem(state: State):
        msg = llm.invoke(f"Write a poem about {state['topic']}")
        return {"poem": msg.text}


    def aggregator(state: State):
        combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
        combined += f"STORY:\n{state['story']}\n\n"
        combined += f"JOKE:\n{state['joke']}\n\n"
        combined += f"POEM:\n{state['poem']}"
        return {"combined_output": combined}


    # workflow
    parallel_builder = StateGraph(State)

    # Add nodes
    parallel_builder.add_node("joke", call_llm_joke)
    parallel_builder.add_node("story", call_llm_story)
    parallel_builder.add_node("poem", call_llm_poem)
    parallel_builder.add_node("agg", aggregator)

    # add parallel edges
    parallel_builder.add_edge(START, "joke")
    parallel_builder.add_edge(START, "story")
    parallel_builder.add_edge(START, "poem")
    parallel_builder.add_edge("joke", "agg")
    parallel_builder.add_edge("story", "agg")
    parallel_builder.add_edge("poem", "agg")
    parallel_builder.add_edge("agg", END)

    parallel_workflow = parallel_builder.compile()

    mo.image(parallel_workflow.get_graph().draw_mermaid_png())

    state = parallel_workflow.invoke({"topic": "cats"})
    mo.output.append(state["combined_output"])
    return (state,)


@app.cell
def _(state):
    from pprint import pprint

    pprint(state["combined_output"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Routing workflow

    ![img](https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/routing.png?w=1100&fit=max&auto=format&n=dL5Sn6Cmy9pwtY0V&q=85&s=9aaa28410da7643f4a2587f7bfae0f21)
    """)
    return


@app.cell
def _(BaseModel, END, Field, START, StateGraph, TypedDict, llm, mo):
    from typing_extensions import Literal
    from langchain.messages import HumanMessage, SystemMessage


    class Route(BaseModel):
        step: Literal["poem", "story", "joke"] = Field(
            None, description="The next step in the routing process"
        )


    router = llm.with_structured_output(Route)


    # state
    class RouterState(TypedDict):
        input: str
        decision: str
        output: str


    # Nodes
    def llm_call_story2(state: RouterState):
        result = llm.invoke(state["input"])
        return {"output": result.text}


    def llm_call_joke2(state: RouterState):
        result = llm.invoke(state["input"])
        return {"output": result.text}


    def llm_call_poem2(state: RouterState):
        result = llm.invoke(state["input"])
        return {"output": result.text}


    def llm_call_router(state: RouterState):
        decision = router.invoke(
            [
                SystemMessage(
                    content="Route the input to story, joke, or poem based on the user's request."
                ),
                HumanMessage(content=state["input"]),
            ]
        )

        return {"decision": decision.step}


    def route_decision(state: RouterState):
        if state["decision"] == "story":
            return "story"
        elif state["decision"] == "joke":
            return "joke"
        elif state["decision"] == "poem":
            return "poem"


    router_builder = StateGraph(RouterState)

    router_builder.add_node("story", llm_call_story2)
    router_builder.add_node("joke", llm_call_joke2)
    router_builder.add_node("poem", llm_call_poem2)
    router_builder.add_node("router", llm_call_router)

    router_builder.add_edge(START, "router")
    router_builder.add_conditional_edges(
        "router", route_decision, ["story", "joke", "poem"]
    )

    router_builder.add_edge("story", END)
    router_builder.add_edge("joke", END)
    router_builder.add_edge("poem", END)

    router_workflow = router_builder.compile()

    mo.image(router_workflow.get_graph().draw_mermaid_png())
    return HumanMessage, SystemMessage, router_workflow


@app.cell
def _(router_workflow):
    router_run = router_workflow.invoke({"input": "Write me a joke about cats"})
    print(router_run["output"])
    return (router_run,)


@app.cell
def _(router_run):
    router_run
    return


@app.cell
def _(router_run):
    print(router_run["output"][0]["text"])
    return


app._unparsable_cell(
    r"""
    # Orchestrator Worker config
    ![img](https://mintcdn.com/langchain-5e9cc07a/ybiAaBfoBvFquMDz/oss/images/worker.png?fit=max&auto=format&n=ybiAaBfoBvFquMDz&q=85&s=2e423c67cd4f12e049cea9c169ff0676)
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(BaseModel, Field, llm):
    from typing import Annotated, List


    class Section(BaseModel):
        name: str = Field(description="Name for this section of the report.")
        description: str = Field(
            description="Brief overview of the main topics and concepts to be covered in this section."
        )


    class Sections(BaseModel):
        sections: List[Section] = Field(description="Sections of the report.")


    planner = llm.with_structured_output(Sections)
    return Annotated, List, Section, planner


@app.cell
def _(
    Annotated,
    END,
    HumanMessage,
    List,
    START,
    Section,
    StateGraph,
    SystemMessage,
    TypedDict,
    llm,
    mo,
    planner,
):
    ## Creating workers

    from langgraph.types import Send
    import operator


    class ReportState(TypedDict):
        topic: str
        sections: List[Section]
        completed_sections: Annotated[
            list, operator.add
        ]  # all workers write to this in parallel
        final_report: str


    class WorkerState(TypedDict):
        section: Section
        completed_sections: Annotated[list, operator.add]


    # Nodes
    def orchestrator(state: ReportState):
        report_sections = planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(
                    content=f"Here is the report topic: {state['topic']}"
                ),
            ]
        )
        # print(report_sections.usage_metadata)
        return {"sections": report_sections.sections}

    def llm_worker(state: WorkerState):
        section = llm.invoke(
            [SystemMessage(content="Write a section of the report based on the provided name and description. Include no preamble for each section. Use markdown formatting."), 
             HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}")]
        )
        print(section.usage_metadata)
        return {"completed_sections": [section.text]}

    def synthesizer(state: ReportState):
        completed_sections = state["completed_sections"]
        # print("Completed sections:", completed_sections)
        combined = "\n\n--- --- ---\n\n".join(completed_sections)
        return {"final_report": combined}

    # Conditional edge function to create llm_call workers that each write a section of the report
    def assign_workers(state: ReportState):
        """Assign a worker to each section in the plan"""

        # Kick off section writing in parallel via Send() API
        return [Send("worker", {"section": s}) for s in state["sections"]]


    # workflow
    orchestrator_worker_builder = StateGraph(ReportState)

    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("worker", llm_worker)
    orchestrator_worker_builder.add_node("synthesizer", synthesizer)

    # edges
    orchestrator_worker_builder.add_edge(START, "orchestrator")
    orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, "worker")
    orchestrator_worker_builder.add_edge("worker", "synthesizer")
    orchestrator_worker_builder.add_edge("synthesizer", END)

    orchestrator_worker_workflow = orchestrator_worker_builder.compile()

    mo.image(orchestrator_worker_workflow.get_graph().draw_mermaid_png())

    report = orchestrator_worker_workflow.invoke({"topic": "The impact of AI on society"})
    mo.md(report["final_report"])
    return (orchestrator_worker_workflow,)


@app.cell
def _(mo, orchestrator_worker_workflow):
    mo.image(orchestrator_worker_workflow.get_graph().draw_mermaid_png())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluator-Optimizer Architecture
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
