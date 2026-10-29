"""Research-aware blog generator built with LangGraph, OpenAI, and Tavily.

Usage:
    python blog_generator.py --topic "State of Multimodal LLMs in 2026"
    python blog_generator.py --topic "Self Attention" --output-dir generated_blogs

Required environment variables:
    OPENAI_API_KEY
    TAVILY_API_KEY  (only needed when the router chooses web research)
"""

from __future__ import annotations

import argparse
import operator
import os
import re
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# Load optional values from a project-level .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Schemas
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        ge=120,
        le=550,
        description="Target word count for this section (120–550).",
    )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str
    output_dir: str


# ---------------------------------------------------------------------------
# 2. Model and prompt definitions
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen topics where correctness does not
  depend on recent facts (concepts and fundamentals).
- hybrid (needs_research=true): mostly evergreen, but up-to-date examples,
  tools, models, or releases would improve the result.
- open_book (needs_research=true): volatile topics such as weekly roundups,
  "this week", "latest", rankings, pricing, or policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries must be scoped and specific, not generic terms like "AI" or "LLM".
- If the user specifies "last week", "this week", or "latest", preserve that
  time constraint in the queries.
"""

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Given raw web search results, produce a deduplicated EvidencePack.

Rules:
- Only include items with a non-empty URL that appears in the supplied raw results.
- Never invent URLs, titles, publication dates, or source details.
- Prefer relevant, authoritative sources (official documentation, company blogs,
  research papers, and reputable publications).
- Preserve a publication date only when it is explicitly provided; never guess.
- Keep snippets concise and deduplicate by URL.
"""

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections suitable for the topic and audience.
- Each task needs a one-sentence goal, 3–6 concrete non-overlapping bullets,
  and a target word count from 120 to 550.
- Bullets should be actionable: build, compare, measure, verify, or debug.
- Across the plan, include at least two of: a minimal code example, edge cases
  or failure modes, performance/cost, security/privacy (when relevant), and
  debugging/observability. Set requires_code=true for a section that needs code.

Grounding rules:
- closed_book: focus on evergreen concepts; evidence is not required.
- hybrid: use evidence for current examples and mark affected sections with
  requires_research=true and requires_citations=true.
- open_book: set blog_kind="news_roundup"; focus on events and implications,
  not tutorials, unless explicitly requested. If evidence is insufficient,
  state that clearly and include only supported claims.

Output must match the Plan schema.
"""

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the Goal and cover ALL bullets in order; do not skip or merge them.
- Stay close to the target word count (±15%).
- Output only section content, without the blog's H1 or extra commentary.
- Start with a '## <Section Title>' heading.

Scope:
- If blog_kind is "news_roundup", do not turn the section into a tutorial.
  Focus on events, evidence, and implications.

Grounding and citations:
- If mode is "open_book", do not introduce specific event, company, model,
  funding, or policy claims unless supported by the supplied evidence URLs.
- For each sourced claim, use a Markdown link such as ([Source](URL)).
- Use only URLs included in the supplied evidence. If a claim cannot be
  supported, say "Not found in provided sources."
- If requires_citations is true, cite outside-world claims using supplied URLs.
- Evergreen reasoning may be uncited when citations are not required.

Code:
- If requires_code is true, include at least one minimal, correct code snippet.

Style:
- Use short paragraphs, helpful bullets, and fenced code blocks when relevant.
- Avoid fluff and marketing language. Be precise and implementation-oriented.
"""


# ---------------------------------------------------------------------------
# 3. Router
# ---------------------------------------------------------------------------

def router_node(state: State) -> dict:
    """Choose closed-book writing, hybrid research, or open-book research."""
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )

    queries = decision.queries[:10] if decision.needs_research else []
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": queries,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# ---------------------------------------------------------------------------
# 4. Research with Tavily
# ---------------------------------------------------------------------------

def _tavily_search(query: str, max_results: int = 6) -> List[dict]:
    """Run one Tavily query and normalize its response across tool versions."""
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is required because the router selected web research. "
            "Add it to your environment or .env file."
        )

    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})

    # langchain-tavily versions may return a list or a dict containing results.
    if isinstance(results, dict):
        results = results.get("results", [])
    if not isinstance(results, list):
        return []

    normalized: List[dict] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        if not url:
            continue
        normalized.append(
            {
                "title": item.get("title") or "",
                "url": url,
                "snippet": item.get("content") or item.get("snippet") or "",
                "published_at": (
                    item.get("published_date") or item.get("published_at")
                ),
                "source": item.get("source"),
            }
        )
    return normalized


def research_node(state: State) -> dict:
    """Search the web, synthesize evidence, and deduplicate by URL."""
    queries = (state.get("queries") or [])[:10]
    raw_results: List[dict] = []

    for query in queries:
        raw_results.extend(_tavily_search(query, max_results=6))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw search results:\n{raw_results[:60]}"),
        ]
    )

    allowed_urls = {item.get("url") for item in raw_results if item.get("url")}
    deduplicated: dict[str, EvidenceItem] = {}
    for item in pack.evidence:
        # Keep the evidence pack grounded in URLs returned by the search tool.
        if item.url in allowed_urls:
            deduplicated[item.url] = item
    return {"evidence": list(deduplicated.values())}


# ---------------------------------------------------------------------------
# 5. Outline planner
# ---------------------------------------------------------------------------

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    "Evidence (use only for fresh claims; may be empty):\n"
                    f"{[item.model_dump() for item in evidence][:20]}"
                )
            ),
        ]
    )

    # Fail early on malformed plans instead of launching workers.
    if not 5 <= len(plan.tasks) <= 9:
        raise ValueError(
            f"The planner returned {len(plan.tasks)} sections; expected 5–9. "
            "Try running again."
        )
    ids = [task.id for task in plan.tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("The generated plan contains duplicate task IDs.")
    return {"plan": plan}


# ---------------------------------------------------------------------------
# 6. Parallel fan-out and section workers
# ---------------------------------------------------------------------------

def fanout(state: State):
    plan = state["plan"]
    if plan is None:
        raise ValueError("The outline planner did not produce a plan.")

    evidence = [item.model_dump() for item in state.get("evidence", [])]
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": plan.model_dump(),
                "evidence": evidence,
            },
        )
        for task in plan.tasks
    ]


def worker_node(payload: dict) -> dict:
    task = Task.model_validate(payload["task"])
    plan = Plan.model_validate(payload["plan"])
    evidence = [
        EvidenceItem.model_validate(item)
        for item in payload.get("evidence", [])
    ]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {item.title} | {item.url} | "
        f"published: {item.published_at or 'unknown'}\n  {item.snippet or ''}"
        for item in evidence[:20]
    ) or "(No evidence was retrieved.)"

    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Section type/tags: {task.tags}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (only cite URLs listed here):\n{evidence_text}\n"
                )
            ),
        ]
    )
    section_md = response.content.strip()
    return {"sections": [(task.id, section_md)]}


# ---------------------------------------------------------------------------
# 7. Reducer: restore section order and save Markdown
# ---------------------------------------------------------------------------

def _safe_filename(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9 _-]", "", title).strip().lower()
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return (slug or "generated_blog")[:120] + ".md"


def reducer_node(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("Cannot combine sections because no plan was generated.")

    ordered_sections = [
        markdown for _, markdown in sorted(state["sections"], key=lambda pair: pair[0])
    ]
    body = "\n\n".join(ordered_sections).strip()
    final_markdown = f"# {plan.blog_title}\n\n{body}\n"

    output_dir = Path(state.get("output_dir") or "generated_blogs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _safe_filename(plan.blog_title)
    output_path.write_text(final_markdown, encoding="utf-8")

    return {"final": final_markdown}


# ---------------------------------------------------------------------------
# 8. Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------

graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker_node)
graph.add_node("reducer", reducer_node)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)
graph.add_edge("research", "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()


def run(topic: str, output_dir: str = "generated_blogs") -> dict:
    """Run the blog generator for a topic and return the final graph state."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    return app.invoke(
        {
            "topic": topic,
            "mode": "closed_book",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "final": "",
            "output_dir": output_dir,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a research-aware Markdown blog using LangGraph."
    )
    parser.add_argument(
        "--topic",
        required=True,
        help='Blog topic, e.g. "State of Multimodal LLMs in 2026"',
    )
    parser.add_argument(
        "--output-dir",
        default="generated_blogs",
        help="Directory where the generated Markdown blog will be saved.",
    )
    args = parser.parse_args()

    result = run(args.topic, args.output_dir)
    print(result["final"])
    print(f"\nSaved blog under: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
