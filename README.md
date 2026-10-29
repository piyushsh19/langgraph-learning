# Python LangGraph Learning

A hands-on learning repository for understanding **LangGraph, LangChain,
LLMs, and AI workflow orchestration** through practical Python projects
and Jupyter notebooks.

The goal is to build a strong foundation in LangGraph, progressing from
basic state management to sequential LLM workflows, parallel execution,
conditional routing, and multi-step AI applications.

## 📚 Learning Projects

### 0. Installation Testing

**Notebook:** `0_test_installation.ipynb`

-   Verify Python and package installations.
-   Test the Jupyter Notebook environment.
-   Confirm the development setup is ready for LangGraph experiments.

### 1. BMI Workflow

**Notebook:** `1_bmi_workflow.ipynb`

A beginner-friendly introduction to state-based workflows.

**Concepts covered:** - Defining workflow state using `TypedDict`. -
Creating Python functions as graph nodes. - Calculating BMI from height
and weight. - Categorizing BMI results. - Connecting nodes using graph
edges. - Compiling and invoking a LangGraph workflow.

**Workflow:**

``` text
Input → Calculate BMI → Label BMI → Output
```

### 2. Simple LLM Workflow

**Notebook:** `2_simple_llm_workflow.ipynb`

A basic question-answering workflow powered by a large language model.

**Concepts covered:** - Defining state using `TypedDict`. - Integrating
an LLM using LangChain's `ChatOpenAI`. - Creating a node to process a
question. - Generating answers using `model.invoke()`. - Updating the
workflow state. - Using `StateGraph`, `START`, and `END`. - Compiling
and invoking an LLM-powered graph.

**Workflow:**

``` text
Question → LLM Node → Answer
```

### 3. Prompt Chaining: Blog Generation and Evaluation

**Notebook:** `3_prompt_chaining.ipynb` *(use your actual filename if
different)*

A multi-step LLM workflow that generates a blog outline, writes the
complete blog, and evaluates the generated content.

**Concepts covered:** - Sequential LLM execution. - Sharing data between
nodes through graph state. - Generating a blog outline from a topic. -
Using the outline to generate a complete blog. - Evaluating generated
content with another LLM call. - Assigning a quality score between 1 and
10. - Storing evaluation feedback and scores in the state. - Connecting
multiple nodes using LangGraph edges.

**Workflow:**

``` text
Input: Blog Title
       |
       v
create_outline
       |
       v
create_blog
       |
       v
evaluate_blog
       |
       v
END: Blog + Score + Feedback
```

**Nodes:** - `create_outline`: generates and stores an outline in
`state["outline"]`. - `create_blog`: uses the title and outline to
generate `state["content"]`. - `evaluate_blog`: evaluates relevance,
structure, clarity, depth, and grammar, then stores feedback and a
score.

Example state:

``` python
class BlogState(TypedDict):
    title: str
    outline: str
    content: str
    evaluation: str
    score: int
```

The score is an LLM-generated quality estimate, not an objective
guarantee of correctness.

### 4. Parallel Workflow: Cricket Batting Statistics

**Notebook:** `4_batsman_workflow.ipynb`

A parallel workflow that calculates multiple batting statistics
independently and combines them into a summary.

**Concepts covered:** - Running independent graph nodes in parallel. -
Calculating strike rate, balls per boundary, and boundary percentage. -
Understanding fan-out and fan-in graph patterns. - Combining results in
a summary node. - Working with shared state and concurrent updates.

**Workflow:**

``` text
                         ┌─ calculate_sr ───────────┐
START ───────────────────┼─ calculate_bpb ──────────┼─> summary ─> END
                         └─ calculate_boundary_percent ┘
```

The calculation nodes run independently. Their results are then used by
the `summary` node. When multiple parallel nodes update the same state
key, configure an appropriate reducer or use separate state keys to
avoid conflicting updates.

### 5. Conditional Workflow: Quadratic Equation

**Notebook:** `6_quadratic_equation_workflow.ipynb`

A conditional workflow that calculates the discriminant of a quadratic
equation and routes execution based on its value.

For (ax\^2 + bx + c = 0), the discriminant is:

``` text
D = b² - 4ac
```

**Concepts covered:** - Creating a typed workflow state. - Calculating a
deterministic value in a graph node. - Using
`add_conditional_edges()`. - Defining a routing function. - Branching
execution based on a result. - Handling three possible outcomes.

**Workflow:**

``` text
START
  |
  v
show_equation
  |
  v
calculate_discriminant
  |
  ├── D < 0 ──> no_real_roots ──> END
  ├── D > 0 ──> real_roots ─────> END
  └── D = 0 ──> repeated_roots ─> END
```

The coefficient `a` must not be zero for the equation to be quadratic.

### 6. LLM-Based Review Handling Workflow

**Notebook:** `7_review_handling_workflow.ipynb` *(suggested filename;
adjust to the actual filename)*

An LLM-powered customer-review workflow that classifies sentiment and
generates an appropriate response. Negative reviews receive additional
diagnosis before a response is written.

**Concepts covered:** - Using an LLM to classify review sentiment. -
Routing to different branches with conditional edges. - Running
diagnosis only for negative reviews. - Generating empathetic responses
to negative reviews. - Generating appreciative responses to positive
reviews. - Passing results between nodes through shared state.

**Workflow:**

``` text
START
  |
  v
find_sentiment
  |
  ├── Negative ──> run_diagnosis ──> negative_response ──> END
  |
  └── Positive ────────────────────> positive_response ──> END
```

**Nodes:** - `find_sentiment`: classifies a review as positive or
negative. - `run_diagnosis`: analyzes the problem described in a
negative review. - `negative_response`: creates an empathetic response
informed by the diagnosis. - `positive_response`: thanks the customer
and responds to positive feedback.

This example demonstrates LLM-based routing: the LLM classifies the
review, while LangGraph's conditional edge selects the execution path.
For production use, validate model output against allowed labels and
consider adding a neutral or uncertain sentiment path.

### 7. Iterative Workflow: X/Twitter Post Generator

**Notebook:** `8_X_post_generator.ipynb`

An iterative LLM workflow that creates a tweet, evaluates it against defined quality criteria, and rewrites it using the evaluator's feedback until it is approved or the maximum number of iterations is reached.

**Concepts covered:**
- Building an iterative workflow with LangGraph.
- Defining shared state with `TypedDict` and `Literal`.
- Generating content with an LLM.
- Evaluating generated content using structured output.
- Defining a Pydantic model for consistent evaluation results.
- Routing conditionally based on the evaluation result.
- Feeding evaluator feedback into an optimization node.
- Looping between optimization and evaluation.
- Tracking the current iteration and enforcing a maximum iteration limit.

**Workflow:**

```text
START
  |
  v
generate_tweet
  |
  v
evaluate_tweet
  |
  v
route_evaluation
  |
  +-- approved --------------------------> END
  |
  +-- needs_improvement --> optimize_tweet
                                |
                                +--------> evaluate_tweet
```

**Shared state:**

```python
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
```

**Nodes:**
- `generate_tweet`: creates an original, concise tweet for the supplied topic.
- `evaluate_tweet`: evaluates originality, humor, punchiness, virality potential, and tweet format. It returns a structured decision and written feedback.
- `optimize_tweet`: rewrites the current tweet using the evaluator's feedback and increments the iteration count.
- `route_evaluation`: sends approved tweets to `END`; otherwise, it routes to optimization while iteration limits allow.

The workflow demonstrates an **evaluator-optimizer pattern**: one LLM generates content, another evaluates it, and the optimizer uses feedback to improve the next version. A maximum iteration limit is important to prevent an endless loop if the evaluator never approves a tweet.


### 9. Basic Chatbot with LangGraph Memory (`9_basic_chatbot.ipynb`)

**Objective:** Build a conversational chatbot using LangGraph, LangChain, and OpenAI, with conversation history maintained through checkpointing.

**Concepts covered:**
- Defining message-based state using `TypedDict` and `Annotated`.
- Using `add_messages` to maintain conversation history.
- Creating a chatbot node with `ChatOpenAI`.
- Building a `StateGraph` with `START`, a chatbot node, and `END`.
- Compiling the graph with `MemorySaver`.
- Using `thread_id` to maintain separate conversation histories.
- Implementing an interactive chatbot loop with exit commands.

**Workflow architecture:**

```text
       START
         |
         v
     chat_node
         |
         v
        END
```

**How it works:**

1. The user submits a message to the chatbot.
2. LangGraph loads the existing checkpoint for the specified `thread_id`.
3. The `chat_node` sends the conversation history to the LLM.
4. The LLM generates a response, which is added to the message history.
5. LangGraph saves the updated state using `MemorySaver`.
6. Subsequent calls with the same `thread_id` continue the conversation.

**Key takeaway:** `MemorySaver` enables conversation continuity across graph invocations during the current process. Different thread IDs maintain separate conversations. For persistence across application restarts, a durable checkpointer is required.

### 10. Persistence and Threads (`10_persistence.ipynb`)

**Objective:** Understand how LangGraph checkpointing saves workflow state and how `thread_id` separates checkpoint histories.

**Concepts covered:**
- Defining shared state with `TypedDict`.
- Building a sequential graph with `StateGraph`, nodes, edges, `START`, and `END`.
- Saving graph checkpoints with `InMemorySaver`.
- Passing `thread_id` through the `configurable` runtime configuration.
- Inspecting saved state with `workflow.get_state(config)`.
- Comparing repeated invocations in the same thread with invocations in separate threads.

**Workflow architecture:**

```text
              START
                |
                v
         generate_joke
                |
                v
      generate_explanation
                |
                v
               END
```

**How it works:**

1. The input state supplies a topic.
2. `generate_joke` calls the LLM and stores the generated joke in state.
3. `generate_explanation` reads the joke, asks the LLM to explain it, and stores the explanation.
4. The compiled graph uses `InMemorySaver` to checkpoint state during execution.
5. A `thread_id` identifies the checkpoint history used for an invocation.
6. `workflow.get_state(config)` retrieves the latest saved state for that thread.

**Thread configuration:**

```python
config = {"configurable": {"thread_id": "1"}}
result = workflow.invoke({"topic": "pizza"}, config=config)
```

Use the same `thread_id` to access the same thread's checkpoint history. Use a different ID, such as `"2"`, to keep another thread separate.

**Important distinction:** Checkpointing saves state between graph invocations, but it does not automatically make a completed workflow continue from its final node. A normal `invoke()` runs the graph from its start using the supplied input and the selected thread's saved state. `InMemorySaver` is process-local and does not survive a Python process restart; durable applications should use a persistent checkpointer.

**Key takeaway:** LangGraph persistence is built around checkpoints and thread identities. A checkpointer stores state, while `thread_id` tells LangGraph which workflow history to access.

### 11.Subgraphs: Local LLM Answer and Hindi Translation
Python example: 15_subgraph_invokeState.py
Workflow: Question → English answer → Hindi translation subgraph

This project demonstrates how to compose a LangGraph parent graph with a compiled child graph (subgraph). A local Hugging Face model, Qwen/Qwen2.5-3B-Instruct, first answers the user's question in English. The parent graph then invokes a translation subgraph to translate that answer into Hindi.

Unlike a cloud-hosted model workflow, this example loads the model from the local Hugging Face cache using local_files_only=True. The model must already be downloaded and available in the active Python environment's Hugging Face cache.
Learning objectives
- Load and call a local Hugging Face instruction model through LangChain.
- Define separate state schemas for a parent graph and a subgraph.
- Build and compile a standalone subgraph.
- Invoke the compiled subgraph from a parent-graph node.
- Pass values into a subgraph and map its output back to the parent state.
- Compose multiple LLM tasks into a sequential workflow.
Architecture
Parent graph
START
  |
  v
[answer]
  | Generates answer_eng from question
  v
[translate]
  | Invokes the compiled translation subgraph
  v
  +------------------------------+
  | Translation subgraph         |
  |                              |
  | START                        |
  |   |                          |
  |   v                          |
  | [translate_text]             |
  |   | Produces translated_text |
  |   v                          |
  | END                          |
  +------------------------------+
  |
  | Returns translated_text to the parent
  v
END
State schemas
Parent state — ParentState
Field	Type	Description
question	str	Question supplied by the user
answer_eng	str	English answer generated by the local LLM
answer_hin	str	Hindi translation returned by the subgraph


### 12.Subgraph state — SubState
Field	Type	Description
input_text	str	English answer passed into the translation subgraph
translated_text	str	Hindi translation produced by the subgraph


The parent and child graphs use different state schemas. The translate_answer node passes answer_eng into the subgraph as input_text, then copies the returned translated_text into the parent's answer_hin field.
Nodes and edges
Parent graph
- answer — creates a concise English answer using the local LLM.
- translate — invokes the translation subgraph and saves the Hindi result.
- Edges: START → answer → translate → END.
Translation subgraph
- translate_text — instructs the local LLM to translate the supplied text into natural, clear Hindi, preserving its meaning and returning only the translation.
- Edges: START → translate_text → END.
Execution flow
1. The script loads the local Qwen model through HuggingFacePipeline.from_model_id.
2. ChatHuggingFace wraps the pipeline so it can receive LangChain messages.
3. The user enters a question.
4. The parent node answer generates an English answer and stores it in answer_eng.
5. The translate node invokes the compiled subgraph with {"input_text": state["answer_eng"]}.
6. The subgraph's translate_text node produces translated_text.
7. The parent graph stores the translation in answer_hin and returns both answers.
The same loaded model instance is reused for both tasks, avoiding a second model load. Both LLM calls still run separately, so generation time depends on the model and hardware.
Prerequisites
- Python environment with the dependencies below installed.
- The model Qwen/Qwen2.5-3B-Instruct already present in the Hugging Face cache.
- Enough RAM/unified memory to load and run the 3B-parameter model.
- A compatible PyTorch and Transformers setup.
Installation
Activate your project's virtual environment, then install the packages:
python -m pip install --upgrade pip
python -m pip install langgraph langchain-core langchain-huggingface transformers torch accelerate python-dotenv
If the model has not been downloaded yet, download it once while internet access is available. The script sets local_files_only=True, so it will not download missing model files at runtime.
Run the example
Save the code as 15_subgraphs.py, then run:
python 15_subgraphs.py
Enter a question when prompted. The script prints the English answer and its Hindi translation.
Configuration notes
- Model: Qwen/Qwen2.5-3B-Instruct
- Inference mode: do_sample=False, which uses deterministic decoding settings.
- Generation limit: max_new_tokens=256 for each model call.
- Offline loading: local_files_only=True requires the model files to exist locally.
- API key: This example does not call ChatOpenAI and does not require an OpenAI API key. load_dotenv() is harmless but not required for this local-model workflow.
Key takeaway
A compiled LangGraph subgraph can be called from a node in a parent graph. The child graph encapsulates one responsibility—translation—while the parent coordinates the overall process and maps data between the two state schemas.

Shared-State Subgraphs: Draft Review and Improvement
Example: 16_shared_state_subgraph.py
Local model: Qwen/Qwen2.5-3B-Instruct

This workflow demonstrates a LangGraph parent graph that calls a compiled improvement subgraph. Both graphs use the same SharedState schema, so the child graph can read fields created by the parent and write additional fields that become available to later parent nodes.

The local Qwen model is loaded once and reused for drafting, review, revision, and summarization.
Learning objectives
- Define a shared state schema for a parent graph and its subgraph.
- Add a compiled subgraph directly as a node in the parent graph.
- Read existing state fields inside child-graph nodes.
- Write new state fields that are visible to the parent graph after the subgraph completes.
- Build a sequential content-improvement workflow using a local LLM.
Architecture
Parent graph
START
  |
  v
[create_draft]
  | Reads: topic
  | Writes: draft
  v
[improvement_subgraph]
  |
  |  Subgraph (same SharedState schema)
  |  START
  |    |
  |    v
  |  [review_draft]
  |    | Reads: topic, draft
  |    | Writes: feedback
  |    v
  |  [revise_draft]
  |    | Reads: topic, draft, feedback
  |    | Writes: revised_draft
  |    v
  |  END
  |
  | Returns updated shared state
  v
[create_final_summary]
  | Reads: topic, revised_draft
  | Writes: final_summary
  v
END
Shared state schema
class SharedState(TypedDict, total=False):
    topic: str
    draft: str
    feedback: str
    revised_draft: str
    final_summary: str
total=False makes the fields optional in the TypedDict definition. This is useful because not every field exists at the start of the workflow: the initial input contains only topic, and later nodes progressively add draft, feedback, revised_draft, and final_summary.

Because both StateGraph(SharedState) definitions use the same schema, the subgraph's nodes can access the parent-created topic and draft fields. The subgraph writes feedback and revised_draft; those updates are available to the parent graph after the subgraph node completes.
Nodes and responsibilities
Parent graph
- create_draft — generates an approximately 150-word initial draft from the topic.
- improvement_subgraph — the compiled child graph, registered directly as a parent node.
- create_final_summary — produces three concise bullet points from the revised draft.
Improvement subgraph
- review_draft — reviews clarity, accuracy, organization, and completeness, then returns three actionable suggestions in feedback.
- revise_draft — uses the original draft and feedback to create revised_draft.
Edges and execution order
Parent graph:
START → create_draft → improvement_subgraph → create_final_summary → END
Subgraph:
START → review_draft → revise_draft → END
Both paths are sequential. There is no conditional routing or loop in this example.
State flow
Field	Created or updated by	Used by
topic	Initial input	create_draft, review_draft, revise_draft, create_final_summary
draft	create_draft	review_draft, revise_draft
feedback	review_draft	revise_draft
revised_draft	revise_draft	create_final_summary
final_summary	create_final_summary	Final graph result


Setup and execution
Install the dependencies in your activated virtual environment:
python -m pip install langgraph langchain-core langchain-huggingface transformers torch accelerate python-dotenv
The model must already be available in the local Hugging Face cache because the code sets local_files_only=True. If it is not cached, download it while internet access is available before running the workflow.

Save the implementation as 16_shared_state_subgraph.py and run:
python 16_shared_state_subgraph.py
Enter a topic when prompted. The program prints the initial draft, editor feedback, revised draft, and final three-point summary.
Important concepts
- Shared schema: Both parent and child graphs use SharedState.
- Shared state values: The subgraph receives the current state, including topic and draft, and returns updates such as feedback and revised_draft.
- Compiled subgraph: improvement_subgraph is compiled first and then added to the parent graph with parent_builder.add_node("improvement_subgraph", improvement_subgraph).
- One model instance: The same local LLM instance is reused for all four model calls, avoiding repeated model loading.
- Run metadata: The config passed to graph.invoke supplies a run name, tags, and model metadata for tracing or observability integrations; it does not change the graph's state schema.
Key takeaway
A subgraph can use the same state schema as its parent graph. This lets the subgraph work with fields already produced upstream and return new state updates for downstream parent nodes, while keeping the review-and-revision logic encapsulated in a reusable component.

## 🛠️ Tech Stack

-   **Python** --- Programming language
-   **LangGraph** --- Graph-based workflow orchestration
-   **LangChain** --- LLM integration and application components
-   **OpenAI API** --- LLM-powered text generation
-   **Jupyter Notebook** --- Interactive development
-   **python-dotenv** --- Environment variable management
-   **Git and GitHub** --- Version control

## ⚙️ Setup and Installation

### Prerequisites

-   Python 3.12 or another Python version compatible with your installed
    dependencies
-   Git
-   VS Code with the Python and Jupyter extensions
-   An OpenAI API key for LLM-powered examples

### 1. Clone the repository

``` bash
git clone https://github.com/piyushsh19/langgraph-learning.git
cd langgraph-learning
```

Skip this step if you already have the repository locally.

### 2. Create a virtual environment

On macOS:

``` bash
python3.12 -m venv .venv
```

### 3. Activate the environment

``` bash
source .venv/bin/activate
```

### 4. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install langgraph langchain-openai python-dotenv jupyter ipykernel
```

If a notebook uses additional packages, install those in the same
virtual environment.

### 5. Register the Jupyter kernel

``` bash
python -m ipykernel install --user --name langgraph-learning --display-name "Python (LangGraph Learning)"
```

Select **Python (LangGraph Learning)** as the kernel in VS Code.

## 🔑 Environment Variables

Create a `.env` file in the project root:

``` env
OPENAI_API_KEY=your_openai_api_key_here
```

Load the key in your notebook:

``` python
from dotenv import load_dotenv

load_dotenv()
```

Initialize the model:

``` python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

Replace the placeholder with your actual API key. Never commit `.env` or
expose API keys in source code or notebooks.

## ▶️ Running the Notebooks

1.  Open the repository in VS Code.
2.  Activate the `.venv` environment.
3.  Select the appropriate Jupyter kernel.
4.  Open a notebook.
5.  Run cells in order, from top to bottom.
6.  Review the output and experiment with different inputs.

For LLM-powered workflows, ensure your API key is configured and your
OpenAI account has access to the selected model.

## 🧠 Core LangGraph Concepts

  Concept              Purpose
  -------------------- ---------------------------------------------------------
  `TypedDict`          Defines the workflow state structure
  `StateGraph`         Creates a graph-based workflow
  Nodes                Execute individual processing steps
  Edges                Define the execution order
  `START`              Marks the workflow entry point
  `END`                Marks workflow completion
  `compile()`          Creates an executable graph
  `invoke()`           Executes the workflow with initial state
  State updates        Pass information between nodes
  Prompt chaining      Uses one LLM step's output in a subsequent step
  Parallel execution   Runs independent nodes concurrently
  Conditional edges    Route execution based on state or node output
  LLM evaluation       Uses an LLM to assess generated content
  Reducers             Define how multiple updates to a state key are combined

## 📁 Repository Structure

The repository may look like this as you add the notebooks:

``` text
langgraph-learning/
├── .gitignore
├── README.md
├── .env.example
├── 0_test_installation.ipynb
├── 1_bmi_workflow.ipynb
├── 2_simple_llm_workflow.ipynb
├── 3_prompt_chaining.ipynb
├── 4_batsman_workflow.ipynb
├── 5_UPSC_essay_workflow.ipynb
├── 6_quadratic_equation_workflow.ipynb
├── 7_review_handling_workflow.ipynb
└── 8_X_post_generator.ipynb
```

This is an illustrative structure based on the notebooks discussed so
far. Keep your actual filenames and omit files that do not exist in your
repository. The UPSC essay workflow is included because it was also part
of the learning work discussed.

## 🔒 Git Best Practices

Commit source files, notebooks, documentation, and dependency files. Do
not commit virtual environments, generated caches, or secrets.

A suitable `.gitignore` for this project is:

``` gitignore
# Virtual environments
.venv/
venv/
myenv/
env/

# Environment variables and secrets
.env
.env.*
!.env.example

# Python cache and build artifacts
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# Jupyter
.ipynb_checkpoints/

# IDE and operating system files
.vscode/
.DS_Store

# Logs and local databases
*.log
*.db
*.sqlite3
```

**Important:** If your virtual environment was already tracked by Git,
adding it to `.gitignore` will not automatically remove it from version
control. To stop tracking it while keeping the local files, run:

``` bash
git rm -r --cached .venv
git add .gitignore
git status
```

Run the `git rm` command only if `.venv` is the name of your environment
and Git is currently tracking it. Review `git status` before committing.

Recommended practices: - Never commit API keys or other secrets. - Clear
unnecessary notebook outputs before committing when appropriate. - Use
meaningful commit messages. - Check `git status` before every commit. -
Keep `.gitignore` in every Python repository.

## 🚀 Future Learning Goals

-   Structured LLM output
-   Error handling and retries
-   Tool calling and AI agents
-   Memory and checkpointing
-   Human-in-the-loop workflows
-   Retrieval-Augmented Generation (RAG)
-   LLM evaluation and testing
-   Workflow deployment and monitoring

## 🎯 Objective

Build practical expertise in LangGraph by progressing from basic
deterministic workflows to sequential LLM applications, parallel
execution, conditional routing, evaluation pipelines, and
production-oriented AI agents.

------------------------------------------------------------------------

**Author:** Piyush Sharma

**Repository:**
[langgraph-learning](https://github.com/piyushsh19/langgraph-learning)
