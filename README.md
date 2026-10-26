# Python LangGraph Learning

A hands-on learning repository for understanding **LangGraph, LangChain, LLMs, and AI workflow orchestration** through practical Python projects and Jupyter notebooks.

The goal is to build a strong foundation in LangGraph, starting with basic state management and progressing to sequential LLM workflows and multi-step AI applications.

## 📚 Learning Projects

### 0. Installation Testing

**Notebook:** `0_test_installation.ipynb`

- Verify Python and package installations.
- Test the Jupyter Notebook environment.
- Confirm the development setup is ready for LangGraph experiments.

### 1. BMI Workflow

**Notebook:** `1_bmi_workflow.ipynb`

A beginner-friendly introduction to state-based workflows.

**Concepts covered:**
- Defining workflow state using `TypedDict`.
- Creating Python functions as graph nodes.
- Calculating BMI from height and weight.
- Categorizing BMI results.
- Connecting nodes using graph edges.
- Compiling and invoking a LangGraph workflow.

**Workflow:**

`Input → Calculate BMI → Label BMI → Output`

### 2. Simple LLM Workflow

**Notebook:** `2_simple_llm_workflow.ipynb`

A basic question-answering workflow powered by a large language model.

**Concepts covered:**
- Defining state using `TypedDict`.
- Integrating an LLM using LangChain's `ChatOpenAI`.
- Creating a node to process a question.
- Generating answers using `model.invoke()`.
- Updating the workflow state.
- Using `StateGraph`, `START`, and `END`.
- Compiling and invoking an LLM-powered graph.

**Workflow:**

`Question → LLM Node → Answer`

### 3. Prompt Chaining with Blog Generation and Evaluation

**Notebook:** `3_prompt_chaining.ipynb` (suggested filename)

A multi-step LLM workflow that generates a blog outline, writes the complete blog, and evaluates the generated content.

**Concepts covered:**
- Sequential LLM execution.
- Sharing data between nodes through graph state.
- Generating a blog outline from a topic.
- Using the outline to generate a complete blog.
- Evaluating generated content with another LLM call.
- Assigning a quality score between 1 and 10.
- Storing evaluation feedback and scores in the state.
- Connecting multiple nodes using LangGraph edges.

#### Workflow architecture

```text
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

#### Node 1: `create_outline`

- Reads the blog title from the state.
- Prompts the LLM to generate a detailed outline.
- Stores the outline in `state["outline"]`.

#### Node 2: `create_blog`

- Reads the title and generated outline.
- Generates the full blog using both inputs.
- Stores the result in `state["content"]`.

#### Node 3: `evaluate_blog`

- Reads the generated blog.
- Evaluates relevance, structure, clarity, depth, and grammar.
- Requests a score from 1 to 10.
- Stores the evaluation in `state["evaluation"]`.
- Stores the numeric score in `state["score"]`.

#### Blog evaluation state

The shared state contains the following fields:

```python
class BlogState(TypedDict):
    title: str
    outline: str
    content: str
    evaluation: str
    score: int
```

The final output includes the blog title, outline, generated content, evaluation feedback, and score.

**Note:** The score is an LLM-generated quality estimate, not an objective guarantee of correctness.

---

## 🛠️ Tech Stack

- **Python** — Programming language
- **LangGraph** — Graph-based workflow orchestration
- **LangChain** — LLM integration and application components
- **OpenAI API** — LLM-powered text generation
- **Jupyter Notebook** — Interactive development
- **python-dotenv** — Environment variable management
- **Git and GitHub** — Version control

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.12 or a compatible Python version for your dependencies
- Git
- VS Code with the Python and Jupyter extensions
- An OpenAI API key for LLM-powered examples

### 1. Clone the repository

```bash
git clone https://github.com/piyushsh19/langgraph-learning.git
cd langgraph-learning
```

Skip this step if you already have the repository locally.

### 2. Create a virtual environment

On macOS:

```bash
python3.12 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install langgraph langchain-openai python-dotenv jupyter ipykernel
```

### 5. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name langgraph-learning --display-name "Python (LangGraph Learning)"
```

Select **Python (LangGraph Learning)** as the kernel in VS Code.

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Load the key in your notebook:

```python
from dotenv import load_dotenv

load_dotenv()
```

Initialize the model:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

Replace the placeholder with your actual API key. Never commit `.env` or expose API keys in source code or notebooks.

## ▶️ Running the Notebooks

1. Open the repository in VS Code.
2. Activate the `.venv` environment.
3. Select the appropriate Jupyter kernel.
4. Open a notebook.
5. Run cells in order, from top to bottom.
6. Review the output and experiment with different inputs.

For LLM-powered workflows, ensure your API key is configured and your OpenAI account has access to the selected model.

## 🧠 Core LangGraph Concepts

| Concept | Purpose |
|---|---|
| `TypedDict` | Defines the workflow state structure |
| `StateGraph` | Creates a graph-based workflow |
| Nodes | Execute individual processing steps |
| Edges | Define the execution order |
| `START` | Marks the workflow entry point |
| `END` | Marks workflow completion |
| `compile()` | Creates an executable graph |
| `invoke()` | Executes the workflow with initial state |
| State updates | Pass information between nodes |
| Prompt chaining | Uses one LLM step's output in a subsequent step |
| LLM evaluation | Uses an LLM to assess generated content |

## 📁 Repository Structure

```text
langgraph-learning/
├── .gitignore
├── README.md
├── 0_test_installation.ipynb
├── 1_bmi_workflow.ipynb
├── 2_simple_llm_workflow.ipynb
└── 3_prompt_chaining.ipynb
```

The notebook names are examples; keep the actual filenames used in your repository.

The following local-only files and directories should not be committed:

```text
.venv/
.env
__pycache__/
.ipynb_checkpoints/
```

## 🔒 Git Best Practices

- Commit Python source files, notebooks, README documentation, and dependency files.
- Exclude virtual environments and generated cache files.
- Never commit API keys or other secrets.
- Clear unnecessary notebook outputs before committing when appropriate.
- Use meaningful commit messages.
- Maintain a `.gitignore` in every Python repository.

## 🚀 Future Learning Goals

- Conditional routing and branching.
- Parallel execution of graph nodes.
- Structured LLM output.
- Error handling and retries.
- Tool calling and AI agents.
- Memory and checkpointing.
- Human-in-the-loop workflows.
- Retrieval-Augmented Generation (RAG).
- LLM evaluation and testing.
- Workflow deployment and monitoring.

## 🎯 Objective

Build practical expertise in LangGraph by progressing from basic deterministic workflows to sequential LLM applications, evaluation pipelines, and production-oriented AI agents.

---

**Author:** Piyush Sharma

**Repository:** [langgraph-learning](https://github.com/piyushsh19/langgraph-learning)
