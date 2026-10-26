
User attachment
User attachment
User attachment
You're building a parallel evaluation workflow in LangGraph for a UPSC answer. Based on your screenshots, you have three independent evaluation nodes — language, analysis, and clarity of thought — followed by a final_evaluation node that combines the feedback and calculates the average score.

Here is the corrected code for the final node and graph.

1. Final evaluation node
Your prompt should include all three feedback fields, and the average score should be calculated from the three individual scores.


def final_evaluation(state: UPSCState):

    # Summarize feedback from all evaluators
    prompt = f"""
    Based on the following feedback, create a summarized evaluation.

    Language feedback:
    {state["language_feedback"]}

    Depth of analysis feedback:
    {state["analysis_feedback"]}

    Clarity of thought feedback:
    {state["clarity_feedback"]}

    Provide an overall assessment with strengths and areas for improvement.
    """

    overall_feedback = model.invoke(prompt).content

    # Calculate average score
    scores = state["individual_scores"]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "overall_feedback": overall_feedback,
        "avg_score": avg_score
    }


2. Define your graph
Your screenshot shows this parallel structure. Keep the three evaluation nodes connected to START, then connect all of them to final_evaluation.


from langgraph.graph import StateGraph, START, END

graph = StateGraph(UPSCState)

# Add nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Start three evaluations in parallel
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

# Join the three branches
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

# Finish
graph.add_edge("final_evaluation", END)

workflow = graph.compile()


3. Important: check your UPSCState
Your state needs to contain these fields (along with any input fields your evaluator nodes require):

class UPSCState(TypedDict):
    # Input fields — adapt to your existing notebook
    answer: str

    # Evaluator outputs
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    individual_scores: list[float]

    # Final output
    overall_feedback: str
    avg_score: float

Keep your existing input fields if they have different names. The important point is that individual_scores must collect the scores from all three parallel evaluators.

Watch out for parallel state updates: if all three evaluators return {"individual_scores": [output.score]}, the default state reducer may overwrite or conflict with these simultaneous updates. Define a list-aggregation reducer for individual_scores in your state, or return each score under a separate field and combine them in final_evaluation. The latter is often simpler and more predictable.

User attachment
User attachment
User attachment
You're building a quadratic equation workflow in LangGraph using 6_quadratic_equation_workflow.ipynb. Based on your screenshots, the workflow should display the equation, calculate its discriminant, and route to one of three nodes depending on the discriminant value.

1. Define the state
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class QuadState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str

2. Create the equation node
Your screenshot has a small formatting issue in the equation string. Use x**2 mathematically, or x² for display.

def show_equation(state: QuadState):
    equation = f'{state["a"]}x² + {state["b"]}x + {state["c"]} = 0'
    return {"equation": equation}

3. Calculate the discriminant
The formula is 
D
=
b
2
−
4
a
c
D=b 
2
 −4ac.

def calculate_discriminant(state: QuadState):
    discriminant = state["b"]**2 - 4 * state["a"] * state["c"]
    return {"discriminant": discriminant}

4. Create the three result nodes
def no_real_roots(state: QuadState):
    return {"result": "No real roots; the solutions are complex."}


def real_roots(state: QuadState):
    import math

    a, b = state["a"], state["b"]
    d = state["discriminant"]

    x1 = (-b + math.sqrt(d)) / (2 * a)
    x2 = (-b - math.sqrt(d)) / (2 * a)

    return {"result": f"Two real roots: x1 = {x1}, x2 = {x2}"}


def repeated_roots(state: QuadState):
    a, b = state["a"], state["b"]
    x = -b / (2 * a)

    return {"result": f"One repeated real root: x = {x}"}

5. Add conditional routing
This function decides which node to execute.

def route_roots(state: QuadState) -> Literal[
    "no_real_roots", "real_roots", "repeated_roots"
]:
    d = state["discriminant"]

    if d < 0:
        return "no_real_roots"
    elif d > 0:
        return "real_roots"
    else:
        return "repeated_roots"

6. Build the graph
Replace your incomplete graph cell with this:


graph = StateGraph(QuadState)

# Add nodes
graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("no_real_roots", no_real_roots)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_roots", repeated_roots)

# Sequential edges
graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")

# Conditional routing
graph.add_conditional_edges(
    "calculate_discriminant",
    route_roots
)

# All branches finish
graph.add_edge("no_real_roots", END)
graph.add_edge("real_roots", END)
graph.add_edge("repeated_roots", END)

# Compile
workflow = graph.compile()


7. Execute the workflow
Try this example first:

initial_state = {
    "a": 1,
    "b": -5,
    "c": 6,
    "equation": "",
    "discriminant": 0.0,
    "result": ""
}

output = workflow.invoke(initial_state)

print("Equation:", output["equation"])
print("Discriminant:", output["discriminant"])
print("Result:", output["result"])

Expected output:

Equation: 1x² + -5x + 6 = 0
Discriminant: 1
Result: Two real roots: x1 = 3.0, x2 = 2.0
The equation formatting can be improved by displaying negative coefficients naturally, but the calculation is correct.

8. Understand the workflow
START

Input coefficients a, b, c

show_equation

Display the quadratic equation

calculate_discriminant

Calculate D = b² − 4ac

D < 0
No real roots

D > 0
Two real roots

D = 0
Repeated root

Each conditional branch leads to END.

Key learning: This example introduces conditional edges in LangGraph. Unlike your previous parallel cricket workflow, which runs several calculations together, this graph uses the discriminant to select exactly one result node. Make sure a is nonzero, because a quadratic equation requires 
a
≠
0
a

=0.

User attachment
like we have build the vonditonal workflow and pravous also the parallle flow example  please keep these context as we will be adding these later in the readme file .

Now focus on the above provide llm based rview handling workflow
We'll build the LLM-Based Review Handling Workflow in your LangGraph learning project, following the diagram you shared.

This example combines two concepts you've already practiced:

Conditional workflow: detect review sentiment and route to the appropriate response path.

LLM-powered processing: use an LLM to analyze negative reviews and generate a diagnosis before responding.

The workflow will have five nodes: find_sentiment, run_diagnosis, negative_response, and positive_response, with START and END defining the entry and exit.

1. Define the workflow state
Create a new notebook, for example 7_review_handling_workflow.ipynb.

Start by defining the state that will pass information between nodes.

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class ReviewState(TypedDict):
    review: str
    sentiment: str
    diagnosis: str
    response: str

What each field stores:

review: the original customer review.

sentiment: whether the review is positive or negative.

diagnosis: analysis of the problem, if the review is negative.

response: the final reply to the customer.

2. Create the find_sentiment node
This node uses the LLM to classify the customer's review.

def find_sentiment(state: ReviewState):
    review = state["review"]

    prompt = f"""
    Classify the sentiment of this customer review.

    Review: {review}

    Return exactly one word:
    positive or negative.

    If the review expresses dissatisfaction, complaints,
    or problems, classify it as negative.
    Otherwise, classify it as positive.
    """

    result = model.invoke(prompt).content.strip().lower()

    # Normalize the LLM output
    sentiment = "negative" if "negative" in result else "positive"

    return {"sentiment": sentiment}

This node produces the sentiment value used by the conditional router.

3. Create the run_diagnosis node
This node runs only when the review is negative. It identifies the customer's problem and suggests what needs to be addressed.

def run_diagnosis(state: ReviewState):
    review = state["review"]

    prompt = f"""
    Analyze this negative customer review.

    Review: {review}

    Identify:
    1. The main problem
    2. The likely cause, if evident
    3. What action could resolve the issue

    Do not invent facts that are not supported by the review.
    """

    diagnosis = model.invoke(prompt).content

    return {"diagnosis": diagnosis}

4. Create the negative_response node
This node uses both the original review and the diagnosis to prepare an empathetic response.

def negative_response(state: ReviewState):
    review = state["review"]
    diagnosis = state["diagnosis"]

    prompt = f"""
    Write a professional and empathetic response
    to this unhappy customer.

    Customer review:
    {review}

    Problem analysis:
    {diagnosis}

    Acknowledge the customer's concern.
    Apologize where appropriate.
    Address the problem and suggest a reasonable next step.
    Do not promise actions that have not been confirmed.
    """

    response = model.invoke(prompt).content

    return {"response": response}

5. Create the positive_response node
Positive reviews take a shorter path. They do not need the diagnosis node.

def positive_response(state: ReviewState):
    review = state["review"]

    prompt = f"""
    Write a warm and professional response
    to this positive customer review.

    Customer review:
    {review}

    Thank the customer, acknowledge their positive feedback,
    and encourage them to continue using the service.
    Keep the response concise and natural.
    """

    response = model.invoke(prompt).content

    return {"response": response}

6. Define the conditional router
This function decides which branch the workflow should follow.

def route_review(
    state: ReviewState
) -> Literal["run_diagnosis", "positive_response"]:

    if state["sentiment"] == "negative":
        return "run_diagnosis"

    return "positive_response"

Notice that this function does not call the LLM. It simply reads the sentiment already stored in the state and selects the next node.

7. Build the graph
Now connect all the nodes according to your diagram.


graph = StateGraph(ReviewState)

# Add nodes
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)
graph.add_node("positive_response", positive_response)

# Entry point
graph.add_edge(START, "find_sentiment")

# Conditional routing
graph.add_conditional_edges(
    "find_sentiment",
    route_review,
    {
        "run_diagnosis": "run_diagnosis",
        "positive_response": "positive_response",
    },
)

# Negative path
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

# Positive path
graph.add_edge("positive_response", END)

# Compile the graph
workflow = graph.compile()


8. Test the negative review
initial_state = {
    "review": (
        "The delivery was three days late, and customer support "
        "did not respond to my messages. Very disappointing."
    ),
    "sentiment": "",
    "diagnosis": "",
    "response": "",
}

output = workflow.invoke(initial_state)

print("SENTIMENT:")
print(output["sentiment"])

print("\nDIAGNOSIS:")
print(output["diagnosis"])

print("\nCUSTOMER RESPONSE:")
print(output["response"])

Expected behavior:

Sentiment is classified as negative.

The diagnosis identifies the delivery delay and lack of customer support.

The negative-response node generates an empathetic reply.

The exact wording will depend on the LLM.

9. Test the positive review
Run this in another cell:

initial_state = {
    "review": (
        "Excellent service! My order arrived on time, "
        "the packaging was great, and the product exceeded "
        "my expectations."
    ),
    "sentiment": "",
    "diagnosis": "",
    "response": "",
}

output = workflow.invoke(initial_state)

print("SENTIMENT:")
print(output["sentiment"])

print("\nCUSTOMER RESPONSE:")
print(output["response"])

Expected behavior:

Sentiment is classified as positive.

The positive-response node generates a thank-you message.

The run_diagnosis node is skipped, so diagnosis remains empty.

10. Visualize the workflow
START

Customer review

find_sentiment

LLM classifies the review

Negative
run_diagnosis

negative_response

Analyze the issue and generate an empathetic reply

END

Positive
positive_response

Generate a thank-you response directly

END

What you've learned
This workflow introduces LLM-based conditional routing: the LLM determines the sentiment, and LangGraph uses that result to choose a path.

Unlike the quadratic-equation workflow, where routing depends on a numerical condition, this workflow routes based on an LLM-generated classification. Unlike the parallel cricket workflow, it executes only the selected branch.

For production use, consider adding a neutral sentiment branch and validating the LLM's classification against a fixed set of allowed labels.

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
git clone 
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


**Repository:** [langgraph-learning]\([https://github.com/piyushsh19/langgraph-learning](https://github.com/piyushsh19/langgraph-learning))




giv eme the update readme.file

…

I've updated your README to include the workflows we've covered so far, including the new quadratic-equation and LLM-based review-handling workflows.

Download the updated README.md

The updated file includes:

Installation testing and BMI workflow

Simple LLM workflow

Prompt chaining with blog generation and evaluation

Parallel cricket batting-statistics workflow

Conditional quadratic-equation workflow

LLM-based review handling workflow

Setup instructions, dependencies, environment variables, and Git best practices

.gitignore guidance to prevent accidentally committing virtual environments and API keys

Replace your existing README
Since your repository is already connected to GitHub, open the project folder in VS Code and replace the contents of your existing README.md with the downloaded file.

Then run:

git status
git add README.md
git commit -m "docs: update LangGraph learning workflows"
git push origin main

Important: Stage only README.md for this documentation commit. Before making any other commits, ensure .venv/ is ignored and your virtual-environment files are not staged.

I have kept the notebook filenames as documented examples where necessary. Adjust any filename in the README if it differs from the actual file in your repository.


README.md
Document
This might also interest you.






Python LangGraph Learning
A hands-on learning repository for understanding LangGraph, LangChain,
LLMs, and AI workflow orchestration through practical Python projects
and Jupyter notebooks.

The goal is to build a strong foundation in LangGraph, progressing from
basic state management to sequential LLM workflows, parallel execution,
conditional routing, and multi-step AI applications.

📚 Learning Projects
0. Installation Testing
Notebook: 0_test_installation.ipynb

Verify Python and package installations.
Test the Jupyter Notebook environment.
Confirm the development setup is ready for LangGraph experiments.
1. BMI Workflow
Notebook: 1_bmi_workflow.ipynb

A beginner-friendly introduction to state-based workflows.

Concepts covered: - Defining workflow state using TypedDict. -
Creating Python functions as graph nodes. - Calculating BMI from height
and weight. - Categorizing BMI results. - Connecting nodes using graph
edges. - Compiling and invoking a LangGraph workflow.

Workflow:

Input → Calculate BMI → Label BMI → Output
2. Simple LLM Workflow
Notebook: 2_simple_llm_workflow.ipynb

A basic question-answering workflow powered by a large language model.

Concepts covered: - Defining state using TypedDict. - Integrating
an LLM using LangChain's ChatOpenAI. - Creating a node to process a
question. - Generating answers using model.invoke(). - Updating the
workflow state. - Using StateGraph, START, and END. - Compiling
and invoking an LLM-powered graph.

Workflow:

Question → LLM Node → Answer
3. Prompt Chaining: Blog Generation and Evaluation
Notebook: 3_prompt_chaining.ipynb (use your actual filename if
different)

A multi-step LLM workflow that generates a blog outline, writes the
complete blog, and evaluates the generated content.

Concepts covered: - Sequential LLM execution. - Sharing data between
nodes through graph state. - Generating a blog outline from a topic. -
Using the outline to generate a complete blog. - Evaluating generated
content with another LLM call. - Assigning a quality score between 1 and
10. - Storing evaluation feedback and scores in the state. - Connecting
multiple nodes using LangGraph edges.

Workflow:

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
Nodes: - create_outline: generates and stores an outline in
state["outline"]. - create_blog: uses the title and outline to
generate state["content"]. - evaluate_blog: evaluates relevance,
structure, clarity, depth, and grammar, then stores feedback and a
score.

Example state:

class BlogState(TypedDict):
    title: str
    outline: str
    content: str
    evaluation: str
    score: int
The score is an LLM-generated quality estimate, not an objective
guarantee of correctness.

4. Parallel Workflow: Cricket Batting Statistics
Notebook: 4_batsman_workflow.ipynb

A parallel workflow that calculates multiple batting statistics
independently and combines them into a summary.

Concepts covered: - Running independent graph nodes in parallel. -
Calculating strike rate, balls per boundary, and boundary percentage. -
Understanding fan-out and fan-in graph patterns. - Combining results in
a summary node. - Working with shared state and concurrent updates.

Workflow:

                         ┌─ calculate_sr ───────────┐
START ───────────────────┼─ calculate_bpb ──────────┼─> summary ─> END
                         └─ calculate_boundary_percent ┘
The calculation nodes run independently. Their results are then used by
the summary node. When multiple parallel nodes update the same state
key, configure an appropriate reducer or use separate state keys to
avoid conflicting updates.

5. Conditional Workflow: Quadratic Equation
Notebook: 6_quadratic_equation_workflow.ipynb

A conditional workflow that calculates the discriminant of a quadratic
equation and routes execution based on its value.

For (ax^2 + bx + c = 0), the discriminant is:

D = b² - 4ac
Concepts covered: - Creating a typed workflow state. - Calculating a
deterministic value in a graph node. - Using
add_conditional_edges(). - Defining a routing function. - Branching
execution based on a result. - Handling three possible outcomes.

Workflow:

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
The coefficient a must not be zero for the equation to be quadratic.

6. LLM-Based Review Handling Workflow
Notebook: 7_review_handling_workflow.ipynb (suggested filename;
adjust to the actual filename)

An LLM-powered customer-review workflow that classifies sentiment and
generates an appropriate response. Negative reviews receive additional
diagnosis before a response is written.

Concepts covered: - Using an LLM to classify review sentiment. -
Routing to different branches with conditional edges. - Running
diagnosis only for negative reviews. - Generating empathetic responses
to negative reviews. - Generating appreciative responses to positive
reviews. - Passing results between nodes through shared state.

Workflow:

START
  |
  v
find_sentiment
  |
  ├── Negative ──> run_diagnosis ──> negative_response ──> END
  |
  └── Positive ────────────────────> positive_response ──> END
Nodes: - find_sentiment: classifies a review as positive or
negative. - run_diagnosis: analyzes the problem described in a
negative review. - negative_response: creates an empathetic response
informed by the diagnosis. - positive_response: thanks the customer
and responds to positive feedback.

This example demonstrates LLM-based routing: the LLM classifies the
review, while LangGraph's conditional edge selects the execution path.
For production use, validate model output against allowed labels and
consider adding a neutral or uncertain sentiment path.

🛠️ Tech Stack
Python --- Programming language
LangGraph --- Graph-based workflow orchestration
LangChain --- LLM integration and application components
OpenAI API --- LLM-powered text generation
Jupyter Notebook --- Interactive development
python-dotenv --- Environment variable management
Git and GitHub --- Version control
⚙️ Setup and Installation
Prerequisites
Python 3.12 or another Python version compatible with your installed
dependencies
Git
VS Code with the Python and Jupyter extensions
An OpenAI API key for LLM-powered examples
1. Clone the repository
git clone https://github.com/piyushsh19/langgraph-learning.git
cd langgraph-learning
Skip this step if you already have the repository locally.

2. Create a virtual environment
On macOS:

python3.12 -m venv .venv
3. Activate the environment
source .venv/bin/activate
4. Install dependencies
python -m pip install --upgrade pip
python -m pip install langgraph langchain-openai python-dotenv jupyter ipykernel
If a notebook uses additional packages, install those in the same
virtual environment.

5. Register the Jupyter kernel
python -m ipykernel install --user --name langgraph-learning --display-name "Python (LangGraph Learning)"
Select Python (LangGraph Learning) as the kernel in VS Code.

🔑 Environment Variables
Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key_here
Load the key in your notebook:

from dotenv import load_dotenv

load_dotenv()
Initialize the model:

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
Replace the placeholder with your actual API key. Never commit .env or
expose API keys in source code or notebooks.

▶️ Running the Notebooks
Open the repository in VS Code.
Activate the .venv environment.
Select the appropriate Jupyter kernel.
Open a notebook.
Run cells in order, from top to bottom.
Review the output and experiment with different inputs.
For LLM-powered workflows, ensure your API key is configured and your
OpenAI account has access to the selected model.

🧠 Core LangGraph Concepts
Concept Purpose

TypedDict Defines the workflow state structure
StateGraph Creates a graph-based workflow
Nodes Execute individual processing steps
Edges Define the execution order
START Marks the workflow entry point
END Marks workflow completion
compile() Creates an executable graph
invoke() Executes the workflow with initial state
State updates Pass information between nodes
Prompt chaining Uses one LLM step's output in a subsequent step
Parallel execution Runs independent nodes concurrently
Conditional edges Route execution based on state or node output
LLM evaluation Uses an LLM to assess generated content
Reducers Define how multiple updates to a state key are combined

📁 Repository Structure
The repository may look like this as you add the notebooks:

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
└── 7_review_handling_workflow.ipynb
This is an illustrative structure based on the notebooks discussed so
far. Keep your actual filenames and omit files that do not exist in your
repository. The UPSC essay workflow is included because it was also part
of the learning work discussed.

🔒 Git Best Practices
Commit source files, notebooks, documentation, and dependency files. Do
not commit virtual environments, generated caches, or secrets.

A suitable .gitignore for this project is:

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
Important: If your virtual environment was already tracked by Git,
adding it to .gitignore will not automatically remove it from version
control. To stop tracking it while keeping the local files, run:

git rm -r --cached .venv
git add .gitignore
git status
Run the git rm command only if .venv is the name of your environment
and Git is currently tracking it. Review git status before committing.

Recommended practices: - Never commit API keys or other secrets. - Clear
unnecessary notebook outputs before committing when appropriate. - Use
meaningful commit messages. - Check git status before every commit. -
Keep .gitignore in every Python repository.

🚀 Future Learning Goals
Structured LLM output
Error handling and retries
Tool calling and AI agents
Memory and checkpointing
Human-in-the-loop workflows
Retrieval-Augmented Generation (RAG)
LLM evaluation and testing
Workflow deployment and monitoring
🎯 Objective
Build practical expertise in LangGraph by progressing from basic
deterministic workflows to sequential LLM applications, parallel
execution, conditional routing, evaluation pipelines, and
production-oriented AI agents.

Author: Piyush Sharma

Repository:
langgraph-learning

