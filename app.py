
import os
import operator
from typing import List, Annotated, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import get_buffer_string


app = Flask(__name__)


# os.environ["GOOGLE_API_KEY"] = ""
# os.environ["TAVILY_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=4096,
    timeout=None,
    max_retries=2
)


tavily_search = TavilySearchResults(max_results=3)


class Analyst(BaseModel):
    name: str = Field(description="Name of the analyst")
    affiliation: str = Field(description="Affiliation of the analyst")
    role: str = Field(description="Role of the analyst")
    description: str = Field(description="Description of the analyst focus area")

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str

# Analyst creation functions
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Identify the key areas of expertise needed to research this topic effectively.

3. Create a list of 3-5 AI analyst personas with:
- Name (use fake names)
- Affiliation
- Role
- Description of their focus area

4. Each analyst should have a distinct perspective on the topic.

5. Format the output as a list of JSON objects with the following keys:
- name
- affiliation
- role
- description"""

def create_analysts(state: GenerateAnalystsState):
    """Create analysts for the research topic"""
    topic = state['topic']
    max_analysts = state['max_analysts']
    
    system_message = analyst_instructions.format(topic=topic)
    analysts = llm.with_structured_output(Analyst).invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Generate {max_analysts} analyst personas")
    ])
    
    return {"analysts": [analysts] if isinstance(analysts, Analyst) else analysts}

def human_feedback(state: GenerateAnalystsState):
    """Get human feedback on analysts"""
    return {}

def should_continue(state: GenerateAnalystsState):
    """Check if human feedback requires changes"""
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    return END

# Analyst graph
def create_analyst_graph():
    builder = StateGraph(GenerateAnalystsState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])
    
    memory = MemorySaver()
    return builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# Interview functions
def generate_question(state: InterviewState):
    """Generate a question for the interview"""
    analyst = state['analyst']
    messages = state['messages']
    
    system_message = f"""You are {analyst.name}, {analyst.role} at {analyst.affiliation}.
    Your focus area is: {analyst.description}
    Ask a question that will help gather information about {state.get('topic', 'the research topic')}."""
    
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
    return {"messages": [question]}

def search_web(state: InterviewState):
    """Search the web for information"""
    messages = state['messages']
    system_message = f"""You will be given a conversation between an analyst and an expert.
    Your goal is to generate a search query to find relevant information."""
    
    search_query = llm.with_structured_output(SearchQuery).invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=get_buffer_string(messages))
    ]).search_query
    
    results = tavily_search.invoke(search_query)
    return {"context": [results]}

def search_wikipedia(state: InterviewState):
    """Search Wikipedia for information"""
    messages = state['messages']
    system_message = f"""You will be given a conversation between an analyst and an expert.
    Your goal is to generate a Wikipedia search query to find relevant information."""
    
    search_query = llm.with_structured_output(SearchQuery).invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=get_buffer_string(messages))
    ]).search_query
    
    try:
        docs = WikipediaLoader(query=search_query, load_max_docs=2).load()
        formatted_docs = [{"content": doc.page_content, "url": doc.metadata.get("source", "")} for doc in docs]
        return {"context": [formatted_docs]}
    except Exception:
        return {"context": [[]]}

def generate_answer(state: InterviewState):
    """Generate an answer based on context"""
    analyst = state['analyst']
    messages = state['messages']
    context = state['context']
    
    system_message = f"""You are {analyst.name}, {analyst.role} at {analyst.affiliation}.
    Your focus area is: {analyst.description}
    Answer the question based on the provided context."""
    
    answer = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {messages[-1].content}")
    ])
    
    return {"messages": [answer]}

def route_messages(state: InterviewState):
    """Route messages based on turn count"""
    num_responses = len([m for m in state['messages'] if m.type == 'ai'])
    if num_responses >= state['max_num_turns']:
        return "save_interview"
    return "ask_question"

def save_interview(state: InterviewState):
    """Save the interview transcript"""
    messages = state['messages']
    interview = get_buffer_string(messages)
    return {"interview": interview}

def write_section(state: InterviewState):
    """Write a section based on the interview"""
    analyst = state['analyst']
    interview = state['interview']
    
    section_writer_instructions = f"""You are a technical writer. Your task is to create a short, easily digestible section of a report based on an interview.
    
    Focus area: {analyst.description}
    
    Interview:
    {interview}
    
    Write a section with:
    1. Title (## header)
    2. Summary (### header)
    3. Sources (### header)
    
    Use markdown formatting."""
    
    section = llm.invoke([
        SystemMessage(content=section_writer_instructions),
        HumanMessage(content="Write a report section based on the interview")
    ])
    
    return {"sections": [section.content]}

# Interview graph with improved flow
def create_interview_graph():
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)
    
    # Improved flow with parallel search
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge(["search_web", "search_wikipedia"], "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    
    memory = MemorySaver()
    return interview_builder.compile(checkpointer=memory)

# Report writing functions
report_writer_instructions = """You are a technical writer creating a report on: {topic}
Combine the sections below into a single coherent report with appropriate section headers.
Each section starts with ## followed by the title.
Make sure to preserve all content from each section.
Add transitions between sections as needed."""

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}
You will be given all of the sections of the report.
Write a detailed technical introduction and conclusion for the report.
Use markdown formatting."""

def initiate_all_interviews(state: ResearchGraphState):
    """Initiate all interviews in parallel"""
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    return "conduct_interview"

def write_report(state: ResearchGraphState):
    """Write the main report content"""
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos.")
    ])
    
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    """Write the introduction"""
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    instructions = intro_conclusion_instructions.format(topic=topic)
    intro = llm.invoke([SystemMessage(content=instructions), HumanMessage(content="Write the report introduction")])
    
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    """Write the conclusion"""
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    instructions = intro_conclusion_instructions.format(topic=topic)
    conclusion = llm.invoke([SystemMessage(content=instructions), HumanMessage(content="Write the report conclusion")])
    
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """Combine all parts into final report"""
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    
    sources = None
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None
    
    final_report = state["introduction"] + "\n\n" + content + "\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n## Sources\n" + sources
    
    return {"final_report": final_report}

# Main research graph with improved flow
def create_research_graph():
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", create_interview_graph())
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)
    
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)
    
    memory = MemorySaver()
    return builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)


analyst_graph = create_analyst_graph()
research_graph = create_research_graph()

# API Routes
@app.route('/generate_analysts', methods=['POST'])
def generate_analysts():
    data = request.json
    topic = data.get('topic', '')
    max_analysts = data.get('max_analysts', 3)
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    config = {"configurable": {"thread_id": "1"}}
    initial_state = {
        "topic": topic,
        "max_analysts": max_analysts,
        "human_analyst_feedback": None,
        "analysts": []
    }
    
    # Run the graph
    for event in analyst_graph.stream(initial_state, config, stream_mode="values"):
        if 'analysts' in event:
            analysts = event['analysts']
            return jsonify({
                "analysts": [
                    {
                        "name": a.name,
                        "affiliation": a.affiliation,
                        "role": a.role,
                        "description": a.description
                    } for a in analysts
                ]
            })
    
    return jsonify({"error": "Failed to generate analysts"}), 500

@app.route('/provide_feedback', methods=['POST'])
def provide_feedback():
    data = request.json
    feedback = data.get('feedback', '')
    config = {"configurable": {"thread_id": "1"}}
    
    # Update state with feedback
    analyst_graph.update_state(config, {"human_analyst_feedback": feedback}, as_node="human_feedback")
    
    # Continue execution
    final_state = analyst_graph.get_state(config)
    return jsonify({"status": "Feedback received", "next": final_state.next})

@app.route('/research', methods=['POST'])
def conduct_research():
    data = request.json
    topic = data.get('topic', '')
    max_analysts = data.get('max_analysts', 3)
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    config = {"configurable": {"thread_id": "2"}}
    initial_state = {
        "topic": topic,
        "max_analysts": max_analysts,
        "human_analyst_feedback": None,
        "analysts": [],
        "sections": [],
        "introduction": "",
        "content": "",
        "conclusion": "",
        "final_report": ""
    }
    
    # Run the research graph
    for event in research_graph.stream(initial_state, config, stream_mode="updates"):
        pass
    
    # Get final state
    final_state = research_graph.get_state(config)
    report = final_state.values.get('final_report', '')
    
    return jsonify({"report": report})

if __name__ == '__main__':
    app.run(debug=True)
