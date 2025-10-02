import os
import json
from flask import Flask, render_template, request, jsonify, session
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pypdf import PdfReader
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
import uuid
import httpx
from langsmith import Client

# Load environment variables
load_dotenv()

# Configure LangSmith tracing (optional but useful for debugging)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print(f"[INIT] LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("[INIT] LangSmith tracing not configured (set LANGCHAIN_API_KEY to enable)")

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize OpenAI LLM with SSL verification disabled
_http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    http_client=_http_client
)
print("[INIT] OpenAI HTTP client initialized with SSL verification disabled")

# Global variable to store handbook content
HANDBOOK_CONTENT = None
HANDBOOK_CHUNKS = []

# Hardcoded path to the employee handbook PDF
# Place the PDF in the same folder as app.py or update this path
HANDBOOK_PATH = r"C:\Users\haley.krueger\Downloads\Employee Handbook - Multi State - 1320.pdf"

# Simple keyword helpers to prioritize relevant chunks without full RAG
def _select_candidate_chunks(question: str, chunks: List[str], max_candidates: int = 3) -> List[int]:
    """Return indices of top candidate chunks based on keyword heuristics."""
    question_lc = question.lower()
    keywords: List[str] = []
    if "pto" in question_lc:
        keywords += ["pto", "paid time off", "vacation", "time off"]
    if "vacation" in question_lc:
        keywords += ["vacation", "paid time off", "pto", "time off"]
    if "leave" in question_lc:
        keywords += ["leave", "leaves of absence", "time off", "pto", "paid time off"]
    if "holiday" in question_lc:
        keywords += ["holiday", "holidays", "time off"]
    # Fallback: use top non-trivial words from the question
    if not keywords:
        keywords = [w for w in question_lc.split() if len(w) >= 3][:5]
    
    scored: List[tuple[int, int]] = []
    for idx, chunk in enumerate(chunks):
        chunk_lc = chunk.lower()
        score = 0
        for kw in keywords:
            score += chunk_lc.count(kw)
        scored.append((idx, score))
    
    # Sort by score desc and take indices with score > 0
    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = [idx for idx, s in scored if s > 0][:max_candidates]
    print(f"[SELECT] Keywords={keywords} | scores_top={scored[:max_candidates]} | candidates={candidates}")
    return candidates

class GraphState(TypedDict):
    """State for the graph"""
    question: str
    answer: str
    found_in_handbook: bool
    is_work_related: bool
    user_accepted_ticket: bool
    ticket_drafted: bool
    ticket_content: Dict[str, str]
    messages: List[str]

def load_pdf(file_path: str) -> str:
    """Load PDF content from file"""
    if not os.path.exists(file_path):
        print(f"[PDF] File not found: {file_path}")
        return None
    
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(f"[PDF] Loaded PDF from {file_path}: pages={len(reader.pages)}, chars={len(text)}")
        return text
    except Exception as e:
        print(f"[PDF] Error loading PDF: {e}")
        return None

def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    """Chunk text based on token count"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    print(f"[CHUNK] Chunking text: total_tokens={len(tokens)}, max_tokens={max_tokens}")
    
    chunks = []
    current_chunk = []
    current_count = 0
    
    # Split by sentences to avoid breaking mid-sentence
    sentences = text.replace('\n', ' ').split('. ')
    
    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)
        if current_count + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_count = len(sentence_tokens)
        else:
            current_chunk.append(sentence)
            current_count += len(sentence_tokens)
    
    # Add the last chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk)
        chunks.append(chunk_text)
    
    # Log chunk summary
    try:
        chunk_token_sizes = [len(encoding.encode(c)) for c in chunks]
        print(f"[CHUNK] Created {len(chunks)} chunks. Token sizes (first 10): {chunk_token_sizes[:10]}")
    except Exception as e:
        print(f"[CHUNK] Error computing chunk token sizes: {e}")

    return chunks

def search_handbook_tool(state: GraphState) -> GraphState:
    """Tool to search the employee handbook for answers"""
    global HANDBOOK_CHUNKS
    
    if not HANDBOOK_CHUNKS:
        state["found_in_handbook"] = False
        state["answer"] = "Employee handbook is not loaded. Please ensure the handbook PDF is in the correct location."
        print("[SEARCH] Handbook not loaded. Aborting search.")
        return state
    
    question = state["question"]
    print(f"[SEARCH] Starting search. Chunks={len(HANDBOOK_CHUNKS)} | Question='{question}'")
    
    # Create a prompt to search through handbook chunks
    search_prompt = ChatPromptTemplate.from_template("""
    You are an assistant that helps employees find information in the Axos employee handbook.
    
    Given the employee's question and the provided handbook content, produce a clear, concise
    answer grounded ONLY in the handbook content. You may synthesize, summarize, and reasonably
    interpret policy implications that follow from the text, as long as they are consistent with
    the content. Do not use external knowledge or invent policies not supported by the text.
    
    If the content is partially relevant, answer using what is present and explicitly note any
    missing specifics or assumptions. Quote or cite short phrases when helpful.
    
    Only respond with exactly "NOT_FOUND" if there is no relevant information in the provided
    content regarding the user's question.
    
    Question: {question}
    
    Handbook Content:
    {content}
    
    Answer:
    """)
    
    # Search through chunks for relevant information
    best_answer = None
    found = False
    
    # Prefer searching candidate chunks first based on simple keyword heuristics
    candidate_indices = _select_candidate_chunks(question, HANDBOOK_CHUNKS, max_candidates=3)
    indices_to_search = candidate_indices if candidate_indices else list(range(len(HANDBOOK_CHUNKS)))
    print(f"[SEARCH] Indices to search (ordered): {indices_to_search}")
    
    for order, idx in enumerate(indices_to_search, start=1):
        chunk = HANDBOOK_CHUNKS[idx]
        
        preview = chunk[:120].replace('\n', ' ')
        print(f"[SEARCH] Checking chunk idx={idx} (order {order}/{len(indices_to_search)}) | chars={len(chunk)} | preview='{preview}'")
        chain = search_prompt | llm
        # Add minimal config for LangSmith tracing on individual LLM calls
        llm_config = {"tags": [f"chunk_{idx}", "search_handbook"]} if os.getenv("LANGCHAIN_API_KEY") else {}
        response = chain.invoke({
            "question": question,
            "content": chunk
        }, config=llm_config)
        
        answer_text = response.content.strip()
        print(f"[SEARCH] Model response len={len(answer_text)} | starts_with_NOT_FOUND={answer_text.startswith('NOT_FOUND')}")
        if answer_text != "NOT_FOUND" and not answer_text.startswith("NOT_FOUND"):
            best_answer = answer_text
            found = True
            print(f"[SEARCH] Found answer in chunk idx={idx}")
            break  # Found an answer, stop searching
    
    if found and best_answer:
        state["found_in_handbook"] = True
        state["answer"] = best_answer
    else:
        state["found_in_handbook"] = False
        # Check if question is work-related
        work_check_prompt = ChatPromptTemplate.from_template("""
        Determine if the following question is related to Axos, work, or the employee handbook.
        Respond with only "YES" or "NO".
        
        Question: {question}
        
        Is this work/Axos/handbook related?
        """)
        
        work_chain = work_check_prompt | llm
        work_response = work_chain.invoke({"question": question})
        
        is_work_related = work_response.content.strip().upper() == "YES"
        state["is_work_related"] = is_work_related
        print(f"[SEARCH] Work-related check: {work_response.content.strip()} -> {is_work_related}")
        
        if is_work_related:
            state["answer"] = "I couldn't find the answer to your question in the handbook. Would you like me to draft a ticket to the helpdesk?"
            print("[SEARCH] No answer found. Offering to draft ticket.")
        else:
            state["answer"] = "I can only answer questions related to the Axos employee handbook. Your question doesn't appear to be related to work or the handbook."
            print("[SEARCH] No answer found and not work-related.")
    
    return state

def draft_ticket_tool(state: GraphState) -> GraphState:
    """Tool to draft a help desk ticket"""
    if not state.get("is_work_related", False) or state.get("found_in_handbook", False):
        print("[TICKET] Skipping ticket draft (either not work-related or answer was found).")
        return state
    
    question = state["question"]
    print(f"[TICKET] Drafting ticket for question: '{question}'")
    
    # Create a ticket draft
    ticket_prompt = ChatPromptTemplate.from_template("""
    Draft a professional help desk ticket for the following question that wasn't found in the employee handbook.
    
    Question: {question}
    
    Create a ticket with:
    1. A clear, concise subject line
    2. A detailed description of the inquiry
    3. Context about why this information is needed
    
    Format as JSON with fields: subject, description, priority (low/medium/high)
    """)
    
    ticket_chain = ticket_prompt | llm
    response = ticket_chain.invoke({"question": question})
    
    try:
        # Parse the JSON response
        ticket_content = json.loads(response.content)
        
        # Add additional fields
        ticket_content["timestamp"] = datetime.now().isoformat()
        ticket_content["ticket_id"] = str(uuid.uuid4())[:8].upper()
        ticket_content["original_question"] = question
        
        state["ticket_drafted"] = True
        state["ticket_content"] = ticket_content
        state["answer"] = f"I've drafted a help desk ticket for your question:\n\n**Subject:** {ticket_content['subject']}\n\n**Description:** {ticket_content['description']}\n\n**Priority:** {ticket_content['priority']}\n\n**Ticket ID:** {ticket_content['ticket_id']}"
        print(f"[TICKET] Ticket drafted: id={ticket_content['ticket_id']} priority={ticket_content['priority']}")
        
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        state["ticket_drafted"] = True
        state["ticket_content"] = {
            "subject": f"Inquiry: {question[:50]}...",
            "description": f"Employee inquiry not found in handbook: {question}",
            "priority": "medium",
            "timestamp": datetime.now().isoformat(),
            "ticket_id": str(uuid.uuid4())[:8].upper(),
            "original_question": question
        }
        state["answer"] = f"I've drafted a help desk ticket for your question."
        print("[TICKET] Ticket drafted using fallback JSON parsing.")
    
    return state

def should_draft_ticket(state: GraphState) -> str:
    """Determine if we should draft a ticket"""
    if state.get("found_in_handbook", False):
        return "end"
    elif state.get("is_work_related", False) and state.get("user_accepted_ticket", False):
        return "draft_ticket"
    else:
        return "end"

# Create the workflow graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("search_handbook", search_handbook_tool)
workflow.add_node("draft_ticket", draft_ticket_tool)

# Add edges
workflow.set_entry_point("search_handbook")
workflow.add_conditional_edges(
    "search_handbook",
    should_draft_ticket,
    {
        "draft_ticket": "draft_ticket",
        "end": END
    }
)
workflow.add_edge("draft_ticket", END)

# Compile the graph
app_graph = workflow.compile()

# Optional: Set a custom run name for LangSmith tracing
def get_langsmith_config(question: str = None):
    """Get LangSmith configuration with custom metadata"""
    if not os.getenv("LANGCHAIN_API_KEY"):
        return {}
    
    config = {
        "callbacks": [],
        "metadata": {
            "application": "employee-handbook-assistant",
            "environment": os.getenv("FLASK_ENV", "production")
        }
    }
    
    if question:
        config["run_name"] = f"Question: {question[:50]}..."
    
    return config

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Handle API questions from the new UI"""
    data = request.json
    question = data.get('question', '')
    awaiting_response = data.get('awaiting_response', False)
    
    # Greeting shortcut
    if question.strip().lower() in {"hi", "hello", "hey"}:
        return jsonify({
            "answer": "Hi! I can answer questions about the employee handbook. Feel free to ask about company policies, benefits, procedures, and more.",
            "found_in_handbook": False,
            "offer_ticket": False,
            "ticket_drafted": False,
            "ticket_content": {}
        })

    # If user is responding to a pending ticket offer
    if awaiting_response or session.get('offer_pending', False):
        normalized = question.strip().lower()
        yes_set = {"yes", "y", "yeah", "yep", "sure", "please", "ok", "okay", "yes please"}
        no_set = {"no", "n", "nope", "nah"}
        original_question = session.get('offer_question', '')
        # Clear the offer by default; set again if unrecognized
        session.pop('offer_pending', None)
        session.pop('offer_question', None)
        if normalized in yes_set:
            # Run the graph with acceptance flag to draft ticket
            initial_state = {
                "question": original_question or question,
                "answer": "",
                "found_in_handbook": False,
                "is_work_related": True,
                "user_accepted_ticket": True,
                "ticket_drafted": False,
                "ticket_content": {},
                "messages": []
            }
            langsmith_config = get_langsmith_config(original_question or question)
            result = app_graph.invoke(initial_state, config=langsmith_config)
            response = {
                "answer": result["answer"],
                "found_in_handbook": result["found_in_handbook"],
                "offer_ticket": False,
                "ticket_drafted": result.get("ticket_drafted", False),
                "ticket_content": result.get("ticket_content", {}),
                "original_question": original_question
            }
            if result.get("ticket_drafted"):
                session['pending_ticket'] = result.get("ticket_content", {})
            return jsonify(response)
        elif normalized in no_set:
            return jsonify({
                "answer": "Okay, I won't create a ticket. You can ask another question about the handbook anytime.",
                "found_in_handbook": False,
                "offer_ticket": False,
                "ticket_drafted": False,
                "ticket_content": {}
            })
        else:
            # Unrecognized response, re-prompt and keep offer pending
            session['offer_pending'] = True
            session['offer_question'] = original_question
            return jsonify({
                "answer": "Please reply with 'yes' or 'no'. Would you like me to draft a ticket to the helpdesk?",
                "found_in_handbook": False,
                "offer_ticket": False,
                "ticket_drafted": False,
                "ticket_content": {}
            })
    # Ensure handbook is loaded from the hardcoded path before answering
    global HANDBOOK_CONTENT, HANDBOOK_CHUNKS
    if not HANDBOOK_CHUNKS and os.path.exists(HANDBOOK_PATH):
        content = load_pdf(HANDBOOK_PATH)
        if content:
            HANDBOOK_CONTENT = content
            HANDBOOK_CHUNKS = chunk_text(content, max_tokens=4000)
    print(f"[API] Received question: '{question}' | chunks_loaded={len(HANDBOOK_CHUNKS)}")
    # Process the question through the graph
    initial_state = {
        "question": question,
        "answer": "",
        "found_in_handbook": False,
        "is_work_related": False,
        "user_accepted_ticket": False,
        "ticket_drafted": False,
        "ticket_content": {},
        "messages": []
    }
        
    # Invoke with LangSmith tracing
    langsmith_config = get_langsmith_config(question)
    result = app_graph.invoke(initial_state, config=langsmith_config)
    print(f"[API] Graph result: found_in_handbook={result['found_in_handbook']} | ticket_drafted={result.get('ticket_drafted', False)}")
    
    response = {
        "answer": result["answer"],
        "found_in_handbook": result["found_in_handbook"],
        "offer_ticket": False,
        "ticket_drafted": result.get("ticket_drafted", False),
        "ticket_content": result.get("ticket_content", {}),
        "original_question": question
    }
    
    # If no answer found but work-related, offer ticket creation
    if not result.get("found_in_handbook", False) and result.get("is_work_related", False):
        response["offer_ticket"] = True
        session['offer_pending'] = True
        session['offer_question'] = question
    
    # Store ticket in session if drafted
    if result.get("ticket_drafted"):
        session['pending_ticket'] = result.get("ticket_content", {})
    
    return jsonify(response)
    
@app.route('/api/submit-ticket', methods=['POST'])
def api_submit_ticket():
    """Handle ticket submission from the new UI"""
    data = request.json
    ticket = data.get('ticket', session.get('pending_ticket', {}))
    
    if ticket:
        # Here you would normally submit to a real ticketing system
        # For now, we'll just confirm submission
        session.pop('pending_ticket', None)
        return jsonify({
            "success": True,
            "message": f"Ticket {ticket.get('ticket_id', 'UNKNOWN')} has been submitted successfully. You will receive a response within 24-48 hours.",
            "ticket_id": ticket.get('ticket_id', 'UNKNOWN')
        })
    else:
        return jsonify({
            "success": False,
            "message": "No pending ticket found."
        })

"""
Removed manual load endpoint; handbook is auto-loaded from HANDBOOK_PATH
"""

# Keep old /chat endpoint for backwards compatibility
@app.route('/chat', methods=['POST'])
def chat_legacy():
    """Legacy chat endpoint - redirects to new API"""
    data = request.json
    return api_ask()

@app.route('/status', methods=['GET'])
def status():
    """Check if handbook is loaded"""
    global HANDBOOK_CHUNKS
    
    return jsonify({
        "handbook_loaded": len(HANDBOOK_CHUNKS) > 0,
        "chunks": len(HANDBOOK_CHUNKS)
    })

if __name__ == '__main__':
    # Try to load handbook on startup
    if os.path.exists(HANDBOOK_PATH):
        content = load_pdf(HANDBOOK_PATH)
        if content:
            HANDBOOK_CONTENT = content
            HANDBOOK_CHUNKS = chunk_text(content, max_tokens=8000)
            print(f"[INIT] Handbook loaded from {HANDBOOK_PATH}: {len(HANDBOOK_CHUNKS)} chunks")
        else:
            print(f"[INIT] Failed to load handbook content from {HANDBOOK_PATH}")
    else:
        print(f"[INIT] Handbook path does not exist: {HANDBOOK_PATH}")
    
    app.run(debug=True, port=5000)
