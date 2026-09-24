import json
import re
import operator
import os
from typing import Annotated, TypedDict, List, Dict, Any, Literal, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_ollama import OllamaLLM
import app

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]

# Connect to Ollama
llm = OllamaLLM(model="qwen2.5-coder:7b", base_url="http://127.0.0.1:11434")

def system_prompt() -> str:
    return """You are INDUS AI — a sovereign agentic AI running in a secure, air-gapped environment.

AVAILABLE TOOLS:
1. `read_file`       — Reads/OCRs a file from disk. Args: {"filename": "report.pdf"}
2. `extract_findings` — Extracts structured findings from report text. Args: {"report_text": "..."}
3. `draft_approval_note` — Drafts a formal approval note. Args: {"findings_text": "...", "note_type": "approval|rejection|review"}
4. `search_knowledge` — Searches the local knowledge base. Args: {"query": "string"}
5. `export_document`  — Saves content to a file. Args: {"content": "...", "format": "docx|pdf|pptx|xlsx"}

TOOL CALL FORMAT (STRICT XML, NO DEVIATION):
<tool>tool_name</tool>
<args>{"arg_name": "arg_value"}</args>

PIPELINE EXAMPLE for "Read inspection report and draft approval note":
Step 1 → <tool>read_file</tool> <args>{"filename": "..."}</args> → wait for Observation
Step 2 → <tool>extract_findings</tool> <args>{"report_text": "..."}</args> → wait for Observation
Step 3 → <tool>draft_approval_note</tool> <args>{"findings_text": "...", "note_type": "approval"}</args> → wait for Observation
Step 4 → <tool>export_document</tool> <args>{"content": "...", "format": "docx"}</args> → wait for Observation
Step 5 → Reply to user with confirmation and filename.

RULES:
- One tool call at a time. Wait for Observation before next step.
- Never make up file contents. Always use read_file first.
- For approval notes, always use draft_approval_note.
- When done, give a clear summary of what was accomplished without XML tags.
"""

# Tools Implementation
def read_file(filename: str) -> str:
    import asyncio
    uploads_dir = os.path.join(os.getcwd(), "agent_uploads")
    filepath = os.path.join(uploads_dir, filename)
    
    if not os.path.exists(filepath):
        # Fallback to the user's available files string in case the directory is missing
        try:
            available = os.listdir(uploads_dir)
        except:
            available = []
        return f"Error: File '{filename}' not found. Available files in agent_uploads: {available}"
    
    try:
        # Use the synchronous OCR runner but call it directly here for the agent
        docs = app._run_processing_sync(filepath, filename)
        if not docs:
            return "No text could be extracted from this file."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error reading file: {str(e)}"

def extract_findings(report_text: str) -> str:
    prompt = f"""You are a forensic document analyst. Analyze the following inspection/audit report text and extract key information.

Return your analysis in this EXACT format:
SUMMARY: <one paragraph summary of the report>
KEY_FINDINGS:
- <finding 1>
- <finding 2>
- <finding 3>
DEFECTS_OR_ISSUES:
- <issue 1 if any>
RECOMMENDATIONS:
- <recommendation 1>
OVERALL_STATUS: <PASS / FAIL / NEEDS_REVIEW>

REPORT TEXT:
{report_text[:6000]}

Analysis:"""
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        return f"Error extracting findings: {str(e)}"

def draft_approval_note(findings_text: str, note_type: str = "approval") -> str:
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    prompt = f"""You are a senior government official drafting a formal {note_type} note.
Based on the following inspection findings, write a complete, professional approval note.

The note MUST include:
1. Official header: "INSPECTION {note_type.upper()} NOTE"
2. Date: {today}
3. Reference Number: AUTO-{datetime.now().strftime('%Y%m%d-%H%M')}
4. Executive Summary (2-3 sentences)
5. Key Findings (as a numbered list)
6. Decision with rationale
7. Required Actions / Conditions (if any)
8. Authorization signature block

FINDINGS:
{findings_text}

Draft the complete official note below:"""
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        return f"Error drafting note: {str(e)}"

def search_knowledge(query: str) -> str:
    try:
        if app.folder_vector_store is None:
            return "Error: Vector store not loaded yet."
        docs = app.folder_vector_store.similarity_search(query, k=5)
        if not docs:
            return "No matching documents found in knowledge base."
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

def export_document(content: str, format: str) -> str:
    from datetime import datetime
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_report_{timestamp}.{format}"
        
        # Ensure exports directory exists
        exports_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(exports_dir, exist_ok=True)
        filepath = os.path.join(exports_dir, filename)
        
        if format == "docx":
            app.compile_to_docx(content, filepath)
        elif format == "pdf":
            app.compile_to_pdf(content, filepath)
        elif format == "pptx":
            app.compile_to_pptx(content, filepath)
        elif format == "xlsx":
            app.compile_to_xlsx(content, filepath)
        else:
            return f"Unsupported format: {format}"
            
        return f"Successfully exported to {filename}. The file is now available in {exports_dir}."
    except Exception as e:
        return f"Error exporting document: {str(e)}"

# Nodes
def agent_node(state: AgentState):
    messages = state["messages"]
    
    # Prepend system prompt
    full_prompt = system_prompt() + "\n\nConversation History:\n"
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Only include observation if it's an observation (which we mock as HumanMessage)
        full_prompt += f"{role}: {msg.content}\n"
    
    full_prompt += "\nAssistant: "
    
    response = llm.invoke(full_prompt)
    
    return {"messages": [AIMessage(content=response)]}

def tool_node(state: AgentState):
    last_msg = state["messages"][-1].content
    
    # Parse XML tool call
    tool_match = re.search(r'<tool>(.*?)</tool>', last_msg)
    args_match = re.search(r'<args>(.*?)</args>', last_msg, re.DOTALL)
    
    if not tool_match or not args_match:
        return {"messages": [HumanMessage(content="Observation: Tool execution failed. Invalid XML format.")]}
        
    tool_name = tool_match.group(1).strip()
    try:
        args = json.loads(args_match.group(1).strip())
    except:
        return {"messages": [HumanMessage(content="Observation: Tool execution failed. Args must be valid JSON.")]}
        
    result = ""
    if tool_name == "search_knowledge":
        result = search_knowledge(args.get("query", ""))
    elif tool_name == "export_document":
        result = export_document(args.get("content", ""), args.get("format", "docx"))
    elif tool_name == "read_file":
        result = read_file(args.get("filename", ""))
    elif tool_name == "extract_findings":
        result = extract_findings(args.get("report_text", ""))
    elif tool_name == "draft_approval_note":
        result = draft_approval_note(args.get("findings_text", ""), args.get("note_type", "approval"))
    else:
        result = f"Unknown tool: {tool_name}"
        
    observation = f"Observation from {tool_name}: {result}"
    return {"messages": [HumanMessage(content=observation)]}

def router(state: AgentState) -> Literal["tools", "end"]:
    last_msg = state["messages"][-1].content
    if "<tool>" in last_msg:
        return "tools"
    return "end"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()
