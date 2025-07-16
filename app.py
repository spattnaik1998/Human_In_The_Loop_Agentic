from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import uuid
from typing import Dict, Optional
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# Set your API keys here
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

app = FastAPI(title="Human-in-the-Loop AI Assistant")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1000)

# Define tools
@tool
def multiply(first_number: int, second_number: int) -> int:
    """Multiply two integer numbers"""
    return first_number * second_number

@tool
def search(query: str):
    """Perform web search on the user query"""
    tavily = TavilySearchResults()
    result = tavily.invoke(query)
    return result

# Tool mapping
tools = [search, multiply]
model_with_tools = llm.bind_tools(tools)
tool_mapping = {tool.name: tool for tool in tools}

# In-memory storage for sessions and pending approvals
sessions: Dict[str, Dict] = {}
pending_approvals: Dict[str, Dict] = {}

# Pydantic models
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ApprovalRequest(BaseModel):
    approval_id: str
    approved: bool

class MessageResponse(BaseModel):
    type: str
    message: Optional[str] = None
    approval_request: Optional[Dict] = None
    session_id: str

def process_ai_response(user_input: str, session_id: str) -> MessageResponse:
    """Core human-in-the-loop logic"""
    
    # Get AI response with tools
    response = model_with_tools.invoke([HumanMessage(content=user_input)])
    
    # Check if AI wants to use tools
    tool_calls = response.additional_kwargs.get("tool_calls", [])
    
    if not tool_calls:
        # No tools needed, return AI response directly
        return MessageResponse(
            type="ai_response",
            message=response.content,
            session_id=session_id
        )
    
    # AI wants to use a tool
    tool_call = tool_calls[0]
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])
    
    # Human-in-the-loop decision point
    if tool_name == "search":
        # For search, ask for human approval
        approval_id = str(uuid.uuid4())
        query = tool_args.get("query", "")
        
        # Store pending approval
        pending_approvals[approval_id] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "original_query": user_input
        }
        
        return MessageResponse(
            type="approval_request",
            approval_request={
                "approval_id": approval_id,
                "tool_name": tool_name,
                "query": query,
                "message": f"I want to search for: '{query}'. Do you approve?"
            },
            session_id=session_id
        )
    
    elif tool_name == "multiply":
        # For math, execute automatically
        try:
            result = tool_mapping[tool_name].invoke(tool_args)
            return MessageResponse(
                type="ai_response",
                message=f"The result is: {result}",
                session_id=session_id
            )
        except Exception as e:
            return MessageResponse(
                type="ai_response",
                message=f"Error in calculation: {str(e)}",
                session_id=session_id
            )
    
    # Unknown tool
    return MessageResponse(
        type="ai_response",
        message="I'm not sure how to handle that request.",
        session_id=session_id
    )

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Main chat endpoint - core of human-in-the-loop logic"""
    
    # Create or get session
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}
    
    # Add user message to session
    sessions[session_id]["messages"].append({
        "type": "user",
        "content": request.message
    })
    
    try:
        # Process with human-in-the-loop logic
        response = process_ai_response(request.message, session_id)
        
        # Store AI response in session (if not approval request)
        if response.type == "ai_response":
            sessions[session_id]["messages"].append({
                "type": "ai",
                "content": response.message
            })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/approve", response_model=MessageResponse)
async def approve_action(request: ApprovalRequest):
    """Handle human approval/denial of AI actions"""
    
    approval_id = request.approval_id
    
    if approval_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    approval_data = pending_approvals[approval_id]
    session_id = approval_data["session_id"]
    
    try:
        if request.approved:
            # Execute the approved tool
            tool_name = approval_data["tool_name"]
            tool_args = approval_data["tool_args"]
            
            result = tool_mapping[tool_name].invoke(tool_args)
            
            # Format search results nicely
            if tool_name == "search" and isinstance(result, list):
                formatted_result = "Search results:\n\n"
                for i, item in enumerate(result[:3], 1):
                    title = item.get("title", "No title")
                    content = item.get("content", "No content")[:200] + "..."
                    url = item.get("url", "")
                    formatted_result += f"{i}. {title}\n{content}\n{url}\n\n"
                message = formatted_result
            else:
                message = f"Tool executed successfully: {result}"
            
        else:
            # User denied the action
            message = "Action cancelled by user."
        
        # Store AI response in session
        if session_id in sessions:
            sessions[session_id]["messages"].append({
                "type": "ai",
                "content": message
            })
        
        return MessageResponse(
            type="ai_response",
            message=message,
            session_id=session_id
        )
        
    except Exception as e:
        error_message = f"Error executing action: {str(e)}"
        if session_id in sessions:
            sessions[session_id]["messages"].append({
                "type": "ai",
                "content": error_message
            })
        return MessageResponse(
            type="ai_response",
            message=error_message,
            session_id=session_id
        )
    
    finally:
        # Clean up pending approval
        if approval_id in pending_approvals:
            del pending_approvals[approval_id]

@app.get("/")
async def get_frontend():
    """Serve the frontend"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human-in-the-Loop AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            height: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .ai-message {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            margin-right: auto;
            white-space: pre-wrap;
        }
        
        .approval-request {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            margin: 0 auto;
            text-align: center;
            padding: 15px;
            max-width: 90%;
        }
        
        .approval-buttons {
            margin-top: 15px;
        }
        
        .approve-btn, .deny-btn {
            padding: 10px 20px;
            margin: 0 5px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .approve-btn {
            background: #28a745;
            color: white;
        }
        
        .approve-btn:hover {
            background: #218838;
        }
        
        .deny-btn {
            background: #dc3545;
            color: white;
        }
        
        .deny-btn:hover {
            background: #c82333;
        }
        
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .message-input:focus {
            border-color: #667eea;
        }
        
        .send-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Human-in-the-Loop AI Assistant</h1>
            <p>AI asks for permission before web searches • Math calculations run automatically</p>
        </div>
        
        <div class="messages" id="messages">
            <div class="message ai-message">
                👋 Hello! I'm your AI assistant with human-in-the-loop controls.
                
                Try asking me:
                • "What is 25 times 48?" (executes automatically)
                • "Who is the current president of the USA?" (asks for approval)
            </div>
        </div>
        
        <div class="input-container">
            <div class="input-group">
                <input type="text" id="messageInput" class="message-input" 
                       placeholder="Type your message here..." maxlength="500">
                <button id="sendBtn" class="send-btn">Send</button>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        function addMessage(content, className) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${className}`;
            messageDiv.textContent = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addApprovalRequest(data) {
            const approvalDiv = document.createElement('div');
            approvalDiv.className = 'message approval-request';
            approvalDiv.innerHTML = `
                <div><strong>🤔 Human Decision Required</strong></div>
                <div style="margin: 10px 0;">${data.message}</div>
                <div class="approval-buttons">
                    <button class="approve-btn" onclick="handleApproval('${data.approval_id}', true)">
                        ✅ Approve
                    </button>
                    <button class="deny-btn" onclick="handleApproval('${data.approval_id}', false)">
                        ❌ Deny
                    </button>
                </div>
            `;
            messagesDiv.appendChild(approvalDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Disable input
            messageInput.disabled = true;
            sendBtn.disabled = true;
            
            // Add user message
            addMessage(message, 'user-message');
            messageInput.value = '';
            
            // Add loading message
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message ai-message loading';
            loadingDiv.textContent = '🤖 Thinking...';
            messagesDiv.appendChild(loadingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: currentSessionId
                    })
                });
                
                const data = await response.json();
                
                // Remove loading message
                loadingDiv.remove();
                
                // Update session ID
                currentSessionId = data.session_id;
                
                if (data.type === 'ai_response') {
                    addMessage(data.message, 'ai-message');
                } else if (data.type === 'approval_request') {
                    addApprovalRequest(data.approval_request);
                }
                
            } catch (error) {
                loadingDiv.remove();
                addMessage(`Error: ${error.message}`, 'ai-message');
            }
            
            // Re-enable input
            messageInput.disabled = false;
            sendBtn.disabled = false;
            messageInput.focus();
        }
        
        async function handleApproval(approvalId, approved) {
            // Remove approval request
            const approvalRequests = document.querySelectorAll('.approval-request');
            approvalRequests.forEach(req => {
                if (req.innerHTML.includes(approvalId)) {
                    req.remove();
                }
            });
            
            // Add loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message ai-message loading';
            loadingDiv.textContent = approved ? '🔍 Executing search...' : '❌ Cancelling...';
            messagesDiv.appendChild(loadingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/approve', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        approval_id: approvalId,
                        approved: approved
                    })
                });
                
                const data = await response.json();
                
                // Remove loading
                loadingDiv.remove();
                
                // Add response
                addMessage(data.message, 'ai-message');
                
            } catch (error) {
                loadingDiv.remove();
                addMessage(`Error: ${error.message}`, 'ai-message');
            }
        }
        
        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Focus on input
        messageInput.focus();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)