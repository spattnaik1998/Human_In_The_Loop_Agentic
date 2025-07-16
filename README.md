# Human-in-the-Loop AI Assistant

A clean, simple FastAPI implementation demonstrating human-in-the-loop AI decision making using OpenAI and Tavily search.

## 🎯 Core Human-in-the-Loop Logic

This application demonstrates the key concept: **AI asks for human permission before taking certain actions.**

- **Math calculations** (multiply tool) → Execute automatically
- **Web searches** (search tool) → Ask for human approval first
- **Clear decision points** → User approves or denies each search

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API keys in `main.py`:**
   ```python
   os.environ["OPENAI_API_KEY"] = "your-actual-openai-key"
   os.environ["TAVILY_API_KEY"] = "your-actual-tavily-key"
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Open your browser:**
   ```
   http://localhost:8000
   ```

## 🧠 How It Works

### The Core Logic (in `process_ai_response` function):

1. **User sends message** → AI processes with tools available
2. **AI wants to use a tool** → Check which tool
3. **Decision point:**
   - If `multiply` → Execute automatically
   - If `search` → Ask human for approval
4. **Human approves/denies** → Execute or cancel accordingly

### Example Flows:

**Math (automatic):**
```
User: "What is 25 times 48?"
AI: Executes multiply tool automatically
Response: "The result is: 1200"
```

**Search (requires approval):**
```
User: "Who is the current president?"
AI: "I want to search for: 'current president'. Do you approve?"
User: Clicks "Approve"
AI: Executes search and returns results
```

## 📁 Project Structure

- `main.py` - Complete FastAPI application with embedded frontend
- `requirements.txt` - Python dependencies
- No separate frontend files needed - everything is self-contained

## 🎨 Features

- **Clean FastAPI backend** - No WebSocket complexity
- **Embedded frontend** - Single file deployment
- **Session management** - Tracks conversation history
- **Error handling** - Graceful failure management
- **Responsive UI** - Works on desktop and mobile

## 🔧 API Endpoints

- `GET /` - Serves the web interface
- `POST /chat` - Main chat endpoint (human-in-the-loop logic here)
- `POST /approve` - Handle approval/denial of AI actions

## 💡 Portfolio Value

This project showcases:
- **AI agent architecture** with tool integration
- **Human-AI collaboration** patterns
- **Clean API design** with FastAPI
- **Decision-making workflows** in AI systems
- **Real-world AI safety** concepts (human oversight)

Perfect for demonstrating understanding of:
- LangChain tool integration
- Human-centered AI design
- RESTful API development
- Modern web application architecture