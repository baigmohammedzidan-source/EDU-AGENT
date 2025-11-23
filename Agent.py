# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


"""
===============================================================================
🎓 EduBridge AI - Multi-Agent Educational & Career Guidance Platform
===============================================================================

Agents:
- QAAgent        → General Q&A, explanations
- CuratorAgent   → Learning resources & courses
- ResearchAgent  → Deep comparisons, trends (A2A with CuratorAgent)
- AboutAgent     → System help & overview + Developer info
- ResumeAgent    → Resume analysis & improvement tips
- StudyAgent     → Study plans & roadmaps
- CareerAgent    → Career path advice
- RAGAgent       → Document / notes Q&A (using pasted text)
- WebAgent       → Web-style informational answers (no real-time web)

Core Features:
- Orchestrator           → Routes query to best agent
- Agent2Agent Protocol   → Agents can collaborate internally
- Session & Memory       → Context-aware replies
- Observatory            → Logs activity and metrics
- Bilingual Support      → English / Tamil

NO file upload.  
Documents / resumes are pasted as text with prefixes:
- "RESUME: <your resume text>"
- "DOC: <your document / notes>"

Developer:
- Mohammed Faizal. M, 19 years
- 2nd year B.Com student (primary), The New College, Chennai
- Placement cell officer, Department of Commerce,
  Achiever's Club 2025-2026, Shift-1
===============================================================================
"""

# ==================== IMPORTS ====================
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

import gradio as gr
import google.generativeai as genai


# ==================== API KEY CONFIGURATION ====================

def get_api_key() -> str:
    """
    Get Gemini API key from multiple sources in priority order:
    1. Environment variable (HuggingFace Spaces, production)
    2. Kaggle secrets
    3. User input (fallback for local testing)
    """
    
    # Try environment variable first (HuggingFace Spaces, Docker, etc.)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        print("✅ API key loaded from environment variable")
        return api_key
    
    # Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GEMINI_API_KEY")
        if api_key:
            print("✅ API key loaded from Kaggle secrets")
            return api_key
    except Exception:
        pass  # Kaggle secrets not available
    
    # Try Google Colab secrets
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY')
        if api_key:
            print("✅ API key loaded from Colab secrets")
            return api_key
    except Exception:
        pass  # Colab not available
    
    
    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY is required!\n"
            "Set it as an environment variable or add it to your platform's secrets."
        )
    
    if not api_key.startswith("AIza"):
        raise ValueError("❌ Invalid API key format. Gemini keys start with 'AIza'.")
    
    print("✅ API key accepted!")
    return api_key


# Initialize API
try:
    API_KEY = get_api_key()
    genai.configure(api_key=API_KEY)
    print("✅ Gemini API configured successfully!\n")
except Exception as e:
    print(f"❌ Error configuring API: {str(e)}")
    raise


# ==================== CONFIG ====================

class Config:
    MODEL_NAME = "gemini-2.5-flash"
    MAX_MEMORY_SIZE = 60


# ==================== MEMORY (Context + Session Data) ====================

class Memory:
    def __init__(self, max_size: int = Config.MAX_MEMORY_SIZE):
        self.max_size = max_size
        self.conversations: List[Dict] = []
        self.session_id = f"session_{int(time.time())}"
        self.created_at = datetime.now()
        self.session_data: Dict[str, Dict] = {}  # resume, doc, profile etc.

    def add(self, role: str, content: str, agent: str = "system"):
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "agent": agent,
            "session_id": self.session_id,
        })
        if len(self.conversations) > self.max_size:
            self.conversations = self.conversations[-self.max_size:]

    def get_context(self, last_n: int = 8) -> str:
        recent = self.conversations[-last_n:]
        return "\n".join(
            f"{m['role']} ({m['agent']}): {m['content'][:200]}"
            for m in recent
        )

    def update_session_data(self, key: str, value):
        self.session_data[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }

    def get_session_data(self, key: str, default=None):
        return self.session_data.get(key, {}).get("value", default)

    def get_info(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.conversations),
            "data_keys": list(self.session_data.keys())
        }

    def clear(self):
        self.conversations = []
        self.session_data = {}


# ==================== OBSERVATORY ====================

class Observatory:
    def __init__(self):
        self.logs: List[Dict] = []
        self.metrics: Dict = {
            "total_queries": 0,
            "agent_calls": {},
            "errors": 0,
            "a2a_calls": 0
        }
        self.start_time = datetime.now()

    def log(self, event_type: str, agent: str, details: str):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "agent": agent,
            "details": details
        })

        if len(self.logs) > 120:
            self.logs = self.logs[-120:]

        if event_type == "query":
            self.metrics["total_queries"] += 1
        if event_type == "error":
            self.metrics["errors"] += 1
        if event_type == "a2a":
            self.metrics["a2a_calls"] += 1

        if agent not in self.metrics["agent_calls"]:
            self.metrics["agent_calls"][agent] = 0
        self.metrics["agent_calls"][agent] += 1

    def report(self) -> str:
        uptime = datetime.now() - self.start_time
        lines = []
        lines.append("📊 EDUBRIDGE SYSTEM OBSERVATORY")
        lines.append(f"⏱ Uptime: {int(uptime.total_seconds())} seconds")
        lines.append(f"💬 Total Queries: {self.metrics['total_queries']}")
        lines.append(f"❌ Errors: {self.metrics['errors']}")
        lines.append(f"🔁 Agent2Agent Calls: {self.metrics['a2a_calls']}")
        lines.append("")
        lines.append("🤖 Agent Call Counts:")
        for agent, count in sorted(self.metrics["agent_calls"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  · {agent}: {count}")
        lines.append("\n📝 Recent Events:")
        for log in self.logs[-6:]:
            t = log["timestamp"].split("T")[1][:8]
            lines.append(f"  [{t}] {log['type']} - {log['agent']}: {log['details'][:80]}")
        return "\n".join(lines)


# ==================== BASE AGENT + A2A ====================

class BaseAgent:
    def __init__(self, name: str, role: str, model, observatory: Observatory):
        self.name = name
        self.role = role
        self.model = model
        self.obs = observatory

    def call_model(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            if not response or not getattr(response, "text", None):
                return "⚠️ No response from model. Try rephrasing your question."
            return response.text
        except Exception as e:
            self.obs.log("error", self.name, f"Model error: {str(e)}")
            return f"⚠️ {self.name} encountered an error: {str(e)}"

    def process(self, query: str, context: str = "") -> str:
        raise NotImplementedError

    def communicate_with_agent(self, target_agent: "BaseAgent", message: str, context: str = "") -> str:
        self.obs.log("a2a", f"{self.name}→{target_agent.name}", f"A2A message: {message[:80]}")
        return target_agent.process(message, context)


# ==================== AGENTS ====================

class QAAgent(BaseAgent):
    def process(self, query: str, context: str = "") -> str:
        prompt = f"""
You are an Educational & Career Q&A Specialist.

Conversation context:
{context}

Student question:
{query}

Explain clearly, step-by-step, with practical examples, tips, and encouragement.
"""
        return self.call_model(prompt)


class CuratorAgent(BaseAgent):
    def process(self, query: str, context: str = "") -> str:
        prompt = f"""
You are an Educational Resource Curator.

User request:
{query}

Task:
- Suggest high-quality learning resources.
- For each resource include:
  · Name
  · Type (course / book / video / website)
  · Platform
  · Difficulty level
  · One-line reason why recommended.
"""
        return self.call_model(prompt)


class ResearchAgent(BaseAgent):
    def __init__(self, name, role, model, observatory, curator_agent: CuratorAgent):
        super().__init__(name, role, model, observatory)
        self.curator_agent = curator_agent

    def process(self, query: str, context: str = "") -> str:
        base_report = self.call_model(f"""
You are a Research Specialist for education & careers.

Topic:
{query}

Provide:
1. Overview
2. Current trends / statistics (if known)
3. Comparison of options
4. Pros & cons
5. Future scope.
""")
        curated = self.communicate_with_agent(
            self.curator_agent,
            f"Suggest learning resources for: {query}",
            context
        )
        return base_report + "\n\n📚 Recommended Learning Resources (via Curator Agent):\n" + curated


class AboutAgent(BaseAgent):
    def __init__(self, name, role, model, observatory, memory: Memory):
        super().__init__(name, role, model, observatory)
        self.memory = memory

    def process(self, query: str, context: str = "") -> str:
        info = self.memory.get_info()
        return f"""
🎓 **EduBridge - Multi-Agent Educational & Career Guidance System**

Available Agents:
- QAAgent → General concepts, explanations, doubts
- CuratorAgent → Courses, books, videos, learning paths
- ResearchAgent → Deep analysis, comparisons, future trends
- ResumeAgent → Resume review & improvements
- StudyAgent → Custom study plans
- CareerAgent → Career options & next steps
- RAGAgent → Q&A over pasted documents (DOC:)
- WebAgent → Web-style info & broad overviews

Key Features:
- Orchestrator routes your query to the best agent
- Agents can talk internally (Agent2Agent)
- Session & Memory keep recent context
- Observatory tracks queries and events
- Supports both English and தமிழ் (Tamil)

Session Info:
- Session ID: {info['session_id']}
- Started: {info['created_at']}
- Messages so far: {info['message_count']}
- Stored Data Keys: {", ".join(info['data_keys']) or "None"}

👨‍💻 **About the Developer**

- Name: *Mohammed Faizal. M*  
- Age: 19  
- Primary Role: 2nd year B.Com student, The New College, Chennai  
- Secondary Role: Placement Cell Officer, Department of Commerce,  
  Achiever's Club 2025–2026, Shift-1  

EduBridge AI is built as a multi-agent educational mentor to support students
with guidance in careers, skills, and learning paths.
"""


class ResumeAgent(BaseAgent):
    def __init__(self, name, role, model, observatory, memory: Memory):
        super().__init__(name, role, model, observatory)
        self.memory = memory

    def process(self, query: str, context: str = "") -> str:
        resume_text = self.memory.get_session_data("resume_text")
        if not resume_text:
            return """📄 No resume found in memory.

Paste your resume like this:
RESUME:
[Your resume text here]

Then ask:
- "Analyze my resume"
- "How can I improve my resume for data analyst roles?"
"""
        prompt = f"""
You are a Resume Reviewer & Career Coach.

Candidate Resume:
{resume_text}

User question:
{query}

Provide:
- Overall evaluation
- Strengths & weaknesses
- 5–10 specific improvements (rewrite bullets, action verbs, quantification)
- ATS friendliness suggestions.
"""
        return self.call_model(prompt)


class StudyAgent(BaseAgent):
    def __init__(self, name, role, model, observatory, memory: Memory):
        super().__init__(name, role, model, observatory)
        self.memory = memory

    def process(self, query: str, context: str = "") -> str:
        profile = self.memory.get_session_data("profile_notes", "")
        prompt = f"""
You are a Study Plan Architect.

Student profile (if any):
{profile}

Request:
{query}

Create:
- A structured study plan (days / weeks)
- Topics with clear order
- Practice suggestions
- Checkpoints / mini projects
- Tips to stay consistent and avoid burnout.
"""
        return self.call_model(prompt)


class CareerAgent(BaseAgent):
    def __init__(self, name, role, model, observatory, memory: Memory):
        super().__init__(name, role, model, observatory)
        self.memory = memory

    def process(self, query: str, context: str = "") -> str:
        profile = self.memory.get_session_data("profile_notes", "")
        resume = self.memory.get_session_data("resume_text", "")
        prompt = f"""
You are a Career Path Advisor.

Student profile (if available):
{profile}

Resume (if available):
{resume}

Question:
{query}

Provide:
- 2–4 suitable career paths
- Why each path fits
- Skills/qualifications needed
- Next 3 concrete steps for each path.
"""
        return self.call_model(prompt)


class RAGAgent(BaseAgent):
    def __init__(self, name, role, model, observatory, memory: Memory):
        super().__init__(name, role, model, observatory)
        self.memory = memory

    def process(self, query: str, context: str = "") -> str:
        doc = self.memory.get_session_data("rag_document")
        if not doc:
            return """📄 No document in memory.

Paste your document like this:
DOC:
[Your document / article / notes here]

Then ask:
- "Summarize the document"
- "What are the key points?"
- "Explain the section about X"
"""
        prompt = f"""
You are a Document Q&A Assistant.

Document:
{doc}

User question:
{query}

Answer strictly using the information in the document.
If unsure, say clearly that the document doesn't answer it.
"""
        return self.call_model(prompt)


class WebAgent(BaseAgent):
    def process(self, query: str, context: str = "") -> str:
        prompt = f"""
You are a 'Web-style' Information Assistant.

You do NOT have live internet access, but you will:
- Use your internal knowledge
- Provide an up-to-date style overview
- Clearly mention that information may not reflect real-time changes.

User query:
{query}
"""
        return self.call_model(prompt)


# ==================== ORCHESTRATOR ====================

class Orchestrator:
    def __init__(self, agents: Dict[str, BaseAgent], memory: Memory, observatory: Observatory):
        self.agents = agents
        self.memory = memory
        self.obs = observatory

    def route(self, query: str) -> str:
        q = query.lower()

        # Explicit tags
        if q.startswith("resume:") or q.startswith("cv:"):
            return "resume"
        if q.startswith("doc:") or q.startswith("document:"):
            return "rag"

        # Intent patterns
        if any(w in q for w in ["study plan", "study schedule", "prepare for exam", "how to study"]):
            return "study"
        if any(w in q for w in ["career path", "which career", "scope of", "job role", "future career"]):
            return "career"
        if any(w in q for w in ["latest", "current trends", "recent", "update", "web search", "google"]):
            return "web"
        if any(w in q for w in ["research", "compare", "analysis", "trends", "market"]):
            return "research"
        if any(w in q for w in ["course", "resources", "learn", "material", "roadmap", "syllabus"]):
            return "curator"
        if any(w in q for w in ["about system", "how does this work", "what can you do", "edubridge"]):
            return "about"
        if any(w in q for w in ["resume", "cv"]):
            return "resume"
        if any(w in q for w in ["document", "pdf", "article", "notes"]):
            return "rag"

        return "qa"

    def process(self, query: str) -> str:
        self.obs.log("query", "orchestrator", f"Received: {query[:80]}")
        context = self.memory.get_context()

        agent_key = self.route(query)
        agent = self.agents.get(agent_key, self.agents["qa"])
        self.obs.log("routing", "orchestrator", f"Routing to {agent.name}")

        response = agent.process(query, context)
        self.memory.add("user", query)
        self.memory.add("assistant", response, agent.name)
        self.obs.log("response", agent.name, "Response generated")

        return f"**🤖 {agent.name}** | {agent.role}\n\n{response}"


# ==================== EDUBRIDGE SYSTEM WRAPPER ====================

class EduBridgeSystem:
    def __init__(self):
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
        self.memory = Memory()
        self.obs = Observatory()

        curator = CuratorAgent("CuratorAgent", "Resource Curator", self.model, self.obs)

        self.agents = {
            "qa": QAAgent("QAAgent", "Q&A Specialist", self.model, self.obs),
            "curator": curator,
            "research": ResearchAgent("ResearchAgent", "Research Specialist", self.model, self.obs, curator),
            "about": AboutAgent("AboutAgent", "System Guide", self.model, self.obs, self.memory),
            "resume": ResumeAgent("ResumeAgent", "Resume Analyzer", self.model, self.obs, self.memory),
            "study": StudyAgent("StudyAgent", "Study Plan Generator", self.model, self.obs, self.memory),
            "career": CareerAgent("CareerAgent", "Career Path Advisor", self.model, self.obs, self.memory),
            "rag": RAGAgent("RAGAgent", "Document Intelligence", self.model, self.obs, self.memory),
            "web": WebAgent("WebAgent", "Web-style Info", self.model, self.obs),
        }

        self.orchestrator = Orchestrator(self.agents, self.memory, self.obs)

    def handle_message(self, message: str, language: str) -> str:
        raw = message.strip()

        # Language wrapping
        if language == "Tamil":
            msg_for_model = "தயவுசெய்து தமிழில் பதிலளிக்கவும்:\n" + raw
        else:
            msg_for_model = raw

        lower = raw.lower()

        # Store resume / document / profile in session data
        if lower.startswith("resume:") or lower.startswith("cv:"):
            resume_text = raw.split(":", 1)[1].strip()
            self.memory.update_session_data("resume_text", resume_text)
        elif lower.startswith("doc:") or lower.startswith("document:"):
            doc_text = raw.split(":", 1)[1].strip()
            self.memory.update_session_data("rag_document", doc_text)
        elif any(w in lower for w in ["about me", "my background", "i am studying", "i am a student"]):
            self.memory.update_session_data("profile_notes", raw)

        return self.orchestrator.process(msg_for_model)

    def get_status(self) -> str:
        return self.obs.report()

    def clear(self):
        self.memory.clear()


# ==================== INIT SYSTEM ====================

system = EduBridgeSystem()


# ==================== GRADIO CALLBACKS ====================

def chat_fn(message, history, language):
    if not message.strip():
        return history, ""
    response = system.handle_message(message, language)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""

def clear_fn():
    system.clear()
    return [], "🧹 Session cleared. Start a new conversation."

def status_fn():
    return system.get_status()


# ==================== GRADIO UI ====================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css="""
    body {
        background: radial-gradient(circle at top, #1f2937 0, #020617 55%) fixed;
        color: #e5e7eb;
        font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .glass-card {
        background: rgba(15,23,42,0.78);
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 18px 45px rgba(0,0,0,0.4);
        backdrop-filter: blur(18px) saturate(140%);
        padding: 16px;
    }
    .header-box {
        background: linear-gradient(135deg, rgba(37,99,235,0.95), rgba(29,78,216,0.98));
        border-radius: 22px;
        padding: 20px 28px;
        margin-bottom: 18px;
        box-shadow: 0 20px 40px rgba(15,23,42,0.7);
        color: #f9fafb;
    }
    .header-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .header-subtitle {
        font-size: 14px;
        opacity: 0.96;
        margin-top: 6px;
    }
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding-top: 10px;
    }
    .gr-button {
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    .gr-button.primary {
        background: linear-gradient(135deg,#3b82f6,#2563eb) !important;
        border: 1px solid rgba(191,219,254,0.7) !important;
        color: white !important;
    }
    .gr-button.secondary {
        background: rgba(15,23,42,0.7) !important;
        border: 1px solid rgba(148,163,184,0.7) !important;
        color: #e5e7eb !important;
    }
    .gr-textbox, .gr-chatbot {
        border-radius: 14px !important;
        border: 1px solid rgba(148,163,184,0.5) !important;
        background: rgba(15,23,42,0.88) !important;
        color: #e5e7eb !important;
    }
    .gr-chatbot > div {
        background: transparent !important;
    }
    .message-user {
        background: linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
    }
    .message-bot {
        background: rgba(15,23,42,0.95) !important;
        color: #e5e7eb !important;
        border-radius: 18px 18px 18px 4px !important;
        border: 1px solid rgba(148,163,184,0.6);
    }
    """,
    title="EduBridge AI - Multi-Agent Guidance"
) as demo:

    # Header
    gr.HTML("""
    <div class="header-box">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="header-title">EDUBRIDGE AI</div>
                <div class="header-subtitle">
                    Multi-Agent Educational & Career Guidance • Q&A • Research • Resume • Career • RAG
                </div>
            </div>
            <div style="font-size:32px;">🎓</div>
        </div>
    </div>
    """)

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=3):
            with gr.Group(elem_classes="glass-card"):
                chatbot = gr.Chatbot(
                    type="messages",
                    label="💬 EduBridge Conversation",
                    height=460
                )
                language = gr.Radio(
                    ["English", "Tamil"],
                    value="English",
                    label="🌐 Language"
                )
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about careers, courses, resume, study plans, documents (DOC:/RESUME:), or the system...",
                    lines=3
                )
                with gr.Row():
                    send_btn = gr.Button("📤 Send", elem_classes="primary")
                    clear_btn = gr.Button("🧹 Clear Session", elem_classes="secondary")

        # Right: Observatory & Help
        with gr.Column(scale=2):
            with gr.Group(elem_classes="glass-card"):
                status_box = gr.Textbox(
                    label="📊 Observatory & System Status",
                    lines=20,
                    interactive=False
                )
                status_btn = gr.Button("🔄 Refresh Status", elem_classes="secondary")
                gr.Markdown("""
**How to use agents:**

- **General Q&A (QAAgent)**  
  · "What is data science?"  
  · "Explain machine learning in simple terms."

- **Resume (ResumeAgent)**  
  · `RESUME:` + paste your resume  
  · Then: "Analyze my resume for analyst roles"

- **Study Plan (StudyAgent)**  
  · "Create a 30-day study plan for SQL and Excel"

- **Career (CareerAgent)**  
  · "Career options after B.Com if I like AI and analytics"

- **Document Q&A (RAGAgent)**  
  · `DOC:` + paste article/notes  
  · Then: "Summarize this document"

- **System Info (AboutAgent)**  
  · "How does this EduBridge system work?"
                """)

    # Events
    send_btn.click(
        chat_fn,
        inputs=[msg, chatbot, language],
        outputs=[chatbot, msg]
    )
    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, language],
        outputs=[chatbot, msg]
    )

    clear_btn.click(
        clear_fn,
        outputs=[chatbot, status_box]
    )

    status_btn.click(
        status_fn,
        outputs=[status_box]
    )

    demo.load(
        status_fn,
        outputs=[status_box]
    )

# ==================== LAUNCH ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 25 + "🎓 EDUBRIDGE AI SYSTEM")
    print("="*80)
    print("\nDeveloper: Mohammed Faizal. M")
    print("Institution: The New College, Chennai")
    print("="*80 + "\n")
    
    # Check if running in notebook environment
    try:
        get_ipython()
        IN_NOTEBOOK = True
        print("📓 Running in Jupyter/Colab/Kaggle notebook environment")
    except NameError:
        IN_NOTEBOOK = False
        print("💻 Running in standard Python environment")
    
    # Configure launch parameters based on environment
    if IN_NOTEBOOK:
        # For notebooks: simpler config, auto-find port, always share
        print("🚀 Launching with notebook configuration...")
        print("   • Auto-finding available port")
        print("   • Generating shareable public URL")
        print("   • Multiple reruns supported\n")
        
        demo.launch(
            share=True,           # Always create public URL
            debug=False,          # Reduce noise in notebooks
            quiet=True,           # Less verbose output
            show_error=True,      # Show errors clearly
            inbrowser=False,      # Don't auto-open browser in notebooks
        )
    else:
        # For production/local: specific config
        print("🚀 Launching with production configuration...")
        print("   • Server: 0.0.0.0:7860")
        print("   • Public URL sharing enabled\n")
        
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=False,
            show_error=True,
        )
