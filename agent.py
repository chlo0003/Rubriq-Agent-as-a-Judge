from IPython.display import display, JSON
import json

def notebook_pretty_print(data):
    # If data is a string, parse it first
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            print(data)
            return
            
    display(JSON(data))


import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Union

from google.genai import types

from google.adk.agents import Agent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import agent_tool

# Optional Kaggle secrets support
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None  # safe fallback


# -------------------------------------------------------------------
# Logging & API key setup
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _ensure_google_api_key() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return

    key = None
    if UserSecretsClient is not None:
        try:
            secrets = UserSecretsClient()
            key = secrets.get_secret("GOOGLE_API_KEY") or secrets.get_secret("GEMINI_API_KEY")
        except Exception:
            key = None

    if key:
        os.environ["GOOGLE_API_KEY"] = key
        return

    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Set it as an environment variable or Kaggle secret "
        "('GOOGLE_API_KEY' or 'GEMINI_API_KEY')."
    )


_ensure_google_api_key()
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")

MODEL_NAME = "gemini-2.0-flash"


# -------------------------------------------------------------------
# Agent instructions
# -------------------------------------------------------------------

ANALYSIS_INSTRUCTION = """
You are the ANALYSIS AGENT in a judge team for a coding project.

You receive a SINGLE user message that is a JSON object.
Your tasks:
1) Summarise the project.
2) Infer 3–8 evaluation criteria.

Output STRICT JSON ONLY:
{
  "summary": "...",
  "criteria": [{ "name": "...", "max_score": ... }]
}
"""

SCORING_INSTRUCTION = """
You are the SCORING AGENT.
You receive JSON containing 'rubric_text', 'project_writeup', 'code_text', and 'analysis_result'.
Score each criterion based on evidence.

Output STRICT JSON ONLY:
{
  "scores": [{ "criterion": "...", "score": ..., "max_score": ..., "reason": "..." }]
}
"""

FEEDBACK_INSTRUCTION = """
You are the FEEDBACK AGENT.
You receive analysis and scoring results. Write an overall comment.

Output STRICT JSON ONLY:
{
  "overall_comment": "..."
}
"""

ORCHESTRATOR_INSTRUCTION = """
You are the ORCHESTRATOR.
The user ALWAYS sends you a JSON object.
1) Call "rubriq_pipeline" exactly once with the user's JSON.
2) Return the tool's JSON result AS-IS.
"""


# -------------------------------------------------------------------
# Build sub-agents (LLM)
# -------------------------------------------------------------------

analysis_agent = Agent(
    name="rubriq_analysis_agent",
    model=MODEL_NAME,
    instruction=ANALYSIS_INSTRUCTION,
    output_key="analysis_result",
)

scoring_agent = Agent(
    name="rubriq_scoring_agent",
    model=MODEL_NAME,
    instruction=SCORING_INSTRUCTION,
    output_key="scoring_result",
)

feedback_agent = Agent(
    name="rubriq_feedback_agent",
    model=MODEL_NAME,
    instruction=FEEDBACK_INSTRUCTION,
    output_key="rubriq_output",
)

pipeline_agent = SequentialAgent(
    name="rubriq_pipeline",
    description="Sequential pipeline: analysis, scoring, feedback.",
    sub_agents=[analysis_agent, scoring_agent, feedback_agent],
)

pipeline_tool = agent_tool.AgentTool(agent=pipeline_agent)

orchestrator_agent = Agent(
    name="rubriq_orchestrator",
    model=MODEL_NAME,
    instruction=ORCHESTRATOR_INSTRUCTION,
    tools=[pipeline_tool],
)

root_agent = orchestrator_agent


# -------------------------------------------------------------------
# Session service (memory) & runner
# -------------------------------------------------------------------

session_service = InMemorySessionService()
ORCH_APP_NAME = "rubriq_orchestrator_app"

orchestrator_runner = Runner(
    agent=orchestrator_agent,
    app_name=ORCH_APP_NAME,
    session_service=session_service,
)

# -------------------------------------------------------------------
# Helper Function: run_session
# -------------------------------------------------------------------

USER_ID = "kaggle_user"

async def run_session(
    runner_instance: Runner,
    user_queries: Union[List[str], str] = None,
    session_name: str = "default",
):
    """
    Helper to run a conversation session in a Kaggle notebook.
    Prints the agent's output as it streams.
    """
    print(f"\n ### Session: {session_name}")

    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query in user_queries:
            # Note: In this specific pipeline, 'query' is expected to be a JSON string
            # but we print a truncated version to keep logs clean if it's huge.
            display_query = (query[:75] + '...') if len(query) > 75 else query
            print(f"\nUser > {display_query}")

            content = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream response
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session_name, new_message=content
            ):
                if event.content and event.content.parts:
                    text_part = event.content.parts[0].text
                    if text_part and text_part != "None":
                        print(f"{MODEL_NAME} > {text_part}")
    else:
        print("No queries!")


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    demo_rubric = """
    Category 1: The Pitch (Problem, Solution, Value)
(30 points total)	This is where you'll be evaluated on the "why" and "what" of your project and how well you communicate your vision.
Core Concept & Value
(15 points)	Your project's central idea, its relevance to the track for the submission; focused on innovation and value. The use of agents should be clear, meaningful and central to your solution.
Writeup
(15 points)	How well your written submission articulates the problem you're solving, your solution, its architecture, and your project's journey.
Category 2: The Implementation
(Architecture, Code)
(70 points total)	This is where you'll be evaluated on the "how" of your project. This includes the quality of your code, technical design, and AI integration.
Technical Implementation
(50 points)	In your submission, you must demonstrate what you’ve learned in this course by applying at least three (3) of the key concepts listed in the Features To Include In Your Agent Submission section.

For this criteria, we will assess the quality of your solution's architecture, and your code, and the meaningful use of agents in your solution.

Your code should contain comments pertinent to implementation, design and behaviors.

Participants are not required to deploy their agents to a live public endpoint for judging purposes; however, if you do deploy, please provide documentation to reproduce the deployment.

🚨REMINDER: DO NOT INCLUDE ANY API KEYS OR PASSWORDS IN YOUR CODE.
Documentation
(20 points)	Your submission (when submitting via GitHub) should contain a README.md file explaining the problem, solution, architecture, instructions for setup, and relevant diagrams or images where appropriate.

If you are solely submitting a Kaggle notebook, please provide documentation directly inline via Markdown Cells of the notebook.
Bonus points (Tooling, Model Use, Deployment, Video)
20 points total	You can earn optional bonus points.
Effective Use of Gemini
(5 points)	Use of Gemini to power your agent (or at least one sub-agent).
Agent Deployment
(5 points)	If you either have code or otherwise show evidence (e.g. in your code or write up) of having deployed your agent using Agent Engine or a similar Cloud-based runtime (e.g. Cloud Run).
YouTube Video Submission
(10 points)	Your video should include clarity, conciseness and quality of messaging. It should be under 3 min long. It should articulate:

Problem Statement: Describe the problem you're trying to solve, and why you think it's an important or interesting problem to solve.

Agents: Why agents? How can agents uniquely help solve that problem?

Architecture: Images and a description of the overall agent architecture.

Demo: demo of your solution, which can include images, an animation, or a video of the agent working.

The Build: How you created it, what tools or technologies you used.
    """

    demo_project_writeup = """
    Project Overview

Edubridge is a free, ethical, and transparent AI guidance platform designed specifically for students and fresh graduates. It addresses the critical challenge of accessible career counselling by providing intelligent, personalised guidance without any cost barriers. Built using a sophisticated multi-agent architecture, Edubridge combines the power of ChatGPT with carefully curated educational resources from trusted platforms.

Introduction

As a second-year B.Com. Student at The New College, Chennai, I've witnessed firsthand how my peers struggle with career decisions, interview preparation, and finding quality learning resources. Many students lack access to affordable career guidance, and the overwhelming amount of online information makes it difficult to identify trustworthy sources.

Edubridge was born from this observation—a commitment to democratise educational guidance and ensure that every student, regardless of their background or financial situation, can access quality career counselling and learning resources.

Core Features

Intelligent Multi-Agent System:

Specialised agents working together to provide comprehensive guidance
Automatic query routing to the most qualified agent
Context-aware responses tailored to individual needs
AI-Powered Responses:

Integration with ChatGPT (GPT-4o-mini) for natural, intelligent conversations
Career path recommendations and skill development advice
Interview preparation strategies and industry insights
Curated Resource Library:

Access to verified learning platforms, including Kaggle Learn, Microsoft Learn, Google Developers, and more
Quality-filtered resources to save research time
Direct links to courses, tutorials, and training materials
Beautiful Gradio Web Interface:

Clean, intuitive chat interface for seamless interaction
Mobile-responsive design accessible from any device
Real-time conversation with instant responses
Session summary and statistics tracking
No technical knowledge required to use
Session Management:

Conversation history tracking
Progress monitoring across sessions
Session summary statistics
Complete Transparency:

Full logging of all interactions
Clear acknowledgement of limitations
Ethical operation with user privacy as priority
The Agent Team

Edubridge operates through four specialised agents, each designed for specific tasks:

Router Agent
Analyses incoming queries to understand user intent
Intelligently routes questions to the appropriate specialist agent
Ensures users receive the most relevant assistance
QA Agent
Provides educational and career guidance using ChatGPT
Answers questions about career paths, skills, and industry trends
Offers interview preparation tips and professional development advice
Curator Agent
Discovers and recommends learning resources from trusted platforms
Filters content by quality and relevance
Provides direct access to courses on Kaggle, Microsoft Learn, Google Cloud, Coursera, and more
About Agent
Shares information about the developer and project
Explains the system architecture and capabilities
Provides contact information and project background
Unique Features That Set Edubridge Apart

Truly Free & Ethical: No hidden costs, no data exploitation, no commercial agenda. Built purely to help students succeed.

Trusted Sources Only: Unlike generic AI chatbots, Edubridge only recommends resources from verified, quality educational platforms—saving students from information overload and low-quality content.

Multi-Agent Architecture: Professional-grade system design with specialised agents, agents, not just a simple chatbot wrapper. Each agent has distinct responsibilities and expertise.

Production-Ready Code: Complete with error handling, logging, session management, and deployment capabilities. This isn't just a demo—it's a working system ready for real users.

Seamless Web Experience: Gradio-powered interface provides an intuitive, beautiful chat experience that works perfectly on desktop and mobile devices.

Memory & Context: Remembers conversation history within sessions, providing coherent multi-turn conversations rather than treating each query in isolation.

Observability: Complete logging system with RUN_ID tracking, making the system transparent and debuggable.

Permanent Deployment: Hosted on Hugging Face Spaces for reliable, continuous access without setup requirements.

Technology Stack

AI Model: ChatGPT (GPT-4o-mini) via OpenAI API
Web Framework: Gradio for an interactive web interface
Backend: Python 3.8+ with modular, clean architecture
Content Tools: BeautifulSoup4 and Requests for web scraping
Design Pattern: Multi-agent system with Agent-to-Agent (A2A) protocol
Deployment: Hugging Face Spaces for permanent, seamless interaction
Try Edubridge Yourself!

Edubridge is now permanently deployed and accessible to everyone!

Access Options:

Hugging Face Spaces (Live Deployment): https://huggingface.co/spaces/MdFaizal6059/Edubridge-Career-and-educational-assistance-Agent

Instant access, no setup required
Permanent availability for seamless interaction
Works on any device with a web browser
Kaggle Notebook (Source Code): https://www.kaggle.com/code/mohammedfaizalm/edubridge-agent-is-good-capstone

View the complete source code
Understand the implementation
Run your own instance
How to Use:

Click the Hugging Face link above
Wait for the interface to load (usually 5-10 seconds)
Start chatting with Edubridge in the chat box
Ask any educational or career-related question
Sample Queries to Try:

"What skills do I need to start a career in data science?"
"Resources for learning Python from scratch"
"How should I prepare for technical interviews?"
"Tell me about the developer of Edubridge"
Feedback I'm Looking For

I would love to hear your thoughts on:

Functionality:

Does the system respond appropriately to different types of queries?
Are the resource recommendations helpful and relevant?
Is the routing between agents logical and effective?
User Experience:

Is the Gradio interface intuitive and easy to use?
Are the responses clear and actionable?
Does the conversation flow feel natural?
How is the performance and response time?
Technical Implementation:

Any suggestions for improving the code architecture?
Ideas for additional features or agents?
Performance optimisation of opportunities?
Impact & Reach:

How can this project better serve students?
What additional features would make it more valuable?
Suggestions for wider adoption and scaling?
Mission: Creating accessible educational technology that empowers students regardless of their background or resources.

Whether you're a student, educator, developer, or someone passionate about educational technology, I invite you to:

Try Edubridge - on Hugging Face and share your experience
Provide feedback - on functionality and user experience
Suggest improvements - or additional features
Share ideas - on how to maximise impact for students
Collaborate - if you're interested in contributing to this mission
Spread the word - to students who could benefit from free guidance
Education should be accessible to all, and technology should serve to democratise it. Let's work together to make quality educational guidance available to every student who needs it.

Thank you for taking the time to review Edubridge.
    """

    demo_code_text = '''
    # ==========================================
# EDUBRIDGE: AI MULTI-AGENT SYSTEM (FINAL UI VERSION)
# Developer: Mohammed Faizal. M
# ==========================================

import os
import time
import google.generativeai as genai
import gradio as gr
from kaggle_secrets import UserSecretsClient

# --- 1. CONFIGURATION ---

# Fetch API Key
try:
    user_secrets = UserSecretsClient()
    API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception:
    API_KEY = "YOUR_API_KEY_HERE"

# Configure Gemini
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-lite"

# PASTE YOUR DIRECT LOGO IMAGE LINK HERE (Must end in .png or .jpg)
LOGO_URL = "https://d2u1z1lopyfwlx.cloudfront.net/thumbnails/498bc06f-8aca-5934-83ae-cba4ce1408bf/10f1907b-2567-564d-b1ad-0ade96af1083.jpg"

# --- 2. MEMORY & DATA (UPDATED) ---

DEV_INFO = "
DEVELOPER PROFILE:
Name: Mohammed Faizal. M
Age: 19
Place: Chennai
Primary Role: Undergraduate student - B.Com (Commerce), The New College, Chennai (Day Shift).
Secondary Role: Placement Officer (Student/Coordinator), Achievers Club 2025-2026, P.G. & Research Department of Commerce.
Motto: To provide free services to the needy people in education and career development rather than other platforms and tools which run for profit motive.
"

PROJECT_INFO =" 
SYSTEM NAME: EduBridge
DESCRIPTION: Bridging the gap between students, freshers, and advanced intelligence to support and guide them to success in their career and future.
COST POLICY: We dont implement any cost for usage. The agent is completely free to use.
KEY FEATURES: Bilingual support (English/Tamil), Multi-agent system, Personalized guidance.
"

# --- 3. AGENT DEFINITIONS ---

class Agent:
    def __init__(self, name, role, color, description, prompt, image_url):
        self.name = name
        self.role = role
        self.color = color 
        self.description = description
        self.base_prompt = prompt
        self.image_url = image_url

agents_db = {
    "Auto (Smart Routing)": None,
    "ResumeAgent": Agent(
        "ResumeAgent", "CV Specialist", "#ef4444", "Reviews resumes & formatting.",
        "You are ResumeAgent. Analyze resumes critically. Focus on impact, ATS keywords, and clarity.",
        "https://cdn-icons-png.flaticon.com/512/942/942748.png"
    ),
    "CuratorAgent": Agent(
        "CuratorAgent", "Resource Hunter", "#10b981", "Finds courses & books.",
        "You are CuratorAgent. Suggest specific, high-quality resources (courses, books, URLs).",
        "https://cdn-icons-png.flaticon.com/512/3079/3079165.png"
    ),
    "ResearchAgent": Agent(
        "ResearchAgent", "Career Analyst", "#06b6d4", "Market trends & salaries.",
        "You are ResearchAgent. Provide data-driven insights into job markets and salary expectations.",
        "https://cdn-icons-png.flaticon.com/512/1651/1651586.png"
    ),
    "AboutAgent": Agent(
        "AboutAgent", "System Guide", "#8b5cf6", "Explains EduBridge features.",
        f"You are AboutAgent. Introduce EduBridge. Developer info: {DEV_INFO}. System info: {PROJECT_INFO}.",
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
    ),
    "StudyAgent": Agent(
        "StudyAgent", "Academic Coach", "#f59e0b", "Study plans & exam tips.",
        "You are StudyAgent. Help create study timetables and explain concepts simply.",
        "https://cdn-icons-png.flaticon.com/512/3406/3406987.png"
    ),
    "CareerAgent": Agent(
        "CareerAgent", "Pathway Advisor", "#ec4899", "Job roles & transitions.",
        "You are CareerAgent. Discuss specific job roles and how to transition between careers.",
        "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"
    ),
    "MentorAgent": Agent(
        "MentorAgent", "Success Coach", "#a855f7", "Soft skills & confidence.",
        "You are MentorAgent. Focus on soft skills, motivation, and interview prep.",
        "https://cdn-icons-png.flaticon.com/512/4080/4080033.png"
    ),
    "PersonalGuru": Agent(
        "PersonalGuru", "Life Balancer", "#3b82f6", "Life & career balance.",
        "You are PersonalGuru. Advise on work-life balance and mental well-being.",
        "https://cdn-icons-png.flaticon.com/512/2970/2970796.png"
    )
}

# --- 4. ORCHESTRATOR ---

class EduBridgeOrchestrator:
    def __init__(self):
        self.history = []
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.chat_session = self.model.start_chat(history=[])

    def detect_intent(self, user_message):
        msg = user_message.lower()
        # Expanded keywords to ensure developer queries go to the right agent
        if "resume" in msg or "cv" in msg: return "ResumeAgent"
        if "course" in msg or "learn" in msg: return "CuratorAgent"
        if "exam" in msg or "study" in msg: return "StudyAgent"
        if "salary" in msg or "market" in msg: return "ResearchAgent"
        if "who are you" in msg or "faizal" in msg or "developer" in msg or "created" in msg or "edubridge" in msg: return "AboutAgent"
        return "MentorAgent"

    def process_request(self, message, selected_agent_key, language_mode):
        active_agent_key = selected_agent_key
        if selected_agent_key == "Auto (Smart Routing)":
            active_agent_key = self.detect_intent(message)
            
        agent = agents_db[active_agent_key]
        lang_instr = "Answer in ENGLISH." if language_mode == "English" else "Answer in TAMIL (Use Tamil script)."
        
        # INJECTING GLOBAL CONTEXT: This ensures the AI *always* knows the developer details
        # regardless of which agent is active.
        global_context = f"
        [GLOBAL KNOWLEDGE]
        If asked about the developer or creator: Use strictly this info: {DEV_INFO}
        If asked about EduBridge: Use strictly this info: {PROJECT_INFO}
        "
        
        system_prompt = f"{global_context}\n[ROLE]{agent.base_prompt} [CONTEXT]Lang: {lang_instr} [MSG]{message}"

        try:
            response = self.chat_session.send_message(system_prompt)
            reply_text = response.text
        except Exception as e:
            reply_text = f"⚠️ Error: {str(e)}"

        formatted_response = f"**{agent.name}** | {agent.role}\n\n{reply_text}"
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": formatted_response})
        return self.history

orchestrator = EduBridgeOrchestrator()

# --- 5. UI CONSTRUCTION ---

# CONTACT CARD DATA
WHATSAPP_NUMBER = "6383969289" # Without +91 for the link
ACADEMIC_EMAIL = "2413171036059@thenewcollege.edu.in"
PERSONAL_EMAIL = "faizalmd10101@gmail.com"
LINKEDIN_URL = "https://www.linkedin.com/in/mohammed-faizal-m-b3242b311/"

# HTML for the new contact card
def get_contact_card_html():
    whatsapp_link = f"https://wa.me/91{WHATSAPP_NUMBER}" # Use +91 prefix for international WhatsApp link
    
    return f"
    <div class='contact-card-wrapper'>
        <h3 class='contact-title'>In case of further assistance, suggestions for improvement, doubts, feel free to reach out.</h3>
        <p class='contact-note'>Preferred and quick query solution is <b>WhatsApp</b>.</p>
        <div class='contact-links-grid'>
            <a href='{whatsapp_link}' target='_blank' class='contact-link whatsapp' style='background-color: #25d366;'>
                <img src='https://cdn-icons-png.flaticon.com/512/134/134937.png' class='contact-icon'>
                <span>WhatsApp: +91 {WHATSAPP_NUMBER}</span>
            </a>
            <a href='mailto:{PERSONAL_EMAIL}' class='contact-link email-personal' style='background-color: #d946ef;'>
                <img src='https://cdn-icons-png.flaticon.com/512/628/628751.png' class='contact-icon'>
                <span>Personal Email</span>
            </a>
            <a href='mailto:{ACADEMIC_EMAIL}' class='contact-link email-academic' style='background-color: #0ea5e9;'>
                <img src='https://cdn-icons-png.flaticon.com/512/3670/3670044.png' class='contact-icon'>
                <span>Academic Email</span>
            </a>
            <a href='{LINKEDIN_URL}' target='_blank' class='contact-link linkedin' style='background-color: #0077b5;'>
                <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' class='contact-icon'>
                <span>LinkedIn Profile</span>
            </a>
        </div>
    </div>
    "

# CSS includes:
# 1. FIXED Typewriter Animation (Loops correctly: Type -> Wait -> Delete -> Retype)
# 2. Header Layout (Flexbox)
# 3. Chat Visibility fixes (forcing white text on dark background)
# 4. NEW Contact Card Styles
# 5. NEW Footer Text Contrast Fix
custom_css = 
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');

body { background: #0f172a; font-family: 'Inter', sans-serif; }
.gradio-container { background: #0f172a; color: white !important; max-width: 1200px !important; margin: auto; }

/* --- HEADER STYLES --- */
.header-wrapper {
    display: flex;
    align-items: center;
    gap: 25px;
    padding: 20px;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
}
.logo-box img {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid #6366f1;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
}
.title-box { flex-grow: 1; }

h1.header-title {
    background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8em !important; 
    font-weight: 900; 
    margin: 0;
    line-height: 1.2;
}

/* --- FIXED TYPEWRITER ANIMATION --- */
.typewriter-container {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1em;
    color: #cbd5e1; /* Light gray text for visibility */
    margin-top: 10px;
    display: inline-block;
    min-height: 1.5em; /* Prevent layout shift */
}

.typewriter-text {
    overflow: hidden; 
    border-right: 3px solid #ec4899; /* The Cursor */
    white-space: nowrap; 
    margin: 0;
    color: #e2e8f0; /* Ensure text is visible against dark bg */
    width: 0; /* Start at 0 */
    /* Animation: Name | Duration | Timing | Infinite Loop */
    animation: type-wait-delete 12s steps(60, end) infinite, blink-caret .75s step-end infinite;
}

/* Animation Stages (Total 12s):
   0-30%  (3.6s): Typing
   30-70% (4.8s): Waiting (Standby ~5s)
   70-85% (1.8s): Deleting
   85-100% (1.8s): Pause before restart
*/
@keyframes type-wait-delete {
    0% { width: 0; }
    30% { width: 100%; } 
    70% { width: 100%; } /* Holds text for standby */
    85% { width: 0; }    /* Deletes text */
    100% { width: 0; }   /* Pause */
}

@keyframes blink-caret { 
    from, to { border-color: transparent } 
    50% { border-color: #ec4899; } 
}

/* --- HORIZONTAL AGENT SCROLL --- */
.agent-row-container {
    display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 15px; padding: 10px 5px;
    scrollbar-width: thin; scrollbar-color: #475569 #0f172a;
}
.agent-card-mini {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255,255,255,0.1);
    min-width: 200px; max-width: 200px;
    padding: 15px; border-radius: 12px;
    display: flex; flex-direction: column; align-items: center; text-align: center;
    transition: transform 0.2s;
}
.agent-card-mini:hover { transform: translateY(-5px); background: rgba(30, 41, 59, 1); border-color: #6366f1; }
.agent-img-mini { width: 45px; height: 45px; margin-bottom: 10px; border-radius: 8px; }
.agent-name-mini { font-weight: 700; font-size: 0.95em; color: white; margin-bottom: 4px; }
.agent-role-mini { font-size: 0.75em; color: #cbd5e1; }

/* --- CONTACT CARD STYLES --- */
.contact-card-wrapper {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 12px;
    margin-top: 30px;
    margin-bottom: 20px;
    text-align: center;
}
.contact-title {
    color: #a855f7;
    font-size: 1.2em;
    font-weight: 700;
    margin-top: 0;
}
.contact-note {
    color: #94a3b8;
    font-size: 0.9em;
    margin-bottom: 20px;
}
.contact-links-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}
.contact-link {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px 15px;
    border-radius: 8px;
    color: white !important;
    text-decoration: none;
    font-weight: 600;
    transition: opacity 0.2s, transform 0.2s;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.contact-link:hover {
    opacity: 0.9;
    transform: translateY(-2px);
}
.contact-icon {
    width: 20px;
    height: 20px;
    margin-right: 10px;
    filter: invert(1); /* Makes icons white for better contrast */
}

/* --- CHAT VISIBILITY FIXES (ChatGPT Style) --- */
.message-wrap { max-width: 85% !important; }

/* User Message: Blue Background, White Text */
.user-row .message { 
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; 
    color: #ffffff !important; 
    border: none !important;
}

/* Bot Message: Dark Gray Background, Light Text */
.bot-row .message { 
    background: #334155 !important; 
    color: #f8fafc !important; 
    border: 1px solid #475569 !important;
}

/* Force markdown content inside bot bubbles to be visible */
.bot-row .message p, .bot-row .message strong, .bot-row .message li {
    color: #f8fafc !important;
}
.bot-row .message code {
    background-color: #1e293b !important;
    color: #fbbf24 !important;
}

/* --- FOOTER STYLES --- */
.footer-text { 
    text-align: center; 
    color: #64748b; 
    font-size: 0.8em; 
    margin-top: 20px; 
    padding-bottom: 20px; 
    border-top: 1px solid #1e293b; 
    padding-top: 20px; 
}
/* NEW RULE: Improve contrast for developer name in the footer */
.footer-text b {
    color: #6366f1; /* Using a vibrant color like primary indigo for contrast */
}
"

def get_agent_row_html():
    html = "<div class='agent-row-container'>"
    for name, agent in agents_db.items():
        if agent:
            html += f"
            <div class='agent-card-mini'>
                <img src='{agent.image_url}' class='agent-img-mini' referrerpolicy='no-referrer'>
                <div class='agent-name-mini' style='color:{agent.color}'>{agent.name}</div>
                <div class='agent-role-mini'>{agent.description}</div>
            </div>
            "
    html += "</div>"
    return html

# Function to build the header HTML
def get_header_html():
    # Note: Text updated to match your description
    desc_text = "Bridging gap between students, freshers and advanced intelligence."
    
    return f"
    <div class='header-wrapper'>
        <div class='logo-box'>
            <img src='{LOGO_URL}' alt='EduBridge Logo'>
        </div>
        <div class='title-box'>
            <h1 class='header-title'>EduBridge AI</h1>
            <div class='typewriter-container'>
                <div class='typewriter-text'>{desc_text}</div>
            </div>
        </div>
    </div>
    "

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=custom_css, title="EduBridge AI") as demo:
    
    # 1. HEADER (Logo + Typewriter)
    gr.HTML(get_header_html())

    # 2. HORIZONTAL AGENT CARDS
    gr.HTML(get_agent_row_html())

    # 3. CHAT INTERFACE
    chatbot = gr.Chatbot(
        label="Live Session", 
        height=500, 
        type="messages", 
        render_markdown=True,
        show_copy_button=True
    )
    
    # 4. CONTROLS
    with gr.Group():
        with gr.Row():
            agent_selector = gr.Dropdown(choices=list(agents_db.keys()), value="Auto (Smart Routing)", label="Choose Agent", scale=1)
            language_toggle = gr.Radio(choices=["English", "Tamil"], value="English", label="Language", scale=1)
        
        with gr.Row():
            msg_input = gr.Textbox(placeholder="Ask about resumes, courses, exams, or career...", show_label=False, scale=4)
            send_btn = gr.Button("🚀 Send", variant="primary", scale=1)
            clear_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)
    
    # 5. CONTACT CARD (NEW FEATURE)
    gr.HTML(get_contact_card_html())

    # 6. FOOTER
    # The name "Mohammed Faizal. M" is bolded, and the CSS above now ensures this bolded text has a high-contrast color.
    gr.HTML(f"
        <div class='footer-text'>
            Made with ❤️ by <b>Mohammed Faizal. M</b> - U.G Student, B.Com, The New College, Chennai, India<br>
            &copy; EduBridge 2025 - All rights reserved.
        </div>
    ")

    # LOGIC
    def respond(msg, hist, agent, lang):
        if not msg.strip(): return hist, ""
        return orchestrator.process_request(msg, agent, lang), ""
    
    def reset():
        orchestrator.history = []
        orchestrator.chat_session = orchestrator.model.start_chat(history=[])
        return []

    msg_input.submit(respond, [msg_input, chatbot, agent_selector, language_toggle], [chatbot, msg_input])
    send_btn.click(respond, [msg_input, chatbot, agent_selector, language_toggle], [chatbot, msg_input])
    clear_btn.click(reset, [], [chatbot])

demo.launch(share=True, debug=False)
"
    '''

    # 1. Prepare the JSON payload expected by the Orchestrator
    input_payload = {
        "rubric_text": demo_rubric,
        "project_writeup": demo_project_writeup,
        "code_text": demo_code_text
    }
    
    # 2. Serialize to string (Agents communicate via text strings)
    json_query_string = json.dumps(input_payload, ensure_ascii=False)

    # 3. Run the session using the helper function
    # Note: We use 'await' directly as Kaggle notebooks support top-level await.
    await run_session(
        runner_instance=orchestrator_runner,
        user_queries=[json_query_string],
        session_name="rubriq_demo_session"
    )
