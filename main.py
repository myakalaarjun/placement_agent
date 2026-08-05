import os
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from pypdf import PdfReader
from ddgs import DDGS

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Placement Agent",
    description="Agentic AI career assistant for students",
    version="1.0"
)


# ============================================================
# GEMINI API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is not set.")


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0.2,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    html_file = os.path.join(
        base_dir,
        "templates",
        "index.html"
    )

    if not os.path.exists(html_file):
        return {
            "error": "index.html not found",
            "expected_location": html_file
        }

    return FileResponse(
        html_file,
        media_type="text/html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Student Placement Agent is running"
    }


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(pdf_path: str) -> str:

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text() or ""

        text += page_text + "\n"

    text = text.strip()

    if not text:

        return "No readable text found in the resume PDF."

    # Limit text sent to Gemini
    return text[:16000]


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    resume_path: str,
    role: str,
    github_id: str
):

    # ========================================================
    # TOOL 1 — READ RESUME
    # ========================================================

    @tool
    def read_resume() -> str:
        """Read and extract text from the student's resume PDF."""

        try:

            return extract_resume_text(
                resume_path
            )

        except Exception as e:

            return f"Resume reading failed: {e}"


    # ========================================================
    # TOOL 2 — JOB SEARCH
    # ========================================================

    @tool
    def job_search(role: str) -> str:
        """Search for jobs related to the target role."""

        query = (
            f"{role} jobs India "
            f"fresher entry level"
        )

        try:

            results = DDGS().text(
                query,
                max_results=5
            )

            output = []

            for item in results:

                output.append({

                    "title": item.get(
                        "title",
                        ""
                    ),

                    "url": item.get(
                        "href",
                        ""
                    ),

                    "snippet": item.get(
                        "body",
                        ""
                    )

                })

            return json.dumps(
                output,
                indent=2,
                ensure_ascii=False
            )

        except Exception as e:

            return f"Job search failed: {e}"


    # ========================================================
    # TOOL 3 — SKILL GAP
    # ========================================================

    @tool
    def skill_gap(role: str) -> str:
        """Compare resume skills with target job role."""

        try:

            resume_text = extract_resume_text(
                resume_path
            )

            prompt = f"""
You are a career skill gap analyzer.

Target role:
{role}

Student resume:
{resume_text}

Compare the student's current skills
with the target role.

Give:

1. Matching skills
2. Missing skills
3. Technologies to learn
4. Priority of each missing skill
5. Simple learning plan

Keep it suitable for a college student.

Do not invent information
that is not supported by the resume.
"""

            response = llm.invoke(prompt)

            return response.text

        except Exception as e:

            return (
                f"Skill gap analysis failed: {e}"
            )


    # ========================================================
    # TOOL 4 — PROJECT IDEAS
    # ========================================================

    @tool
    def project_ideas(role: str) -> str:
        """Search for student project ideas related to the target role."""

        query = (
            f"{role} student "
            f"project ideas GitHub"
        )

        try:

            results = DDGS().text(
                query,
                max_results=5
            )

            output = []

            for item in results:

                output.append({

                    "title": item.get(
                        "title",
                        ""
                    ),

                    "url": item.get(
                        "href",
                        ""
                    ),

                    "snippet": item.get(
                        "body",
                        ""
                    )

                })

            return json.dumps(
                output,
                indent=2,
                ensure_ascii=False
            )

        except Exception as e:

            return (
                f"Project search failed: {e}"
            )


    # ========================================================
    # TOOL 5 — GITHUB CHECK
    # ========================================================

    @tool
    def github_check(github_id: str) -> str:
        """Check public GitHub repository activity."""

        username = (
            github_id
            .strip()
            .rstrip("/")
            .split("/")[-1]
        )

        url = (
            f"https://api.github.com/"
            f"users/{username}/repos"
        )

        try:

            import requests

            response = requests.get(
                url,
                params={
                    "sort": "pushed",
                    "direction": "desc",
                    "per_page": 10
                },
                timeout=20
            )

            if response.status_code == 404:

                return (
                    f"GitHub user "
                    f"'{username}' not found."
                )

            response.raise_for_status()

            repos = response.json()

            result = {

                "username": username,

                "repositories": []

            }

            for repo in repos:

                result[
                    "repositories"
                ].append({

                    "name": repo.get(
                        "name"
                    ),

                    "description": repo.get(
                        "description"
                    ),

                    "language": repo.get(
                        "language"
                    ),

                    "stars": repo.get(
                        "stargazers_count"
                    ),

                    "url": repo.get(
                        "html_url"
                    ),

                    "last_pushed": repo.get(
                        "pushed_at"
                    )

                })

            return json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )

        except Exception as e:

            return (
                f"GitHub check failed: {e}"
            )


    # ========================================================
    # TOOLS LIST
    # ========================================================

    tools = [

        read_resume,

        job_search,

        skill_gap,

        project_ideas,

        github_check

    ]

    tool_map = {

        t.name: t

        for t in tools

    }


    # ========================================================
    # AGENT CORE
    # ========================================================

    agent_llm = llm.bind_tools(
        tools
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    SYSTEM_PROMPT = """

You are a Student Career Agent.

Your job is to analyze a student's
resume and target career role.

Available tools:

1. read_resume
2. job_search
3. skill_gap
4. project_ideas
5. github_check

Process:

1. Read the resume.
2. Analyze skill gaps.
3. Search relevant jobs.
4. Find useful project ideas.
5. Check GitHub activity.
6. Combine all results.
7. Give a final career recommendation.

Use the tools when useful.

Do not invent information.

The final response should contain:

- Resume summary
- Target role
- Matching skills
- Skill gaps
- Jobs
- Project ideas
- GitHub activity
- Recommended next steps

Keep the explanation simple
and useful for a college student.

"""


    # ========================================================
    # INITIAL MESSAGE
    # ========================================================

    messages = [

        SystemMessage(
            content=SYSTEM_PROMPT
        ),

        HumanMessage(
            content=f"""

Analyze this student's career profile.

Target role:
{role}

GitHub:
{github_id}

The resume is available
through the read_resume tool.

Use the available tools and provide
a complete career analysis.

"""
        )

    ]


    # ========================================================
    # AGENT LOOP
    # ========================================================

    trace = []

    final_answer = ""

    for step in range(10):

        try:

            response = agent_llm.invoke(
                messages
            )

        except Exception as e:

            return {

                "error": "Agent model failed",

                "details": str(e)

            }


        messages.append(
            response
        )


        # ====================================================
        # AGENT FINISHED
        # ====================================================

        if not response.tool_calls:

            final_answer = response.text

            break


        # ====================================================
        # EXECUTE TOOLS
        # ====================================================

        for call in response.tool_calls:

            tool_name = call["name"]

            tool_args = call.get(
                "args",
                {}
            )

            trace.append({

                "step": step + 1,

                "tool": tool_name,

                "arguments": tool_args

            })


            # ------------------------------------------------
            # Check tool exists
            # ------------------------------------------------

            if tool_name not in tool_map:

                tool_result = (
                    f"Unknown tool requested: "
                    f"{tool_name}"
                )

            else:

                try:

                    selected_tool = (
                        tool_map[tool_name]
                    )

                    tool_result = (
                        selected_tool.invoke(
                            tool_args
                        )
                    )

                except Exception as e:

                    tool_result = (
                        f"Tool execution failed: "
                        f"{e}"
                    )


            # ------------------------------------------------
            # Add result to agent conversation
            # ------------------------------------------------

            messages.append(

                ToolMessage(

                    content=str(
                        tool_result
                    ),

                    tool_call_id=call["id"]

                )

            )


    # ========================================================
    # FALLBACK
    # ========================================================

    if not final_answer:

        final_answer = (
            "The agent could not "
            "complete the analysis."
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "student_input": {

            "resume": "uploaded PDF",

            "role": role,

            "github": github_id

        },

        "model": "gemma-4-31b-it",

        "tool_trace": trace,

        "final_synthesis": final_answer

    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_student(

    resume: UploadFile = File(...),

    role: str = Form(...),

    github_id: str = Form(...)

):

    # ========================================================
    # VALIDATE PDF
    # ========================================================

    if not resume.filename:

        raise HTTPException(

            status_code=400,

            detail="Resume file is required."

        )


    if not resume.filename.lower().endswith(".pdf"):

        raise HTTPException(

            status_code=400,

            detail="Please upload a PDF resume."

        )


    # ========================================================
    # VALIDATE ROLE
    # ========================================================

    if not role.strip():

        raise HTTPException(

            status_code=400,

            detail="Job role is required."

        )


    # ========================================================
    # VALIDATE GITHUB
    # ========================================================

    if not github_id.strip():

        raise HTTPException(

            status_code=400,

            detail="GitHub username or URL is required."

        )


    temp_path = None


    try:

        # ====================================================
        # READ UPLOADED PDF
        # ====================================================

        file_content = await resume.read()


        if not file_content:

            raise HTTPException(

                status_code=400,

                detail="Uploaded PDF is empty."

            )


        # ====================================================
        # CREATE TEMPORARY PDF
        # ====================================================

        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        ) as temp_file:

            temp_file.write(
                file_content
            )

            temp_path = (
                temp_file.name
            )


        # ====================================================
        # RUN AGENT
        # ====================================================

        result = run_agent(

            resume_path=temp_path,

            role=role.strip(),

            github_id=github_id.strip()

        )


        return result


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


    finally:

        # ====================================================
        # DELETE TEMPORARY PDF
        # ====================================================

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass
