from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from dotenv import load_dotenv
from services.ai_service import AIService

import os
import ollama
import math
import json
import re


# =========================
# LOAD ENVIRONMENT
# =========================

load_dotenv()


# =========================
# FLASK APP
# =========================

app = Flask(__name__)


# =========================
# AI SERVICE
# =========================

gemini_key = os.getenv("GEMINI_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

print("🔑 GEMINI_API_KEY:", "موجودة ✅" if gemini_key else "غير موجودة ❌")
print("🔑 MISTRAL_API_KEY:", "موجودة ✅" if mistral_key else "غير موجودة ❌")

ai = AIService(
    gemini_key,
    mistral_key
)

# =========================
# GLOBAL PDF DATA
# =========================

pdf_text = ""

# Original sections used by Quiz
pdf_sections = []

# Small chunks used by RAG
pdf_chunks = []

# Embeddings for RAG
pdf_embeddings = []


# =========================
# EMBEDDING
# =========================

def create_embedding(text):

    try:

        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )

        return response["embedding"]

    except Exception as e:

        print("❌ Embedding error:", e)

        return None


# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(a, b):

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norm_a == 0 or norm_b == 0:

        return 0

    return dot_product / (
        norm_a * norm_b
    )


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================
# UPLOAD PDF
# =========================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    global pdf_text
    global pdf_sections
    global pdf_chunks
    global pdf_embeddings


    # =========================
    # CHECK FILE
    # =========================

    if "file" not in request.files:

        return jsonify({
            "error": "No PDF file uploaded."
        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({
            "error": "No file selected."
        }), 400


    if not file.filename.lower().endswith(".pdf"):

        return jsonify({
            "error": "Please upload a PDF file."
        }), 400


    # =========================
    # RESET OLD DATA
    # =========================

    pdf_text = ""
    pdf_sections = []
    pdf_chunks = []
    pdf_embeddings = []


    try:

        # =========================
        # READ PDF
        # =========================

        reader = PdfReader(file)

        total_pages = len(
            reader.pages
        )

        print(
            "📄 Total pages:",
            total_pages
        )


        # =========================
        # EXTRACT FULL TEXT
        # =========================

        all_text = []

        for page in reader.pages:

            page_text = (
                page.extract_text()
                or ""
            )

            all_text.append(
                page_text
            )


        pdf_text = "\n".join(
            all_text
        )


        # =========================
        # CHUNK SIZE
        # =========================

        chunk_size = request.form.get(
            "chunk_size",
            10
        )

        try:

            chunk_size = int(
                chunk_size
            )

        except:

            chunk_size = 10


        if chunk_size not in [5, 10, 20]:

            chunk_size = 10


        # =========================
        # CREATE SECTIONS
        # =========================

        pdf_sections = []
        pdf_chunks = []


        for i in range(
            0,
            total_pages,
            chunk_size
        ):

            section_text = ""


            for j in range(
                i,
                min(
                    i + chunk_size,
                    total_pages
                )
            ):

                page_text = (
                    reader.pages[j]
                    .extract_text()
                    or ""
                )

                section_text += (
                    page_text + "\n"
                )


            if section_text.strip():

                # Keep original section
                # for Quiz

                pdf_sections.append(
                    section_text
                )


                # =========================
                # SMALL RAG CHUNKS
                # =========================

                words = (
                    section_text.split()
                )

                words_per_embedding = 500


                for k in range(
                    0,
                    len(words),
                    words_per_embedding
                ):

                    small_chunk = " ".join(
                        words[
                            k:k + words_per_embedding
                        ]
                    )


                    if small_chunk.strip():

                        pdf_chunks.append(
                            small_chunk
                        )


        print(
            "📚 Total sections:",
            len(pdf_sections)
        )

        print(
            "🧩 Total embedding chunks:",
            len(pdf_chunks)
        )


        # =========================
        # CREATE EMBEDDINGS
        # =========================

        pdf_embeddings = []

        valid_chunks = []


        for i, chunk in enumerate(
            pdf_chunks
        ):

            print(
                f"🧠 Creating embedding "
                f"{i + 1}/{len(pdf_chunks)}..."
            )


            embedding = create_embedding(
                chunk
            )


            if embedding is not None:

                valid_chunks.append(
                    chunk
                )

                pdf_embeddings.append(
                    embedding
                )


        pdf_chunks = valid_chunks


        print(
            "🧠 Embeddings created:",
            len(pdf_embeddings)
        )


        # =========================
        # CREATE SUMMARY
        # =========================

        summary_text = pdf_text


        # Prevent extremely large
        # summary prompts

        MAX_SUMMARY_CHARS = 30000


        if len(summary_text) > MAX_SUMMARY_CHARS:

            summary_text = (
                summary_text[
                    :MAX_SUMMARY_CHARS
                ]
            )


        summary_prompt = f"""
You are an AI study assistant.

Analyze the following PDF content.

Create a useful study summary.

Return TWO sections:

ENGLISH SUMMARY:
- Main topic
- Important concepts
- Important facts
- Key points
- Useful conclusions

ARABIC SUMMARY:
- الموضوع الرئيسي
- المفاهيم المهمة
- الحقائق المهمة
- أهم النقاط
- الخلاصة

Keep the summary clear and useful
for a student.

PDF CONTENT:

{summary_text}
"""


        try:

            summary = ai.generate(
                summary_prompt
            )


        except Exception as e:

            print(
                "❌ Summary error:",
                e
            )

            summary = (
                "Could not generate summary."
            )


        # =========================
        # RETURN SECTIONS
        # =========================

        sections = []


        for i in range(
            math.ceil(
                total_pages /
                chunk_size
            )
        ):

            sections.append({

                "index": i,

                "start_page":
                    i * chunk_size + 1,

                "end_page":
                    min(
                        (i + 1) * chunk_size,
                        total_pages
                    )

            })


        return jsonify({

            "pages":
                total_pages,

            "analyzed_pages":
                total_pages,

            "chunk_size":
                chunk_size,

            "chunks":
                len(pdf_chunks),

            "summary":
                summary,

            "sections":
                sections

        })


    except Exception as e:

        print(
            "❌ PDF upload error:",
            e
        )


        return jsonify({

            "error":
                "Could not process the PDF."

        }), 500


# =========================
# CHATBOT
# =========================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    global pdf_chunks
    global pdf_embeddings


    # =========================
    # CHECK PDF
    # =========================

    if not pdf_chunks:

        return jsonify({

            "error":
                "Please upload a PDF first."

        }), 400


    # =========================
    # GET QUESTION
    # =========================

    data = request.get_json(
        silent=True
    ) or {}


    question = (
        data.get("question")
        or data.get("message")
        or ""
    ).strip()


    if not question:

        return jsonify({

            "error":
                "Please enter a question."

        }), 400


    # =========================
    # CREATE QUESTION EMBEDDING
    # =========================

    question_embedding = (
        create_embedding(
            question
        )
    )


    if question_embedding is None:

        return jsonify({

            "error":
                "Could not process the question."

        }), 500


    # =========================
    # CALCULATE SIMILARITY
    # =========================

    similarities = []


    for i, embedding in enumerate(
        pdf_embeddings
    ):

        similarity = cosine_similarity(
            question_embedding,
            embedding
        )


        similarities.append(
            (
                similarity,
                i
            )
        )


    # Highest similarity first

    similarities.sort(
        reverse=True
    )


    # =========================
    # TOP 3 CHUNKS
    # =========================

    top_results = similarities[:3]


    context_parts = []


    for similarity, index in top_results:

        context_parts.append(
            pdf_chunks[index]
        )


    context = "\n\n".join(
        context_parts
    )


    # =========================
    # CHAT PROMPT
    # =========================

    prompt = f"""
You are an AI study assistant.

Answer the student's question using
ONLY the provided PDF context.

If the answer is not present in the
context, clearly say that the information
is not available in the provided PDF.

Do not invent information.

Give a clear and helpful answer.

PDF CONTEXT:

{context}

STUDENT QUESTION:

{question}
"""


    # =========================
    # AI GENERATION
    # =========================

    try:

        answer = ai.generate(
            prompt
        )


        return jsonify({

            "answer":
                answer

        })


    except Exception as e:

        print(
            "❌ Chat error:",
            e
        )


        return jsonify({

            "error":
                "AI could not answer the question."

        }), 500


# =========================
# QUIZ
# =========================

@app.route(
    "/quiz",
    methods=["POST"]
)
def quiz():

    global pdf_sections


    # =========================
    # CHECK PDF
    # =========================

    print(
        "📚 PDF sections:",
        len(pdf_sections)
    )


    if not pdf_sections:

        return jsonify({

            "error":
                "Please upload a PDF first."

        }), 400


    # =========================
    # GET REQUEST DATA
    # =========================

    data = request.get_json(
        silent=True
    ) or {}


    sections = data.get(
        "sections",
        []
    )


    question_count = data.get(
        "question_count",
        10
    )


    difficulty = data.get(
        "difficulty",
        "medium"
    )


    general = data.get(
        "general",
        False
    )


    # =========================
    # VALIDATE QUESTION COUNT
    # =========================

    try:

        question_count = int(
            question_count
        )

    except:

        question_count = 10


    if question_count not in [
        5,
        10,
        15,
        20
    ]:

        question_count = 10


    # =========================
    # VALIDATE DIFFICULTY
    # =========================

    if difficulty not in [
        "easy",
        "medium",
        "hard"
    ]:

        difficulty = "medium"


    # =========================
    # GENERAL QUIZ
    # =========================

    if general:

        quiz_text = "\n\n".join(
            pdf_sections
        )


    # =========================
    # SELECTED SECTIONS
    # =========================

    else:

        selected_texts = []


        for index in sections:

            try:

                index = int(
                    index
                )

            except:

                continue


            if (
                0 <= index
                < len(pdf_sections)
            ):

                selected_texts.append(
                    pdf_sections[index]
                )


        if not selected_texts:

            return jsonify({

                "error":
                    "Please select at least one section."

            }), 400


        quiz_text = "\n\n".join(
            selected_texts
        )


    # =========================
    # LIMIT QUIZ CONTEXT
    # =========================

    MAX_QUIZ_CHARS = 50000


    if len(quiz_text) > MAX_QUIZ_CHARS:

        print(
            "⚠️ Quiz content too large."
        )


        selected_sections = []


        for section in pdf_sections:

            selected_sections.append(
                section
            )


            current_text = (
                "\n\n".join(
                    selected_sections
                )
            )


            if len(
                current_text
            ) >= MAX_QUIZ_CHARS:

                break


        quiz_text = "\n\n".join(
            selected_sections
        )


    print(
        "🧠 Quiz content length:",
        len(quiz_text)
    )

    print(
        "📝 Questions:",
        question_count
    )

    print(
        "🎯 Difficulty:",
        difficulty
    )


    # =========================
    # QUIZ PROMPT
    # =========================

    prompt = f"""
You are an AI quiz generator.

Create a multiple-choice quiz from
the provided PDF content.

Difficulty: {difficulty}

Number of questions: {question_count}

Rules:

- Create exactly {question_count} questions.
- Each question must have exactly 4 options.
- Only ONE option is correct.
- Questions must be based ONLY on the provided content.
- Do not invent information.
- Include the correct answer.
- Include a short explanation for the correct answer.
- Return ONLY valid JSON.
- The answer must be the INDEX of the correct option.
- Index starts from 0.

JSON format:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0,
            "explanation": "Explanation"
        }}
    ]
}}

PDF CONTENT:

{quiz_text}
"""


    # =========================
    # GENERATE QUIZ
    # =========================

    try:

        result = ai.generate(
            prompt
        )


        print(
            "🤖 Raw Quiz Response:"
        )

        print(result)


        # =========================
        # CLEAN JSON
        # =========================

        result = result.strip()


        result = re.sub(
            r"^```json\s*",
            "",
            result,
            flags=re.IGNORECASE
        )


        result = re.sub(
            r"^```\s*",
            "",
            result
        )


        result = re.sub(
            r"\s*```$",
            "",
            result
        )


        result = result.strip()


        # =========================
        # PARSE JSON
        # =========================

        quiz_data = json.loads(
            result
        )


        # =========================
        # CHECK QUESTIONS
        # =========================

        if "questions" not in quiz_data:

            raise ValueError(
                "Quiz JSON does not contain questions."
            )


        if not isinstance(
            quiz_data["questions"],
            list
        ):

            raise ValueError(
                "Questions must be a list."
            )


        # =========================
        # VALIDATE QUESTIONS
        # =========================

        questions = (
            quiz_data["questions"]
        )


        if len(questions) == 0:

            raise ValueError(
                "Quiz contains no questions."
            )


        # Keep only requested
        # number of questions

        questions = questions[
            :question_count
        ]


        quiz_data["questions"] = (
            questions
        )


        # =========================
        # VALIDATE EACH QUESTION
        # =========================

        for i, question in enumerate(
            questions
        ):


            # Question object

            if not isinstance(
                question,
                dict
            ):

                raise ValueError(
                    f"Question {i + 1} is invalid."
                )


            # Question text

            if not question.get(
                "question"
            ):

                raise ValueError(
                    f"Question {i + 1} has no text."
                )


            # Options

            options = question.get(
                "options"
            )


            if not isinstance(
                options,
                list
            ):

                raise ValueError(
                    f"Question {i + 1} options are invalid."
                )


            if len(options) != 4:

                raise ValueError(
                    f"Question {i + 1} must have exactly 4 options."
                )


            # Answer

            answer = question.get(
                "answer"
            )


            # Gemini can sometimes return
            # "0" instead of 0

            if isinstance(
                answer,
                str
            ):

                try:

                    answer = int(
                        answer
                    )

                    question["answer"] = (
                        answer
                    )

                except:

                    raise ValueError(
                        f"Question {i + 1} answer is invalid."
                    )


            if not isinstance(
                answer,
                int
            ):

                raise ValueError(
                    f"Question {i + 1} answer is invalid."
                )


            if answer < 0 or answer > 3:

                raise ValueError(
                    f"Question {i + 1} answer must be between 0 and 3."
                )


            # Explanation

            if not question.get(
                "explanation"
            ):

                raise ValueError(
                    f"Question {i + 1} has no explanation."
                )


        # =========================
        # FINAL VALIDATION
        # =========================

        print(
            "✅ Quiz validation passed:",
            len(questions),
            "questions"
        )


        # =========================
        # RETURN QUIZ
        # =========================

        return jsonify({

            "quiz":
                quiz_data

        })


    except json.JSONDecodeError as e:

        print(
            "❌ Quiz JSON error:",
            e
        )


        return jsonify({

            "error":
                "AI generated invalid JSON."

        }), 500


    except Exception as e:

        print(
            "❌ Quiz error:",
            e
        )


        return jsonify({

            "error":
                "AI generated an invalid quiz."

        }), 500


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port
    )