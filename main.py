from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from dotenv import load_dotenv
from services.ai_service import AIService
import os
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

cloudflare_token = os.getenv("CLOUDFLARE_API_TOKEN")
cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
cloudflare_model = os.getenv("CLOUDFLARE_MODEL_NAME")
ai_service = AIService(
    gemini_key,
    mistral_key,
    cloudflare_token,
    cloudflare_account_id,
    cloudflare_model
)

print("🔑 GEMINI_API_KEY:", "موجودة ✅" if gemini_key else "غير موجودة ❌")
print("🔑 MISTRAL_API_KEY:", "موجودة ✅" if mistral_key else "غير موجودة ❌")
print("🔑 CLOUDFLARE_API_TOKEN:", "موجودة ✅" if cloudflare_token else "غير موجودة ❌")
print("🔑 CLOUDFLARE_ACCOUNT_ID:", "موجودة ✅" if cloudflare_account_id else "غير موجودة ❌")
print("🤖 CLOUDFLARE_MODEL:", cloudflare_model or "غير موجود ❌")

ai = AIService(
    gemini_key,
    mistral_key,
    cloudflare_token,
    cloudflare_account_id,
    cloudflare_model
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
        response = ai.gemini.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embedding[0].values
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
        # EMBEDDINGS DISABLED
        # =========================

        pdf_embeddings = []

        print("ℹ️ Embeddings disabled.")
        print("🧩 PDF chunks available:", len(pdf_chunks))

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
# TRANSLATE
# =========================

@app.route("/translate", methods=["POST"])
def translate():

    data = request.get_json()

    text = (data.get("text") or "").strip()
    language = data.get("language")

    languages = {
        "en": "English",
        "fr": "French",
        "ar": "Arabic"
    }

    if not text:
        return jsonify({
            "error": "No text provided."
        }), 400

    if language not in languages:
        return jsonify({
            "error": "Unsupported language."
        }), 400

    language_name = languages[language]

    prompt = f"""
Translate the following text into {language_name}.

Instructions:
- Preserve the exact meaning.
- Preserve important facts and technical terms.
- Keep the same structure when possible.
- Do not add explanations.
- Return ONLY the translation.

TEXT:
{text}
"""

    try:

        translation = ai.generate(prompt)

        return jsonify({
            "translation": translation
        })

    except Exception as e:

        print("Translation error:", e)

        return jsonify({
            "error": "Translation failed."
        }), 500


# =========================
# CHATBOT
# =========================

@app.route("/chat", methods=["POST"])
def chat():

    global pdf_text
    global pdf_chunks
    global pdf_embeddings

    print("================================")
    print("💬 CHATBOT DEBUG")
    print("📄 PDF TEXT LENGTH:", len(pdf_text))
    print("🧩 PDF CHUNKS:", len(pdf_chunks))
    print("🧠 PDF EMBEDDINGS:", len(pdf_embeddings))
    print("================================")

    if not pdf_text.strip():
        return jsonify({
            "error": "Please upload a PDF first."
        }), 400

    # =========================
    # GET QUESTION
    # =========================

    data = request.get_json(silent=True) or {}

    question = (
        data.get("question")
        or data.get("message")
        or ""
    ).strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    # =========================
    # CREATE CONTEXT
    # =========================

    context = pdf_text[:50000]

    if not context.strip():
     return jsonify({
        "error": "Please upload a PDF first."
    }), 400

    # -------------------------
    # TRY RAG / EMBEDDINGS
    # -------------------------

    if pdf_chunks and pdf_embeddings:

        question_embedding = create_embedding(question)

        if question_embedding is not None:

            similarities = []

            for i, embedding in enumerate(pdf_embeddings):

                similarity = cosine_similarity(
                    question_embedding,
                    embedding
                )

                similarities.append(
                    (similarity, i)
                )

            similarities.sort(
                reverse=True
            )

            # Take the 3 most relevant chunks
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
    # FALLBACK
    # =========================

    # If embeddings are not available,
    # use the extracted PDF text directly.

    if not context.strip():

        context = pdf_text[:50000]

    # =========================
    # AI PROMPT
    # =========================

    prompt = f"""
You are an AI study assistant.

You are answering questions about a PDF
uploaded by the student.

IMPORTANT RULES:

1. Answer using ONLY the information
   contained in the PDF context.

2. Do NOT invent information.

3. If the answer is not present in the PDF,
   say clearly:

   "This information is not available
   in the provided PDF."

4. Give a clear and helpful explanation.

5. If the student asks for a definition,
   explain it simply.

6. If the student asks for steps,
   give them in numbered points.

7. If the student asks for a comparison,
   organize the answer clearly.

8. Answer in the same language used by
   the student whenever possible.

PDF CONTEXT:

{context}

STUDENT QUESTION:

{question}

ANSWER:
"""

    # =========================
    # GENERATE ANSWER
    # =========================

    try:

        answer = ai.generate(prompt)

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("❌ Chat error:", e)

        return jsonify({
            "error": "AI could not answer the question."
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
- Include a short topic describing the concept tested.
- The topic must be based ONLY on the PDF content.
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
            "explanation": "Explanation",
            "topic": "Main topic of this question"

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

        result = ai.generate(prompt)

        print("🤖 Raw Quiz Response:")
        print(repr(result))

        if not result:
            raise ValueError(
                "AI returned an empty response."
            )

        result = str(result).strip()

        if not result:
            raise ValueError(
                "AI returned an empty response after cleaning."
    )

        


# =========================
        # CLEAN JSON
        # =========================

        result = str(result).strip()

        print("🧹 Before JSON cleaning:")
        print(repr(result))

        # Remove markdown code fences
        result = re.sub(
            r"^```(?:json)?\s*",
            "",
            result,
            flags=re.IGNORECASE
        )

        result = re.sub(
            r"\s*```$",
            "",
            result
        )

        result = result.strip()

        # If AI added text before/after JSON,
        # extract the JSON object
        start = result.find("{")
        end = result.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "AI response does not contain a valid JSON object."
            )

        result = result[start:end + 1]

        print("🧹 Cleaned Quiz JSON:")
        print(result)


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
        if len(questions) != question_count:

           raise ValueError(
        f"AI generated {len(questions)} questions instead of {question_count}."
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
            # Topic

            if not question.get("topic"):

                  raise ValueError(
                       f"Question {i + 1} has no topic."
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
# SOLVE EXERCISE / IMAGE
# =========================

@app.route("/solve-image", methods=["POST"])
def solve_image():
    print("================================")
    print("📸 IMAGE SOLVER DEBUG")
    print("================================")

    # 1. استقبال الـ Prompt أو السؤال النصي المرفق مع الصورة
    prompt = request.form.get("prompt", "").strip()
    
    # 2. التحقق من وجود الملف (الصورة)
    if "exercise_image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_file = request.files["exercise_image"]

    if image_file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    try:
        # 3. حفظ الصورة مؤقتاً في السيرفر للمعالجة
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(image_path)

        # 4. استدعاء AIService لتحليل الصورة عبر Gemini-1.5-flash
        print("🤖 Sending image to Gemini...")
        answer = ai.generate_with_image(prompt, image_path)

        # 5. تنظيف ومسح الصورة من السيرفر بعد الانتهاء
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        print("❌ Image solving error:", e)
        return jsonify({
            "error": "AI could not process the image."
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