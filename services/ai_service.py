from google import genai
from google.genai import errors
from mistralai.client import Mistral
import ollama


class AIService:

    def __init__(self, gemini_key, mistral_key):

        # =========================
        # Gemini
        # =========================

        self.gemini = genai.Client(
            api_key=gemini_key
        )

        # =========================
        # Mistral
        # =========================

        self.mistral = Mistral(
            api_key=mistral_key
        )

        # =========================
        # Ollama
        # =========================

        self.ollama_model = "qwen3:8b"


    def generate(self, prompt):

        # =========================
        # 1️⃣ Try Gemini
        # =========================

        try:

            response = self.gemini.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            print("✅ Used Gemini")

            return response.text

        except errors.ClientError as e:

            if e.code == 429:

                print("⚠️ Gemini quota exceeded!")
                print("🔄 Switching to Mistral...")

            else:

                print("❌ Gemini error:", e)
                print("🔄 Switching to Mistral...")


        # =========================
        # 2️⃣ Try Mistral
        # =========================

        try:

            response = self.mistral.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            print("✅ Used Mistral")

            return response.choices[0].message.content

        except Exception as e:

            print("❌ Mistral error:", e)
            print("🔄 Switching to Ollama...")


        # =========================
        # 3️⃣ Try Ollama
        # =========================

        try:

            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            print("✅ Used Ollama (Local AI)")

            return response["message"]["content"]

        except Exception as e:

            print("❌ Ollama error:", e)

            return "AI_ERROR"


# =========================
# TEST
# =========================

if __name__ == "__main__":

    from dotenv import load_dotenv
    import os

    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    ai = AIService(
        gemini_key,
        mistral_key
    )

    result = ai.generate(
        "Say hello in one sentence."
    )

    print("\nRESULT:")
    print(result)