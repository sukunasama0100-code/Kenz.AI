from google import genai
from google.genai import errors
from mistralai.client import Mistral
import ollama
import time


class AIService:

    def __init__(self, gemini_key, mistral_key):

        # =========================
        # API KEYS
        # =========================

        self.gemini_key = gemini_key
        self.mistral_key = mistral_key

        print(
            "🔑 Gemini Key:",
            "موجودة ✅" if gemini_key else "غير موجودة ❌"
        )

        print(
            "🔑 Mistral Key:",
            "موجودة ✅" if mistral_key else "غير موجودة ❌"
        )

        # =========================
        # Gemini
        # =========================

        self.gemini = None

        if gemini_key:

            self.gemini = genai.Client(
                api_key=gemini_key
            )

        # =========================
        # Mistral
        # =========================

        self.mistral = None

        if mistral_key:

            self.mistral = Mistral(
                api_key=mistral_key
            )

        # =========================
        # Ollama
        # =========================

        self.ollama_model = "qwen3:8b"


    # =====================================================
    # GEMINI
    # =====================================================

    def generate_with_gemini(self, prompt):

        if self.gemini is None:

            raise Exception(
                "Gemini API key is not available."
            )

        response = self.gemini.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text


    # =====================================================
    # MISTRAL
    # =====================================================

    def generate_with_mistral(self, prompt):

        if self.mistral is None:

            raise Exception(
                "Mistral API key is not available."
            )

        response = self.mistral.chat.complete(

            model="mistral-small-latest",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content


    # =====================================================
    # OLLAMA
    # =====================================================

    def generate_with_ollama(self, prompt):

        response = ollama.chat(

            model=self.ollama_model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


    # =====================================================
    # MAIN GENERATE FUNCTION
    # =====================================================

    def generate(self, prompt):

        # =================================================
        # 1️⃣ GEMINI
        # =================================================

        try:

            print("🤖 Trying Gemini...")

            result = self.generate_with_gemini(
                prompt
            )

            if result and result.strip():

                print("✅ Used Gemini")

                return result


        except errors.ClientError as e:

            # ---------------------------------------------
            # Gemini 429
            # ---------------------------------------------

            if getattr(e, "code", None) == 429:

                print(
                    "⚠️ Gemini quota/rate limit exceeded."
                )

            # ---------------------------------------------
            # Gemini 503
            # ---------------------------------------------

            elif getattr(e, "code", None) == 503:

                print(
                    "⚠️ Gemini is temporarily unavailable (503)."
                )

            # ---------------------------------------------
            # Other Gemini client errors
            # ---------------------------------------------

            else:

                print(
                    "❌ Gemini ClientError:",
                    e
                )

            print(
                "🔄 Switching to Mistral..."
            )


        except Exception as e:

            print(
                "❌ Gemini error:",
                e
            )

            print(
                "🔄 Switching to Mistral..."
            )


        # =================================================
        # 2️⃣ MISTRAL
        # =================================================

        try:

            print("🤖 Trying Mistral...")

            result = self.generate_with_mistral(
                prompt
            )

            if result and result.strip():

                print("✅ Used Mistral")

                return result


        except Exception as e:

            print(
                "❌ Mistral error:",
                e
            )

            print(
                "🔄 Switching to Ollama..."
            )


        # =================================================
        # 3️⃣ OLLAMA
        # =================================================

        try:

            print("🤖 Trying Ollama...")

            result = self.generate_with_ollama(
                prompt
            )

            if result and result.strip():

                print(
                    "✅ Used Ollama (Local AI)"
                )

                return result


        except Exception as e:

            print(
                "❌ Ollama error:",
                e
            )


        # =================================================
        # ALL AI PROVIDERS FAILED
        # =================================================

        print(
            "❌ All AI providers failed."
        )

        return (
            "Sorry, all AI services are currently "
            "unavailable. Please try again later."
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from dotenv import load_dotenv
    import os

    load_dotenv()

    gemini_key = os.getenv(
        "GEMINI_API_KEY"
    )

    mistral_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    ai = AIService(
        gemini_key,
        mistral_key
    )

    result = ai.generate(
        "Say hello in one sentence."
    )

    print("\nRESULT:")
    print(result)