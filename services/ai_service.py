from google import genai
from google.genai import errors
from mistralai.client import Mistral
from PIL import Image
import os
import requests
import base64


class AIService:

    def __init__(
        self,
        gemini_key,
        mistral_key,
        cloudflare_token=None,
        cloudflare_account_id=None,
        cloudflare_model=None
    ):

        # =========================
        # API KEYS
        # =========================

        self.gemini_key = gemini_key
        self.mistral_key = mistral_key

        self.cloudflare_token = cloudflare_token
        self.cloudflare_account_id = cloudflare_account_id
        self.cloudflare_model = cloudflare_model

        print(
            "🔑 Gemini Key:",
            "موجودة ✅" if gemini_key else "غير موجودة ❌"
        )

        print(
            "🔑 Mistral Key:",
            "موجودة ✅" if mistral_key else "غير موجودة ❌"
        )

        print(
            "🔑 Cloudflare Token:",
            "موجودة ✅" if cloudflare_token else "غير موجودة ❌"
        )

        print(
            "🆔 Cloudflare Account ID:",
            "موجودة ✅" if cloudflare_account_id else "غير موجودة ❌"
        )

        print(
            "🤖 Cloudflare Model:",
            cloudflare_model if cloudflare_model else "غير موجودة ❌"
        )

        # =========================
        # GEMINI
        # =========================

        self.gemini = None

        if gemini_key:

            self.gemini = genai.Client(
                api_key=gemini_key
            )

        # =========================
        # MISTRAL
        # =========================

        self.mistral = None

        if mistral_key:

            self.mistral = Mistral(
                api_key=mistral_key
            )


    # =====================================================
    # GEMINI TEXT
    # =====================================================

    def generate_with_gemini(self, prompt):

        if self.gemini is None:

            raise Exception(
                "Gemini API key is not available."
            )

        print(
            "🤖 Gemini model: gemini-3.6-flash"
        )

        response = self.gemini.models.generate_content(

            model="gemini-3.6-flash",

            contents=prompt
        )

        if not response.text:

            raise Exception(
                "Gemini returned an empty response."
            )

        return response.text


    # =====================================================
    # GEMINI IMAGE
    # =====================================================

    def generate_with_image(
        self,
        prompt,
        image_file_or_path
    ):

        if self.gemini is None:

            raise Exception(
                "Gemini API key is not available."
            )

        if not image_file_or_path:

            raise Exception(
                "No image provided."
            )

        print(
            "📸 Opening image..."
        )

        image = Image.open(
            image_file_or_path
        )

        print(
            "📐 Image size:",
            image.size
        )

        print(
            "🖼️ Image format:",
            image.format
        )

        user_prompt = (

            prompt

            if prompt

            else

            """
            Analyze this exercise carefully.

            Solve the exercise step by step.

            Explain the reasoning clearly and simply.

            If there are mathematical calculations,
            show the calculations.

            If there is a diagram, table, graph,
            or handwritten information, analyze it carefully.

            Answer in the same language as the user's question.
            """
        )

        print(
            "🤖 Sending image to Gemini..."
        )

        response = self.gemini.models.generate_content(

            model="gemini-3.6-flash",

            contents=[
                user_prompt,
                image
            ]
        )

        if not response.text:

            raise Exception(
                "Gemini returned an empty image response."
            )

        print(
            "✅ Gemini successfully analyzed the image."
        )

        return response.text


    # =====================================================
    # MISTRAL TEXT
    # =====================================================

    def generate_with_mistral(self, prompt):

        if self.mistral is None:

            raise Exception(
                "Mistral API key is not available."
            )

        print(
            "🤖 Mistral model: mistral-small-latest"
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

        result = response.choices[0].message.content

        if not result:

            raise Exception(
                "Mistral returned an empty response."
            )

        return result


    # =====================================================
    # CLOUDFLARE IMAGE
    # =====================================================

    def generate_with_cloudflare(
        self,
        prompt,
        image_path
    ):

        if not self.cloudflare_token:

            raise Exception(
                "Cloudflare API token is not available."
            )

        if not self.cloudflare_account_id:

            raise Exception(
                "Cloudflare Account ID is not available."
            )

        if not self.cloudflare_model:

            raise Exception(
                "Cloudflare model is not available."
            )

        if not image_path:

            raise Exception(
                "No image provided."
            )

        print(
            "☁️ Sending image to Cloudflare..."
        )

        # =========================
        # READ IMAGE
        # =========================

        with open(
            image_path,
            "rb"
        ) as image_file:

            image_base64 = base64.b64encode(
                image_file.read()
            ).decode("utf-8")


        # =========================
        # PROMPT
        # =========================

        user_prompt = (

            prompt

            if prompt

            else

            """
            Analyze this exercise carefully.

            Solve the exercise step by step.

            Explain the reasoning clearly and simply.

            If there are mathematical calculations,
            show the calculations.

            If there is a diagram, table, graph,
            or handwritten information, analyze it carefully.

            Answer in the same language as the user's question.
            """
        )


        # =========================
        # CLOUDFLARE URL
        # =========================

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.cloudflare_account_id}/ai/run/"
            f"{self.cloudflare_model}"
        )


        # =========================
        # HEADERS
        # =========================

        headers = {

            "Authorization":
                f"Bearer {self.cloudflare_token}",

            "Content-Type":
                "application/json"
        }


        # =========================
        # PAYLOAD
        # =========================

        payload = {

            "messages": [

                {

                    "role": "user",

                    "content": [

                        {
                            "type": "text",
                            "text": user_prompt
                        },

                        {
                            "type": "image_url",

                            "image_url": {

                                "url":
                                    f"data:image/jpeg;base64,{image_base64}"
                            }
                        }

                    ]
                }

            ]
        }


        # =========================
        # REQUEST
        # =========================

        response = requests.post(

            url,

            headers=headers,

            json=payload,

            timeout=120
        )


        # =========================
        # CHECK RESPONSE
        # =========================

        if response.status_code != 200:

            print(
                "❌ Cloudflare status:",
                response.status_code
            )

            print(
                "❌ Cloudflare response:",
                response.text
            )

            raise Exception(
                "Cloudflare image request failed."
            )


        data = response.json()


        # =========================
        # CHECK SUCCESS
        # =========================

        if not data.get("success"):

            raise Exception(
                f"Cloudflare API error: {data}"
            )


        result = data.get(
            "result",
            {}
        )


        answer = result.get(
            "response"
        )


        if not answer:

            raise Exception(
                "Cloudflare returned an empty response."
            )


        print(
            "✅ Cloudflare successfully analyzed the image."
        )

        return answer


    # =====================================================
    # MAIN GENERATE
    # =====================================================

    def generate(self, prompt):

        # =========================
        # 1️⃣ GEMINI
        # =========================

        if self.gemini:

            try:

                print(
                    "🤖 Trying Gemini..."
                )

                result = self.generate_with_gemini(
                    prompt
                )

                if result and result.strip():

                    print(
                        "✅ Used Gemini"
                    )

                    return result

            except errors.ClientError as e:

                code = getattr(
                    e,
                    "code",
                    None
                )

                print(
                    "❌ Gemini ClientError:",
                    e
                )

                if code == 429:

                    print(
                        "⚠️ Gemini quota/rate limit exceeded."
                    )

                elif code == 404:

                    print(
                        "⚠️ Gemini model not found."
                    )

                elif code == 503:

                    print(
                        "⚠️ Gemini temporarily unavailable."
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


        # =========================
        # 2️⃣ MISTRAL
        # =========================

        if self.mistral:

            try:

                print(
                    "🤖 Trying Mistral..."
                )

                result = self.generate_with_mistral(
                    prompt
                )

                if result and result.strip():

                    print(
                        "✅ Used Mistral"
                    )

                    return result

            except Exception as e:

                print(
                    "❌ Mistral error:",
                    e
                )

                print(
                    "🔄 Mistral failed."
                )


        # =========================
        # 3️⃣ ALL FAILED
        # =========================

        print(
            "❌ All cloud AI providers failed."
        )

        return (
            "Sorry, all AI services are currently "
            "unavailable. Please try again later."
        )