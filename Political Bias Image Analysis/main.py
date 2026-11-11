
import os
import base64
import json

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

app = FastAPI(title="Political Image Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

ANALYSIS_PROMPT = """
You are an analytical assistant evaluating ideological messaging
and communicative framing in individual political images.

Analyze the uploaded image AND its Instagram caption together.

Keep these three concepts separate:

1. PARTY AFFILIATION

Possible values:
- democratic_associated
- republican_associated
- other
- none_identified
- unclear

Party affiliation does not automatically establish ideology.

2. IDEOLOGICAL ORIENTATION

Possible values:
- liberal-associated
- conservative-associated
- mixed
- insufficient_evidence

STRICT CLASSIFICATION RULE:

Before assigning liberal-associated or conservative-associated,
identify at least one specific ideological proposition, policy
position, or argument communicated by the post.

Party logos, party colors, candidate identities, election
victories, and general partisan celebration are NOT sufficient.

A supportive stance toward a political party does not
automatically establish support for its entire ideology.

Do not infer ideology from facial expressions, appearance,
or the identity of a politician.

IDEOLOGICAL NEUTRALITY RULE:

Do not classify a message as liberal-associated or
conservative-associated merely because it supports or
criticizes a particular politician, political party,
government, or foreign policy decision.

Do not assume that positions on national sovereignty,
patriotism, territorial integrity, democracy, or
government accountability belong exclusively to one
ideological orientation.

Identify the specific ideological principle or policy
position being communicated.

If the argument could reasonably be supported by people
from multiple ideological orientations, classify it as
insufficient_evidence unless additional evidence
establishes a clearer ideological orientation.

3. COMMUNICATIVE STANCE

Possible values:
- supportive
- critical
- neutral_reporting
- mixed
- unclear

Identify what specific message, person, policy, or movement
the creator appears to support or criticize.

Do not assume that displaying a slogan means endorsing it.
Consider quotation, satire, criticism, reporting, and context.

EVIDENCE REQUIREMENTS:

A. VISUAL EVIDENCE
Describe observable visual elements:
- Objects and symbols
- Political materials
- Actions and events
- Composition and cropping
- Visual hierarchy and prominence
- Scene context

B. EMBEDDED TEXT EVIDENCE
Extract relevant text appearing inside the image.
Do not include Instagram caption text here.

C. CAPTION ANALYSIS
Analyze the caption separately, including wording,
tone, context, and its relationship to the image.

D. REASONING
Explain how the evidence supports or fails to support
each classification.

Distinguish observations from interpretations.
If evidence is insufficient, say so.
Do not invent political events or policy positions.

Return ONLY valid JSON using this structure:

{
  "party_affiliation": "democratic_associated | republican_associated | other | none_identified | unclear",
  "ideological_orientation": "liberal-associated | conservative-associated | mixed | insufficient_evidence",
  "communicative_stance": "supportive | critical | neutral_reporting | mixed | unclear",
  "stance_target": "",
  "visual_evidence": [],
  "embedded_text_evidence": [],
  "caption_analysis": "",
  "reasoning": "",
  "limitations": [],
  "confidence_note": "Not calibrated"
}

Do not provide numerical confidence scores.
Do not include Markdown formatting or code fences.
"""

@app.get("/")
def home():
    return {
        "message": "Political Image Analysis API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    caption: str = Form("")
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WEBP image."
        )

    image_bytes = await image.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    mime_type = image.content_type

    image_data_url = (
        f"data:{mime_type};base64,{encoded_image}"
    )

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                ANALYSIS_PROMPT
                                + "\n\nINSTAGRAM CAPTION:\n"
                                + (caption or "[No caption provided]")
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            extra_body={
                "reasoning": {
                "effort": "low"
                }
            },
            temperature=0.2,
            max_tokens=6000,
            response_format={"type": "json_object"}
        )

        if not response.choices:
            raise ValueError("The AI returned no response choices.")

        choice = response.choices[0]
        result_text = choice.message.content

        if not result_text:
            usage = getattr(response, "usage", None)
            finish_reason = getattr(
                choice, "finish_reason", "unknown"
            )

            raise ValueError(
                f"Empty AI response. "
                f"Finish reason: {finish_reason}. "
                f"Usage: {usage}"
            )

        # Remove possible Markdown code fences.
        result_text = result_text.strip()

        if result_text.startswith("```"):
            result_text = result_text.replace(
                "```json", "", 1
            )
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
        try:
            analysis = json.loads(result_text)

        except json.JSONDecodeError as error:
            finish_reason = getattr(choice, "finish_reason", "unknown")

            raise ValueError(
                f"Invalid JSON returned by AI. "
                f"Line: {error.lineno}, "
                f"Column: {error.colno}, "
                f"Finish reason: {finish_reason}. "
                f"Raw response: {result_text[:1000]}"
            )

        return {
            "success": True,
            "analysis": analysis,
            "model_used": response.model
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Analysis failed: {str(error)}"
        )
