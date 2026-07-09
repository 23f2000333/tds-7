import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestBody(BaseModel):
    document_id: str
    text: str
    schema: dict


def clean_vendor(v):
    if v is None:
        return None
    v = str(v).strip()
    v = re.sub(r"[.,;:]+$", "", v)  # remove trailing punctuation only
    v = re.sub(r"\s+", " ", v)
    return v


def clean_email(v):
    if v is None:
        return None
    return str(v).strip().lower()


def clean_currency(v):
    if v is None:
        return None

    v = str(v).strip().upper()

    mapping = {
        "RS": "INR",
        "RUPEES": "INR",
        "RUPEE": "INR",
        "₹": "INR",
        "$": "USD",
        "US$": "USD",
        "EUROS": "EUR",
        "EURO": "EUR",
        "POUNDS STERLING": "GBP",
    }

    return mapping.get(v, v)


def clean_priority(v):
    if v is None:
        return None
    return str(v).strip().lower()


@app.post("/extract")
def extract(req: RequestBody):

    prompt = f"""
Extract the requested information from this invoice.

Return ONLY valid JSON.

Follow the supplied JSON schema exactly.

Rules:

- Return exactly the schema fields.
- No markdown.
- No explanation.
- Dates must be YYYY-MM-DD.
- Currency must be ISO4217 (USD, EUR, GBP, INR, JPY).
- total_amount, quantity, unit_price etc. must be JSON numbers.
- Preserve vendor names exactly as written except remove trailing punctuation.

Invoice text:

{req.text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=req.schema,
            temperature=0,
        ),
    )

    result = json.loads(response.text)

    if "vendor" in result:
        result["vendor"] = clean_vendor(result["vendor"])

    if "contact_email" in result:
        result["contact_email"] = clean_email(result["contact_email"])

    if "currency" in result:
        result["currency"] = clean_currency(result["currency"])

    if "priority" in result:
        result["priority"] = clean_priority(result["priority"])

    return result
