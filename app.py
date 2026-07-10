import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

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
    v = re.sub(r"[.,;:]+$", "", v)
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
        "RUPEE": "INR",
        "RUPEES": "INR",
        "₹": "INR",
        "$": "USD",
        "US$": "USD",
        "EURO": "EUR",
        "EUROS": "EUR",
        "£": "GBP",
        "POUNDS STERLING": "GBP",
        "YEN": "JPY",
    }

    return mapping.get(v, v)


def clean_priority(v):
    if v is None:
        return None
    return str(v).strip().lower()


@app.post("/extract")
def extract(req: RequestBody):

    prompt = f"""
You are an invoice extraction engine.

Extract information from the invoice.

Return ONLY valid JSON.

Follow this schema EXACTLY:

{json.dumps(req.schema, indent=2)}

Rules:

- Return EXACTLY the fields defined in the schema.
- Do NOT invent extra fields.
- Missing fields must be null.
- Dates must be YYYY-MM-DD.
- Currency must be ISO4217 codes (USD, EUR, GBP, INR, JPY).
- Numbers must be JSON numbers.
- Preserve vendor names exactly as written except remove trailing punctuation.
- Return JSON ONLY.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract structured invoice information.",
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nInvoice:\n{req.text}",
            },
        ],
    )

    text = response.choices[0].message.content.strip()

    try:
        result = json.loads(text)
    except Exception:
        result = {}

    # ---------- Normalization ----------

    if "vendor" in result:
        result["vendor"] = clean_vendor(result["vendor"])

    if "contact_email" in result:
        result["contact_email"] = clean_email(result["contact_email"])

    if "currency" in result:
        result["currency"] = clean_currency(result["currency"])

    if "priority" in result:
        result["priority"] = clean_priority(result["priority"])

    # ---------- Ensure EXACT schema ----------

    if "properties" in req.schema:
        expected_keys = list(req.schema["properties"].keys())
    else:
        expected_keys = list(req.schema.keys())

    final = {}

    for key in expected_keys:
        final[key] = result.get(key)

    return final
