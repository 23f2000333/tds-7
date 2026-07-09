import json
import os

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


@app.post("/extract")
def extract(req: RequestBody):

    prompt = f"""
Extract the requested information from this invoice.

Return ONLY JSON.

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

    return json.loads(response.text)
