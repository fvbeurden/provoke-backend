"""
PROVOKE RAG Backend — FastAPI
==============================
Endpoints:
  POST /api/initieel   → Stap 1: Claude genereert initiële DD (JSON-lijst)
  POST /api/verfijn    → Stap 2: Pinecone RAG + Claude streaming verfijnd verslag

Start:
  cd backend && uvicorn main:app --reload --port 8000
"""

import os, json, asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import anthropic

# ── Setup ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="PROVOKE RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
pinecone_index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(
    os.getenv("PINECONE_INDEX_NAME", "medische-kennisbasis-nl")
)
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "huidziekten")


# ── URL helper ─────────────────────────────────────────────────────────────────
def maak_huidziekten_url(bestandsnaam: str) -> str:
    """Construeer huidziekten.nl URL vanuit PDF-bestandsnaam.
    Voorbeeld: EczemaSeborroicum.pdf → .../etxt/EczemaSeborroicum.htm
    Let op: originele hoofdletters bewaren (huidziekten.nl is hoofdlettergevoelig).
    """
    if not bestandsnaam:
        return ""
    slug = bestandsnaam
    if slug.lower().endswith(".pdf"):
        slug = slug[:-4]
    if not slug:
        return ""
    first_letter = slug[0].lower() if slug[0].isalpha() else "a"
    return f"https://www.huidziekten.nl/zakboek/dermatosen/{first_letter}txt/{slug}.htm"


# ── Request modellen ───────────────────────────────────────────────────────────
class InitieelRequest(BaseModel):
    provoke_tekst: str   # de ingevulde PROVOKE-samenvatting

class VerfijnRequest(BaseModel):
    provoke_tekst: str
    diagnoses: list[str]  # namen van de initiële DD (van Stap 1)


# ── Helpers ────────────────────────────────────────────────────────────────────
def embed(tekst: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=tekst[:8000]
    )
    return response.data[0].embedding


def zoek_pinecone(query: str, top_k: int = 5) -> list[dict]:
    """Query Pinecone en geef matches terug als lijst van dicts."""
    vector = embed(query)
    results = pinecone_index.query(
        vector=vector,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True
    )
    return results.get("matches", [])


def bouw_rag_context(matches: list[dict]) -> tuple[str, list[dict]]:
    """
    Bouw context-string en bronnen-lijst uit Pinecone matches.
    Elke bron bevat nu ook de huidziekten.nl-URL voor in het verslag.
    """
    context_delen = []
    bronnen = []

    for m in matches:
        meta         = m.get("metadata", {})
        tekst        = meta.get("tekst", "")
        naam         = meta.get("aandoening", "Onbekend")
        icd10        = meta.get("icd10", "")
        score        = round(m.get("score", 0), 3)
        bestandsnaam = meta.get("bestandsnaam", "")
        afb_str      = meta.get("afbeeldingen", "")
        afbeeldingen = [a for a in afb_str.split(",") if a] if afb_str else []
        url          = maak_huidziekten_url(bestandsnaam)

        context_delen.append(
            f"### {naam}{f' (ICD10: {icd10})' if icd10 else ''}\n"
            f"Bron: {url}\n\n"
            f"{tekst}"
        )
        bronnen.append({
            "aandoening": naam,
            "icd10": icd10,
            "score": score,
            "afbeeldingen": afbeeldingen,
            "url": url
        })

    return "\n\n---\n\n".join(context_delen), bronnen


# ── Endpoint 1: Initiële DD ───────────────────────────────────────────────────
@app.post("/api/initieel")
async def initieel(req: InitieelRequest):
    """
    Stap 1: Stuur PROVOKE-tekst naar Claude → geeft JSON-lijst van diagnoses terug.
    Geen RAG — Claude gebruikt zijn eigen kennis voor de initiële DD.
    """
    prompt = f"""Je bent een ervaren dermatoloog. Analyseer de volgende PROVOKE-beschrijving
en geef een lijst van 3 tot 7 meest waarschijnlijke differentiaaldiagnoses.

Geef ALLEEN een JSON-array terug in dit exacte formaat, niets anders:
[
  {{"naam": "Diagnose 1", "waarschijnlijkheid": "hoog"}},
  {{"naam": "Diagnose 2", "waarschijnlijkheid": "middel"}},
  {{"naam": "Diagnose 3", "waarschijnlijkheid": "laag"}}
]

PROVOKE BESCHRIJVING:
{req.provoke_tekst}"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    tekst = response.content[0].text.strip()

    # Parse JSON uit de response
    try:
        # Zoek JSON-array in de tekst
        match = __import__('re').search(r'\[.*\]', tekst, __import__('re').DOTALL)
        diagnoses = json.loads(match.group(0)) if match else []
    except:
        diagnoses = []

    return {"diagnoses": diagnoses}


# ── Endpoint 2: Verfijnd verslag (streaming) ───────────────────────────────────
@app.post("/api/verfijn")
async def verfijn(req: VerfijnRequest):
    """
    Stap 2: Zoek Pinecone per diagnose afzonderlijk, bouw RAG-context,
    stream het verfijnd verslag via Claude.
    """
    # Query Pinecone afzonderlijk per diagnose → elk krijgt zijn eigen correcte URL
    all_matches: list[dict] = []
    seen_ids: set[str] = set()
    diagnose_url_map: dict[str, str] = {}   # diagnose-naam → huidziekten.nl URL

    for diagnose_naam in req.diagnoses:
        # Zoek op diagnose-naam alleen (niet de hele PROVOKE-tekst) voor nauwkeurige URL-koppeling
        query = f"{diagnose_naam} huidziekte behandeling"
        matches = zoek_pinecone(query, top_k=2)
        for i, m in enumerate(matches):
            mid = m.get("id", "")
            if mid not in seen_ids:
                seen_ids.add(mid)
                all_matches.append(m)
            # Eerste match = beste URL voor deze diagnose
            if i == 0 and diagnose_naam not in diagnose_url_map:
                meta = m.get("metadata", {})
                bestandsnaam = meta.get("bestandsnaam", "")
                url = maak_huidziekten_url(bestandsnaam)
                if url:
                    diagnose_url_map[diagnose_naam] = url

    context, bronnen = bouw_rag_context(all_matches)

    # Bouw een expliciete koppeling diagnose → URL voor Claude
    url_mapping_regels = "\n".join(
        f"- {naam}: {url}"
        for naam, url in diagnose_url_map.items()
    )

    systeem_prompt = """Je bent een ervaren dermatoloog-assistent die huisartsen ondersteunt in de eerste lijn.

VERPLICHTE OPMAAKREGELS — wijk hier nooit van af:
1. Begin DIRECT met "## Differentiaaldiagnoses". Voeg NOOIT toe: titels als
   "Differentiaaldiagnose-verslag", "Datum", "Casus", "Klinische samenvatting" of "Opsteller".
2. Gebruik 🔴 voor meest waarschijnlijk, 🟡 voor mogelijk, ⚪ voor minder waarschijnlijk.
3. Elk diagnose-blok heeft dit exacte formaat:

### [nr]. [emoji] [Diagnose naam]

**Argumenten vóór:**
- ...

**Argumenten tégen:**
- ...

**Beleid (conform kennisbasis/richtlijn):**
- ...

**Bron:** [URL uit de DIAGNOSE-URL KOPPELING die hoort bij DEZE diagnose, als Markdown-hyperlink: [https://www.huidziekten.nl/...](https://www.huidziekten.nl/...)]

4. KRITISCH: Gebruik voor elke **Bron:** UITSLUITEND de URL die in de DIAGNOSE-URL KOPPELING
   staat bij die specifieke diagnose. Kopieer NOOIT dezelfde URL naar meerdere diagnoses.
5. Sluit af met een ## Conclusie-sectie (meest waarschijnlijke diagnose + vervolgadvies).
6. Gebruik uitsluitend de aangeleverde KENNISBASIS als bron voor beleid en verwijscriteria.
7. Schrijf in het Nederlands, helder en klinisch. Alles links uitgelijnd."""

    gebruiker_bericht = f"""PROVOKE BESCHRIJVING:
{req.provoke_tekst}

INITIËLE DIFFERENTIAALDIAGNOSES:
{chr(10).join(f'- {d}' for d in req.diagnoses)}

DIAGNOSE-URL KOPPELING (gebruik voor elke diagnose UITSLUITEND de bijbehorende URL hieronder):
{url_mapping_regels}

KENNISBASIS (huidziekten.nl):
{context}

Stel nu een volledig differentiaaldiagnose-verslag op."""

    def stream_claude():
        # Stuur eerst de bronnen als eerste chunk — volledige objecten incl. afbeeldingen en URL
        bronnen_payload = [
            {
                "aandoening": b["aandoening"],
                "icd10": b.get("icd10", ""),
                "score": b["score"],
                "afbeeldingen": b.get("afbeeldingen", []),
                "url": b.get("url", "")
            }
            for b in bronnen
        ]
        bronnen_json = json.dumps({"type": "bronnen", "bronnen": bronnen_payload})
        yield f"data: {bronnen_json}\n\n"

        # Stream het verslag
        with claude_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=systeem_prompt,
            messages=[{"role": "user", "content": gebruiker_bericht}]
        ) as stream:
            for chunk in stream.text_stream:
                if chunk:
                    payload = json.dumps({"type": "tekst", "inhoud": chunk})
                    yield f"data: {payload}\n\n"

        yield "data: {\"type\": \"klaar\"}\n\n"

    return StreamingResponse(
        stream_claude(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    stats = pinecone_index.describe_index_stats()
    vectors = stats.get("namespaces", {}).get(NAMESPACE, {}).get("vector_count", 0)
    return {
        "status": "ok",
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME"),
        "namespace": NAMESPACE,
        "vectoren": vectors
    }
