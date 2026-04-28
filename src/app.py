"""
app.py — FastAPI backend for LegalEase (multi-act Indian legal RAG chatbot).

Retrieval pipeline:
  1. Query expansion  (first-match-wins rules, act-aware)
  2. Dense retrieval  BGE-base embeddings + IndexFlatIP, top-k=15
  3. Sparse retrieval BM25Okapi, top-k=15
  4. Reciprocal Rank Fusion
  5. Guarded adjacency boost (dense score >= 0.45, sorted-section index)
  6. Return top-5 chunks to LLM

Document offer detection (zero extra LLM calls):
  - BGE semantic gate: cosine similarity against pre-embedded intent anchors
  - Act-key scoring: retrieved act keys break ties and confirm doc type
  - Returns doc_offer:{type, label, law_ref} or null in every /chat response

Field extraction:
  - POST /extract_fields — one targeted LLM call to pre-fill the doc form
    from user's narrative. Called only when user clicks Draft in the chat.
"""

import os
import re
import json
import pickle
import faiss
import ollama
import numpy as np
import time

# Load .env file automatically — install with: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from doc_generator import generate_document, DOCUMENT_TYPES

app = FastAPI()

@app.on_event("startup")
async def _startup_log():
    key_status = f"SET ({RESEND_API_KEY[:8]}...)" if RESEND_API_KEY else "NOT SET — email will fail"
    print(f"\n✅ LegalEase backend started")
    print(f"   RESEND_API_KEY  : {key_status}")
    print(f"   RESEND_FROM     : {RESEND_FROM_EMAIL}")
    print(f"   ⚠  Resend sandbox: you can only send TO your own Resend account email.\n")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
_sessions: dict = {}
SESSION_TTL = 3600

def _clean_sessions():
    now = time.time()
    stale = [k for k, v in _sessions.items()
             if now - v.get("created_at", 0) > SESSION_TTL]
    for k in stale:
        del _sessions[k]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VECTOR_PATH               = "vectorstore"
EMBED_MODEL               = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX              = "Represent this legal question for retrieval: "
MAX_QUESTION_LEN          = 500
SCORE_THRESHOLD           = 0.25
TOP_K_DENSE               = 15
TOP_K_FINAL               = 5
ADJACENCY_MIN_DENSE_SCORE = 0.45

# BGE document-intent gate threshold.
# Queries below this score are informational — no doc offer shown.
DOC_INTENT_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Embedding model (shared for RAG + intent detection)
# ---------------------------------------------------------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

# ---------------------------------------------------------------------------
# Document intent anchor matrix
# Pre-embedded at startup (~50 ms). Reused every request at ~2 ms per query.
# Each anchor is a victim-situation sentence that maps to a doc type.
# ---------------------------------------------------------------------------
_DOC_INTENT_ANCHORS = [
    # complaint_letter — criminal offences, police involvement needed
    ("complaint_letter",
     "Someone physically attacked and assaulted me and I want to report it"),
    ("complaint_letter",
     "I was beaten up and need to file a police complaint"),
    ("complaint_letter",
     "Someone threatened to kill me and I fear for my safety"),
    ("complaint_letter",
     "My belongings were stolen and I need to report the theft"),
    ("complaint_letter",
     "I was robbed and want to report the crime to police"),
    ("complaint_letter",
     "A person committed a serious crime against me"),
    ("complaint_letter",
     "I was kidnapped or held against my will"),
    ("complaint_letter",
     "I was sexually harassed or assaulted by someone"),
    ("complaint_letter",
     "Someone broke into my house and I want to file a complaint"),

    # legal_notice — civil/monetary disputes
    ("legal_notice",
     "My employer is not paying my salary and I need a legal notice"),
    ("legal_notice",
     "Someone borrowed money from me and refuses to repay it"),
    ("legal_notice",
     "I was cheated and defrauded out of a large sum of money"),
    ("legal_notice",
     "The other party has breached our contract or agreement"),
    ("legal_notice",
     "I received a defective product and want compensation"),
    ("legal_notice",
     "My landlord is refusing to return my security deposit"),
    ("legal_notice",
     "I need to send a formal demand notice to recover money"),

    # cybercrime_complaint — IT Act offences
    ("cybercrime_complaint",
     "Someone hacked into my bank account or email"),
    ("cybercrime_complaint",
     "I was scammed online and lost money through fraud"),
    ("cybercrime_complaint",
     "My identity was stolen and used online without my consent"),
    ("cybercrime_complaint",
     "I am receiving threatening and abusive messages online"),
    ("cybercrime_complaint",
     "My private photos were shared on the internet without permission"),
    ("cybercrime_complaint",
     "I was cheated by a fake website or fraudulent mobile app"),
    ("cybercrime_complaint",
     "Someone gained unauthorized access to my computer or accounts"),
]

print("Building document intent anchor matrix...")
_ANCHOR_LABELS = [label for label, _ in _DOC_INTENT_ANCHORS]
_ANCHOR_SENTS  = [sent  for _,     sent in _DOC_INTENT_ANCHORS]
_ANCHOR_MATRIX = embed_model.encode(
    _ANCHOR_SENTS,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
).astype(np.float32)
print(f"Anchor matrix: {_ANCHOR_MATRIX.shape[0]} anchors ready.")

# Fallback multilingual keywords — only fires when BGE score is borderline
# (keeps the anchor matrix as primary and regex as a safety net for
#  obvious Hindi / Malayalam document request words that BGE may miss)
_DOC_KEYWORDS_FALLBACK = re.compile(
    r"\b(complaint letter|shikayat patra|shikayat|parishikayat|"
    r"legal notice|notice bhejo|notice chahiye|notice banao|"
    r"cyber complaint|cybercrime report|online fraud complaint)\b",
    re.IGNORECASE,
)

# Act-key → doc_type mapping (used to confirm type from retrieved chunks)
_ACT_TO_DOC = {
    "BNS":      "complaint_letter",
    "BNSS":     "complaint_letter",
    "BSA":      "complaint_letter",
    "IT":       "cybercrime_complaint",
    "CONSUMER": "legal_notice",
    "CONTRACT": "legal_notice",
    "MV":       "complaint_letter",
}

# Human-readable law references per doc type (shown in the action card)
_DOC_LAW_REF = {
    "complaint_letter":     "BNSS 2023, Section 173",
    "legal_notice":         "BNS 2023, Section 318 / Contract Act",
    "cybercrime_complaint": "IT Act 2000, Section 66C / BNS 2023",
}

_DOC_CARD_LABEL = {
    "complaint_letter":     "Draft Complaint Letter to SHO",
    "legal_notice":         "Draft Legal Notice",
    "cybercrime_complaint": "Draft Cybercrime Complaint",
}


def detect_doc_offer(query_vec: np.ndarray, query: str,
                     retrieved_meta: list) -> Optional[dict]:
    """
    Determine whether to show a document action card and which type.

    Decision pipeline (highest score wins):
      1. Cosine similarity between query embedding and each intent anchor.
         Winner = highest-scoring anchor across all types.
      2. Act-key voting from retrieved chunks confirms / adjusts the type.
      3. Multilingual keyword fallback if BGE score is in the border zone.

    Returns a dict with {type, label, law_ref} or None.
    """
    # Step 1: BGE similarity against all anchors
    # query_vec is already L2-normalised (unit vector) so dot product = cosine
    sims = (_ANCHOR_MATRIX @ query_vec.T).flatten()  # shape (N,)

    best_idx   = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    bge_type   = _ANCHOR_LABELS[best_idx]

    # Hard gate — below threshold means the query is informational
    if best_score < DOC_INTENT_THRESHOLD:
        # Check multilingual keyword fallback before giving up
        kw_match = _DOC_KEYWORDS_FALLBACK.search(query)
        if not kw_match:
            return None
        # Keyword matched — use BGE best type if score is >= 0.20, else complaint
        bge_type = bge_type if best_score >= 0.20 else "complaint_letter"

    # Step 2: Act-key vote from retrieved chunks
    # Count votes per doc type based on which acts were actually retrieved
    act_votes: dict = {}
    for meta in retrieved_meta:
        ak = meta.get("act_key", "")
        dt = _ACT_TO_DOC.get(ak)
        if dt:
            act_votes[dt] = act_votes.get(dt, 0) + 1

    if act_votes:
        # Highest-voted act type wins (breaks ties by BGE)
        act_winner = max(act_votes, key=act_votes.get)
        # Only override BGE if act voting is confident (>=2 chunks agree)
        if act_votes[act_winner] >= 2:
            final_type = act_winner
        else:
            # One-chunk retrieval: trust BGE over single chunk
            final_type = bge_type
    else:
        final_type = bge_type

    return {
        "type":    final_type,
        "label":   _DOC_CARD_LABEL[final_type],
        "law_ref": _DOC_LAW_REF[final_type],
    }


# ---------------------------------------------------------------------------
# Vectorstore
# ---------------------------------------------------------------------------
print("Loading vectorstore...")
index = faiss.read_index(f"{VECTOR_PATH}/index.faiss")
with open(f"{VECTOR_PATH}/chunks.pkl",   "rb") as f:
    chunks: list = pickle.load(f)
with open(f"{VECTOR_PATH}/metadata.pkl", "rb") as f:
    metadata: list = pickle.load(f)
print(f"Loaded {len(chunks)} chunks.")

print("Building BM25 index...")
bm25 = BM25Okapi([c.lower().split() for c in chunks])

# ---------------------------------------------------------------------------
# Adjacency structures
# ---------------------------------------------------------------------------
def _sec_sort_key(s):
    m = re.match(r"^(\d+)([A-Z]?)$", s)
    if m:
        return (int(m.group(1)), m.group(2))
    return (0, s)

_act_sections: dict = {}
for idx, meta in enumerate(metadata):
    ak  = meta["act_key"]
    sid = meta["section_number"]
    if ak not in _act_sections:
        _act_sections[ak] = {}
    _act_sections[ak][sid] = idx

act_sorted_ids: dict = {}
for ak, sid_map in _act_sections.items():
    act_sorted_ids[ak] = sorted(sid_map.keys(), key=_sec_sort_key)

def compound_key_to_idx(act_key: str, sec_id: str):
    return _act_sections.get(act_key, {}).get(sec_id)

def get_neighbours(act_key: str, sec_id: str) -> list:
    sorted_ids = act_sorted_ids.get(act_key, [])
    try:
        pos = sorted_ids.index(sec_id)
    except ValueError:
        return []
    neighbours = []
    if pos > 0:
        prev = compound_key_to_idx(act_key, sorted_ids[pos - 1])
        if prev is not None:
            neighbours.append(prev)
    if pos < len(sorted_ids) - 1:
        nxt = compound_key_to_idx(act_key, sorted_ids[pos + 1])
        if nxt is not None:
            neighbours.append(nxt)
    return neighbours

print("Adjacency structures built.")

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
EXPANSION_RULES = [
    (r"\b(definition of document|document.*defined|what is.*document|document under)\b",
     "nyaya document letters figures marks substance electronic digital record intended"),
    (r"\b(definition of|what is defined as|meaning of|defined under)\b",
     "definitions sanhita adhiniyam context otherwise requires"),
    (r"\b(threatening to kill|death threat|kill me|murder me|threaten my life)\b",
     "criminal intimidation threat causing death fear"),
    (r"\b(hit me|beat me|punch|slap|physically attack|bodily harm|"
     r"injured me|attacked me|someone attacked|assaulted me)\b",
     "hurt bodily pain disease infirmity simple voluntarily causing hurt assault criminal force"),
    (r"\b(grievous|serious injur|broken bone|permanent damage|disfigur|acid attack)\b",
     "grievous hurt voluntarily causing grievous hurt acid"),
    (r"\b(threaten(?!.*kill)|intimidat|blackmail|extort)\b",
     "criminal intimidation threat injury extortion"),
    (r"\b(salary|wages|not paid|employer|employee|payment due|unpaid)\b",
     "criminal breach of trust entrusted dishonestly"),
    (r"\b(borrow|lend|loan|didn.t return|not return|not repay|money back)\b",
     "criminal breach of trust cheating dishonestly entrusted"),
    (r"\b(defective product|defective goods|consumer complaint|refund|replacement|"
     r"warranty|misleading advertisement|unfair trade|product liability|"
     r"spurious goods|adulterant)\b",
     "consumer complaint defect deficiency unfair trade practice "
     "district commission redressal product liability"),
    (r"\b(hacked|hacking|hack into|unauthorized access|cyber attack|"
     r"broke into.*computer|computer intrusion)\b",
     "computer contravention punishment computer related offence unauthorized "
     "access damage data system"),
    (r"\b(identity theft|stole my identity|impersonat.*online|"
     r"someone using my identity|fraud using my name|fake.*my.*identity)\b",
     "identity theft cheating personation electronic resource fraudulently "
     "computer signature"),
    (r"\b(obscene.*online|online.*obscene|posted.*about.*me.*online|"
     r"intimate image|revenge porn|privacy violated|voyeur|"
     r"obscene content|obscene material|obscene publication)\b",
     "violation privacy capture transmit publish image intimate obscene "
     "electronic record"),
    (r"\b(online fraud|e-commerce|online shopping|fake website|phishing|"
     r"cyber fraud|data breach|computer offence|cyber crime)\b",
     "computer related offences unauthorized access cyber terrorism "
     "electronic record information technology"),
    (r"\b(contract|agreement|breach of contract|void agreement|consideration|"
     r"offer and acceptance|damages for breach|specific performance|penalty clause)\b",
     "contract agreement offer acceptance consideration void voidable "
     "breach compensation damages"),
    (r"\b(steal|stole|stolen|snatched|pickpocket|burglar|theft)\b",
     "theft dishonestly movable property took without consent"),
    (r"\b(rob(?!bery)|mugged|held at gunpoint|armed theft|dacoity)\b",
     "robbery dacoity extortion force hurt criminal force"),
    (r"\b(kidnap|abduct|taken away|missing person|held captive|wrongful confinement)\b",
     "kidnapping abduction wrongful confinement compelling"),
    (r"\b(rape|sexual assault|molestat|outrage modesty|sexual violence)\b",
     "rape sexual intercourse punishment rigorous imprisonment consent"),
    (r"\b(sexual harass|eve teas|unwanted touch|groping|stalking)\b",
     "sexual harassment assault criminal force modesty stalking"),
    (r"\b(cheated|fraud|scam|deceiv|fake identity|impersonat)\b",
     "cheating dishonestly inducing person deceiving"),
    (r"\b(forged|fake document|counterfeit|falsif|fabricated)\b",
     "forgery false document making forged"),
    (r"\b(defam|slander|libel|false statement about me|spoil reputation)\b",
     "defamation imputation reputation words spoken"),
    (r"\b(compensation.*accident|accident.*compensation|met with.*accident|"
     r"road accident.*compensation|hit by.*vehicle.*compensation|"
     r"compensation.*injur|accident claim|injury claim)\b",
     "no fault liability compensation death permanent disablement "
     "claims tribunal application personal injury award"),
    (r"\b(drunk driv|drunken driv|driving under influence|rash driv|reckless driv|"
     r"overspeed|speed limit|penalty.*drunk|drunk.*penalty|"
     r"alcohol.*driv|driv.*alcohol)\b",
     "drunken driving influence alcohol drugs breath analyser "
     "penalty fine imprisonment offence speed"),
    (r"\b(driving licen|licen\w*\s+requir|requir\w*\s+licen|need\s+\w*licen|"
     r"get\s+\w*licen|licen\w*\s+age|age\s+\w*licen|without\s+\w*licen)",
     "no person shall drive public place holds effective driving licence "
     "issued authorising age eighteen years transport"),
    (r"\b(motor vehicle|road accident|car accident|vehicle registration|"
     r"traffic offence|traffic rule|hit by vehicle)\b",
     "motor vehicle act registration certificate insurance penalty offence transport"),
    (r"\b(privacy|personal data|data stolen|data leak)\b",
     "violation privacy electronic record personal data publishing"),
    (r"\b(self[\s-]defenc|defend myself|right to protect|can i fight back)",
     "right of private defence body property every person subject "
     "restrictions extends things done"),
    (r"\b(attempt|tried to commit|incomplete crime)\b",
     "attempt punishment attempting imprisonment"),
    (r"\b(murder|culpable homicide|killed someone|causing death)\b",
     "murder culpable homicide punishment imprisonment death"),
]

def expand_query(query: str) -> str:
    for pattern, terms in EXPANSION_RULES:
        if re.search(pattern, query.lower()):
            return query + " " + terms
    return query


# ---------------------------------------------------------------------------
# Input sanitiser
# ---------------------------------------------------------------------------
_INJECTION = re.compile(
    r"(ignore previous|disregard|system\s*:|assistant\s*:|###|"
    r"you are now|new prompt|forget everything)",
    re.IGNORECASE,
)

def sanitize(text: str) -> str:
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", " ", text)
    text = _INJECTION.sub("[removed]", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ---------------------------------------------------------------------------
# Greeting / off-topic detectors
# ---------------------------------------------------------------------------
_GREETINGS = re.compile(
    r"^(hi+|hello+|hey+|greetings?|good\s*(morning|afternoon|evening|night)|"
    r"howdy|sup|what'?s up|how are you|how r u|hru|namaste|namaskar|"
    r"vanakkam|salaam|thank\s*you|thanks|bye+|goodbye|see\s*you|ok+|okay+|"
    r"yes|no|sure|great|nice|cool|awesome|wow|lol|haha|test|testing|"
    r"who are you|what are you|what can you do|help me|"
    r"are you (a )?(bot|ai|robot|human|legal|lawyer))[\s!?.]*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "Hello! I am LegalEase, an AI legal assistant for Indian law. "
    "I can help you understand your legal rights and options under acts like the "
    "Bharatiya Nyaya Sanhita, Bharatiya Nagarik Suraksha Sanhita, "
    "Bharatiya Sakshya Adhiniyam, Information Technology Act, and Motor Vehicles Act.\n\n"
    "Please describe your legal issue in plain language and I will do my best to help. "
    "For example:\n"
    "• \"Someone hit me on the street\"\n"
    "• \"My employer is not paying my salary\"\n"
    "• \"Someone hacked into my computer\"\n"
    "• \"I met with a road accident and need compensation\""
)

_LEGAL_WORDS = re.compile(
    r"\b(law|legal|court|police|fir|crime|offence|offense|section|act|rights?|"
    r"arrest|complaint|case|judge|lawyer|advocate|jail|bail|charge|accused|victim|"
    r"theft|fraud|scam|assault|rape|murder|kidnap|harass|defam|contract|"
    r"property|tenant|landlord|divorce|custody|salary|employer|employee|accident|"
    r"compensation|hit me|beat|punch|slap|attack|threat|blackmail|extort|"
    r"stole|stolen|hacked|hack|cyber|cheated|cheat|borrow|lend|loan|"
    r"injur|violence|weapon|register|sue|sued|petition|notice|warrant|"
    r"job|fired|termination|dismiss|unfair|wrongful|workplace|work|labour|"
    r"evict|rent|lease|property|consumer|refund|defective|cheque|bounce|"
    r"domestic|dowry|marriage|will|inherit|succession|deed|agreement)\b",
    re.IGNORECASE,
)

_OFFTOPIC_RESPONSE = (
    "I am LegalEase, an AI legal assistant specializing in Indian law. "
    "I can only help with legal questions related to Indian laws such as the "
    "Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), "
    "Information Technology Act, Motor Vehicles Act, and more.\n\n"
    "Please describe a legal issue you are facing — for example:\n"
    "• \"Someone hit me on the street\"\n"
    "• \"My employer is not paying my salary\"\n"
    "• \"Someone hacked into my account\"\n"
    "• \"I met with a road accident\""
)


def is_greeting(text: str) -> bool:
    return bool(_GREETINGS.match(text.strip()))

# Phrases that are clearly non-legal regardless of word count
_OFFTOPIC_PHRASES = re.compile(
    r"\b(fill\s*(the\s*)?(form|document|field|box)|manually\s*fill|"
    r"how\s*(to\s*)?(use|open|navigate|click|upload|download|login|logout|sign\s*(in|up))|"
    r"where\s*(is|are|do|can)\s*(i|we|the|a)\s*(find|see|go|get|click|upload|download)|"
    r"what\s*(is|are)\s*(this|that|the|a)\s*(button|page|feature|option|menu|tab)|"
    r"(this\s*)?(app|website|site|page|interface)\s*(is|does|works?|looks?)|"
    r"(nice|good|bad|great|awesome|cool|terrible)\s*(app|website|work|job|design)|"
    r"(i\s*)?can\s*(i\s*)?(manually|easily|quickly|just)\s*fill|"
    r"weather|cricket|football|movie|music|food|recipe|cook|sport|news|"
    r"stock\s*price|bitcoin|crypto|currency|exchange\s*rate)\b",
    re.IGNORECASE,
)

def is_meaningful_legal_query(text: str) -> bool:
    # Block clearly non-legal phrases first, regardless of length
    if _OFFTOPIC_PHRASES.search(text):
        return False
    words = text.split()
    if len(words) >= 4:   # 4 words is enough — "lost my job unfairly"
        return True
    return bool(_LEGAL_WORDS.search(text))


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------
def hybrid_retrieve(query: str, top_k_dense: int = TOP_K_DENSE,
                    top_k_final: int = TOP_K_FINAL):
    """
    Returns (chunks, metadata, rrf_scores, query_vec).
    query_vec is returned so the caller can reuse it for intent detection
    without a second encode call.
    """
    expanded = expand_query(query)

    # Dense retrieval
    query_vec = embed_model.encode(
        [QUERY_PREFIX + expanded],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    dense_scores, dense_indices = index.search(query_vec, top_k_dense)

    # Sparse retrieval
    bm25_scores = bm25.get_scores(expanded.lower().split())
    bm25_top    = np.argsort(bm25_scores)[::-1][:top_k_dense]

    # Reciprocal Rank Fusion
    rrf:             dict = {}
    dense_score_map: dict = {}

    for rank, (idx, score) in enumerate(zip(dense_indices[0], dense_scores[0])):
        if idx != -1 and score >= SCORE_THRESHOLD:
            rrf[int(idx)]             = rrf.get(int(idx), 0.0) + 1.0 / (rank + 60)
            dense_score_map[int(idx)] = float(score)

    for rank, idx in enumerate(bm25_top):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rank + 60)

    # Guarded adjacency boost
    adjacency_bonus = 1.0 / (top_k_dense * 2 + 60)
    for idx, dense_score in dense_score_map.items():
        if dense_score >= ADJACENCY_MIN_DENSE_SCORE:
            meta = metadata[idx]
            for n_idx in get_neighbours(meta["act_key"], meta["section_number"]):
                rrf[n_idx] = rrf.get(n_idx, 0.0) + adjacency_bonus

    if not rrf:
        return [], [], [], query_vec

    sorted_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k_final]
    return (
        [chunks[i]   for i in sorted_ids],
        [metadata[i] for i in sorted_ids],
        [rrf[i]      for i in sorted_ids],
        query_vec,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_prompt(question: str, context: str) -> str:
    return f"""You are a strict Indian legal information assistant.

STRICT RULES:
1. Use ONLY the text in the CONTEXT block below. Do not use any external knowledge.
2. Every section you cite MUST appear in the CONTEXT. Include both the full act name
   and section number exactly as shown.
3. If the context covers multiple acts, cite each relevant act separately.
4. Do NOT mention the Indian Penal Code (IPC), Code of Criminal Procedure (CrPC),
   Indian Evidence Act, or any act that does NOT appear in the context.
5. If the context does not contain enough information to answer, respond ONLY with:
   "Insufficient legal information available in the provided documents."
6. Do not repeat these instructions. Do not add preamble before your answer.

FORMAT YOUR ANSWER EXACTLY AS:

1. Relevant Law / Section(s):
   [List each applicable act and section number]

2. Explanation in simple language:
   [Plain English explanation a non-lawyer can understand]

3. Example scenario:
   [Concrete example matching the user's situation]

4. Possible legal actions:
   [Step-by-step actions the user can take, citing which act each step comes from]

5. Advice / next steps:
   [Practical advice]

6. Disclaimer:
   This information is for general awareness only and does not constitute legal advice.
   Please consult a qualified advocate for advice specific to your situation.

---
CONTEXT:
{context}
---

USER QUESTION: {question}

ANSWER:"""


def build_followup_prompt(question: str, context: str, topic: str) -> str:
    return (
        f'You are LegalEase, an AI Indian legal assistant. A user previously asked about '
        f'"{topic}" and received a detailed legal analysis. The relevant legal provisions '
        f"for their case are provided below.\n\n"
        f"Answer the follow-up question naturally and helpfully, like a knowledgeable "
        f"legal advisor. Be conversational, clear, and practical. "
        f"Do not use the numbered format — just answer directly.\n\n"
        f"LEGAL CONTEXT:\n{context}\n\n"
        f"FOLLOW-UP QUESTION: {question}\n\n"
        f"ANSWER:"
    )


# ---------------------------------------------------------------------------
# Ghost citation guard
# ---------------------------------------------------------------------------
_GHOST_ACTS = re.compile(
    r"\bIPC\b|Indian Penal Code|Code of Criminal Procedure|CrPC\b|Indian Evidence Act",
    re.IGNORECASE,
)

def flag_ghost_citations(text: str) -> str:
    if _GHOST_ACTS.search(text):
        return (
            "[WARNING: Response may reference an act not present in the retrieved "
            "context. Please verify before relying on this answer.]\n\n" + text
        )
    return text


# ---------------------------------------------------------------------------
# Follow-up classifier (1 fast LLM call, ~0.5s)
# ---------------------------------------------------------------------------
def is_followup_llm(question: str, topic: str) -> bool:
    prompt = (
        f'Previous legal case topic: "{topic}"\n'
        f'New message: "{question}"\n\n'
        f"Is this new message a follow-up question about the same legal case or topic? "
        f"Reply with YES or NO only.\n\nAnswer:"
    )
    try:
        resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 5},
        )
        return "YES" in resp["message"]["content"].upper()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Query(BaseModel):
    question:   str
    session_id: Optional[str] = None


class DocRequest(BaseModel):
    doc_type: str
    fields:   dict


class ExtractRequest(BaseModel):
    doc_type:  str
    narrative: str


# ---------------------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
def chat_endpoint(query: Query):
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Question exceeds {MAX_QUESTION_LEN} characters.")

    safe_q     = sanitize(question)
    session_id = query.session_id or str(uuid4())
    _clean_sessions()

    # 1. Greeting / pure off-topic (regex, instant)
    if is_greeting(safe_q):
        return {
            "response":  _GREETING_RESPONSE,
            "sources":   [],
            "session_id": session_id,
            "doc_offer":  None,
        }

    # 2. Meaningful query check
    if not is_meaningful_legal_query(safe_q):
        return {
            "response":  _OFFTOPIC_RESPONSE,
            "sources":   [],
            "session_id": session_id,
            "doc_offer":  None,
        }

    # 3. Follow-up on existing session
    session = _sessions.get(session_id)
    if session:
        if is_followup_llm(safe_q, session["topic"]):
            prompt = build_followup_prompt(
                safe_q, session["context"], session["topic"])
            try:
                resp = ollama.chat(
                    model="llama3:8b",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3, "num_predict": 600},
                )
                answer_text = resp["message"]["content"]
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"LLM error: {exc}")
            return {
                "response":   answer_text,
                "sources":    session.get("metadata", []),
                "session_id": session_id,
                "doc_offer":  None,
            }

    # 4. New legal question: RAG
    retrieved_chunks, retrieved_meta, scores, query_vec = hybrid_retrieve(safe_q)

    if not retrieved_chunks:
        return {
            "response":  ("I could not find specific legal provisions for your query. "
                          "Please try rephrasing or describe your situation in more detail."),
            "sources":   [],
            "session_id": session_id,
            "doc_offer":  None,
        }

    context_parts = [
        f"[{m['act_name']} | Section {m['section_number']}]\n{c}"
        for c, m in zip(retrieved_chunks, retrieved_meta)
    ]
    context = "\n\n---\n\n".join(context_parts)
    prompt  = build_prompt(safe_q, context)

    try:
        resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 900},
        )
        answer_text = resp["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    answer_text = flag_ghost_citations(answer_text)

    # 5. Document offer detection (BGE + act-key scoring, ~2ms, no LLM call)
    doc_offer = detect_doc_offer(query_vec[0], safe_q, retrieved_meta)

    # 6. Store session
    _sessions[session_id] = {
        "context":    context,
        "metadata":   retrieved_meta,
        "topic":      safe_q[:150],
        "created_at": time.time(),
    }

    # Deduplicate sources
    seen, unique_sources = set(), []
    for meta in retrieved_meta:
        key = (meta["act_name"], meta["section_number"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(meta)

    return {
        "response":   answer_text,
        "sources":    unique_sources,
        "session_id": session_id,
        "doc_offer":  doc_offer,
    }


# ---------------------------------------------------------------------------
# /extract_fields endpoint
# ---------------------------------------------------------------------------
@app.post("/extract_fields")
def extract_fields(req: ExtractRequest):
    """
    Extract structured document fields from the user's narrative using the LLM.
    Called once when the user clicks 'Draft' on the action card.
    Returns a JSON object with field_name -> extracted_value pairs.
    Only known fields for the requested doc_type are returned.
    """
    if req.doc_type not in DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown doc_type: {req.doc_type}")

    field_names = [
        f["name"] for f in DOCUMENT_TYPES[req.doc_type]["fields"]
    ]

    prompt = (
        f"You are a legal document assistant. Extract structured information "
        f"from the user's situation description to pre-fill a {req.doc_type.replace('_', ' ')} form.\n\n"
        f"USER SITUATION:\n{req.narrative[:800]}\n\n"
        f"Extract ONLY information that is explicitly present in the text above. "
        f"Do NOT invent or assume any information. "
        f"If a field cannot be determined from the text, omit it.\n\n"
        f"Return a JSON object with ONLY these field names as keys:\n"
        f"{json.dumps(field_names, indent=2)}\n\n"
        f"Rules:\n"
        f"- Return ONLY valid JSON. No explanation, no markdown, no backticks.\n"
        f"- Use empty string \"\" for any field you cannot determine.\n"
        f"- Do not include fields with empty string values.\n\n"
        f"JSON:"
    )

    try:
        resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 600},
        )
        raw = resp["message"]["content"].strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",          "", raw)
        raw = raw.strip()

        # Extract first JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"fields": {}}

        extracted = json.loads(m.group(0))

        # Sanitize: keep only known field names, strip whitespace
        clean = {}
        for k, v in extracted.items():
            if k in field_names and isinstance(v, str) and v.strip():
                clean[k] = v.strip()

        return {"fields": clean}

    except json.JSONDecodeError:
        return {"fields": {}}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Field extraction failed: {exc}")


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "chunks_loaded": len(chunks),
        "acts_loaded":   list({m["act_name"] for m in metadata}),
    }


# ---------------------------------------------------------------------------
# Document generation endpoints
# ---------------------------------------------------------------------------
@app.get("/doc_types")
def get_doc_types():
    """Return available document types and their field definitions."""
    return {
        k: {
            "label":       v["label"],
            "description": v["description"],
            "fields":      v["fields"],
        }
        for k, v in DOCUMENT_TYPES.items()
    }


@app.post("/generate")
def generate_doc(req: DocRequest):
    """Generate a legal document and return it as a downloadable DOCX."""
    try:
        docx_bytes, filename = generate_document(req.doc_type, req.fields)
        if not docx_bytes or len(docx_bytes) < 100:
            raise HTTPException(
                status_code=500,
                detail="Document generation produced an empty file. "
                       "Please check your inputs and try again.")
        return Response(
            content=docx_bytes,
            media_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"),
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Document Summarizer
# ---------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    text:     str            # extracted PDF text (frontend extracts using PDF.js)
    filename: str = "document"

@app.post("/summarize")
def summarize_document(req: SummarizeRequest):
    """
    Summarize a legal document in plain language.
    Frontend extracts text from PDF using PDF.js and sends it here.
    Returns structured summary with key points, parties, dates, and plain explanation.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Document text is too short to summarize.")

    # ── Step 1: Legal document detection ─────────────────────────────────
    # Quick LLM check before running the full summary.
    # Prevents non-legal documents (study material, recipes, etc.) from being processed.
    sample = text[:1200]
    detect_prompt = (
        f"You are a legal document classifier. Read this document excerpt and determine "
        f"if it is an Indian legal document such as an FIR, court order, legal notice, "
        f"affidavit, complaint, agreement, bail application, judgement, legislation, "
        f"or any other official legal document.\n\n"
        f"DOCUMENT EXCERPT:\n{sample}\n\n"
        f"Reply with exactly one word: LEGAL or NOT_LEGAL"
    )
    try:
        detect_resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": detect_prompt}],
            options={"temperature": 0, "num_predict": 5},
        )
        verdict = detect_resp["message"]["content"].strip().upper()
        is_legal = "LEGAL" in verdict and "NOT_LEGAL" not in verdict
    except Exception:
        is_legal = True  # If detection fails, proceed with summary rather than blocking

    if not is_legal:
        return {
            "summary": None,
            "is_legal": False,
            "message": (
                "This document does not appear to be an Indian legal document. "
                "LegalEase can only summarize legal documents such as FIRs, court orders, "
                "legal notices, agreements, affidavits, complaints, and similar official legal texts.\n\n"
                "If you believe this is a legal document, please ensure the PDF contains "
                "readable text (not a scanned image) and try again."
            ),
            "filename": req.filename,
            "was_truncated": False,
            "char_count": len(text),
        }

    # ── Step 2: Full summarization ────────────────────────────────────────
    # Truncate to avoid LLM context overflow — 4000 chars is enough for a meaningful summary
    truncated = text[:4000]
    was_truncated = len(text) > 4000

    prompt = f"""You are an expert Indian legal document analyst.

A user has uploaded a legal document and needs a plain-language summary.

DOCUMENT TEXT:
{truncated}
{"[Note: Document was truncated for processing. Summary is based on the first section.]" if was_truncated else ""}

Provide a structured summary in the following format EXACTLY:

1. Document Type:
   [What kind of document is this? e.g. FIR, Legal Notice, Court Order, Agreement, Complaint]

2. Parties Involved:
   [List all named parties — complainant, accused, advocate, company, court, etc.]

3. Key Dates & References:
   [Any important dates, case numbers, section numbers, or reference IDs mentioned]

4. What This Document Is About (Plain Language):
   [2-3 sentences explaining what this document does in simple language a non-lawyer can understand]

5. Key Legal Provisions Cited:
   [List any specific sections, acts, or laws mentioned in the document]

6. Important Points to Note:
   [3-5 bullet points of the most important things the reader should know]

7. Recommended Next Steps:
   [What should the person holding this document do next?]

Keep all explanations in simple, clear English. Avoid legal jargon where possible.

SUMMARY:"""

    try:
        resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 800},
        )
        summary = resp["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    return {
        "summary":       summary,
        "is_legal":      True,
        "filename":      req.filename,
        "was_truncated": was_truncated,
        "char_count":    len(text),
    }


# ---------------------------------------------------------------------------
# Legal Section Auto-Tagger
# ---------------------------------------------------------------------------

class TagRequest(BaseModel):
    situation: str   # user's plain-language description

@app.post("/tag_sections")
def tag_sections(req: TagRequest):
    """
    Given a plain-language situation description, return the most relevant
    Indian law sections using BGE retrieval (same pipeline as /chat).
    Returns top sections with act name, section number, and brief explanation.
    """
    situation = sanitize(req.situation.strip())
    if not situation:
        raise HTTPException(status_code=400, detail="Situation description cannot be empty.")
    if len(situation) < 10:
        raise HTTPException(status_code=400, detail="Please describe your situation in more detail.")

    # Use the same hybrid retrieval pipeline
    retrieved_chunks, retrieved_meta, scores, query_vec = hybrid_retrieve(situation, top_k_final=8)

    if not retrieved_chunks:
        return {"sections": [], "message": "No specific sections found for this situation."}

    # Ask LLM to explain why each section applies
    context_parts = [
        f"[{m['act_name']} | Section {m['section_number']}]\n{c[:300]}"
        for c, m in zip(retrieved_chunks, retrieved_meta)
    ]
    context = "\n\n".join(context_parts)

    prompt = f"""You are an Indian legal expert. A person described their situation and I retrieved these potentially relevant law sections.

SITUATION: {situation}

RETRIEVED SECTIONS:
{context}

For each section that is GENUINELY relevant to this situation, provide:
- Section number and act name
- One sentence explaining WHY it applies to this situation
- Relevance level: HIGH / MEDIUM / LOW

Only include sections that actually apply. Skip irrelevant ones.

Format each as:
SECTION: [Act Name] — Section [Number]
WHY IT APPLIES: [One clear sentence]
RELEVANCE: [HIGH/MEDIUM/LOW]

List them from most to least relevant."""

    try:
        resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 600},
        )
        explanation = resp["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    # Build structured response
    sections = []
    seen = set()
    for chunk, meta, score in zip(retrieved_chunks, retrieved_meta, scores):
        key = (meta["act_name"], meta["section_number"])
        if key not in seen:
            seen.add(key)
            sections.append({
                "act_name":       meta["act_name"],
                "section_number": meta["section_number"],
                "act_key":        meta["act_key"],
                "relevance_score": round(float(score), 4),
            })

    return {
        "sections":    sections,
        "explanation": explanation,
        "situation":   situation,
    }


# ---------------------------------------------------------------------------
# Case Outcome Predictor
# ---------------------------------------------------------------------------

class OutcomeRequest(BaseModel):
    situation:  str            # description of the case
    role:       str = "victim" # "victim" or "accused"

@app.post("/predict_outcome")
def predict_outcome(req: OutcomeRequest):
    """
    Given a case description, predict likely legal outcome, timeline,
    success probability, and recommended actions.
    Uses RAG context + LLM reasoning. Clearly labeled as an estimate.
    """
    situation = sanitize(req.situation.strip())
    role      = req.role.strip().lower()

    if not situation:
        raise HTTPException(status_code=400, detail="Case description cannot be empty.")
    if len(situation) < 20:
        raise HTTPException(status_code=400, detail="Please describe your case in more detail.")

    # Retrieve relevant law sections
    retrieved_chunks, retrieved_meta, scores, query_vec = hybrid_retrieve(situation, top_k_final=5)

    context = ""
    if retrieved_chunks:
        context_parts = [
            f"[{m['act_name']} | Section {m['section_number']}]\n{c[:400]}"
            for c, m in zip(retrieved_chunks, retrieved_meta)
        ]
        context = "\n\n".join(context_parts)

    prompt = f"""You are an experienced Indian legal analyst. Analyze this case and provide an honest, realistic prediction.

CASE DESCRIPTION: {situation}
PERSPECTIVE: {role.capitalize()} seeking guidance

{"RELEVANT LAW SECTIONS:" + chr(10) + context if context else ""}

Provide a structured case outcome analysis in this EXACT format:

1. Case Strength Assessment:
   [Rate as: Strong / Moderate / Weak, with a 1-sentence reason]

2. Estimated Probability of Favorable Outcome:
   [Give a percentage range e.g. 60-75%, with brief reasoning]

3. Most Likely Legal Outcome:
   [What is the most realistic result if this case proceeds through the legal system?]

4. Typical Timeline:
   [How long does this type of case usually take in Indian courts?]

5. Key Factors That Will Determine the Outcome:
   [3-4 bullet points of what matters most]

6. Strongest Arguments Available:
   [Top 2-3 legal arguments based on the retrieved sections]

7. Risks and Challenges:
   [What could weaken this case?]

8. Recommended Immediate Actions:
   [Top 3 concrete steps to take right now]

9. Disclaimer:
   This is an AI-generated estimate based on general legal patterns and is NOT legal advice. Actual outcomes depend on evidence, jurisdiction, judge, and many other factors. Consult a qualified lawyer before taking any action.

ANALYSIS:"""

    try:
        resp = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 900},
        )
        analysis = resp["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    # Extract sections for display
    sections = []
    seen = set()
    for meta in retrieved_meta:
        key = (meta["act_name"], meta["section_number"])
        if key not in seen:
            seen.add(key)
            sections.append({
                "act_name":       meta["act_name"],
                "section_number": meta["section_number"],
            })

    return {
        "analysis":  analysis,
        "sections":  sections,
        "situation": situation,
        "role":      role,
    }


# ---------------------------------------------------------------------------
# Legal Notice by Email (Resend API)
# ---------------------------------------------------------------------------
# Set your Resend API key in .env or environment:
#   RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxx
# Get a free key at resend.com — 3,000 emails/month free

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")

# From address is hardcoded to the only valid sandbox address on Resend's free plan.
# Once you verify a custom domain at resend.com/domains, change this line directly.
RESEND_FROM_EMAIL = "onboarding@resend.dev"


class EmailRequest(BaseModel):
    doc_type:         str
    fields:           dict
    recipient_email:  str
    recipient_name:   str = ""
    sender_name:      str = ""


@app.post("/send_email")
def send_legal_notice_email(req: EmailRequest):
    """
    Generate a legal document and send it as a DOCX attachment via Resend API.

    Requirements:
      - pip install requests
      - RESEND_API_KEY set in .env
      - RESEND_FROM_EMAIL defaults to onboarding@resend.dev (free sandbox)
    """
    try:
        import requests as _requests
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Missing dependency: run 'pip install requests' and restart the server."
        )
    import base64

    if not RESEND_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Email service not configured. Set RESEND_API_KEY in your .env file."
        )

    # Validate email
    if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", req.recipient_email):
        raise HTTPException(status_code=400, detail="Invalid recipient email address.")

    if req.doc_type not in DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown document type: {req.doc_type}")

    # Generate document
    try:
        docx_bytes, filename = generate_document(req.doc_type, req.fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document generation failed: {e}")

    if not docx_bytes or len(docx_bytes) < 100:
        raise HTTPException(status_code=500, detail="Document generation produced an empty file.")

    # Encode attachment as base64
    doc_b64      = base64.b64encode(docx_bytes).decode("utf-8")
    doc_type_cfg = DOCUMENT_TYPES[req.doc_type]
    doc_label    = doc_type_cfg["label"]

    # Compose email
    sender_display = req.sender_name or "LegalEase User"
    recip_display  = req.recipient_name or req.recipient_email
    subject        = f"Legal Notice — {doc_label} | LegalEase"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#f7f9fc;padding:32px 24px;">
      <div style="background:linear-gradient(135deg,#1a2a44,#223558);border-radius:14px;padding:28px 24px;text-align:center;margin-bottom:24px;">
        <div style="font-size:22px;font-weight:800;color:white;font-family:Georgia,serif;">
          Legal<span style="color:#e6a817;">Ease</span>
        </div>
        <div style="font-size:13px;color:rgba(255,255,255,0.6);margin-top:6px;">AI-Powered Indian Legal Guidance</div>
      </div>
      <div style="background:white;border-radius:12px;padding:24px;margin-bottom:16px;border:1px solid #e2e8f0;">
        <p style="font-size:15px;color:#1a2236;margin-bottom:12px;">Dear {recip_display},</p>
        <p style="font-size:14px;color:#374151;line-height:1.7;margin-bottom:12px;">
          You have received a <strong>{doc_label}</strong> from <strong>{sender_display}</strong> via LegalEase.
        </p>
        <p style="font-size:14px;color:#374151;line-height:1.7;margin-bottom:16px;">
          Please find the document attached to this email. This is a formally drafted legal document
          generated under the applicable provisions of Indian law. You are advised to read it carefully
          and respond within the stipulated time period if applicable.
        </p>
        <div style="background:#fef9ee;border:1px solid rgba(230,168,23,0.3);border-left:3px solid #e6a817;border-radius:8px;padding:12px 16px;font-size:13px;color:#92400e;">
          ⚠ This document was generated using LegalEase AI and is intended as a draft.
          Please consult a qualified advocate before relying on this document in official proceedings.
        </div>
      </div>
      <div style="text-align:center;font-size:12px;color:#9aabb8;margin-top:16px;">
        Sent via LegalEase · AI Legal Guidance for India<br>
        This is an automated email. Do not reply directly to this message.
      </div>
    </div>
    """

    text_body = (
        f"Dear {recip_display},\n\n"
        f"You have received a {doc_label} from {sender_display} via LegalEase.\n\n"
        f"Please find the document attached.\n\n"
        f"Note: This document was generated using LegalEase AI. "
        f"Please consult a qualified advocate before using it in official proceedings.\n\n"
        f"Sent via LegalEase — AI Legal Guidance for India"
    )

    payload = {
        "from":    RESEND_FROM_EMAIL,
        "to":      [req.recipient_email],
        "subject": subject,
        "html":    html_body,
        "text":    text_body,
        "attachments": [
            {
                "filename": filename,
                "content":  doc_b64,
            }
        ],
    }

    try:
        resp = _requests.post(
            "https://api.resend.com/emails",
            json=payload,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type":  "application/json",
                "User-Agent":    "LegalEase/1.0",
            },
            timeout=20,
        )

        if resp.status_code == 200 or resp.status_code == 201:
            result = resp.json()
            return {
                "ok":       True,
                "email_id": result.get("id", ""),
                "message":  f"Document sent successfully to {req.recipient_email}",
            }

        # Parse Resend error
        try:
            err = resp.json()
            detail = err.get("message", err.get("error", resp.text))
        except Exception:
            detail = resp.text

        # Give a helpful message for common errors
        if resp.status_code == 403:
            detail = (
                "Resend 403 — two possible causes: "
                "(1) Wrong from address: your .env RESEND_FROM_EMAIL must be exactly 'onboarding@resend.dev' (no quotes, no display name) on the free plan. "
                "(2) Sandbox recipient restriction: on the free plan you can only send TO your own Resend account email address. "
                "To send to anyone, verify a custom domain at resend.com/domains."
            )
        elif resp.status_code == 401:
            detail = "Invalid RESEND_API_KEY. Check your .env file."
        elif resp.status_code == 422:
            detail = f"Invalid request: {detail}"

        raise HTTPException(status_code=502, detail=f"Email delivery failed ({resp.status_code}): {detail}")

    except _requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Email service timed out. Please try again.")
    except _requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Could not connect to Resend API. Check your internet connection.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Email service error: {e}")


@app.get("/news")
def get_legal_news():
    """
    Fetch latest Indian legal news from public RSS feeds.
    Tries LiveLaw first, then Bar & Bench as fallback.
    Returns up to 8 news items as JSON.
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    feeds = [
        ("LiveLaw",      "https://www.livelaw.in/rss/news"),
        ("Bar & Bench",  "https://www.barandbench.com/feed"),
    ]

    for source_name, url in feeds:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "LegalEase/1.0 (legal news aggregator)"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()

            root    = ET.fromstring(xml_data)
            channel = root.find("channel")
            if channel is None:
                continue

            items = []
            for item in channel.findall("item")[:8]:
                title   = (item.findtext("title",       "") or "").strip()
                link    = (item.findtext("link",        "") or "").strip()
                desc    = (item.findtext("description", "") or "").strip()
                pub     = (item.findtext("pubDate",     "") or "").strip()
                # Strip HTML tags from description
                desc    = re.sub(r"<[^>]+>", "", desc)[:180].strip()
                if title and link:
                    items.append({
                        "title":  title,
                        "link":   link,
                        "desc":   desc,
                        "date":   pub,
                        "source": source_name,
                    })

            if items:
                return {"items": items, "ok": True}

        except Exception:
            continue

    return {"items": [], "ok": False}

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)