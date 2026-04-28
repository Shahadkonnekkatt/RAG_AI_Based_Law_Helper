"""
ingest.py — Multi-act legal RAG ingestion pipeline.

Supported acts (place trimmed PDFs in the data/ folder):
  BNS.pdf                  Bharatiya Nyaya Sanhita, 2023            s.1-358
  BNSS.pdf                 Bharatiya Nagarik Suraksha Sanhita, 2023 s.1-531
  BSA.pdf                  Bharatiya Sakshya Adhiniyam, 2023        s.1-170
  CONSUMER ACT.pdf         Consumer Protection Act, 2019            s.1-107
  CONTRACT ACT.pdf         Indian Contract Act, 1872                s.1-238
  IT ACT.pdf               Information Technology Act, 2000         s.1-94+lettered
  MOTOR VEHICLES ACT.pdf   Motor Vehicles Act, 1988                 s.1-217+lettered

Run:
    python ingest.py

Outputs (in vectorstore/):
    index.faiss     FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
    chunks.pkl      list[str]  — raw chunk texts
    metadata.pkl    list[dict] — section_number, act_name, act_key, source per chunk

Parser design:
    - No lookahead limit: accumulates title lines until EITHER an em-dash is
      found (real section) OR the next valid section number appears without any
      em-dash (= TOC stub → silently discard).
    - Section numbers are alphanumeric strings: "66", "66A", "215B" etc.
    - Sections whose body < MIN_BODY_LEN chars or starts with "[Omitted" are discarded.
    - Each stored chunk is prefixed with "[Act Name | Section N]" so that act
      name and section number are always present for BM25 term matching.
"""

import os
import re
import pickle

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = "data"
VECTOR_PATH = "vectorstore"

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
DOC_PREFIX   = "Represent this legal passage for retrieval: "
BATCH_SIZE   = 32
MIN_BODY_LEN = 80   # characters — sections shorter than this are stub noise

# ---------------------------------------------------------------------------
# Act configuration
# ---------------------------------------------------------------------------
# key         : short identifier used in compound adjacency dict keys
# name        : full act name cited verbatim in LLM answers
# max_section : highest *numeric* base section (lettered sections share the
#               same numeric base, so 66A's base is 66 which is ≤ 94)
ACT_CONFIG = {
    "BNS.pdf": {
        "key":  "BNS",
        "name": "Bharatiya Nyaya Sanhita, 2023",
        "max_section": 358,
    },
    "BNSS.pdf": {
        "key":  "BNSS",
        "name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "max_section": 531,
    },
    "BSA.pdf": {
        "key":  "BSA",
        "name": "Bharatiya Sakshya Adhiniyam, 2023",
        "max_section": 170,
    },
    "CONSUMER ACT.pdf": {
        "key":  "CONSUMER",
        "name": "Consumer Protection Act, 2019",
        "max_section": 107,
    },
    "CONTRACT ACT.pdf": {
        "key":  "CONTRACT",
        "name": "Indian Contract Act, 1872",
        "max_section": 238,
    },
    "IT ACT.pdf": {
        "key":  "IT",
        "name": "Information Technology Act, 2000",
        "max_section": 94,
    },
    "MOTOR VEHICLES ACT.pdf": {
        "key":  "MV",
        "name": "Motor Vehicles Act, 1988",
        "max_section": 217,
    },
}

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
# Matches a section-start line: optional spaces, 1-3 digits, optional SINGLE
# uppercase letter (66A, 215B), a dot, whitespace, then at least one non-space.
SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]?)\.\s+\S")

# Any unicode dash variant used as title/body separator in Indian legal PDFs
DASH_RE = re.compile(r"[\u2013\u2014\u2015\u2012]|(?:--)")

# Omitted section body patterns
OMITTED_RE = re.compile(r"^\[?omitted", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Section sort key
# ---------------------------------------------------------------------------
def sec_sort_key(sec_id: str):
    """Sort "1","2","66","66A","66B","67","215A" in correct legal order."""
    m = re.match(r"^(\d+)([A-Z]?)$", sec_id)
    if m:
        return (int(m.group(1)), m.group(2))
    return (0, sec_id)


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------
def parse_act(pdf_path: str, cfg: dict) -> dict:
    """
    Parse one act PDF.  Returns dict { sec_id : metadata_dict }.

    State machine:
        SCAN   → looking for a valid section-start line
        TITLE  → found section number, accumulating title lines until em-dash
        BODY   → past the em-dash, accumulating body until next section start
    """
    reader   = PdfReader(pdf_path)
    max_num  = cfg["max_section"]
    act_name = cfg["name"]
    act_key  = cfg["key"]
    source   = os.path.basename(pdf_path)

    # Flatten all pages to a line list, stripping blank lines
    lines = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        for ln in raw.split("\n"):
            ln = ln.strip()
            if ln:
                lines.append(ln)

    # ------------------------------------------------------------------ #
    # Helper: is this line a valid section start for this act?
    # ------------------------------------------------------------------ #
    def is_section_start(line):
        m = SECTION_RE.match(line)
        if not m:
            return False, None
        sec_id = m.group(1)
        base   = int(re.match(r"^(\d+)", sec_id).group(1))
        if 1 <= base <= max_num:
            return True, sec_id
        return False, None

    sections = {}
    i        = 0
    n        = len(lines)

    while i < n:
        ok, sec_id = is_section_start(lines[i])
        if not ok:
            i += 1
            continue

        # ---- TITLE phase ------------------------------------------------
        # Accumulate lines (including the first one) until we find an em-dash
        # OR hit the next valid section (= TOC stub).
        title_lines = [lines[i]]
        has_dash    = bool(DASH_RE.search(lines[i]))
        j           = i + 1

        while not has_dash and j < n:
            nxt_ok, _ = is_section_start(lines[j])
            if nxt_ok:
                # Next section appeared before any dash → this was a TOC stub
                break
            title_lines.append(lines[j])
            if DASH_RE.search(lines[j]):
                has_dash = True
            j += 1

        if not has_dash:
            # TOC stub — skip to wherever j landed
            i = j
            continue

        # ---- BODY phase -------------------------------------------------
        body_lines = []
        while j < n:
            nxt_ok, _ = is_section_start(lines[j])
            if nxt_ok:
                break
            body_lines.append(lines[j])
            j += 1

        # Reconstruct full text and isolate body (after first dash)
        full_text  = " ".join(title_lines + body_lines)
        dash_match = DASH_RE.search(full_text)
        body       = full_text[dash_match.end():].strip()

        # Discard [Omitted] sections
        if OMITTED_RE.match(body):
            i = j
            continue

        # Discard stubs too short to be useful
        if len(body) < MIN_BODY_LEN:
            i = j
            continue

        # Build chunk text — prefix carries act name + section so BM25
        # always sees those terms even for vague queries
        chunk = f"[{act_name} | Section {sec_id}]\n{full_text.strip()}"

        sections[sec_id] = {
            "chunk":          chunk,
            "section_number": sec_id,
            "act_name":       act_name,
            "act_key":        act_key,
            "source":         source,
        }

        i = j

    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(VECTOR_PATH, exist_ok=True)

    # Find PDFs in data/ that match ACT_CONFIG
    try:
        found_files = os.listdir(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: data/ folder not found. Create it and place trimmed PDFs inside.")
        return

    pdf_files = sorted([f for f in found_files if f.endswith(".pdf") and f in ACT_CONFIG])

    if not pdf_files:
        print(f"No recognised PDFs in {DATA_PATH}/")
        print(f"Expected one or more of: {list(ACT_CONFIG.keys())}")
        return

    unknown = [f for f in found_files if f.endswith(".pdf") and f not in ACT_CONFIG]
    if unknown:
        print(f"Warning — ignoring unrecognised PDFs: {unknown}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks   = []
    all_metadata = []
    act_counts   = {}

    for filename in pdf_files:
        cfg  = ACT_CONFIG[filename]
        path = os.path.join(DATA_PATH, filename)
        print(f"\nParsing {filename} ...")
        sections = parse_act(path, cfg)
        count    = len(sections)
        act_counts[cfg["name"]] = count
        print(f"  → {count} sections kept")

        if count == 0:
            print(f"  WARNING: 0 sections extracted. Check that the PDF is trimmed "
                  f"and that section format matches expected pattern.")

        # Keep sections in correct legal order so adjacency indexing is stable
        for sec_id in sorted(sections.keys(), key=sec_sort_key):
            d = sections[sec_id]
            all_chunks.append(d["chunk"])
            all_metadata.append({
                "section_number": d["section_number"],
                "act_name":       d["act_name"],
                "act_key":        d["act_key"],
                "source":         d["source"],
            })

    total = len(all_chunks)
    if total == 0:
        print("\nNo sections extracted. Aborting.")
        return

    print(f"\n{'─'*60}")
    print(f"Total sections across all acts: {total}")
    print("Sections per act:")
    for name, cnt in act_counts.items():
        print(f"  {cnt:4d}  {name}")
    print(f"{'─'*60}")

    print("\nGenerating embeddings (may take 1–3 minutes) ...")
    embeddings = model.encode(
        [DOC_PREFIX + c for c in all_chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_PATH, "index.faiss"))
    with open(os.path.join(VECTOR_PATH, "chunks.pkl"),   "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(VECTOR_PATH, "metadata.pkl"), "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"\nVectorstore saved → {VECTOR_PATH}/")
    print(f"  index.faiss  :  {total} vectors, dim={dim}")
    print("Done.")


if __name__ == "__main__":
    main()