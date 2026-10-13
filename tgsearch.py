# /// script
# requires-python = ">=3.11"
# dependencies = ["telethon", "ocrmac", "qrcode", "iterfzf", "rapidfuzz", "fastapi", "uvicorn",
#                 "torch", "transformers>=5", "numpy", "pillow", "imagehash", "tqdm"]
# ///
"""Index photos (and images sent as files) from selected Telegram chats (OCR via Apple Vision) and search them.

  export TG_API_ID=... TG_API_HASH=...   # from https://my.telegram.org -> API development tools
  uv run tgsearch.py login                 # QR login, once (login codes often never arrive for API apps)
  uv run tgsearch.py chats                 # pick chats in fzf (multi-select), then sync them
  uv run tgsearch.py sync <id|name>...     # download + OCR new photos/image files (resumable)
  uv run tgsearch.py sync                  # same for every chat already indexed
  uv run tgsearch.py search "пароль" --from Вася --after 2024-01-01 --open
  uv run tgsearch.py search --meaning "собака в снегу"   # by what's in the picture (embeddings)
  uv run tgsearch.py embed [--model HF_ID]   # vectors for downloaded photos still missing them (sync does this too)
  uv run tgsearch.py dedup                 # merge copies of the same picture already indexed (sync does this for new ones)
  uv run tgsearch.py repair [<id|name>...] # mark messages deleted in Telegram, update captions, re-download lost files
  uv run tgsearch.py serve                 # web UI on http://127.0.0.1:8000 (build it first: cd web && bun run build)
"""
import argparse
import asyncio
import functools
import os
import sqlite3
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = ["ru-RU", "en-US"]
# M1: OCR alone is ~0.18 s/photo, ~0.08 with 4 threads; downloads (~1.1 s each) are the bottleneck.
# Raise DOWNLOADS if Telegram doesn't answer with FloodWaits.
DOWNLOADS, OCR_THREADS = 8, 4
STALL = 120  # s without a saved photo -> reconnect and resume (Telethon has no request timeouts)
# Embedding model for search by meaning: any transformers CLIP/SigLIP-family id (get_image/text_features).
# Vectors are stored per model, so several can coexist; `embed --model X` adds X for already downloaded photos.
MODEL = "google/siglip2-base-patch16-256"
MODEL_LOCK = threading.Lock()  # one model load / GPU call at a time: the web server runs requests in threads
DUP_HASH_BITS = 6  # pHash distance for duplicate *candidates*; same_picture() decides


def norm(s):
    return (s or "").replace("ё", "е").replace("Ё", "Е")


def link(chat_id, msg_id):
    if str(chat_id).startswith("-100"):  # channel/supergroup: official link to the message, works for members
        return f"https://t.me/c/{str(chat_id)[4:]}/{msg_id}"
    if chat_id > 0:  # private chat: Telegram has no message links there, so open the chat (mention-style link)
        return f"tg://user?id={chat_id}"
    # ponytail: basic groups have no id-based link at all; undocumented openmessage works on some clients only
    return f"tg://openmessage?chat_id={-chat_id}&message_id={msg_id}"


# is this media row's message gone from Telegram (as of the last `repair`)?
DELETED = "EXISTS (SELECT 1 FROM deleted d WHERE d.chat_id = media.chat_id AND d.msg_id = media.msg_id)"
# search rows: these, then a snippet
COLS = "date, chat, sender, path, tglink(chat_id, msg_id), ocr, caption, chat_id, msg_id, image, " + DELETED
IMAGE = 9  # index of `image` in a search row
COL_OF = {"ocr": 5, "caption": 6}  # ...and of the searchable texts

# A picture is stored, OCR'd and embedded once (`images`, `emb`); every message it was sent in is a `media` row
# (an appearance) pointing at it, carrying a copy of its OCR text so FTS can search appearances directly.
# images.photo_id is Telegram's photo or document id (an image sent as a file): both random 64-bit, no clashes.
SCHEMA = [
    "CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, path TEXT, phash TEXT, photo_id INTEGER UNIQUE)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS media USING fts5(ocr, caption, sender, chat,
        chat_id UNINDEXED, msg_id UNINDEXED, date UNINDEXED, path UNINDEXED, image UNINDEXED, tokenize='trigram')""",
    "CREATE TABLE IF NOT EXISTS emb (model TEXT, image INTEGER, vec BLOB, PRIMARY KEY (model, image)) WITHOUT ROWID",
    # Telegram albums: each photo is its own message, sharing grouped_id. Row present = checked, NULL = no album.
    """CREATE TABLE IF NOT EXISTS albums (chat_id INTEGER, msg_id INTEGER, grouped_id INTEGER,
        PRIMARY KEY (chat_id, msg_id)) WITHOUT ROWID""",
    "CREATE INDEX IF NOT EXISTS albums_group ON albums (chat_id, grouped_id)",
    # Chats whose image files older than their newest indexed message were looked for: sync used to fetch
    # photos only, and its resume point (newest indexed message) would never reach back to those files.
    "CREATE TABLE IF NOT EXISTS files_scanned (chat_id INTEGER PRIMARY KEY)",
    # messages `repair` found deleted in Telegram; their rows (and files) stay searchable
    "CREATE TABLE IF NOT EXISTS deleted (chat_id INTEGER, msg_id INTEGER, PRIMARY KEY (chat_id, msg_id)) WITHOUT ROWID",
]


def connect(path=HERE / "media.db"):
    # WAL: searches (`serve`) read a consistent snapshot while `sync`/`embed`/`dedup` commit, instead of
    # taking turns; two writers queue, waiting up to 60 s (dedup's merge is one multi-second transaction).
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.create_function("tglink", 2, link, deterministic=True)
    if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'media'").fetchone() and \
            "image" not in [c[0] for c in db.execute("SELECT * FROM media LIMIT 0").description]:
        migrate(db, path)
    for sql in SCHEMA:
        db.execute(sql)
    return db


def migrate(db, path):
    """Pre-dedup layout (row and vector per message) -> one image per message; `dedup` then merges copies."""
    if str(path) != ":memory:":
        with sqlite3.connect(f"{path}.bak-pre-dedup") as bak:
            db.backup(bak)  # unlike a file copy, includes anything still in the WAL file
        print(f"migrating index to images + appearances (backup: {path}.bak-pre-dedup)")
    db.execute("BEGIN")  # all or nothing
    db.execute("ALTER TABLE media RENAME TO media_old")
    has_emb = db.execute("SELECT 1 FROM sqlite_master WHERE name = 'emb'").fetchone()
    if has_emb:
        db.execute("ALTER TABLE emb RENAME TO emb_old")
    for sql in SCHEMA:
        db.execute(sql)
    image_of = {}
    for rowid, chat_id, msg_id, path_ in db.execute("SELECT rowid, chat_id, msg_id, path FROM media_old").fetchall():
        image_of[chat_id, msg_id] = img = db.execute("INSERT INTO images (path) VALUES (?)", (path_,)).lastrowid
        db.execute("""INSERT INTO media SELECT ocr, caption, sender, chat, chat_id, msg_id, date, path, ?
                      FROM media_old WHERE rowid = ?""", (img, rowid))
    if has_emb:
        db.executemany("INSERT INTO emb VALUES (?,?,?)",
                       [(m, image_of[c, i], v) for m, c, i, v in db.execute("SELECT * FROM emb_old") if (c, i) in image_of])
        db.execute("DROP TABLE emb_old")
    db.execute("DROP TABLE media_old")
    db.commit()


def add(db, chat_id, chat, msg_id, date, sender, caption, ocr, path, image=None):
    """Record one appearance; image=None -> it's a new picture."""
    with db:
        image = image or db.execute("INSERT INTO images (path) VALUES (?)", (str(path),)).lastrowid
        db.execute("INSERT INTO media VALUES (?,?,?,?,?,?,?,?,?)",
                   (norm(ocr), norm(caption), norm(sender), norm(chat), chat_id, msg_id, date, str(path), image))


def first_per_image(items, image_of, limit=None):
    """Best-ranked appearance of each picture, in rank order, at most `limit` pictures."""
    seen, out = set(), []
    for it in items:
        if (img := image_of(it)) not in seen:
            seen.add(img)
            out.append(it)
            if len(out) == limit:
                break
    return out


def appearances(db, images):
    """image -> [(date, chat, sender, link, msg_id, deleted)], oldest first: every message a picture was sent in."""
    out = {}
    # ponytail: `image` is UNINDEXED, so this scans all appearances (ms for 10k); index table if it grows a lot
    q = f"SELECT image, date, chat, sender, tglink(chat_id, msg_id), msg_id, {DELETED} FROM media WHERE image IN ({','.join('?' * len(images))})"
    for img, *rest in db.execute(q + " ORDER BY date", list(images)):
        out.setdefault(img, []).append(tuple(rest))
    return out


ORDERS = ("new", "old", "relevance")


def album_sizes(db, keys):
    """(chat_id, msg_id) -> number of indexed photos in its album, for messages that are part of one."""
    if not keys:
        return {}
    q = f"""SELECT a.chat_id, a.msg_id, (SELECT count(*) FROM albums b WHERE b.chat_id = a.chat_id
            AND b.grouped_id = a.grouped_id) FROM albums a WHERE a.grouped_id IS NOT NULL
            AND (a.chat_id, a.msg_id) IN (VALUES {','.join(['(?,?)'] * len(keys))})"""
    return {(c, m): n for c, m, n in db.execute(q, [x for k in keys for x in k])}


def search(*args, **kw):
    """search_page() without the total: just the rows."""
    return search_page(*args, **kw)[0]


def search_page(db, text=None, caption=None, sender=None, chat=None, after=None, before=None, limit=20,
                min_score=None, model=MODEL, similar=None, msg_id=None, order="new", album=None, meaning=None, ocr=None,
                offset=0, chat_id=None):
    """(rows `offset`..`offset + limit`, total pictures matching). One row per picture; every given criterion applies. Text queries decide what matches: `ocr` in the
    picture's text, `caption` in the message's, `text` in either; fuzzy (typos, OCR errors) if min_score is
    set, else exact substring. `meaning` (a description) and `similar` (an image id) score what's in the
    picture via `model`'s embeddings and decide relevance when given.
    order: "new"/"old" = by message date (a picture sent several times counts by its newest/oldest matching
    appearance; with meaning/similar, which match everything a bit, it's the `limit` best, sorted by date),
    "relevance" = best match first. album=(chat_id or None, msg_id): only the photos of that message's album,
    in message order."""
    said = [(q, cols) for q, cols in ((text, ("ocr", "caption")), (ocr, ("ocr",)), (caption, ("caption",))) if q]
    vector = bool(meaning or similar)
    fuzzy = bool(said and min_score is not None)
    quote = lambda v: '"' + norm(v).replace('"', '""') + '"'
    # ponytail: trigram index -> exact queries shorter than 3 chars match nothing
    terms = ([f"{{{' '.join(cols)}}} : {quote(q)}" for q, cols in said if not fuzzy] +
             [f"{col} : {quote(v)}" for col, v in (("sender", sender), ("chat", chat)) if v])
    where, params = (["model = ?"], [model]) if vector else ([], [])
    if terms:
        where.append("media MATCH ?")
        params.append(" AND ".join(terms))
    if after:
        where.append("date >= ?")
        params.append(after)
    if before:
        where.append("date < ?")
        params.append(before)
    if msg_id:
        where.append("msg_id = ?")
        params.append(msg_id)
    if chat_id:  # exact chat; `chat` is a name substring (names repeat: a group upgraded to a supergroup gets a new id)
        where.append("chat_id = ?")
        params.append(chat_id)
    if album:
        where.append("""(chat_id, msg_id) IN (SELECT a.chat_id, a.msg_id FROM albums a JOIN albums b
            ON a.chat_id = b.chat_id AND a.grouped_id = b.grouped_id WHERE b.msg_id = ? AND (? IS NULL OR b.chat_id = ?))""")
        params += [album[1], album[0], album[0]]
    where = " WHERE " + " AND ".join(where) if where else ""
    if vector or fuzzy:
        return rank(db, where, params, said, min_score, model, meaning, similar, limit, offset, order)
    # best appearance per picture first, then snippets for just those rows (snippet() can't run in a grouped query)
    by = {"new": "date DESC, msg_id DESC", "old": "date, msg_id",
          "relevance": "rank, date DESC" if terms else "date DESC, msg_id DESC"}[order]
    if album:
        by = "msg_id"  # an album reads in the order it was sent
    best = first_per_image(db.execute(f"SELECT rowid, image FROM media{where} ORDER BY {by}", params), lambda r: r[1])
    ids = [r[0] for r in best[offset:offset + limit]]
    if not ids:
        return [], len(best)
    snip = "snippet(media, -1, '\x1b[1m', '\x1b[0m', '…', 64)" if terms else "substr(ocr, 1, 60)"
    where += (" AND " if where else " WHERE ") + f"rowid IN ({','.join('?' * len(ids))})"
    return db.execute(f"SELECT {COLS}, {snip} FROM media{where} ORDER BY {by}", (*params, *ids)).fetchall(), len(best)


def rank(db, where, params, said, min_score, model, meaning, similar, limit, offset, order):
    """search() for fuzzy text and/or picture vectors; exact text is already a filter in `where`."""
    import numpy as np
    from rapidfuzz import fuzz, process

    queries = [f"similarity to #{similar}"] * bool(similar) + ["meaning similarity"] * bool(meaning)
    rows = db.execute(f"SELECT {COLS}{', vec' if queries else ''} FROM media"
                      f"{' JOIN emb USING (image)' if queries else ''}{where}", params).fetchall()
    if not rows:
        return [], 0
    keep, rel, shown = set(range(len(rows))), np.zeros(len(rows)), []
    for query, cols in said:  # each text query must match (fuzzy: score >= min_score); scores add up
        q = norm(query).lower()
        # partial_ratio searches the shorter string inside the longer one, so a doc shorter than the query
        # (e.g. "" for a textless image) would "contain itself" in the query and score 100. Padding with \0
        # (matches nothing) keeps the query as the needle.
        docs = {c: [(r[COL_OF[c]] or "").lower().ljust(len(q), "\0") for r in rows] for c in cols}
        scores = {}
        if min_score is not None:
            # ponytail: scans every row passing the other filters (fine for ~100k); prefilter with a trigram OR-query if slow
            for c in cols:  # best column, not their concatenation: no matches across the seam, no lost edge matches
                for _, score, i in process.extract(q, docs[c], scorer=fuzz.partial_ratio, score_cutoff=min_score,
                                                   limit=None):
                    scores[i] = max(scores.get(i, 0), score)
            keep &= scores.keys()
            rel[list(scores)] += list(scores.values())
        shown.append((q, cols, docs, scores))
    if queries:  # ponytail: brute-force cosine over every vector passing the filters (ms for 10k); faiss near ~1M
        vecs = []
        if similar:
            row = db.execute("SELECT vec FROM emb WHERE model = ? AND image = ?", (model, similar)).fetchone()
            if not row:
                return [], 0
            vecs.append(np.frombuffer(row[0], np.float16))
        if meaning:
            vecs.append(encode(model, text=meaning)[0])
        lib = np.frombuffer(b"".join(r[-1] for r in rows), np.float16).reshape(len(rows), -1)
        sims = lib.astype(np.float32) @ np.stack(vecs).astype(np.float32).T
        rows = [r[:-1] for r in rows]
        # look-alike similarities run ~0.3-1, text-to-picture ~0-0.2: z-score each before adding them up
        rel = ((sims - sims.mean(0)) / (sims.std(0) + 1e-9)).sum(1) if len(vecs) > 1 else sims[:, 0]
    # best first (picture score if any, else the fuzzy one); ties and each picture's appearance by date, per `order`
    idx = sorted(keep, key=lambda i: rows[i][0], reverse=order != "old")
    if queries or order == "relevance":
        idx.sort(key=lambda i: -rel[i])  # stable
    best = first_per_image(idx, lambda i: rows[i][IMAGE])
    top = best[offset:offset + limit]
    if queries and order != "relevance":  # everything matches a picture query a bit: each page = the next best, by date
        top.sort(key=lambda i: rows[i][0], reverse=order == "new")
    out = []
    for i in top:
        snip = []
        for q, cols, docs, scores in shown:
            a, c = max(((fuzz.partial_ratio_alignment(q, docs[c][i]), c) for c in cols), key=lambda x: x[0].score)
            t, s, e = rows[i][COL_OF[c]] or "", a.dest_start, a.dest_end
            snip.append(f"…{t[max(0, s - 40):s]}\x1b[1m{t[s:e]}\x1b[0m{t[e:e + 40]}…" +
                        (f" ({scores[i]:.0f}%)" if i in scores else ""))
        snip += [f"{sims[i, j]:.3f} {label}" for j, label in enumerate(queries)]
        out.append((*rows[i], " · ".join(snip)))
    return out, len(best)


@functools.cache
def load_model(name):
    import torch
    from transformers import AutoModel, AutoProcessor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(name, dtype=torch.float16 if dev == "mps" else torch.float32)
    return model.to(dev).eval(), AutoProcessor.from_pretrained(name), dev


def encode(name, images=None, text=None):
    """Unit-length float16 vectors for a batch of PIL images, or for text (one query or a list)."""
    import torch

    with MODEL_LOCK, torch.inference_mode():
        model, proc, dev = load_model(name)
        if images is not None:
            out = model.get_image_features(**proc(images=images, return_tensors="pt").to(dev))
        else:  # SigLIP was trained on lowercased, max_length-padded text; CLIP is fine with both
            texts = [text] if isinstance(text, str) else text
            out = model.get_text_features(**proc(text=[t.lower() for t in texts], padding="max_length",
                                                  max_length=64, return_tensors="pt").to(dev))
        out = getattr(out, "pooler_output", out)  # transformers 5 returns a model output, 4.x a tensor
        return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy().astype("float16")


TAG_SPACES = {}  # model -> (labels, label vectors, offset), built on first use


def tags(db, images, model=MODEL, k=4):
    """image -> the k labels from tags.txt the model finds closest to it (zero-shot, from stored vectors).
    Raw closeness made a few labels ("meme", "chat screenshot") top almost everything, so each label's mean
    closeness over the library is half subtracted; fully subtracting it surfaced odd ones on 5.3k photos."""
    import numpy as np

    if not db.execute("SELECT 1 FROM emb WHERE model = ? LIMIT 1", (model,)).fetchone():
        return {}  # nothing embedded with this model (yet)
    if model not in TAG_SPACES:
        labels = [ln.strip() for ln in (HERE / "tags.txt").read_text().splitlines() if ln.strip() and ln[0] != "#"]
        vecs = encode(model, text=labels).astype(np.float32)
        lib = [v for (v,) in db.execute("SELECT vec FROM emb WHERE model = ?", (model,))]
        lib = np.frombuffer(b"".join(lib), np.float16).reshape(len(lib), -1).astype(np.float32)
        TAG_SPACES[model] = labels, vecs, 0.5 * (lib @ vecs.T).mean(0)
    labels, vecs, offset = TAG_SPACES[model]
    q = f"SELECT image, vec FROM emb WHERE model = ? AND image IN ({','.join('?' * len(images))})"
    return {img: [labels[j] for j in np.argsort(-(np.frombuffer(v, np.float16).astype(np.float32) @ vecs.T - offset))[:k]]
            for img, v in db.execute(q, (model, *images))}


def embed(db, name=MODEL, batch=32):
    from tqdm import tqdm

    skipped = []

    def load(path):
        try:
            return open_image(path).convert("RGB")
        except OSError:  # missing/corrupt file: skip it, don't block the rest
            skipped.append(path)

    rows = db.execute("""SELECT id, path FROM images WHERE NOT EXISTS (SELECT 1 FROM emb
        WHERE model = ? AND emb.image = images.id)""", (name,)).fetchall()
    if not rows:
        return
    print(f"== embedding {len(rows)} photos with {name}")
    t0 = time.time()
    try:
        with tqdm(total=len(rows), unit="pic") as bar:
            for i in range(0, len(rows), batch):
                chunk = [(image, img) for image, p in rows[i:i + batch] if (img := load(p))]
                if chunk:
                    vecs = encode(name, images=[img for _, img in chunk])
                    with db:  # committed per batch: Ctrl+C or a crash loses at most one batch
                        db.executemany("INSERT INTO emb VALUES (?,?,?)",
                                       [(name, image, v.tobytes()) for (image, _), v in zip(chunk, vecs)])
                bar.update(len(rows[i:i + batch]))
    except KeyboardInterrupt:
        return print("stopped, run embed again to resume")
    print(f"  done in {time.time() - t0:.0f}s" + (f", {len(skipped)} unreadable skipped" if skipped else ""))


def open_image(path):
    """Image.open, with Pillow's decompression-bomb limit (~179M pixels) raised for huge scans/panoramas sent as
    files. Files claiming even more raise OSError, which callers treat as unreadable: a crafted few-KB PNG can
    claim billions of pixels, and decoding it would eat all memory on every sync retry."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 500_000_000  # Pillow refuses 2x this: 1G pixels, ~3 GB decoded as RGB
    try:
        return Image.open(path)
    except Image.DecompressionBombError as e:
        raise OSError(e) from e


def phash(path):
    """64-bit perceptual hash as int; None if the file can't be read."""
    import imagehash

    try:
        return int(str(imagehash.phash(open_image(path))), 16)
    except OSError:
        return None


def text_conflict(a, b):
    """Do two OCR texts say different things? Noise garbles words or drops a line; another caption adds words
    on both sides or barely overlaps. On 348 real copy pairs this flagged 1 true copy (garbled both ways)."""
    import re
    from rapidfuzz import fuzz, process

    if not (a and b):
        return False
    if max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) < 45:  # copies were >= 54, a different meme ~33
        return True
    wa, wb = (set(re.findall(r"\w{3,}", t.lower())) for t in (a, b))

    def unmatched(xs, ys):
        return sum(process.extractOne(x, ys, scorer=fuzz.ratio, score_cutoff=75) is None for x in xs)

    return unmatched(wa, wb) >= 2 and unmatched(wb, wa) >= 2


def same_picture(a, b, ocr_a, ocr_b):
    """Are two files (already close by pHash) the same picture? Measured on 5.7k real photos: true copies
    (recompressed/resized) differ in <0.1% of pixels; most same-template memes with other text from ~0.17%,
    so between 0.1% and 0.5% the OCR must also agree. A caption change in a small area can vanish in the
    downscaled compare (different text on a paper someone holds: 0.05%), so conflicting OCR always vetoes.
    Errs toward keeping both: a missed duplicate is clutter, a false merge hides a picture."""
    import numpy as np
    from PIL import Image
    from rapidfuzz import fuzz

    if text_conflict(ocr_a, ocr_b):
        return False
    try:
        px = [np.asarray(open_image(p).convert("L").resize((160, 160), Image.BILINEAR), np.int16) for p in (a, b)]
    except OSError:
        return False
    diff = (np.abs(px[0] - px[1]) > 40).mean()
    return bool(diff < 0.001 or (diff < 0.005 and ocr_a and ocr_b and fuzz.ratio(ocr_a, ocr_b) >= 90))


def pictures(db):
    """image -> [phash int | None, path, ocr]: what new photos get compared against."""
    ocr = dict(db.execute("SELECT image, ocr FROM media GROUP BY image"))
    return {i: [h and int(h, 16), p, ocr.get(i, "")] for i, h, p in db.execute("SELECT id, phash, path FROM images")}


def find_copy(pics, path, text, h):
    return next((i for i, (h2, p2, o2) in pics.items()
                 if h2 is not None and (h ^ h2).bit_count() <= DUP_HASH_BITS and same_picture(path, p2, text, o2)), None)


def trash(path):
    """Move to the macOS Trash (Finder's "Put Back" works); plain delete where that's unavailable."""
    try:
        from Foundation import NSURL, NSFileManager
    except ImportError:
        return Path(path).unlink(missing_ok=True)
    NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(NSURL.fileURLWithPath_(str(path)), None, None)


def dedup(db):
    """Merge copies of the same picture already indexed: one file and vector per picture; every appearance
    kept, with its own caption and OCR text. The copies' files go to the Trash."""
    import numpy as np

    todo = db.execute("SELECT id, path FROM images WHERE phash IS NULL").fetchall()
    if todo:
        print(f"== fingerprinting {len(todo)} pictures")
        with ThreadPoolExecutor(8) as ex:
            hashes = list(ex.map(phash, [p for _, p in todo]))
        with db:
            db.executemany("UPDATE images SET phash = ? WHERE id = ?",
                           [(f"{h:016x}", i) for (i, _), h in zip(todo, hashes) if h is not None])
    pics = pictures(db)
    ids = [i for i, (h, *_) in pics.items() if h is not None]
    hs = np.array([pics[i][0] for i in ids], np.uint64)
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    print("== comparing")
    for a in range(len(ids)):  # ponytail: all-pairs hash compare, vectorized (~1 s for 6k); BK-tree/LSH past ~100k
        for b in np.nonzero(np.bitwise_count(hs[a] ^ hs[a + 1:]) <= DUP_HASH_BITS)[0] + a + 1:
            i, j = ids[a], ids[b]
            if root(i) != root(j) and same_picture(pics[i][1], pics[j][1], pics[i][2], pics[j][2]):
                parent[root(j)] = root(i)
    groups = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)

    def area(path):
        try:
            w, h = open_image(path).size
            return w * h
        except OSError:
            return 0

    gone = []
    with db:
        for g in (g for g in groups.values() if len(g) > 1):
            keep = max(g, key=lambda i: (area(pics[i][1]), -i))  # biggest copy: best for OCR and viewing
            path = pics[keep][1]
            for i in g:
                if i == keep:
                    continue
                pid = db.execute("SELECT photo_id FROM images WHERE id = ?", (i,)).fetchone()[0]
                db.execute("UPDATE media SET image = ?, path = ? WHERE image = ?", (keep, path, i))
                db.execute("DELETE FROM emb WHERE image = ?", (i,))
                db.execute("DELETE FROM images WHERE id = ?", (i,))
                db.execute("UPDATE images SET photo_id = coalesce(photo_id, ?) WHERE id = ?", (pid, keep))
                gone.append(pics[i][1])
    size = sum(os.path.getsize(p) for p in gone if os.path.exists(p))
    for p in gone:  # only after the commit: the index never points at a trashed file
        trash(p)
    print(f"  {len(gone)} copies merged into {sum(len(g) > 1 for g in groups.values())} pictures, "
          f"{size / 1e6:.1f} MB moved to the Trash")


def client():
    import logging
    from telethon.sync import TelegramClient
    logging.basicConfig(format="  telethon: %(message)s")  # its warnings (reconnects, failed attempts)
    logging.getLogger("telethon.client.users").setLevel(logging.INFO)  # + its otherwise silent flood-wait sleeps
    return TelegramClient(str(HERE / "tgsearch"), int(os.environ["TG_API_ID"]), os.environ["TG_API_HASH"])


def login():
    import getpass
    import qrcode
    from telethon.errors import SessionPasswordNeededError

    tg = client()
    tg.connect()
    try:
        if tg.is_user_authorized():
            return print("already logged in")
        qr = tg.qr_login()
        while True:
            q = qrcode.QRCode()
            q.add_data(qr.url)
            q.print_ascii(tty=True)
            print("Telegram app -> Settings -> Devices -> Link Desktop Device, scan this")
            try:
                tg.loop.run_until_complete(qr.wait())
                break
            except TimeoutError:  # token lives ~30s, show a fresh one
                tg.loop.run_until_complete(qr.recreate())
            except SessionPasswordNeededError:
                tg.sign_in(password=getpass.getpass("2FA cloud password: "))
                break
        print("logged in as", tg.get_me().first_name)
    finally:
        tg.disconnect()


def find(tg, targets):
    targets = set(targets)
    for d in tg.iter_dialogs():
        if {str(d.id), d.name} & targets:
            targets -= {str(d.id), d.name}
            yield d
            if not targets:
                return
    print("not found:", ", ".join(targets))


def pick(tg, db):
    from iterfzf import iterfzf

    done = dict(db.execute("SELECT chat_id, count(*) FROM media GROUP BY chat_id"))
    seen = {}

    def lines():  # streamed: fzf shows chats while later pages still load
        for d in tg.iter_dialogs():
            line = f"{' '.join(d.name.split())}\t{d.id}" + (f"\t✓ {done[d.id]} indexed" if d.id in done else "")
            seen[line] = d
            yield line

    try:
        picked = iterfzf(lines(), multi=True, header="type to filter · TAB select · ENTER sync · ESC quit")
    except KeyboardInterrupt:  # iterfzf raises it on ESC
        return []
    return [seen[line] for line in picked or []]


def analyze(path):
    return ocr(path), phash(path)


def ocr(path):
    from ocrmac import ocrmac
    from tqdm import tqdm

    try:
        return " ".join(t for t, *_ in ocrmac.OCR(str(path), language_preference=LANGS).recognize())
    except Exception as e:  # one broken image must not block resume forever
        tqdm.write(f"  OCR failed on {path}: {e}")
        return ""


def sync(tg, db, dialogs):
    for d in dialogs:
        while True:
            task = tg.loop.create_task(sync_chat(tg, db, d))
            try:
                tg.loop.run_until_complete(task)
                break
            except KeyboardInterrupt:  # cancel in-flight work so nothing runs on during disconnect
                task.cancel()
                tg.loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                return print("\nstopped, run sync again to resume")
            except TimeoutError:  # a silently dead connection: what Ctrl+C + rerun fixed by hand
                print("  reconnecting, resuming after the last saved photo")
                tg.disconnect()
                tg.connect()
    embed(db)  # vectors for search by meaning, for whatever isn't embedded yet


async def merge(a, b):
    """Two async iterables of messages, each in ascending id order, as one in ascending id order."""
    a, b = aiter(a), aiter(b)  # Telethon's iterators set up their state in __aiter__
    x, y = await anext(a, None), await anext(b, None)
    while x is not None or y is not None:
        if y is None or (x is not None and x.id < y.id):
            yield x
            x = await anext(a, None)
        else:
            yield y
            y = await anext(b, None)


async def sync_chat(tg, db, d):
    from telethon.tl.types import InputMessagesFilterDocument as Files, InputMessagesFilterPhotos as Photos
    from telethon.utils import get_display_name
    from tqdm import tqdm

    last, have = db.execute("SELECT max(msg_id), count(*) FROM media WHERE chat_id = ?", (d.id,)).fetchone()
    scanned = db.execute("SELECT 1 FROM files_scanned WHERE chat_id = ?", (d.id,)).fetchone()
    done = {m for (m,) in db.execute("SELECT msg_id FROM media WHERE chat_id = ?", (d.id,))}
    # Telegram can't filter by mime: counts every file, shrunk as non-images turn up. ponytail: ones older than
    # the resume point never turn up, so on short re-syncs the bar starts too long and snaps to 100% at the end.
    todo = sum([(await tg.get_messages(d, limit=0, filter=f)).total for f in (Photos, Files)]) - have
    print(f"== {d.name} ({d.id})")
    # album ids for photos indexed before albums were tracked (100 messages per request, no downloads)
    unchecked = [m for (m,) in db.execute("""SELECT msg_id FROM media WHERE chat_id = ? AND NOT EXISTS
        (SELECT 1 FROM albums a WHERE a.chat_id = media.chat_id AND a.msg_id = media.msg_id)""", (d.id,))]
    for i in range(0, len(unchecked), 100):
        ids = unchecked[i:i + 100]
        msgs = await tg.get_messages(d, ids=ids)  # None for deleted messages
        with db:
            db.executemany("INSERT OR REPLACE INTO albums VALUES (?,?,?)",
                           [(d.id, mid, m and m.grouped_id) for mid, m in zip(ids, msgs)])
        print(f"  album info: {min(i + 100, len(unchecked))}/{len(unchecked)} earlier photos", end="\r", flush=True)
    if unchecked:
        print()
    loop, pool = asyncio.get_running_loop(), ThreadPoolExecutor(OCR_THREADS)
    window, n, copies, t0 = deque(), 0, 0, time.time()
    bar = tqdm(total=todo, desc=d.name, unit="pic")  # log lines go through tqdm.write, above the bar
    pics = pictures(db)
    by_photo = dict(db.execute("SELECT photo_id, id FROM images WHERE photo_id IS NOT NULL"))
    in_flight = set()  # photo ids being downloaded right now (an original and its forward in one window)
    stage = {}  # msg id -> what its photo is waiting on, to report a stall

    async def fetch(m):
        if not m.file:  # e.g. expired self-destructing photo: nothing to save
            return None
        pid = m.file.media.id
        if pid in by_photo or pid in in_flight:  # forward/re-send of a picture we have: no download
            return m, None, None, None
        in_flight.add(pid)
        path = HERE / "media" / str(d.id) / f"{m.id}{m.file.ext or ''}"
        path.parent.mkdir(parents=True, exist_ok=True)
        stage[m.id] = "download"
        if not await tg.download_media(m, file=str(path)):
            return None
        stage[m.id] = "OCR"
        return m, path, *await loop.run_in_executor(pool, analyze, path)

    def save(res):  # runs in message order, so copies within one batch are caught too
        nonlocal n, copies
        if res is None:
            return
        m, path, text, h = res
        pid = m.file.media.id
        image = by_photo.get(pid) or (h is not None and find_copy(pics, path, text, h)) or None
        if not image and not path:  # forward of a photo whose own download failed (expired): nothing to keep
            return
        if is_copy := bool(image):  # one file and vector per picture; each appearance keeps its own OCR text
            if path:
                path.unlink(missing_ok=True)
            _, path, known_text = pics[image]
            text = known_text if text is None else text  # a forward wasn't downloaded, so wasn't read
            copies += 1
        else:
            with db:
                image = db.execute("INSERT INTO images (path, phash, photo_id) VALUES (?,?,?)",
                                   (str(path), None if h is None else f"{h:016x}", pid)).lastrowid
            pics[image] = [h, str(path), text]
        if pid:
            by_photo.setdefault(pid, image)
        n += 1
        date = m.date.astimezone().strftime("%Y-%m-%d %H:%M")
        add(db, d.id, d.name, m.id, date, get_display_name(m.sender), m.message, text, path, image)
        with db:
            db.execute("INSERT OR REPLACE INTO albums VALUES (?,?,?)", (d.id, m.id, m.grouped_id))
        tqdm.write(f"  {date} #{m.id} {'(copy) ' if is_copy else ''}{text[:60]!r}")

    # Up to DOWNLOADS photos in flight (downloading or in OCR), but rows are saved in message order,
    # so max(msg_id) stays a valid resume point after Ctrl+C, a crash or a stall.
    try:
        # Telethon waits forever on a dead connection (no request timeouts; connections to other DCs, where
        # many photos live, get no keepalive pings), so give up after STALL seconds without progress
        async with asyncio.timeout(STALL) as deadline:
            async def advance():
                save(await window[0][1])
                window.popleft()
                bar.update()
                deadline.reschedule(loop.time() + STALL)

            async def image_files():  # until this chat is `scanned`, also those older than the resume point
                async for m in tg.iter_messages(d, filter=Files, min_id=scanned and last or 0, reverse=True):
                    if m.id in done:
                        continue
                    if m.document and m.document.mime_type.startswith("image/") and not m.sticker:
                        yield m
                    else:  # not a picture, but counted in the bar's total
                        bar.total -= 1

            # ponytail: videos, GIFs and stickers skipped (see README backlog)
            photos = tg.iter_messages(d, filter=Photos, min_id=last or 0, reverse=True)
            async for m in merge(photos, image_files()):
                window.append((m.id, asyncio.ensure_future(fetch(m))))
                if len(window) == DOWNLOADS:
                    await advance()
            while window:
                await advance()
            bar.total = bar.n
    except TimeoutError:
        mid = window[0][0] if window and window[0][1].cancelled() else None  # cancelled = the one awaited
        tqdm.write(f"  no progress for {STALL}s, stuck on " +
              (f"msg #{mid} ({stage.get(mid, 'queued')})" if mid else "fetching the message list"))
        raise
    finally:
        for _, t in window:
            t.cancel()
        await asyncio.gather(*(t for _, t in window), return_exceptions=True)
        pool.shutdown(cancel_futures=True)
        bar.close()
    with db:
        db.execute("INSERT OR IGNORE INTO files_scanned VALUES (?)", (d.id,))
    print(f"  done: {n} photos ({copies} copies of pictures already indexed) in {time.time() - t0:.0f}s "
          f"({n / max(time.time() - t0, 1e-9):.1f}/s)")


def repair(tg, db, dialogs):
    """Bring the index in line with Telegram: messages deleted there get marked (rows and files stay), edited
    captions and renamed chats get updated, and pictures whose file is missing are downloaded again."""
    from tqdm import tqdm

    missing = {i: p for i, p in db.execute("SELECT id, path FROM images") if not os.path.exists(p)}
    # paths are absolute: if the folder was moved, don't download everything into the old location
    moved = [missing.pop(i) for i, p in list(missing.items()) if not Path(p).is_relative_to(HERE)]
    sent_in = {}  # missing picture -> every message it was sent in
    for img, cid, mid in db.execute("SELECT image, chat_id, msg_id FROM media"):
        if img in missing:
            sent_in.setdefault(img, []).append((cid, mid))
    wanted = {k for ks in sent_in.values() for k in ks}
    alive = {}  # of those messages, the ones still in Telegram
    try:
        for d in dialogs:
            with db:
                db.execute("UPDATE media SET chat = ? WHERE chat_id = ? AND chat != ?", (norm(d.name), d.id, norm(d.name)))
            rows = db.execute(f"SELECT rowid, msg_id, caption FROM media WHERE chat_id = ? AND NOT {DELETED}",
                              (d.id,)).fetchall()
            gone = edited = 0
            with tqdm(total=len(rows), desc=d.name, unit="msg") as bar:
                for i in range(0, len(rows), 100):
                    batch = rows[i:i + 100]
                    msgs = tg.get_messages(d, ids=[mid for _, mid, _ in batch])  # None for deleted messages
                    with db:
                        for (rowid, mid, caption), m in zip(batch, msgs):
                            if m is None:
                                db.execute("INSERT OR IGNORE INTO deleted VALUES (?,?)", (d.id, mid))
                                gone += 1
                                continue
                            if norm(m.message) != caption:
                                db.execute("UPDATE media SET caption = ? WHERE rowid = ?", (norm(m.message), rowid))
                                edited += 1
                            if (d.id, mid) in wanted:
                                alive[d.id, mid] = m
                    bar.update(len(batch))
            print(f"  {gone} deleted in Telegram, {edited} captions edited")
        restored = 0
        for img, path in tqdm(missing.items(), desc="missing files", unit="file"):
            # its own message first: other appearances may be smaller copies (dedup keeps the biggest)
            for k in sorted(sent_in[img], key=lambda k: Path(path).with_suffix("").parts[-2:] != tuple(map(str, k))):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                if k in alive and tg.download_media(alive[k], file=path):
                    restored += 1
                    break
    except KeyboardInterrupt:
        return print("\nstopped, run repair again (it checks everything anew)")
    print(f"  {restored} of {len(missing)} missing files downloaded again" +
          (f", the rest are gone from Telegram (or from the chats checked)" if restored < len(missing) else "") +
          (f"; {len(moved)} skipped: their paths point outside {HERE} (folder moved?)" if moved else ""))
    embed(db)  # restored files that never got a vector


def serve(port):
    from typing import Literal

    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()

    @app.get("/api/search")
    def api_search(q: str = "", ocr: str = "", caption: str = "", sender: str = "", chat: str = "", after: str = "",
                   before: str = "", exact: bool = False, min_score: int = 80, limit: int = 60,
                   meaning: str = "", model: str = MODEL, similar: int = 0, msg: int = 0,
                   order: Literal[ORDERS] = "new", album: str = "", page: int = 1, chat_id: int = 0):
        db = connect()
        limit = min(limit, 1000)
        rows, total = search_page(db, q or None, caption or None, sender or None, chat or None, after or None,
                                  before or None, limit, None if exact else min_score, model, similar or None,
                                  msg or None, order, tuple(int(x) for x in album.split(":")) if album else None,
                                  meaning or None, ocr or None, (max(page, 1) - 1) * limit,  # album: "chat_id:msg_id"
                                  chat_id or None)
        seen = appearances(db, [r[IMAGE] for r in rows])
        sizes = album_sizes(db, [(r[7], r[8]) for r in rows])
        return {"total": total, "hits": [
            {"image_id": img, "msg_id": mid, "chat_id": cid, "album": sizes.get((cid, mid), 0), "date": date, "chat": chat,
             "sender": sender, "link": url, "snippet": snip, "ocr": ocr, "caption": cap, "deleted": bool(gone),
             "image": "/media/" + Path(path).relative_to(HERE / "media").as_posix(),
             "appearances": [dict(zip(("date", "chat", "sender", "link", "msg_id"), a), deleted=bool(a[5]))
                             for a in seen[img]]}
            for date, chat, sender, path, url, ocr, cap, cid, mid, img, gone, snip in rows]}

    @app.get("/api/tags")
    def api_tags(images: str, model: str = MODEL):  # separate call so text search never waits for the model
        return tags(connect(), [int(i) for i in images.split(",") if i], model)

    @app.get("/api/models")
    def api_models():  # embedding models present in the index, default first
        rows = connect().execute("SELECT model, count(*) FROM emb GROUP BY model ORDER BY model != ?, 2 DESC", (MODEL,))
        return [{"name": name, "count": count} for name, count in rows]

    @app.get("/api/chats")
    def api_chats():
        # by id: names repeat (e.g. a basic group and the supergroup it was upgraded to); max(date) -> latest name
        rows = connect().execute("SELECT chat_id, chat, max(date), count(*) FROM media GROUP BY chat_id ORDER BY 4 DESC")
        return [{"id": cid, "name": name, "count": count} for cid, name, _, count in rows]

    if connect().execute("SELECT 1 FROM emb WHERE model = ? LIMIT 1", (MODEL,)).fetchone():  # warm up (~10 s)
        threading.Thread(target=lambda: tags(connect(), []), daemon=True).start()  # model + tag labels
    app.mount("/media", StaticFiles(directory=HERE / "media", check_dir=False))
    app.mount("/", StaticFiles(directory=HERE / "web" / "dist", html=True, check_dir=False))
    # ponytail: no auth, so localhost only; add auth before binding to anything else
    uvicorn.run(app, host="127.0.0.1", port=port)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("help").add_argument("topic", nargs="?", choices=["login", "chats", "sync", "search", "embed", "dedup", "repair", "serve"])
    sub.add_parser("login")
    sub.add_parser("chats")
    sub.add_parser("sync").add_argument("chat", nargs="*", help="dialog id or exact name; none = every chat "
                                        "already indexed (fetches what's new since its last sync)")
    s = sub.add_parser("search")
    s.add_argument("text", nargs="?", help="text in the picture (OCR) or in the message")
    s.add_argument("--ocr", metavar="TEXT", help="text in the picture only")
    s.add_argument("--caption", metavar="TEXT", help="text in the message only")
    s.add_argument("--from", dest="sender", help="sender name substring")
    s.add_argument("--chat", help="chat name substring")
    s.add_argument("--after", help="YYYY-MM-DD")
    s.add_argument("--before", help="YYYY-MM-DD")
    s.add_argument("-n", type=int, default=20, help="max results (default 20)")
    s.add_argument("--open", action="store_true", help="open results in Preview")
    s.add_argument("--exact", action="store_true", help="exact substring match instead of fuzzy (indexed, newest first)")
    s.add_argument("--min-score", type=int, default=80, help="fuzzy match threshold 0-100 (default 80)")
    s.add_argument("--meaning", metavar="DESCRIPTION", help="what's in the picture, e.g. 'dog in snow' (embeddings); "
                   "combines with TEXT (which then filters) and --similar")
    s.add_argument("--model", default=MODEL, help=f"embedding model for --meaning/--similar/--tags (default {MODEL})")
    s.add_argument("--similar", type=int, metavar="ID", help="pictures that look like picture #ID (ids are shown in results)")
    s.add_argument("--tags", action="store_true", help="show what the model sees in each picture (loads the model)")
    s.add_argument("--msg", type=int, metavar="ID", help="message id (unique among private chats/small groups; "
                                                          "per chat in channels/supergroups, add --chat)")
    s.add_argument("--album", type=int, metavar="MSG_ID", help="all photos of the album that message belongs to")
    s.add_argument("--sort", choices=ORDERS, help="new / old: by message date; relevance: best match first "
                   "(default: relevance when searching text/--meaning/--similar, else new)")
    sub.add_parser("embed").add_argument("--model", default=MODEL, help=f"HF model id (default {MODEL})")
    sub.add_parser("dedup", help="merge copies of the same picture already indexed (copies' files go to the Trash)")
    sub.add_parser("repair", help="mark messages deleted in Telegram (rows stay), update edited captions and chat "
                   "names, re-download files missing on disk").add_argument(
        "chat", nargs="*", help="dialog id or exact name; none = every chat already indexed")
    sub.add_parser("serve").add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    if a.cmd in (None, "help"):
        p.parse_args([a.topic, "-h"] if getattr(a, "topic", None) else ["-h"])  # -h prints help and exits
    elif a.cmd == "login":
        login()
    elif a.cmd == "serve":
        serve(a.port)
    elif a.cmd == "embed":
        embed(connect(), a.model)
    elif a.cmd == "dedup":
        dedup(connect())
    elif a.cmd == "chats":
        db = connect()
        with client() as tg:
            sync(tg, db, pick(tg, db))
    elif a.cmd in ("sync", "repair"):
        db = connect()
        targets = a.chat or [str(c) for (c,) in db.execute("SELECT DISTINCT chat_id FROM media")]
        if not targets:
            return print("nothing indexed yet: name the chats to sync, or pick them with `chats`")
        with client() as tg:
            (sync if a.cmd == "sync" else repair)(tg, db, find(tg, targets))
    else:
        db = connect()
        rows = search(db, a.text, a.caption, a.sender, a.chat, a.after, a.before, a.n,
                      None if a.exact else a.min_score, a.model, a.similar, a.msg,
                      a.sort or ("relevance" if a.text or a.ocr or a.caption or a.meaning or a.similar else "new"),
                      (None, a.album) if a.album else None, a.meaning, a.ocr)
        seen = appearances(db, [r[IMAGE] for r in rows])
        labels = tags(db, [r[IMAGE] for r in rows], a.model) if a.tags else {}
        sizes = album_sizes(db, [(r[7], r[8]) for r in rows])
        for date, chat, sender, path, url, *_, cid, mid, img, gone, snip in rows:
            in_album = f"  (album of {sizes[cid, mid]}: --album {mid})" if (cid, mid) in sizes else ""
            print(f"#{img}  {date}  {chat} / {sender}  msg {mid}{' (deleted in Telegram)' * gone}{in_album}\n"
                  f"  {path}\n  {url}\n  {snip}")
            if img in labels:
                print("  tags: " + ", ".join(labels[img]))
            if len(seen[img]) > 1:
                print("  sent " + ", ".join(f"{d} in {c} by {s} (msg {m}{', deleted' * g})" for d, c, s, _, m, g in seen[img]))
            print()
        if a.open and rows:
            subprocess.run(["open", *(r[3] for r in rows)])


if __name__ == "__main__":
    main()
