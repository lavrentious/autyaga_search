# tgsearch

Local search over media from selected Telegram chats. Photos are pulled via the user's own account (Telethon),
OCR'd locally with Apple Vision (ru + en), and indexed in SQLite FTS5. Nothing leaves the machine except
the Telegram API calls themselves.

Long-term goal: one local search across all personal media sources (Telegram, VK, local folders), by text in
images, captions, metadata, and eventually by meaning (AI tags / embeddings), including videos.

## Usage

```sh
# TG_API_ID / TG_API_HASH from https://my.telegram.org -> API development tools
uv run --env-file .env tgsearch.py login          # QR login, once
uv run --env-file .env tgsearch.py chats          # fzf picker: type to filter, TAB multi-select, ENTER syncs
uv run --env-file .env tgsearch.py sync <id|name>...   # same, non-interactive (exact name or id from picker)
uv run --env-file .env tgsearch.py sync                # refresh: every chat already indexed, new photos only
uv run tgsearch.py search "пароль от wifi" --from вася --chat работа --after 2024-01-01 --open
uv run tgsearch.py search --meaning "собака в снегу"   # by what's in the picture (embeddings)
uv run tgsearch.py search "кот" --meaning "мем"         # combined: text filters, meaning ranks
uv run tgsearch.py search --similar 5718 --tags      # look-alikes of picture #5718 (ids shown as #N), with tags
uv run tgsearch.py search --msg 659070               # by exact message id (shown as "msg N" in results)
uv run tgsearch.py search --album 568363             # all photos of the album that message belongs to
uv run tgsearch.py search "пароль" --sort relevance  # new (default) / old / relevance
uv run --env-file .env tgsearch.py embed [--model HF_ID]  # vectors for photos missing them (sync runs this too)
uv run tgsearch.py dedup                          # merge copies of a picture already indexed (copies -> Trash)
uv run --env-file .env tgsearch.py repair [<id|name>...]  # check indexed messages against Telegram (see below)
uv run tgsearch.py serve [--port 8000]            # web UI + API on http://127.0.0.1:8000 (localhost only)
uv run tgsearch.py help [command]                 # also shown when run without arguments
uv run test_tgsearch.py                           # logic self-check (no Telegram/OCR needed)
```

Search flags: positional text matches the picture's text (OCR) **or** the message text; `--ocr TEXT` / `--caption TEXT`
match just one (UI: "Text in the picture" / "Message text" boxes); all given must match. Text is **fuzzy by
default** (typos/OCR errors, score shown); `--exact` for indexed substring match; `--min-score` fuzziness knob
(default 80; lower = looser; 3-letter fuzzy queries are noisy, use `--exact` for those). `--from`, `--chat` are
exact substring filters; `--after`/`--before`
take `YYYY-MM-DD`; `-n` limit (default 20); `--open` opens hits in Preview. Each hit prints a message link (see below).

## Files

| Path | What |
|---|---|
| `tgsearch.py` | the whole tool; deps declared inline (PEP 723), `uv run` installs them |
| `test_tgsearch.py` | assert-based check of indexing/search logic |
| `tags.txt` | label vocabulary for the "what the model sees" tags (one per line; edit, restart `serve`) |
| `web/` | frontend: React + Vite + TS + Mantine, bun-managed; `main.tsx` is the whole app |
| `web/dist/` | built frontend, served by `serve` (rebuild: `cd web && bun run build`) |
| `media.db` (+ `-wal`, `-shm`) | WAL-mode SQLite: keep the three together, never copy `media.db` alone. The index: `images` (one per picture), `media` (FTS5, one row per appearance), `emb` |
| `media.db.bak-pre-dedup`, `.bak-pre-repair` | backups from the 2026-10-08 dedup migration; delete when happy |
| `media/<chat_id>/<msg_id>.<ext>` | downloaded photos and image files (kept: re-OCR/viewing without re-downloading) |
| `tgsearch.session` | **Telethon login = full account access. Never share/commit.** |
| `.env` | API id/hash. **Secret.** Template: `.env.example` |
| `.gitignore` | keeps secrets (`.env`, `*.session`) and personal data (`media.db*`, `media/`) out of git |

## How it works

- **Ingest** (`sync`): `iter_messages(filter=Photos, min_id=<max indexed msg_id>, reverse=True)` → oldest
  first, merged in message order with `filter=Document` messages whose mime is `image/*` (images sent as files;
  stickers skipped). Chats synced before files were fetched (not in `files_scanned`) get their older image files
  backfilled once (the files stream starts at 0, already indexed messages are skipped). Pipeline: up to
  `DOWNLOADS`=8 photos in flight (each: async download → OCR in a 4-thread pool), but rows are saved strictly
  in message order (sliding window, await the head), so the max indexed id is always a valid resume point.
  Ctrl+C cancels in-flight work cleanly; rerun resumes (half-downloaded files are just overwritten). A tqdm bar
  per chat (count, rate, ETA) with one log line per picture above it. Telethon runs in sync mode; the pipeline
  is one coroutine per chat run via `tg.loop.run_until_complete`.
- **Chat selection**: `chats` streams `iter_dialogs()` into fzf (via `iterfzf`, which bundles the fzf binary),
  so the list appears while pages still load and picking early stops further fetching. Already-indexed chats
  show `✓ N indexed`. `sync` resolves ids/exact names and stops iterating dialogs once all are found.
- **OCR**: `ocrmac` → Apple Vision, `accurate` mode, `["ru-RU", "en-US"]` (Russian is unavailable in `fast`
  mode). Recognized lines joined with spaces so phrases can match across line breaks. OCR failure on one
  image stores empty text instead of blocking resume.
- **Index**: one FTS5 table, `tokenize='trigram'` → case-insensitive substring search (Cyrillic included),
  tolerant to word endings/OCR noise. Columns: `ocr, caption, sender, chat` (indexed) +
  `chat_id, msg_id, date, path` (UNINDEXED). `ё` is normalized to `е` on insert and query.
  Dates stored as local time `YYYY-MM-DD HH:MM` (string compare works for ranges).
- **Fuzzy search**: SQL applies the non-text filters, then `rapidfuzz` `partial_ratio` (best-matching window
  of the query inside OCR+caption, Levenshtein-based) scores every remaining row; matched span is highlighted.
  Linear scan: ~0.1 s on 2.3k rows. Queries ≤ 4 chars are effectively exact at the default 80.
  Gotcha (fixed): `partial_ratio` searches the *shorter* string in the longer, so textless images (doc `" "`)
  scored 100% for any query with a space → docs are padded with `\0` to the query length.
- **Pictures vs appearances (dedup)**: a picture is stored, fingerprinted and embedded once (`images`: path,
  pHash, Telegram photo id; `emb` keyed by image); every message it was sent in is a `media` row (chat, sender,
  date, caption, its own OCR text, link) pointing at it. Search ranks appearances, then keeps each picture's
  best one (`first_per_image`); `appearances()` lists all of them (CLI "sent …", UI "Sent N times: …" links).
  Copy detection: (1) same Telegram `photo.id` (forwards/re-sends) -> linked without downloading (also for an
  original + forward in one download window); (2) otherwise pHash distance <= 6 picks candidates and
  `same_picture()` decides: pixels at 160×160 differ < 0.1% (or < 0.5% with OCR >= 90% similar) **and** OCR
  texts don't conflict (`text_conflict`: overall similarity < 45, or each side has 2+ words the other lacks).
  Sync applies this on arrival (the duplicate download is deleted); `dedup` applies it to what's indexed,
  keeps the biggest copy, moves the others to the macOS Trash (NSFileManager, "Put Back" works).
  Old-layout databases are migrated automatically on first `connect()` (backup `media.db.bak-pre-dedup`).
- **Embeddings / search by meaning**: `google/siglip2-base-patch16-256` (multilingual SigLIP 2, 768-d) via
  `transformers` on the M1 GPU (MPS, fp16). Table `emb(model, chat_id, msg_id, vec)`: unit-length float16
  vectors, **keyed by model** (and by picture, not message), so models are swappable/coexisting — `embed --model <any CLIP/SigLIP HF id>`
  adds that model's vectors for already downloaded photos (no re-download), `--model` picks one at search time.
  `embed` only processes photos missing vectors for that model, commits per batch (Ctrl+C-safe), skips
  unreadable files; `sync` runs it after syncing. Search = brute-force cosine in numpy over vectors passing the
  SQL filters (`media JOIN emb USING (image)`); "Similar" uses a stored photo vector as the query.
  Query text is lowercased + padded to 64 tokens (SigLIP training format). Model load ~10 s (cached in `serve`).
- **Tags ("what the model sees")**: the vectors themselves are unreadable (768 numbers), so they're translated
  zero-shot: every label in `tags.txt` is encoded once with the same model; a picture's tags = the 4 labels
  closest to its stored vector, after subtracting **half** of each label's mean closeness over the library
  (raw closeness put "meme"/"chat screenshot" on 40–45% of pictures; fully subtracting surfaced odd labels).
  Computed at query time from stored vectors (nothing stored), via `GET /api/tags?images=…` so text search
  never waits for the model; the server warms model + labels at start (~20 s). Click a tag = put it in the Meaning box.
- **"Similar"** = a meaning query that is the picture's own stored vector instead of a text's: pictures
  whose vectors are closest (same subject/layout/style to the model, not pixel matching — that's dedup).
  Clicking it clears the Text/Meaning boxes; whatever is typed afterwards combines with it.
- **Combined search**: every criterion given applies (`search()` / `rank()`). Text queries (picture text, message
  text, or either = the better of the two fields; exact or fuzzy) decide *what matches*; Meaning and/or Similar decide *relevance* when given. Meaning +
  Similar together: each similarity is z-scored over the candidates and summed (look-alike similarities run
  ~0.3–1, text-to-picture ~0–0.2, so a raw sum would be all look-alike). Pictures without a vector (no file)
  drop out as soon as a picture query is given. Exact text alone stays all-SQL (FTS rank + `snippet()`).
- **Web UI** (`serve`): FastAPI inside `tgsearch.py` reuses `search_page()` as is. `GET /api/search` (CLI's params:
  `q, ocr, caption, meaning, similar, sender, chat, after, before, msg, album, exact, min_score, order, model`, plus
  `page` + `limit`, `chat_id` = comma-separated ids) → `{total, hits}`, each hit with its file `size` (null =
  missing on disk); `GET /api/chats` (names, counts and bytes in `media/<chat_id>/` for the multi-select chat
  filter; their sum is the "on disk" total), `/media` and `/` are static mounts (`media/`, `web/dist/`).
  Frontend searches as you type (250 ms debounce, stale requests aborted; the spinner sits in the sticky status
  line, results stay usable meanwhile). **Pages** of 60: every page is a slice of one ranking (for
  Meaning/Similar sorted by date: page n = the n-th 60 best, by date), total from the same pass. **URL = search
  state**: non-default filters + page are the query string (shareable, survive reload; the same string is the API
  query); a page change is a history step (Back = previous page), typing replaces the entry. After a reload the
  Similar banner only knows the id ("picture #N"); the album banner is derived from the results. Matches highlighted by
  splitting the snippet on the CLI's ANSI bold codes (no HTML injection). Each hit: highlighted snippet, then
  caption + full OCR text behind "Show full text" (Mantine `Spoiler`, query words highlighted); without a query
  the first lines are shown. UI kit: Mantine (`@mantine/core`, `hooks`, `notifications`), default theme,
  `defaultColorScheme="auto"`; plain list layout, no custom CSS. Dev with hot reload:
  `uv run tgsearch.py serve` + `cd web && bun run dev` (Vite proxies `/api`, `/media` to :8000).
- `chat_id` is Telethon's marked dialog id (`-100…` for channels/supergroups, positive for users).
- **Message links** are computed at query time (SQL function `tglink`), no extra data stored:
  channels/supergroups → `https://t.me/c/<id without -100>/<msg_id>` (official, jumps to the message; UI "Open
  message"); private chats → `tg://user?id=<id>`, which opens the **chat** (UI "Open chat"): Telegram has no
  message links in private chats — the undocumented `tg://openmessage` didn't work in the macOS app — so the
  UI shows the date + msg id to jump via the chat's calendar. Basic groups → `tg://openmessage?chat_id=…`
  (best effort, no id-based link exists).
- **Message ids**: private chats/basic groups share one per-account counter (an id is unique across them);
  channels/supergroups count per chat (combine `--msg` with `--chat`). Shown as "msg N" on every result.
- **Sorting** (`order`/`--sort`): default = `relevance` when a text/meaning/Similar query is given, else `new`
  (date order put 80% fuzzy matches above an exact one). `new` / `old` by message date — a picture sent several times
  counts by its newest/oldest matching appearance; for Meaning/Similar (everything matches a little) it's the
  `limit` most similar, ordered by date; `relevance` = best match first. Ties broken by msg id.
- **Albums**: Telegram sends a multi-photo post as several messages sharing `grouped_id`; table
  `albums(chat_id, msg_id, grouped_id)` (row present = checked, NULL = no album). Sync records it for new
  photos and backfills earlier ones of that chat (`get_messages(ids=…)`, 100 per request, no downloads).
  `album=(chat_id|None, msg_id)` filter shows the album in message order; results carry the album size
  (UI "Album (N)" button + banner; CLI "(album of N: --album …)").
- **Repair** (`repair`, same chat arguments as `sync`): re-fetches every indexed message (`get_messages(ids=…)`,
  100 per request). Gone from Telegram → row in `deleted(chat_id, msg_id)`, the row itself stays searchable
  (UI "deleted in Telegram" badge / struck-through appearance, CLI "(deleted in Telegram)"); edited caption →
  updated; chat renamed → updated. Pictures whose file is missing on disk are downloaded again from a message
  still there, their own message first (other appearances may be smaller copies). Missing paths outside the
  folder (moved, see Robustness) are skipped, not recreated. Idempotent; ends with `embed`.

## Decisions & gotchas

- **Immich** (already self-hosted) was considered: has OCR since v2.2 but no Cyrillic support, and Telegram
  media shouldn't mix with the personal photo library → own tool.
- **Telegram export** (Desktop) is per-chat only via each chat's menu and manual every time → API sync instead.
- **Login codes for API apps often never arrive** (API reports `SentCodeTypeApp`, nothing delivered) → QR
  login (`login` command). 2FA password prompt is the cloud password, not a code.
- Trigram index: exact queries < 3 characters match nothing.
- SigLIP 2 test on 5.7k real photos: 13.5 photos/s embedding on M1; top-12 for "собака"/"a dog"/"рукописный
  текст"/"мем с котом" ~all on target, Russian as good as English. Raw cosine/sigmoid scores are **not
  comparable across queries/languages** (Russian queries score higher overall) → rank only, no cutoff.
- Dedup measurements (5.7k photos, 1.4k candidate pairs): 435 pairs differ in < 0.1% of pixels — all true
  copies even when OCR disagreed (OCR noise down to 54% similarity). Same-template memes with another caption
  start at ~0.17% — **except** a caption change in a tiny area (text on a paper someone holds) can be 0.05%;
  pixels can't see it, OCR can (33% similar) -> the OCR veto. First run without the veto merged 2 different
  memes and overwrote copies' OCR with the kept copy's (one became unsearchable); repaired from backup + Trash
  the same day, and now each appearance keeps its own OCR text. Result: 5,717 appearances -> 5,359 pictures,
  303 pictures sent more than once (175 across chats, up to 7×), 35 MB of copies to the Trash.
- Expired self-destructing photos: `download_media` returns nothing → sync now skips them. 17 older rows from
  before the fix have a caption but no image file; kept (caption is searchable), UI shows a placeholder.
- OCR is not the bottleneck: on M1, Vision takes ~0.18 s/photo (~0.08 with 4 threads, ~0.05 with 4 processes);
  the old sequential sync's ~1.3 s/photo was mostly download latency.
- `search` doesn't need Telegram creds or network.
- **Concurrency**: the index is in WAL mode, so `serve`/`search` read a consistent snapshot while `sync`/`embed`/
  `dedup` commit; two writers queue (busy wait up to 60 s). Measured on a copy: 160/160 web searches OK during
  per-photo commits and an 8 s write transaction, median 0.15 s vs 0.14 s idle. In the default rollback mode
  it also never errored (5 s busy wait), but readers and writers took turns: ~6× fewer searches/s.
- Don't run two Telegram commands (`chats`/`sync`/`login`) at once: they share `tgsearch.session` (SQLite lock).
- CLI commands that need the model (`--meaning`, `--similar`, `--tags`) load it each run (~10–20 s); `serve`
  keeps it loaded.
- **Chat names aren't unique**: a basic group upgraded to a supergroup (Telegram does it when the group grows
  or gets certain settings) leaves the old group deactivated (shown as deleted, old history) and continues under
  a new `-100…` id, same name. The UI chat filter is by id (`chat_id`; duplicate names show their id); the CLI's
  `--chat` stays a name substring (matches both). The old dropdown used names as values and crashed on the repeat.
- Telethon sleeps on FloodWaits up to 60 s (now printed as "telethon: Sleeping for …").
- **Sync hangs** (seen around 1,300 of 2k photos; Ctrl+C + rerun resumed fine): Telethon 1.45 has no request
  timeouts, and connections to other DCs (photos often live on a DC other than the account's) get no keepalive
  pings, unlike the main connection — a silently dead socket (Wi-Fi hiccup, Mac sleep, NAT drop) waits forever,
  and the in-order pipeline then waits on that one photo. Now: no saved photo for `STALL`=120 s → print what it
  was stuck on (download / OCR / message list), reconnect, resume after the last saved photo. Root cause
  inferred from Telethon's code + symptoms, not caught live; the stall line will tell next time.
- **macOS-only parts**: OCR (`ocrmac` → Apple Vision via PyObjC, which only builds on macOS, so
  `uv run` can't even install the deps elsewhere without a `sys_platform == 'darwin'` marker) and dedup's Trash
  (`NSFileManager`, already falls back to plain delete). Embeddings pick MPS, else CPU (no CUDA branch yet).
  Everything else (Telethon, SQLite FTS5, FastAPI, fzf, numpy/torch) is cross-platform.

## Progress

- **2026-10-08**: MVP working end to end: QR login → sync one chat → OCR → search. Verified: logic tests,
  OCR on mixed ru/en text (exact), picker behavior (filter, early pick, ESC, odd names).
  First real chat (2,341 photos, ~400 videos) synced sequentially at **~1.3 s/photo** (~50 min); videos not
  indexed yet.
- **2026-10-09**: Combinable search: separate Text and Meaning boxes (CLI `--meaning DESCRIPTION`), both + Similar
  together; Similar clears the boxes but combines with anything typed after. `fuzzy_rank` + `semantic_rank`
  merged into `rank()`.
- **2026-10-10**: Default sort = best match when searching (Newest when just browsing); an explicit choice wins.
- **2026-10-10**: Fixed: UI crashed when two indexed chats share a name (group → supergroup upgrade); chat
  filter now by id. Initial git commit.
- **2026-10-09**: `sync` without names = refresh every chat already in the index (most recently active first;
  each resumes after its last indexed message, backfills album info; chats you left are reported as not found).
- **2026-10-09**: Web UI pagination (total + pages instead of "Load more"; `search_page()` with `offset`) and
  search state in the URL (reload/share/back). Verified in a browser: page 2 ≠ page 1, Back returns to page 1,
  reload restores boxes/Exact/results, Similar and album views survive a reload; pager fits one line at 390 px.
- **2026-10-09**: Separate picture-text and message-text queries (UI boxes, CLI `--ocr`/`--caption`; message text
  is fuzzy now too). Fixed: fuzzy "either" search scored OCR + caption joined, which lost matches at a field's
  edge (9 of ~100 for one word); now the better field's score counts.
- **2026-10-09**: Sync stall watchdog (auto-reconnect + resume, says where it was stuck), flood-wait
  sleeps printed. Verified with a fake client hanging in a download and in the message list.
- **2026-10-09**: Fixed "Copy link" on pages opened over plain http from another device (no
  `navigator.clipboard` outside https/localhost → legacy copy fallback).
- **2026-10-09**: Message-id lookup + "msg N" everywhere, sorting (newest default / oldest / relevance),
  private-chat links open the chat (message links impossible there), albums (recorded on sync, backfilled for
  earlier photos on next sync; UI "Album (N)" view, CLI `--album`).
- **2026-10-08**: WAL mode + 60 s busy wait (serve during sync measured fine); CLI `--similar`, `--tags`, `#id`.
- **2026-10-08**: Zero-shot tags per picture in the UI (clickable → Meaning search), `tags.txt` vocabulary.
- **2026-10-08**: Dedup: pictures stored once, every appearance kept (chat, sender, date, caption, own OCR,
  link); copies detected on sync (photo id / pHash + pixels + OCR veto) and merged in the existing index.
- **2026-10-08**: Search by meaning (SigLIP 2 embeddings, per-model, swappable): `embed`, `--meaning`, UI
  Text/Meaning switch, "Similar" button, model picker when several are indexed. Fixed: sync stored rows for
  undownloadable (expired) photos.
- **2026-10-08**: Web UI moved to Mantine; full OCR text + caption returned by search and expandable in the UI.
  Checked OCR limits: none in practice (Vision read all 40 lines of a synthetic 591×1280 screenshot, even at
  ~8 px text; stored texts go up to 4.7k chars) — the "cropped" text was only the UI snippet.
- **2026-10-08**: Web UI (`serve` + `web/`). Fixed: textless images scored 100% in fuzzy search;
  `--min-score 0` was treated as exact. Message links (`t.me/c/…` / `tg://openmessage`) in results.
- **2026-10-08**: `chats` fzf picker, `help` command. Sync overhaul: parallel downloads + OCR pool with
  in-order commits (simulated latency: 24 s → 3 s for 48 photos; crash/Ctrl+C resume verified). Real-world
  speedup not yet measured. Fuzzy search (default) + `--exact`.

## Backlog

**Performance**
- Measure real sync rate with the new pipeline; tune `DOWNLOADS` (watch for FloodWaits).
- Fuzzy search at scale: prefilter candidates with a trigram OR-query before rapidfuzz if scans get slow.

**Coverage**
- Videos: download → `ffmpeg` scene keyframes → OCR; audio → `mlx-whisper`. Cheap first pass: OCR Telegram's
  video thumbnail.
- Stickers, maybe.
- Album captions: propagate caption to all photos with the same `grouped_id`.
- More metadata: forwarded-from, reply context, post author.
- Edits/deletions are only picked up by `repair` (not by `sync`); `repair` doesn't notice a photo replaced by
  editing the message, nor renamed senders.
- Verify which client opens `tg://openmessage` links; `--tg` flag to open the top hit in Telegram.

**Dedup follow-ups**
- The 17 caption-only pictures (no file) get re-fingerprinted on every `dedup` run (harmless, ~nothing).
- A re-uploaded copy's own Telegram photo id isn't stored (one `photo_id` per picture), so its later forwards
  are downloaded once more and then matched by fingerprint. A `photo_ids` table would skip that download.
- All-pairs hash compare is fine for ~100k pictures; BK-tree/LSH beyond.

**Search**
- Local vision-language model ("beast mode"): captions/tags/descriptions per image (Qwen-VL via `mlx-vlm`/
  Ollama), stored as text → searchable with FTS/fuzzy, shown in UI; seconds per image, run as background job.
- Tags as filters (e.g. "only screenshots"), per-tag thresholds instead of top-4, Russian label names in UI.
- Hybrid ranking: text currently only filters when a picture query is given; fusing its score in (RRF) could
  help "best match" when the text query is loose.
- Faster `embed`: decode images in a thread pool (currently single-threaded, ~13.5/s); `--meaning` "similar"
  from the CLI.
- Short queries (< 3 chars): fallback to `LIKE`.
- Open hit in Telegram (deep link to message) besides Preview.

**Robustness**
- Paths are stored absolute: moving/renaming the `tgsearch` folder breaks images and the web UI (500s).
  Store them relative to the folder (migration: strip the prefix) — do before moving it.

**Portability** (e.g. run on the home server next to Immich)
- OCR backend for Linux/Windows (candidates: RapidOCR/PaddleOCR ONNX, EasyOCR, Tesseract `rus+eng`) — benchmark
  ru/en quality against Vision on the same photos before switching; keep Vision on macOS.
- `cuda` device in `load_model`; relative paths (see Robustness) so the folder can be copied over.

**Sources / UI**
- VK — researched 2026-10-10 (details below); next step needs the user's own data archive. Local folders.

**VK source: research (2026-10-10)**
- **User tokens**: Implicit / Authorization Code Flow switched off 2024-06-25; new apps only via VK ID (OAuth 2.1 +
  PKCE, 1 h tokens + refresh). VK ID grants `vkid.personal_info`, `email`, `phone`; `photos` (and `friends`,
  `video`, …) only "in exceptional cases" after business-profile verification + a request to devsupport.
  `messages` with a user token: no longer granted at all ("contradicts product strategy"). Old non-revoked
  offline tokens keep working.
- **Quota since 2026-09-07**: 10,000 API calls/month per unverified VK ID account (100M verified; more = paid).
  Fine for personal use (`photos.get` returns ≤1000 per call; image downloads are CDN, not API) — but Kate
  Mobile died of it (shared app quota gone in 1.5 days), so borrowed client tokens are no option.
- **Liked photos**: `fave.getPhotos` ("photos the current user has liked") still answers but left the schema by
  2019; photo objects carry the upload `date`, no like date found anywhere.
- **Saved photos**: `photos.get` with `album_id=saved` (user token only → blocked like above).
- **Group walls**: `wall.get` (≤100 posts/call) works with the VK ID app's *service key* → doable via API.
- **Data archive** (vk.com/data_protection → request; official, manual, hours–days): as of 2024 `photos/` = album
  pages with direct image URLs + dates, `messages/` = chat pages with direct photo attachment URLs (no login
  needed), `likes/photo` = `vk.com/photo…` links only (resolved via the page's og:image with browser cookies,
  e.g. VKArchiveDownloader). Encoding cp1251. Structure must be re-checked on a 2026 archive.
- **Schema prerequisite**: a `source` (telegram/vk) next to chat/msg ids — ids collide across sources; links
  stop being `tglink()`-derived; `images.photo_id` must not mix id spaces.
- Web UI auth (login/password or token) — required before binding `serve` to anything but 127.0.0.1.
- Web UI: lightbox instead of opening the image in a new tab, collapsible filters on phones,
  sync trigger/progress from the UI.
