# /// script
# dependencies = ["rapidfuzz", "numpy", "pillow", "imagehash", "tqdm"]
# ///
"""uv run test_tgsearch.py — checks indexing/search/dedup logic (no Telegram, no OCR, no model)."""
import asyncio
import sqlite3
import struct
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import tgsearch
from tgsearch import add, appearances, connect, dedup, embed, link, phash, same_picture, search, search_page

db = connect(":memory:")
add(db, -1001234567890, "Работа", 1, "2024-03-01 10:00", "Вася Пупкин", "глянь", "Пароль от WiFi: ёлка2024", "/a.jpg")
add(db, -1001234567890, "Работа", 2, "2024-06-01 10:00", "Петя", None, "Встреча в пятницу 18:00", "/b.jpg")
add(db, 42, "Мама", 3, "2025-01-01 10:00", "Мама", 'рецепт "борща"', "Свекла 2 шт", "/c.jpg")
add(db, 7, "Другой", 4, "2025-06-01 10:00", "Кто-то", None, "Пороль неверный", "/d.jpg")
add(db, 8, "Пусто", 5, "2025-07-01 10:00", "Кто-то", None, "", "/e.jpg")  # image without text
add(db, 8, "Пусто", 6, "2025-07-01 10:00", "Кто-то", None, "mid", "/f.jpg")  # text shorter than query
# picture 1 (/a.jpg) sent again, in another chat: a second appearance of the same image
add(db, 42, "Мама", 10, "2025-08-01 10:00", "Мама", "пересылаю", "Пароль от WiFi: ёлка2024", "/a.jpg", image=1)


def paths(**kw):
    return [r[3] for r in search(db, **kw)]


# exact (FTS trigram)
assert paths(text="ПАРОЛЬ от wifi") == ["/a.jpg"]  # case-insensitive, Cyrillic + Latin; one row per picture
assert paths(text="елка") == paths(text="ёлка") == ["/a.jpg"]  # ё == е
assert paths(text="борщ") == ["/c.jpg"]  # text also matches captions
assert paths(text='"борща"') == ["/c.jpg"]  # quotes in query don't break FTS syntax
assert paths(caption="глянь", sender="вася") == ["/a.jpg"]
assert paths(chat="работа") == ["/b.jpg", "/a.jpg"]  # newest first
assert paths(chat="работа", after="2024-04-01") == ["/b.jpg"]
assert paths(chat_id=42) == ["/a.jpg", "/c.jpg"] and paths(chat_id=-1001234567890) == ["/b.jpg", "/a.jpg"]  # exact id
assert paths(before="2024-04-01") == ["/a.jpg"]  # no text filter at all
assert paths(text="пятниц", sender="вася") == []
assert paths(text="пороль от wifi") == []  # typo: exact misses
assert paths(text="пароль") == ["/a.jpg"]  # once, though sent twice
assert search(db, text="пароль")[0][1] == "Мама"  # shown via its newest appearance (2025-08)
assert paths(caption="пересылаю") == ["/a.jpg"]  # captions belong to appearances
assert paths(caption="борщ") == ["/c.jpg"] and paths(ocr="борщ") == []  # text = either, ocr/caption = just one
assert paths(ocr="свекла") == ["/c.jpg"] and paths(caption="свекла") == []
assert paths(ocr="свекла", caption="борщ") == ["/c.jpg"] and paths(ocr="свекла", caption="пароль") == []  # AND

# fuzzy
assert paths(text="пороль от wifi", min_score=80) == ["/a.jpg"]  # typo tolerated
assert paths(text="пароль", min_score=80, order="relevance") == ["/a.jpg", "/d.jpg"]  # exact beats newer typo
assert paths(text="пароль", min_score=80, chat="работа") == ["/a.jpg"]  # other filters still apply
assert paths(text="ёлка", min_score=80) == ["/a.jpg"]  # short query: no noise ("свекла" scores 75)
assert paths(text="зебра", min_score=80) == []
assert paths(text="solo mid", min_score=80) == []  # short/empty docs must not "contain" the query
assert paths(text="mid", min_score=80) == ["/f.jpg"]
assert len(paths(text="зебра", min_score=0)) == 6  # 0 = match everything (6 pictures, 7 appearances)
assert paths(ocr="борща", min_score=80) == [] and paths(ocr="свекла", min_score=80) == ["/c.jpg"]  # per field
assert paths(caption="рецепт борша", min_score=80) == ["/c.jpg"]  # message text is fuzzy too now
assert paths(ocr="свекло", caption="борщ", min_score=80) == ["/c.jpg"]  # both must pass
assert paths(ocr="свекло", caption="пароль", min_score=80) == []
edge = connect(":memory:")  # "either" = best of the two fields, not their concatenation (which lost this one)
add(edge, 1, "c", 1, "2025-01-01 10:00", "s", None, "завтра будет пого", "/x.jpg")
assert search(edge, text="погода", min_score=80) == search(edge, ocr="погода", min_score=80) != []
two = search(db, ocr="свекло", caption="борщ", min_score=80)[0][-1]
assert "\x1b[1mСвекл\x1b[0m" in two and "\x1b[1mборщ\x1b[0m" in two, two  # a highlight per query
assert "\x1b[1mПароль от WiFi\x1b[0m" in search(db, text="пороль от wifi", min_score=80)[0][-1]  # match highlighted

# message links, appearances
assert search(db, text="встреча")[0][4] == "https://t.me/c/1234567890/2"  # supergroup/channel
assert search(db, text="борщ", min_score=80)[0][4] == "tg://user?id=42"  # private chat: no message links, open chat
assert link(-555, 9) == "tg://openmessage?chat_id=555&message_id=9"  # basic group: best effort
assert appearances(db, [1, 2]) == {
    1: [("2024-03-01 10:00", "Работа", "Вася Пупкин", "https://t.me/c/1234567890/1", 1, 0),
        ("2025-08-01 10:00", "Мама", "Мама", "tg://user?id=42", 10, 0)],
    2: [("2024-06-01 10:00", "Работа", "Петя", "https://t.me/c/1234567890/2", 2, 0)]}
db.execute("INSERT INTO deleted VALUES (42, 10)")  # `repair` found msg 10 deleted in Telegram: still found, flagged
assert appearances(db, [1])[1][1][-1] == 1 and search(db, msg_id=10)[0][-2] == 1 and search(db, msg_id=1)[0][-2] == 0
db.execute("DELETE FROM deleted")

# message id lookup: picks that appearance (and shows the picture through it)
assert paths(msg_id=10) == ["/a.jpg"] and search(db, msg_id=10)[0][1] == "Мама"
assert search(db, text="пароль", msg_id=1)[0][1] == "Работа"
assert paths(msg_id=10, chat="работа") == []

# albums: msgs 1 and 2 were sent as one album (grouped_id 77), msg 3 alone
db.executemany("INSERT INTO albums VALUES (?,?,?)", [(-1001234567890, 1, 77), (-1001234567890, 2, 77), (42, 3, None)])
assert paths(album=(None, 2)) == ["/a.jpg", "/b.jpg"]  # in message order; /a shown via msg 1, its album appearance
assert paths(album=(-1001234567890, 1), order="new") == ["/a.jpg", "/b.jpg"]  # album order wins over sort
assert paths(album=(None, 3)) == [] and paths(album=(42, 1)) == []  # not an album / wrong chat
assert tgsearch.album_sizes(db, [(-1001234567890, 1), (42, 3), (42, 10)]) == {(-1001234567890, 1): 2}

# sort order: new (default) / old by message date, relevance = best match first
assert paths(chat="работа", order="old") == ["/a.jpg", "/b.jpg"]
early = dict(text="пароль", min_score=80, before="2025-07-01")  # /a only via its 2024-03 appearance here
assert paths(**early) == ["/d.jpg", "/a.jpg"]  # newest first, though /a matches better
assert paths(**early, order="old") == ["/a.jpg", "/d.jpg"]
assert paths(**early, order="relevance") == ["/a.jpg", "/d.jpg"]
assert search(db, text="пароль", min_score=80)[0][0] == "2025-08-01 10:00"  # sent twice: dated by newest appearance
assert search(db, text="пароль", min_score=80, order="old")[0][0] == "2024-03-01 10:00"


# search by meaning (encoder faked: vectors are hand-made, so ranking is known)
def vec(*xs):
    v = np.array(xs, np.float32)
    return (v / np.linalg.norm(v)).astype(np.float16)


for image, v in {1: vec(1, 0, 0), 2: vec(0, 1, 0), 3: vec(.9, .1, 0)}.items():
    db.execute("INSERT INTO emb VALUES ('fake', ?, ?)", (image, v.tobytes()))
tgsearch.encode = lambda name, images=None, text=None: vec(1, 0, 0)[None]
assert paths(meaning="x", model="fake", order="relevance") == ["/a.jpg", "/c.jpg", "/b.jpg"]  # once per picture
assert paths(meaning="x", model="fake", before="2025-07-01") == ["/c.jpg", "/b.jpg", "/a.jpg"]  # newest first
assert paths(meaning="x", model="fake", limit=2, order="old") == ["/a.jpg", "/c.jpg"]  # 2 most similar, by date
assert paths(meaning="x", model="fake", chat="работа", order="relevance") == ["/a.jpg", "/b.jpg"]  # filters apply
assert paths(similar=2, model="fake", order="relevance")[0] == "/b.jpg"  # a picture is most like itself
assert paths(meaning="x", model="other") == []  # vectors are per model
assert search(db, meaning="x", model="fake", order="relevance")[0][-1] == "1.000 meaning similarity"
# pages: each a slice of one ranking + the total; for picture queries, each page = the next best, by date
assert [r[3] for r in search_page(db, chat="работа", limit=1, offset=1)[0]] == ["/a.jpg"]
assert search_page(db, chat="работа", limit=1, offset=1)[1] == 2
assert search_page(db, chat="работа", offset=5) == ([], 2)  # past the end
assert [r[3] for r in search_page(db, text="пароль", min_score=80, limit=1, offset=1)[0]] == ["/d.jpg"]  # fuzzy
assert [r[3] for r in search_page(db, meaning="x", model="fake", limit=1, offset=1, order="relevance")[0]] == ["/c.jpg"]
assert search_page(db, meaning="x", model="fake", limit=2)[1] == 3
# combined: text decides what matches (exact or fuzzy), the picture query ranks it
assert paths(text="встреча", meaning="x", model="fake") == ["/b.jpg"]
assert paths(text="пароль", min_score=80, meaning="x", model="fake", order="relevance") == ["/a.jpg"]  # /d: no vector
assert paths(chat="работа", similar=2, model="fake", order="relevance") == ["/b.jpg", "/a.jpg"]
hit = search(db, text="встреча", min_score=80, meaning="x", model="fake")[0][-1]
assert "\x1b[1mВстреча\x1b[0m" in hit and "(100%)" in hit and hit.endswith("0.000 meaning similarity"), hit
# meaning + similar: z-scored sum. /c (.9,.1) is near both (1,0,0) and (0,1,0)'s mix -> beats each extreme
both = search(db, meaning="x", similar=2, model="fake", order="relevance")
assert [r[3] for r in both][0] == "/c.jpg" and "similarity to #2" in both[0][-1], both

with tempfile.TemporaryDirectory() as d:
    # embed: once per picture, only those missing vectors for that model; unreadable files skipped
    calls = []
    tgsearch.encode = lambda name, images=None, text=None: calls.append(len(images)) or np.ones((len(images), 3), np.float16)
    e = connect(":memory:")
    for i in range(3):
        Image.new("RGB", (8, 8)).save(f"{d}/{i}.jpg")
        add(e, 1, "c", i, "2024-01-01 00:00", "s", None, "", f"{d}/{i}.jpg")
    add(e, 2, "c2", 5, "2024-01-01 00:00", "s", None, "", f"{d}/0.jpg", image=1)  # same picture elsewhere
    add(e, 1, "c", 9, "2024-01-01 00:00", "s", None, "", f"{d}/missing.jpg")
    embed(e, "m1", batch=2)
    embed(e, "m1")  # nothing left to do
    embed(e, "m2")  # another model gets its own vectors
    assert calls == [2, 1, 3], calls
    assert dict(e.execute("SELECT model, count(*) FROM emb GROUP BY model")) == {"m1": 3, "m2": 3}

    # same_picture: recompressed/resized copy = same; same template with different text = different
    def meme(text, size=(400, 300)):
        img = Image.linear_gradient("L").resize(size).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.ellipse((120, 60, 280, 220), fill=(200, 80, 40))
        draw.text((20, 255), text, fill="black", font_size=28)
        return img

    meme("SOME CAPTION HERE").save(f"{d}/m.png")
    meme("SOME CAPTION HERE").resize((300, 225)).save(f"{d}/m_copy.jpg", quality=70)
    meme("OTHER WORDS THERE").save(f"{d}/m_other.png")
    assert (phash(f"{d}/m.png") ^ phash(f"{d}/m_copy.jpg")).bit_count() <= tgsearch.DUP_HASH_BITS
    # a PNG whose header claims 2.5G pixels: unreadable (no fingerprint), not a crash, nothing decoded
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    Path(f"{d}/bomb.png").write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 50000, 50000,
                                      8, 2, 0, 0, 0)) + chunk(b"IEND", b""))
    assert phash(f"{d}/bomb.png") is None
    assert same_picture(f"{d}/m.png", f"{d}/m_copy.jpg", "", "")
    assert not same_picture(f"{d}/m.png", f"{d}/m_other.png", "SOME CAPTION HERE", "OTHER WORDS THERE")
    assert not same_picture(f"{d}/m.png", f"{d}/nope.png", "", "")  # unreadable: never merge
    # identical pixels but clearly different text (caption changed in a tiny area): OCR vetoes
    assert not same_picture(f"{d}/m.png", f"{d}/m_copy.jpg", "THE PRESIDENT Sequence of Events YOU enter the room",
                            "THE PRESIDENT Sequence of Events УТРОМ ПРЕСС КАЧАТ ТУРНИК")
    assert same_picture(f"{d}/m.png", f"{d}/m_copy.jpg", "THEN ДЕН", "ДЕН")  # OCR noise is not a veto

    # dedup: copies merge into the biggest file, every appearance kept, other picture untouched
    tgsearch.trash = lambda p: Path(p).unlink()  # not the real Trash in tests
    x = connect(":memory:")
    add(x, 1, "A", 1, "2024-01-01 00:00", "s", "first", "SOME CAPTION HERE", f"{d}/m_copy.jpg")
    add(x, 2, "B", 7, "2024-02-01 00:00", "s", "second", "S0ME CAPTlON HERE", f"{d}/m.png")
    add(x, 3, "C", 9, "2024-03-01 00:00", "s", None, "OTHER WORDS THERE", f"{d}/m_other.png")
    x.execute("INSERT INTO emb VALUES ('m', 1, x'00'), ('m', 2, x'00')")
    dedup(x)
    assert x.execute("SELECT id, path FROM images ORDER BY id").fetchall() == [(2, f"{d}/m.png"), (3, f"{d}/m_other.png")]
    assert sorted(x.execute("SELECT msg_id, image, path, ocr FROM media")) == [  # each keeps its own OCR text
        (1, 2, f"{d}/m.png", "SOME CAPTION HERE"), (7, 2, f"{d}/m.png", "S0ME CAPTlON HERE"),
        (9, 3, f"{d}/m_other.png", "OTHER WORDS THERE")]
    assert x.execute("SELECT image FROM emb").fetchall() == [(2,)]
    assert not Path(f"{d}/m_copy.jpg").exists() and Path(f"{d}/m.png").exists()
    assert [r[1] for r in search(x, caption="first")] == ["A"]  # appearance's own caption still searchable

    # tags: labels closest to a picture, minus half their library-wide closeness ("meme" fits everything)
    (Path(d) / "tags.txt").write_text("# comment\nmeme\ncat\ndog\n")
    tgsearch.HERE = Path(d)
    lab = {"meme": vec(1, 1, 1), "cat": vec(1, 0, 0), "dog": vec(0, 1, 0)}
    tgsearch.encode = lambda name, images=None, text=None: np.stack([lab[t] for t in text])
    g = connect(":memory:")
    for image, v in {1: vec(1, .1, .3), 2: vec(.1, 1, .3), 3: vec(.3, .3, 1)}.items():
        g.execute("INSERT INTO emb VALUES ('m', ?, ?)", (image, v.tobytes()))
    assert tgsearch.tags(g, [1, 2, 3], "m", k=2) == {1: ["cat", "meme"], 2: ["dog", "meme"], 3: ["meme", "cat"]}
    assert tgsearch.tags(g, [1], "nope") == {}  # model without vectors: no tags, no crash

    # migration from the pre-dedup layout (row + vector per message): lossless, one image per message
    old = sqlite3.connect(f"{d}/old.db")
    old.execute("""CREATE VIRTUAL TABLE media USING fts5(ocr, caption, sender, chat, chat_id UNINDEXED,
        msg_id UNINDEXED, date UNINDEXED, path UNINDEXED, tokenize='trigram')""")
    old.execute("CREATE TABLE emb (model TEXT, chat_id INTEGER, msg_id INTEGER, vec BLOB, PRIMARY KEY (model, chat_id, msg_id))")
    old.execute("INSERT INTO media VALUES ('текст', 'подпись', 'Вася', 'Чат', 5, 50, '2024-01-01 00:00', '/x.jpg')")
    old.execute("INSERT INTO media VALUES ('', '', 'Петя', 'Чат', 5, 51, '2024-01-02 00:00', '/y.jpg')")
    old.execute("INSERT INTO emb VALUES ('m', 5, 51, x'01')")
    old.commit()
    old.close()
    new = connect(f"{d}/old.db")
    assert new.execute("SELECT * FROM images").fetchall() == [(1, "/x.jpg", None, None), (2, "/y.jpg", None, None)]
    assert new.execute("SELECT msg_id, image, caption FROM media").fetchall() == [(50, 1, "подпись"), (51, 2, "")]
    assert new.execute("SELECT * FROM emb").fetchall() == [("m", 2, b"\x01")]
    assert Path(f"{d}/old.db.bak-pre-dedup").exists()
    assert [r[3] for r in search(new, text="текст")] == ["/x.jpg"]

# sync: photos and images sent as files come as two streams, merged back into message order (= resume order)
async def msgs(*ids):
    for i in ids:
        yield SimpleNamespace(id=i)


async def merged(a, b):
    return [m.id async for m in tgsearch.merge(msgs(*a), msgs(*b))]
assert asyncio.run(merged([1, 4, 5], [2, 3, 6, 7])) == [1, 2, 3, 4, 5, 6, 7]
assert asyncio.run(merged([], [2])) == [2] and asyncio.run(merged([1], [])) == [1] and asyncio.run(merged([], [])) == []

# repair, against a fake Telegram: msg 3 deleted there, msg 4's caption edited, the picture of msgs 1 and 2 lost
# its file (saved from msg 2: downloaded from there, not from the copy in msg 1), msg 6's path is outside the folder
with tempfile.TemporaryDirectory() as d:
    tgsearch.HERE = Path(d)
    tgsearch.embed = lambda db: None  # tested above
    (Path(d) / "media/5").mkdir(parents=True)
    for mid in (3, 4):
        Image.new("RGB", (8, 8)).save(f"{d}/media/5/{mid}.jpg")
    r = connect(":memory:")
    add(r, 5, "Old name", 1, "2024-01-01 00:00", "s", "", "", f"{d}/media/5/2.jpg")
    add(r, 5, "Old name", 2, "2024-01-02 00:00", "s", "", "", f"{d}/media/5/2.jpg", image=1)
    add(r, 5, "Old name", 3, "2024-01-03 00:00", "s", "", "", f"{d}/media/5/3.jpg")
    add(r, 5, "Old name", 4, "2024-01-04 00:00", "s", "old", "", f"{d}/media/5/4.jpg")
    add(r, 5, "Old name", 6, "2024-01-06 00:00", "s", "", "", "/elsewhere/media/5/6.jpg")
    alive = {i: SimpleNamespace(id=i, message="new ё" if i == 4 else "") for i in (1, 2, 4, 6)}
    fetched = []

    class FakeTelegram:
        def get_messages(self, chat, ids):
            return [alive.get(i) for i in ids]

        def download_media(self, m, file):
            fetched.append(m.id)
            Image.new("RGB", (8, 8)).save(file, "JPEG")
            return file
    tgsearch.repair(FakeTelegram(), r, [SimpleNamespace(id=5, name="Новое имя")])
    assert r.execute("SELECT * FROM deleted").fetchall() == [(5, 3)] and search(r, msg_id=3)[0][-2] == 1
    assert r.execute("SELECT caption FROM media WHERE msg_id = 4").fetchone() == ("new е",)
    assert fetched == [2] and Path(f"{d}/media/5/2.jpg").exists() and not Path("/elsewhere").exists()
    assert {c for (c,) in r.execute("SELECT chat FROM media")} == {"Новое имя"}
    tgsearch.repair(FakeTelegram(), r, [SimpleNamespace(id=5, name="Новое имя")])  # again: nothing to do
    assert fetched == [2] and r.execute("SELECT count(*) FROM deleted").fetchone() == (1,)
print("ok")
