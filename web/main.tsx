import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import {
  Anchor,
  Badge,
  Box,
  Button,
  Checkbox,
  CloseButton,
  Container,
  Divider,
  Group,
  Highlight,
  Image,
  Loader,
  MantineProvider,
  Mark,
  NumberInput,
  Pagination,
  Select,
  Spoiler,
  Stack,
  Text,
  TextInput,
} from "@mantine/core";
import { useDebouncedValue } from "@mantine/hooks";
import { Notifications, notifications } from "@mantine/notifications";
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

type Appearance = { date: string; chat: string; sender: string; link: string; msg_id: number };
type Hit = {
  image_id: number; // a picture is stored once; `appearances` = every message it was sent in
  appearances: Appearance[];
  msg_id: number;
  chat_id: number;
  album: number; // photos in this message's album (0 = not part of one)
  date: string;
  chat: string;
  sender: string;
  image: string;
  link: string;
  snippet: string;
  ocr: string;
  caption: string;
};
type Chat = { name: string; count: number }; // also used for /api/models

const PAGE = 60;
const EMPTY = {
  ocr: "", // text in the picture (OCR)
  chat: "",
  sender: "",
  caption: "", // text in the message
  after: "",
  before: "",
  exact: false,
  min_score: 80 as number | string,
  meaning: "", // description of what's in the picture (embeddings); text boxes then only filter
  model: "", // embedding model; empty = server default
  similar: "", // image id of a picture to find look-alikes of
  msg: "", // exact message id
  album: "", // "chat_id:msg_id": show that message's whole album
  order: "new", // new / old: by message date; relevance: best match first
  page: 1,
};
const NO_IMAGE = // shown for messages whose photo is gone (e.g. expired self-destructing ones)
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="140" height="140"><rect width="140" height="140" fill="#8883"/>' +
      '<text x="70" y="74" font-family="sans-serif" font-size="12" fill="#888" text-anchor="middle">no image</text></svg>',
  );
type Filters = typeof EMPTY;

// filters <-> query string, non-defaults only: it's both the page URL (shareable, survives reload/back) and the API query
const toQuery = (f: Filters) =>
  new URLSearchParams(
    Object.entries(f)
      .filter(([k, v]) => v !== "" && v !== EMPTY[k as keyof Filters])
      .map(([k, v]) => [k, String(v)]),
  ).toString();
function fromUrl(): Filters {
  const f: Record<string, unknown> = { ...EMPTY };
  new URLSearchParams(location.search).forEach((v, k) => {
    const d = EMPTY[k as keyof Filters];
    if (d !== undefined) f[k] = typeof d === "boolean" ? v === "true" : typeof d === "number" ? (Number.isNaN(+v) ? d : +v) : v;
  });
  return f as Filters;
}

const fail = (message: string) => notifications.show({ color: "red", message });

// navigator.clipboard exists only on https/localhost pages; opened as http://<lan-ip> (e.g. from a phone) it's
// undefined, so fall back to the legacy copy command, which still works inside a click handler
function copyText(text: string) {
  if (navigator.clipboard) return navigator.clipboard.writeText(text);
  const area = Object.assign(document.createElement("textarea"), { value: text, readOnly: true }); // no keyboard
  area.style.cssText = "position:fixed;top:0;opacity:0"; // no scroll jump
  document.body.append(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  return ok ? Promise.resolve() : Promise.reject();
}

function Result({
  hit,
  ocrQuery,
  captionQuery,
  tags,
  onSimilar,
  onTag,
  onAlbum,
}: {
  hit: Hit;
  ocrQuery: string;
  captionQuery: string;
  tags?: string[]; // what the embedding model sees in the picture (zero-shot labels)
  onSimilar?: () => void;
  onTag: (tag: string) => void;
  onAlbum: () => void;
}) {
  // search snippets mark matches with the CLI's ANSI bold codes: split on them, odd parts are matches
  const parts = hit.snippet.split(/\x1b\[[01]m/);
  const matched = parts.length > 1;
  const copy = () =>
    copyText(hit.link).then(
      () => notifications.show({ color: "green", message: "Link copied" }),
      () => fail("Couldn't copy"),
    );

  return (
    <Group align="flex-start" wrap="nowrap" gap="md" py="md">
      <Anchor href={hit.image} target="_blank" rel="noreferrer" style={{ flexShrink: 0 }}>
        <Image
          src={hit.image}
          fallbackSrc={NO_IMAGE}
          w={{ base: 88, sm: 140 }}
          h={{ base: 88, sm: 140 }}
          fit="cover"
          radius="sm"
          loading="lazy"
        />
      </Anchor>
      <Stack gap={6} miw={0} style={{ flex: 1 }}>
        <Text size="xs" c="dimmed">
          {hit.date} · {hit.chat} / {hit.sender} · msg {hit.msg_id}
        </Text>
        {tags && tags.length > 0 && (
          <Group gap={4} title="What the model sees · click to search by meaning">
            {tags.map((t) => (
              <Badge key={t} size="sm" variant="light" color="gray" tt="none" style={{ cursor: "pointer" }} onClick={() => onTag(t)}>
                {t}
              </Badge>
            ))}
          </Group>
        )}
        {hit.appearances.length > 1 && (
          <Text size="xs" c="dimmed">
            Sent {hit.appearances.length} times:{" "}
            {hit.appearances.map((a, i) => (
              <span key={i}>
                {i > 0 && " · "}
                <Anchor href={a.link} inherit title={`msg ${a.msg_id} · by ${a.sender}`}>
                  {a.chat}, {a.date.slice(0, 10)}
                </Anchor>
              </span>
            ))}
          </Text>
        )}
        {matched && (
          <Text size="sm" style={{ overflowWrap: "anywhere" }}>
            {parts.map((s, i) => (i % 2 ? <Mark key={i}>{s}</Mark> : s))}
          </Text>
        )}
        {(hit.caption || hit.ocr) && (
          // with a match shown above, the full text starts collapsed; otherwise show its first lines
          <Spoiler
            maxHeight={matched ? 0 : 64}
            showLabel="Show full text"
            hideLabel="Hide full text"
            styles={{ control: { fontSize: "var(--mantine-font-size-sm)" } }}
          >
            <Stack gap={4} style={{ overflowWrap: "anywhere" }}>
              {hit.caption && (
                <Highlight highlight={captionQuery} size="sm">
                  {hit.caption}
                </Highlight>
              )}
              {hit.ocr && (
                <Highlight highlight={ocrQuery} size="sm" c="dimmed">
                  {hit.ocr}
                </Highlight>
              )}
            </Stack>
          </Spoiler>
        )}
        <Group gap="xs">
          {hit.link.startsWith("https://t.me/") ? (
            <Button component="a" href={hit.link} size="compact-sm" variant="light">
              Open message
            </Button>
          ) : (
            <Button
              component="a"
              href={hit.link}
              size="compact-sm"
              variant="light"
              title={`Telegram has no links to messages in private chats: this opens the chat. Jump to ${hit.date.slice(0, 10)} with its calendar.`}
            >
              Open chat
            </Button>
          )}
          <Button size="compact-sm" variant="subtle" onClick={copy}>
            Copy link
          </Button>
          {onSimilar && (
            <Button size="compact-sm" variant="subtle" onClick={onSimilar}>
              Similar
            </Button>
          )}
          {hit.album > 1 && (
            <Button size="compact-sm" variant="subtle" onClick={onAlbum}>
              Album ({hit.album})
            </Button>
          )}
        </Group>
      </Stack>
    </Group>
  );
}

function App() {
  const [filters, setFilters] = useState(fromUrl);
  const [debounced] = useDebouncedValue(filters, 250);
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [models, setModels] = useState<Chat[]>([]);
  const [similarTo, setSimilarTo] = useState<Hit | null>(null);
  const [tags, setTags] = useState<Record<number, string[]>>({});
  const pages = Math.ceil(total / PAGE);
  const set = <K extends keyof Filters>(key: K, value: Filters[K]) =>
    setFilters({
      ...filters,
      [key]: value,
      page: key === "page" ? (value as number) : 1, // any other change starts over at page 1
    });
  const searchTag = (tag: string) => {
    setFilters({ ...filters, meaning: tag, page: 1 });
    window.scrollTo({ top: 0 });
  };
  const openAlbum = (hit: Hit) => {
    setFilters({ ...filters, album: `${hit.chat_id}:${hit.msg_id}`, ocr: "", caption: "", meaning: "", msg: "", similar: "", page: 1 });
    window.scrollTo({ top: 0 });
  };
  const findSimilar = (hit: Hit) => {
    setSimilarTo(hit); // starts from scratch; text/meaning typed afterwards narrow it down
    setFilters({ ...filters, similar: String(hit.image_id), ocr: "", caption: "", meaning: "", album: "", page: 1 });
    window.scrollTo({ top: 0 });
  };

  useEffect(() => {
    for (const [url, setter] of [["/api/chats", setChats], ["/api/models", setModels]] as const)
      fetch(url)
        .then((r) => r.json())
        .then(setter)
        .catch(() => fail(`Couldn't load ${url}`));
  }, []);

  useEffect(() => {
    const back = () => setFilters(fromUrl()); // browser back/forward: restore that search
    window.addEventListener("popstate", back);
    return () => window.removeEventListener("popstate", back);
  }, []);

  useEffect(() => {
    const query = toQuery(debounced);
    if (query !== location.search.slice(1)) {
      // a new page is a history step (back = previous page); typing just updates the current entry
      const paged = String(debounced.page) !== (new URLSearchParams(location.search).get("page") ?? "1");
      history[paged ? "pushState" : "replaceState"](null, "", query ? `?${query}` : location.pathname);
    }
    const ctl = new AbortController();
    setLoading(true);
    fetch(`/api/search?${query}&limit=${PAGE}`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((r: { total: number; hits: Hit[] }) => {
        setHits(r.hits);
        setTotal(r.total);
      })
      .catch((e) => e.name !== "AbortError" && fail(`Search failed: ${e.message}`))
      .finally(() => !ctl.signal.aborted && setLoading(false));
    return () => ctl.abort();
  }, [debounced]);

  useEffect(() => {
    const missing = (hits ?? []).map((h) => h.image_id).filter((id) => !(id in tags));
    if (!missing.length || !models.length) return;
    const ctl = new AbortController();
    fetch(`/api/tags?images=${missing.join(",")}`, { signal: ctl.signal })
      .then((r) => r.json())
      .then((t) => setTags((prev) => ({ ...prev, ...t })))
      .catch((e) => e.name !== "AbortError" && fail("Couldn't load tags"));
    return () => ctl.abort();
  }, [hits, models.length]);

  return (
    <Container size="md" pb="xl">
      <Box pos="sticky" top={0} py="sm" bg="var(--mantine-color-body)" style={{ zIndex: 1 }}>
        <Group gap="xs">
          <TextInput
            size="md"
            autoFocus
            style={{ flex: "1 1 200px" }}
            rightSection={loading && <Loader size="xs" />}
            placeholder="Text in the picture"
            value={filters.ocr}
            onChange={(e) => set("ocr", e.currentTarget.value)}
          />
          <TextInput
            size="md"
            style={{ flex: "1 1 200px" }}
            placeholder="Message text"
            value={filters.caption}
            onChange={(e) => set("caption", e.currentTarget.value)}
          />
          {models.length > 0 && (
            <TextInput
              size="md"
              style={{ flex: "1 1 200px" }}
              placeholder="What's in it: dog in snow, мем…"
              value={filters.meaning}
              onChange={(e) => set("meaning", e.currentTarget.value)}
            />
          )}
        </Group>
        <Group gap="xs" mt="xs">
          <Select
            size="xs"
            placeholder="All chats"
            searchable
            clearable
            data={chats.map((c) => ({ value: c.name, label: `${c.name} (${c.count})` }))}
            value={filters.chat || null}
            onChange={(v) => set("chat", v ?? "")}
          />
          <TextInput size="xs" placeholder="Sender" value={filters.sender} onChange={(e) => set("sender", e.currentTarget.value)} />
          <TextInput
            size="xs"
            type="date"
            leftSection={<Text size="xs" c="dimmed">From</Text>}
            leftSectionWidth={40}
            value={filters.after}
            onChange={(e) => set("after", e.currentTarget.value)}
          />
          <TextInput
            size="xs"
            type="date"
            leftSection={<Text size="xs" c="dimmed">Before</Text>}
            leftSectionWidth={50}
            value={filters.before}
            onChange={(e) => set("before", e.currentTarget.value)}
          />
          <TextInput
            size="xs"
            w={110}
            placeholder="Message id"
            inputMode="numeric"
            value={filters.msg}
            onChange={(e) => set("msg", e.currentTarget.value.replace(/\D/g, ""))}
          />
          <Select
            size="xs"
            w={120}
            allowDeselect={false}
            data={[
              { value: "new", label: "Newest" },
              { value: "old", label: "Oldest" },
              { value: "relevance", label: "Best match" },
            ]}
            value={filters.order}
            onChange={(v) => set("order", v ?? "new")}
            title={filters.meaning || filters.similar ? "By date = the most similar pictures, ordered by date" : undefined}
          />
          <Checkbox size="xs" label="Exact" checked={filters.exact} onChange={(e) => set("exact", e.currentTarget.checked)} />

          {!filters.exact && (
            <NumberInput
              size="xs"
              w={110}
              prefix="score ≥ "
              min={0}
              max={100}
              title="Fuzzy match threshold: lower finds more, with more noise"
              value={filters.min_score}
              onChange={(v) => set("min_score", v)}
            />
          )}
          {models.length > 1 && (filters.meaning || filters.similar) && (
            <Select
              size="xs"
              allowDeselect={false}
              data={models.map((m) => ({ value: m.name, label: `${m.name} (${m.count})` }))}
              value={filters.model || models[0].name}
              onChange={(v) => set("model", v ?? "")}
            />
          )}
        </Group>
        {filters.album && (
          <Group gap={4} mt="xs">
            <Text size="sm">
              Album{hits?.length ? ` of ${total} photos · ${hits[0].date} · ${hits[0].chat}` : ""}
            </Text>
            <CloseButton size="sm" aria-label="Close album" onClick={() => set("album", "")} />
          </Group>
        )}
        {filters.similar && (
          <Group gap={4} mt="xs">
            <Text size="sm">
              Similar to{" "}
              {similarTo?.image_id === Number(filters.similar) // after a reload only the id is known
                ? `the photo from ${similarTo.date} · ${similarTo.chat}`
                : `picture #${filters.similar}`}
            </Text>
            <CloseButton size="sm" aria-label="Clear" onClick={() => set("similar", "")} />
          </Group>
        )}
      </Box>

      <Text size="sm" c="dimmed" my="xs">
        {hits === null
          ? "Loading…"
          : `${total} result${total === 1 ? "" : "s"}${pages > 1 ? ` · page ${debounced.page} of ${pages}` : ""}`}
      </Text>
      <Stack gap={0}>
        {hits?.map((h) => (
          <div key={h.image_id}>
            <Divider />
            <Result
              hit={h}
              ocrQuery={debounced.ocr}
              captionQuery={debounced.caption}
              tags={tags[h.image_id]}
              onTag={searchTag}
              onAlbum={() => openAlbum(h)}
              onSimilar={models.length ? () => findSimilar(h) : undefined}
            />
          </div>
        ))}
      </Stack>
      {pages > 1 && (
        <Group justify="center" mt="md">
          <Pagination
            total={pages}
            value={filters.page}
            siblings={1}
            size="sm"
            gap={4} // one line at phone width even mid-range: ‹ 1 … 149 150 151 … 303 ›
            onChange={(p) => {
              set("page", p);
              window.scrollTo({ top: 0 });
            }}
          />
        </Group>
      )}
    </Container>
  );
}

createRoot(document.getElementById("root")!).render(
  <MantineProvider defaultColorScheme="auto">
    <Notifications position="bottom-center" />
    <App />
  </MantineProvider>,
);
