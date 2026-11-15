// The best free AI model aggregator: one server in front of ten simulated providers.
//
// What it does, in order:
//   1. Reads the trial brief at $MAKEFUN_PROMPT_FILE (the provider directory, the sealed quality
//      table, the workload manifest) and probes every provider's catalog route in its own wire.
//   2. Builds the catalog document from the listings and registers it at $MAKEFUN_ENV_URL/register
//      with its port. `--list` prints the same document and exits; outside a trial there is no brief
//      and the document is empty.
//   3. Serves the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in SSE
//      streaming, translating between wires through a canonical form (lib/wires.ts) so the system
//      prompt, the last user turn, the tool names, max_tokens, the stops, an image and a JSON-mode
//      ask all echo exactly, and the provider's usage and answer text come back unchanged.
//   4. Polls $MAKEFUN_ENV_URL/work and exits 0 when the trial reports done.
//
// Routing is the game the scoring sets. Only free models serve. Each provider spends a per-trial
// budget on every call that names one of its free models, scheduled refusals included; a spent
// provider refuses for the rest of the trial, and one provider in each trial stops answering partway
// through. So the entry plans before the first request arrives: the manifest is assigned greedily,
// heaviest requests first, each to the best free model that meets its requirements, rationed by the
// smallest budget each provider is known to carry — the weight-2 requests and the scarce-requirement
// ones (vision, a 200k context, JSON mode) get first claim on the top-rated models, and the plain
// requests, which anyone can serve at nearly the same rating, take what is left. At run time the
// planned provider is tried first and any refusal (402, 429, 5xx) fails over down the rating order —
// with a fresh provider, whose first call cannot be a scheduled refusal, preferred over a called one
// unless the called one rates far higher — skipping providers that have failed twice in a row or
// answered with an outage; every call is made
// one at a time through a single queue, so the one provider that serves a single call at a time is
// never asked twice at once, and no speculative duplicate call is ever spent. Nothing in a decision
// path reads a clock or a random source: the plan and every failover follow from the brief and from
// the answers the providers gave, in order, so a replay walks the same path.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  joinBase,
  modelFree,
  probeProvider,
  readBrief,
  type Brief,
  type Catalog,
  type Probe,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
  type WorkloadRequest,
} from './lib/brief.ts';
import {
  buildAnthropic,
  buildGemini,
  buildOpenAI,
  extractAnthropic,
  extractOpenAI,
  isRecord,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type CanonAnswer,
  type CanonRequest,
  type Json,
} from './lib/wires.ts';

const ANTHROPIC_VERSION = '2023-06-01';
const CALL_TIMEOUT_MS = 25_000;
/**
 * How much rating a provider that has already been called gives up against a fresh one when attempts
 * are ordered: refusals are scheduled from a provider's second call on, so a fresh provider's first
 * call is the one call in the trial that cannot be a scheduled refusal. The value is a cap, not a
 * nudge — freshness wins any near contest, but a far higher rating still goes first. Tuned on the
 * practice instances: 8 and 20 changed nothing, 40 and this value cut the wasted calls, and an
 * unbounded preference scored no better than the cap.
 */
const FRESH_PENALTY_MILLI = 200;

/**
 * The budget bounds each provider is known to carry, from the competition's provider table: the plan
 * rations by the smallest budget a provider can have, and no provider is asked past the largest.
 * A provider the table does not name is rationed as if it carried the smallest budget seen here.
 */
const CALLS_MIN: ReadonlyMap<string, number> = new Map([
  ['groq', 9],
  ['cerebras', 5],
  ['gemini', 6],
  ['zai', 4],
  ['openrouter', 10],
  ['sambanova', 4],
  ['pollinations', 10],
  ['cloudflare', 7],
  ['ovh', 5],
  ['kilol', 12],
]);
const CALLS_MAX: ReadonlyMap<string, number> = new Map([
  ['groq', 12],
  ['cerebras', 6],
  ['gemini', 8],
  ['zai', 5],
  ['openrouter', 12],
  ['sambanova', 5],
  ['pollinations', 14],
  ['cloudflare', 8],
  ['ovh', 6],
  ['kilol', 16],
]);
const callsMin = (providerId: string): number => CALLS_MIN.get(providerId) ?? 4;
const callsMax = (providerId: string): number => CALLS_MAX.get(providerId) ?? 6;

/** One routable free model: the provider that serves it and the quality table's row for it. */
interface Candidate {
  provider: ProviderEntry;
  row: RatingRow;
  /** The provider's place in the brief's directory, the stable tie-break. */
  order: number;
}

/** What the trial has taught the router about one provider so far. */
interface ProviderState {
  /** Inference calls attempted, in order; a provider's refusals are scheduled by this count. */
  made: number;
  /** Refusals since the last success; two in a row mark the provider down. */
  consecFails: number;
  /** Down for the rest of the trial: out of budget, or in its outage. */
  down: boolean;
}

/** Whether a candidate meets a request's requirements, on the quality table's word. */
function meets(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (typeof requires.min_context === 'number' && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

/**
 * Every free model that can be routed: named by the quality table for a provider whose catalog probe
 * answered, and present in that provider's own listing (a model the listing does not name would be
 * refused as unknown). Paid models never become candidates.
 */
function buildCandidates(brief: Brief, probes: readonly Probe[]): Candidate[] {
  const out: Candidate[] = [];
  brief.providers.forEach((provider, order) => {
    const probe = probes.find((p) => p.provider.id === provider.id);
    if (probe === undefined || !probe.ok) return;
    const listed = new Set(probe.models.map((m) => m.id));
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      if (!listed.has(row.model)) continue;
      const listedModel = probe.models.find((m) => m.id === row.model);
      if (listedModel === undefined || !modelFree(listedModel, row)) continue;
      out.push({ provider, row, order });
    }
  });
  return out;
}

/** The candidates for one request, best first: rating, then the brief's order, then the model id. */
function ranked(candidates: readonly Candidate[], requires: Requirement, jsonMode: boolean): Candidate[] {
  return candidates
    .filter((c) => meets(c.row, requires))
    .filter((c) => !(jsonMode && c.provider.wire === 'anthropic'))
    .sort((a, b) => b.row.rating_milli - a.row.rating_milli || a.order - b.order || (a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0));
}

/**
 * The plan, made before the first request: the manifest in weight order (heaviest first, manifest
 * order within a weight), each request assigned the best candidate whose provider still has planned
 * room under the smallest budget it can carry. A request whose requirements the manifest states as
 * JSON mode on the OpenAI API will carry a response_format, which an Anthropic-wire provider cannot
 * echo, so those are planned away from that wire from the start.
 */
function planAssignments(candidates: readonly Candidate[], manifest: readonly WorkloadRequest[]): Map<number, Candidate> {
  const planned = new Map<string, number>();
  const order = manifest.map((entry, index) => ({ entry, index })).sort((a, b) => b.entry.weight - a.entry.weight || a.index - b.index);
  const plan = new Map<number, Candidate>();
  for (const { entry, index } of order) {
    const jsonMode = entry.api === 'openai' && entry.requires.json === true;
    const options = ranked(candidates, entry.requires, jsonMode).sort(
      (a, b) =>
        b.row.rating_milli - a.row.rating_milli ||
        (planned.get(a.provider.id) ?? 0) - (planned.get(b.provider.id) ?? 0) ||
        a.order - b.order ||
        (a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0),
    );
    if (options.length === 0) continue;
    const pick = options.find((c) => (planned.get(c.provider.id) ?? 0) < callsMin(c.provider.id)) ?? options[0];
    if (pick === undefined) continue;
    plan.set(index, pick);
    planned.set(pick.provider.id, (planned.get(pick.provider.id) ?? 0) + 1);
  }
  return plan;
}

/** The loaded trial: the brief, the probes, the catalog, the candidates and the plan. */
interface Loaded {
  brief: Brief;
  catalog: Catalog;
  candidates: Candidate[];
  plan: Map<number, Candidate>;
}

/** Read the brief and probe every provider once, in directory order. */
async function load(envUrl: string): Promise<Loaded> {
  const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
  const probes: Probe[] = [];
  for (const provider of brief.providers) probes.push(await probeProvider(envUrl, provider));
  const candidates = buildCandidates(brief, probes);
  return { brief, catalog: buildCatalog(brief, probes), candidates, plan: planAssignments(candidates, brief.manifest) };
}

/** One provider call, queued: the outcome is the parsed answer, or the refusal's status. */
type CallOutcome = { ok: true; answer: CanonAnswer } | { ok: false; status: number };

/** The request body for one provider: a same-wire body passes through with only the model renamed. */
function upstreamBody(candidate: Candidate, canon: CanonRequest, raw: Json, clientApi: 'openai' | 'anthropic'): { url: string; body: Json } {
  const base = joinBase(envUrl(), candidate.provider.base);
  const wire = candidate.provider.wire;
  if (wire === 'gemini') {
    const verb = canon.stream ? 'streamGenerateContent' : 'generateContent';
    return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.row.model)}:${verb}`, body: buildGemini(canon, candidate.row.model) };
  }
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const body: Json = { ...raw, model: candidate.row.model };
      if (canon.stream) body.stream_options = { include_usage: true };
      return { url: `${base}/v1/chat/completions`, body };
    }
    return { url: `${base}/v1/chat/completions`, body: buildOpenAI(canon, candidate.row.model) };
  }
  if (clientApi === 'anthropic') return { url: `${base}/v1/messages`, body: { ...raw, model: candidate.row.model } };
  return { url: `${base}/v1/messages`, body: buildAnthropic(canon, candidate.row.model) };
}

/** Read one provider answer, in its wire, JSON or streamed as the request asked. */
function parseAnswer(wire: ProviderEntry['wire'], stream: boolean, text: string): CanonAnswer | null {
  if (wire === 'openai') return stream ? parseOpenAIStream(text) : parseJsonAnswer(text, parseOpenAIJson);
  if (wire === 'anthropic') return stream ? parseAnthropicStream(text) : parseJsonAnswer(text, parseAnthropicJson);
  return stream ? parseGeminiStream(text) : parseJsonAnswer(text, parseGeminiJson);
}

function parseJsonAnswer(text: string, parse: (body: unknown) => CanonAnswer | null): CanonAnswer | null {
  try {
    return parse(JSON.parse(text));
  } catch {
    return null;
  }
}

/** The env's origin, read where a call is made so `--list` and the server share one spelling. */
function envUrl(): string {
  return process.env.MAKEFUN_ENV_URL ?? '';
}

class Aggregator {
  private readonly loaded: Loaded;
  private readonly states = new Map<string, ProviderState>();
  private queue: Promise<unknown> = Promise.resolve();
  private arrivals = 0;

  constructor(loaded: Loaded) {
    this.loaded = loaded;
    for (const provider of loaded.brief.providers) this.states.set(provider.id, { made: 0, consecFails: 0, down: false });
  }

  /** Run one provider call at a time, in the order requests arrive: no provider is ever asked twice at once. */
  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn, fn);
    this.queue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  private state(providerId: string): ProviderState {
    let s = this.states.get(providerId);
    if (s === undefined) {
      s = { made: 0, consecFails: 0, down: false };
      this.states.set(providerId, s);
    }
    return s;
  }

  /** One call to one provider for one request: the answer, or the refusal learned from. */
  private async call(candidate: Candidate, canon: CanonRequest, raw: Json, clientApi: 'openai' | 'anthropic'): Promise<CallOutcome> {
    const state = this.state(candidate.provider.id);
    state.made += 1;
    const { url, body } = upstreamBody(candidate, canon, raw, clientApi);
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      accept: canon.stream ? 'text/event-stream' : 'application/json',
      ...authHeaders(candidate.provider.auth, candidate.provider.key),
    };
    if (candidate.provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
    let status = 0;
    let text = '';
    try {
      const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
      status = res.status;
      text = await res.text();
    } catch {
      return { ok: false, status: 0 };
    }
    if (status < 200 || status >= 300) return { ok: false, status };
    const answer = parseAnswer(candidate.provider.wire, canon.stream, text);
    if (answer === null) return { ok: false, status };
    return { ok: true, answer };
  }

  /** Note a refusal: an outage ends the provider for the trial; two refusals in a row do the same. */
  private noteFailure(candidate: Candidate, status: number): void {
    const state = this.state(candidate.provider.id);
    if (status >= 500) {
      state.down = true;
      return;
    }
    state.consecFails += 1;
    if (state.consecFails >= 2) state.down = true;
  }

  /**
   * Serve one client request. The first attempt goes to the planned provider; after that, attempts
   * are ordered by rating with a freshness adjustment — a provider not yet called in this trial
   * always answers its first call (refusals are scheduled from a provider's second call on), so a
   * fresh provider is worth up to FRESH_PENALTY_MILLI of rating against one whose next call may be
   * a scheduled refusal. Providers in good standing are preferred, then any provider with budget
   * room, then
   * anyone at all; the order is recomputed after every refusal, and no provider is tried twice for
   * the same request. The first answer that parses wins.
   */
  async serve(clientApi: 'openai' | 'anthropic', raw: Json): Promise<{ status: number; contentType: string; body: string }> {
    const index = this.arrivals;
    this.arrivals += 1;
    const canon = clientApi === 'openai' ? extractOpenAI(raw) : extractAnthropic(raw);
    const entry = this.loaded.brief.manifest[index];
    const requires: Requirement = { ...(entry?.requires ?? {}) };
    if (canon.tools.length > 0) requires.tools = true;
    if (canon.messages.some((m) => m.role === 'user' && m.parts.some((p) => p.kind === 'image' || p.kind === 'imageUrl'))) requires.vision = true;
    if (canon.jsonMode) requires.json = true;
    const options = ranked(this.loaded.candidates, requires, canon.jsonMode);
    const planned = entry === undefined ? undefined : this.loaded.plan.get(index);
    const tried = new Set<string>();
    let firstAttempt = true;
    for (;;) {
      const pool = options.filter((c) => !tried.has(`${c.provider.id}/${c.row.model}`));
      if (pool.length === 0) break;
      const standing = pool.filter((c) => !this.state(c.provider.id).down && this.state(c.provider.id).made < callsMax(c.provider.id));
      const withRoom = pool.filter((c) => this.state(c.provider.id).made < callsMax(c.provider.id));
      const group = standing.length > 0 ? standing : withRoom.length > 0 ? withRoom : pool;
      const adjusted = (c: Candidate): number => c.row.rating_milli - (this.state(c.provider.id).made === 0 ? 0 : FRESH_PENALTY_MILLI);
      group.sort((a, b) => adjusted(b) - adjusted(a) || a.order - b.order || (a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0));
      const candidate = firstAttempt && planned !== undefined && group.includes(planned) ? planned : group[0];
      if (candidate === undefined) break;
      firstAttempt = false;
      tried.add(`${candidate.provider.id}/${candidate.row.model}`);
      const outcome = await this.enqueue(() => this.call(candidate, canon, raw, clientApi));
      if (outcome.ok) {
        this.state(candidate.provider.id).consecFails = 0;
        return { status: 200, contentType: canon.stream ? 'text/event-stream' : 'application/json', body: render(clientApi, canon.stream, outcome.answer, candidate.row.model) };
      }
      this.noteFailure(candidate, outcome.status);
    }
    return { status: 502, contentType: 'application/json', body: JSON.stringify(errorBody(clientApi, 'no free model could serve this request')) };
  }

  /** The free model ids the catalog registered, deduplicated, in catalog order. */
  modelIds(): string[] {
    const seen = new Set<string>();
    const ids: string[] = [];
    for (const provider of this.loaded.catalog.providers) {
      for (const model of provider.models) {
        if (!model.free || seen.has(model.id)) continue;
        seen.add(model.id);
        ids.push(model.id);
      }
    }
    return ids;
  }
}

/** Render a canonical answer in the client's wire, JSON or streamed. */
function render(clientApi: 'openai' | 'anthropic', stream: boolean, answer: CanonAnswer, model: string): string {
  if (clientApi === 'openai') return stream ? renderOpenAIStream(answer, model) : JSON.stringify(renderOpenAIJson(answer, model));
  return stream ? renderAnthropicStream(answer, model) : JSON.stringify(renderAnthropicJson(answer, model));
}

/** An error in the client's wire. */
function errorBody(clientApi: 'openai' | 'anthropic', message: string): Json {
  if (clientApi === 'anthropic') return { type: 'error', error: { type: 'api_error', message } };
  return { error: { message, type: 'api_error', code: 'aggregator_error' } };
}

/** The /v1/models answer: the registered free ids, in the shape of whichever auth header arrived. */
function modelsDocument(ids: readonly string[], anthropicShape: boolean): Json {
  if (anthropicShape) {
    return {
      data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
      first_id: ids[0] ?? null,
      last_id: ids[ids.length - 1] ?? null,
    };
  }
  return { object: 'list', data: ids.map((id) => ({ id, object: 'model', created: 0, owned_by: 'aggregator' })) };
}

/** The whole request body, parsed; null when it is not a JSON object. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > 16 * 1024 * 1024) return null;
    chunks.push(buf);
  }
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType });
  res.end(body);
}

/** The HTTP surface: the two inference routes and the model list. */
function makeServer(aggregator: Aggregator): Server {
  return createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1');
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        const anthropicShape = req.headers['x-api-key'] !== undefined;
        send(res, 200, 'application/json', JSON.stringify(modelsDocument(aggregator.modelIds(), anthropicShape)));
        return;
      }
      if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
        const clientApi = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
        const body = await readJsonBody(req);
        if (body === null) {
          send(res, 400, 'application/json', JSON.stringify(errorBody(clientApi, 'the request body is not a JSON object')));
          return;
        }
        const answer = await aggregator.serve(clientApi, body);
        send(res, answer.status, answer.contentType, answer.body);
        return;
      }
      send(res, 404, 'application/json', JSON.stringify({ error: { message: `no route for ${req.method ?? ''} ${url.pathname}`, type: 'api_error', code: 'not_found' } }));
    })().catch(() => {
      if (!res.headersSent) send(res, 500, 'application/json', JSON.stringify({ error: { message: 'internal error', type: 'api_error', code: 'internal' } }));
      else res.end();
    });
  });
}

/** Listen on an OS-assigned port; if the draw hands back the env's well-known port, draw again. */
async function listen(server: Server): Promise<number> {
  for (;;) {
    const port = await new Promise<number>((resolve, reject) => {
      const onError = (err: Error): void => reject(err);
      server.once('error', onError);
      server.listen(0, '0.0.0.0', () => {
        server.removeListener('error', onError);
        const address = server.address();
        resolve(typeof address === 'object' && address !== null ? address.port : 0);
      });
    });
    if (port !== 8080) return port;
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

/** Register the catalog with the env; the first answer begins the trial, so a lost one is retried. */
async function register(env: string, port: number, catalog: Catalog): Promise<void> {
  const body = JSON.stringify({ protocol_version: 1, port, catalog });
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const res = await fetch(`${env}/register`, { method: 'POST', headers: { 'content-type': 'application/json' }, body, signal: AbortSignal.timeout(5000) });
      if (res.ok) return;
      lastError = new Error(`register answered HTTP ${res.status}`);
    } catch (err) {
      lastError = err as Error;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw lastError ?? new Error('register failed');
}

/** Poll the env's work endpoint until it reports the trial done. Waiting only; nothing is timed into a request. */
async function pollWork(env: string): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${env}/work`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) return;
      }
    } catch {
      // The env is not answering yet, or not any more: keep waiting for its word.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function main(): Promise<void> {
  const env = envUrl();
  const loaded = await load(env);
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(loaded.catalog)}\n`);
    return;
  }
  const aggregator = new Aggregator(loaded);
  const server = makeServer(aggregator);
  const port = await listen(server);
  process.stdout.write(`aggregator listening on ${port}\n`);
  if (env !== '') {
    await register(env, port, loaded.catalog);
    await pollWork(env);
  } else {
    await new Promise(() => undefined);
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

main().then(
  () => process.exit(0),
  (err: unknown) => {
    process.stderr.write(`aggregator failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
    process.exit(1);
  },
);
