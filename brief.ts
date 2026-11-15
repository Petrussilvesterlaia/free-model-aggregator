// The trial brief: three JSON documents found by their `schema` member inside $MAKEFUN_PROMPT_FILE,
// plus the catalog probe that asks every provider for its model list in its own wire.
//
//   - providers/v1: the directory. Ten providers, each with an id, a wire (openai, anthropic or
//     gemini), an auth scheme, the key variable a real deployment would read, the key the simulated
//     service takes, and the base path the entry joins to $MAKEFUN_ENV_URL to dial it.
//   - ratings/v1: the quality table. One row per free model: its rating in milli-points and what it
//     can do (tools, vision, JSON mode, context length). Paid models have no row here; a provider's
//     own listing names them, flagged paid, and they are never routed.
//   - workload/v1: the manifest. The twenty requests in the order the client sends them, each with
//     its api, streaming flag, weight and the requirements a model must meet to serve it.
//
// The probe reads each provider's catalog route once (with retries) and the catalog document is built
// from the listings, with the quality table filling any field a listing leaves out. Everything in this
// file is a pure function of the brief and the listings: no clock, no randomness, so the document
// `--list` prints is byte for byte the one the register body carries.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief's directory lists it. `base` is its path (or an absolute base URL). */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table: a free model, its rating and what it can do. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from a model; a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One manifest row: the request the client will send, in arrival order. */
export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

/** The brief, parsed. `error` is set when the file could not be read at all. */
export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  /** The best legal rating per request id, as the brief's table states it (may be empty). */
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ListedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: the listing it answered with, or why it did not answer. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
  error: string | null;
}

/** One model of the catalog document: its id, whether it is free, and its context length. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider of the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: what `--list` prints and what the register body carries. */
export interface Catalog {
  providers: CatalogProvider[];
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a document on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); an absolute base may instead be
  // spelled base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. An empty path reads as an empty brief. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directoryDoc === undefined) return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best as unknown[]) {
      if (isRecord(v) && stringOf(v.id) !== '' && typeof v.rating_milli === 'number') best.set(stringOf(v.id), v.rating_milli);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workloadDoc === undefined
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

/** A base path joined to the env's URL for dialling; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in; `none` sends no key at all. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a listed model is free: the quality table's row where it has one (the sealed truth for the
 * models it lists), the listing's own flag where it does not, and paid when neither says.
 */
export function modelFree(listed: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return listed.free ?? false;
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** A listing's price fields read as a free flag: true when every price is zero, null when it is silent. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** The models one catalog route's answer listed, in its wire's shape. */
function listedModels(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/**
 * Probe one provider's catalog route in its own wire. A failure is retried (the env is a local service
 * and a listing is worth having: the catalog is scored on every provider), and a provider that still
 * does not answer is left out of the catalog and gets no traffic.
 */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let error: string | null = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) {
        error = `GET ${url} answered HTTP ${res.status}`;
        continue;
      }
      const body: unknown = await res.json();
      return { provider, ok: true, models: listedModels(provider.wire, body), error: null };
    } catch (err) {
      error = `GET ${url} failed: ${(err as Error).message}`;
    }
  }
  return { provider, ok: false, models: [], error };
}

/**
 * The catalog document the probes and the quality table make: every provider that answered, in the
 * brief's order, its models sorted by id, each with the free flag and context length the table or the
 * listing states. Its base_url is the directory's own path, never joined to the env's URL: the env's
 * origin differs between a live pass and its replay, and the catalog is part of what a replay must
 * reproduce byte for byte.
 */
export function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const listed of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(listed.id)) continue;
      seen.add(listed.id);
      const row = ratingRow(brief, probe.provider.id, listed.id);
      models.push({ id: listed.id, free: modelFree(listed, row), context_length: row?.context_length ?? listed.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
