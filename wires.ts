// The three wires, translated through one canonical form.
//
// A client body (OpenAI Chat Completions or Anthropic Messages) is read into a CanonRequest: the
// conversation as neutral messages, and the echo fields the far end will be judged on — the system
// text, the last user turn's text, the tool names in order, max_tokens, the stop sequences, the last
// image, and whether JSON mode was asked for. Those fields are read exactly the way a provider reads
// them, so a body rebuilt for another wire echoes identically: the system prompt travels as one
// message holding the joined text, tool definitions keep their order and names, an image stays an
// image in whichever spelling the target wire uses, and a JSON-mode ask is only ever carried to a
// wire that can state it (the router keeps Anthropic-wire providers away from JSON-mode requests,
// because that wire cannot state one and its echo would read false).
//
// A provider's answer, JSON or streamed, is read into a CanonAnswer — the whole text and the usage
// pair — and the client's answer is rendered from it in the client's wire, with the stop reason in
// the family the provider's own stop belongs to and the usage carried across unchanged.

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

const asArray = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** The text of a content value: the string itself, or its text blocks joined with newlines. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  return asArray(content)
    .filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => (b as Json).text as string)
    .join('\n');
}

/** One part of a neutral message. */
export type CanonPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; toolUseId: string; text: string; isError: boolean };

/** One neutral message: a user or assistant turn, or a tool result standing on its own. */
export type CanonMessage =
  | { role: 'user' | 'assistant'; parts: CanonPart[] }
  | { role: 'tool'; toolUseId: string; text: string; isError: boolean };

/** One tool definition, wire-neutral. */
export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

/** A client request in canonical form, with the fields an echo is judged on. */
export interface CanonRequest {
  systemText: string;
  userText: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: 'auto' | 'none' | 'required' | { name: string } | null;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  responseFormat: unknown;
  stream: boolean;
  temperature: number | null;
  topP: number | null;
}

/** A provider's answer in canonical form: the whole text and the usage it reported. */
export interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A data URL split into its media type and base64 payload, or null when it is not one. */
function dataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (m === null || m[2] === undefined) return null;
  return { mediaType: m[1] ?? 'application/octet-stream', data: m[3] ?? '' };
}

/** The parts of one OpenAI content value (a string, or an array of typed blocks). */
function openAiParts(content: unknown): CanonPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  const parts: CanonPart[] = [];
  for (const block of asArray(content)) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const parsed = dataUrl(block.image_url.url);
      if (parsed !== null) parts.push({ kind: 'image', mediaType: parsed.mediaType, data: parsed.data });
      else parts.push({ kind: 'imageUrl', url: block.image_url.url });
    }
  }
  return parts;
}

/** The parts of one Anthropic content value (a string, or an array of typed blocks). */
function anthropicParts(content: unknown): CanonPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  const parts: CanonPart[] = [];
  for (const block of asArray(content)) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ kind: 'image', mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ kind: 'imageUrl', url: block.source.url });
      }
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      parts.push({ kind: 'toolUse', id: typeof block.id === 'string' ? block.id : '', name: block.name, input: isRecord(block.input) ? block.input : {} });
    } else if (block.type === 'tool_result' && typeof block.tool_use_id === 'string') {
      parts.push({ kind: 'toolResult', toolUseId: block.tool_use_id, text: blocksText(block.content), isError: block.is_error === true });
    }
  }
  return parts;
}

/** Read an OpenAI Chat Completions body into canonical form. */
export function extractOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  let userText = '';
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
    } else if (raw.role === 'user') {
      userText = blocksText(raw.content);
      messages.push({ role: 'user', parts: openAiParts(raw.content) });
    } else if (raw.role === 'assistant') {
      const parts = openAiParts(raw.content);
      for (const call of asArray(raw.tool_calls)) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            input = {};
          }
        }
        parts.push({ kind: 'toolUse', id: typeof call.id === 'string' ? call.id : '', name: call.function.name, input });
      }
      messages.push({ role: 'assistant', parts });
    } else if (raw.role === 'tool') {
      messages.push({ role: 'tool', toolUseId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', text: blocksText(raw.content), isError: false });
    }
  }
  const tools: CanonTool[] = [];
  for (const raw of asArray(body.tools)) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : {},
    });
  }
  const stops = typeof body.stop === 'string' ? [body.stop] : asArray(body.stop).filter((s): s is string => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    systemText: systemParts.join('\n'),
    userText,
    messages,
    tools,
    toolChoice: openAiToolChoice(body.tool_choice),
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops,
    jsonMode: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
    responseFormat: body.response_format ?? null,
    stream: body.stream === true,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
  };
}

function openAiToolChoice(v: unknown): CanonRequest['toolChoice'] {
  if (v === 'auto' || v === 'none' || v === 'required') return v;
  if (isRecord(v) && isRecord(v.function) && typeof v.function.name === 'string') return { name: v.function.name };
  return null;
}

/** Read an Anthropic Messages body into canonical form. */
export function extractAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  let userText = '';
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'user') {
      userText = blocksText(raw.content);
      const parts = anthropicParts(raw.content);
      const results = parts.filter((p) => p.kind === 'toolResult');
      const rest = parts.filter((p) => p.kind !== 'toolResult');
      for (const r of results) if (r.kind === 'toolResult') messages.push({ role: 'tool', toolUseId: r.toolUseId, text: r.text, isError: r.isError });
      if (rest.length > 0 || results.length === 0) messages.push({ role: 'user', parts: rest });
    } else if (raw.role === 'assistant') {
      messages.push({ role: 'assistant', parts: anthropicParts(raw.content) });
    }
  }
  const tools: CanonTool[] = [];
  for (const raw of asArray(body.tools)) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : {} });
  }
  return {
    systemText: blocksText(body.system),
    userText,
    messages,
    tools,
    toolChoice: anthropicToolChoice(body.tool_choice),
    maxTokens: numOrNull(body.max_tokens),
    stops: asArray(body.stop_sequences).filter((s): s is string => typeof s === 'string'),
    jsonMode: false,
    responseFormat: null,
    stream: body.stream === true,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
  };
}

function anthropicToolChoice(v: unknown): CanonRequest['toolChoice'] {
  if (!isRecord(v)) return null;
  if (v.type === 'auto') return 'auto';
  if (v.type === 'none') return 'none';
  if (v.type === 'any') return 'required';
  if (v.type === 'tool' && typeof v.name === 'string') return { name: v.name };
  return null;
}

/** The OpenAI content blocks for one neutral message's parts. */
function openAiContent(parts: readonly CanonPart[]): unknown {
  const blocks: Json[] = [];
  let plain = '';
  let onlyText = true;
  for (const part of parts) {
    if (part.kind === 'text') {
      plain += (plain === '' ? '' : '\n') + part.text;
      blocks.push({ type: 'text', text: part.text });
    } else if (part.kind === 'image') {
      onlyText = false;
      blocks.push({ type: 'image_url', image_url: { url: `data:${part.mediaType};base64,${part.data}` } });
    } else if (part.kind === 'imageUrl') {
      onlyText = false;
      blocks.push({ type: 'image_url', image_url: { url: part.url } });
    }
  }
  return onlyText ? plain : blocks;
}

/** Build an OpenAI Chat Completions body for `model` from canonical form. */
export function buildOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: message.toolUseId, content: message.text });
    } else if (message.role === 'user') {
      messages.push({ role: 'user', content: openAiContent(message.parts) });
    } else {
      const text = message.parts.filter((p) => p.kind === 'text').map((p) => (p.kind === 'text' ? p.text : '')).join('\n');
      const calls = message.parts
        .filter((p) => p.kind === 'toolUse')
        .map((p) => (p.kind === 'toolUse' ? { id: p.id, type: 'function', function: { name: p.name, arguments: JSON.stringify(p.input) } } : null))
        .filter((c) => c !== null);
      messages.push({ role: 'assistant', content: calls.length > 0 ? (text === '' ? null : text) : text, ...(calls.length > 0 ? { tool_calls: calls } : {}) });
    }
  }
  // The last user text a reader takes from this body must be the client's: when the conversation ends
  // on a tool result, a trailing empty user message states the empty last turn explicitly.
  const last = canon.messages[canon.messages.length - 1];
  if (last !== undefined && last.role === 'tool' && canon.userText === '') messages.push({ role: 'user', content: '' });
  const body: Json = { model, messages, stream: canon.stream };
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    if (canon.toolChoice === 'auto' || canon.toolChoice === 'none' || canon.toolChoice === 'required') body.tool_choice = canon.toolChoice;
    else if (canon.toolChoice !== null) body.tool_choice = { type: 'function', function: { name: canon.toolChoice.name } };
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = [...canon.stops];
  if (canon.responseFormat !== null) body.response_format = canon.responseFormat;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  if (canon.stream) body.stream_options = { include_usage: true };
  return body;
}

/** Build an Anthropic Messages body for `model` from canonical form. */
export function buildAnthropic(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: message.toolUseId, content: message.text, ...(message.isError ? { is_error: true } : {}) }] });
    } else if (message.role === 'user') {
      const blocks: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } });
        else if (part.kind === 'imageUrl') blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
      }
      messages.push({ role: 'user', content: blocks });
    } else {
      const blocks: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.kind === 'toolUse') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
      }
      messages.push({ role: 'assistant', content: blocks });
    }
  }
  // Mirror of the OpenAI case: a conversation ending on a tool result reads as an empty last user
  // turn on this wire, so restate the client's last user text when it was not empty.
  const last = canon.messages[canon.messages.length - 1];
  if (last !== undefined && last.role === 'tool' && canon.userText !== '') messages.push({ role: 'user', content: [{ type: 'text', text: canon.userText }] });
  const body: Json = { model, messages, stream: canon.stream };
  if (canon.systemText !== '') body.system = canon.systemText;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    if (canon.toolChoice === 'auto') body.tool_choice = { type: 'auto' };
    else if (canon.toolChoice === 'none') body.tool_choice = { type: 'none' };
    else if (canon.toolChoice === 'required') body.tool_choice = { type: 'any' };
    else if (canon.toolChoice !== null) body.tool_choice = { type: 'tool', name: canon.toolChoice.name };
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop_sequences = [...canon.stops];
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  return body;
}

/** Build a Gemini generateContent body for `model` from canonical form. */
export function buildGemini(canon: CanonRequest, model: string): Json {
  const contents: Json[] = [];
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { content: message.text } } }] });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') parts.push({ inlineData: { mimeType: part.mediaType, data: part.data } });
      else if (part.kind === 'imageUrl') parts.push({ fileData: { fileUri: part.url } });
      else if (part.kind === 'toolUse') parts.push({ functionCall: { name: part.name, args: part.input } });
      else if (part.kind === 'toolResult') parts.push({ functionResponse: { name: '', response: { content: part.text } } });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const generationConfig: Json = {};
  if (canon.maxTokens !== null) generationConfig.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) generationConfig.stopSequences = [...canon.stops];
  if (canon.jsonMode) generationConfig.responseMimeType = 'application/json';
  if (canon.temperature !== null) generationConfig.temperature = canon.temperature;
  if (canon.topP !== null) generationConfig.topP = canon.topP;
  const body: Json = { contents };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    body.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
    if (canon.toolChoice === 'none') body.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (canon.toolChoice === 'required') body.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (canon.toolChoice !== null && typeof canon.toolChoice === 'object') body.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [canon.toolChoice.name] } };
    else body.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  }
  if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
  return body;
}

/** The `data:` payloads of an SSE text, in order, with the terminal marker removed. */
export function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const data = trimmed.slice('data:'.length).trim();
    if (data !== '' && data !== '[DONE]') out.push(data);
  }
  return out;
}

function usagePair(usage: unknown, inputKey: string, outputKey: string): { input: number; output: number } | null {
  if (!isRecord(usage)) return null;
  const input = usage[inputKey];
  const output = usage[outputKey];
  if (typeof input !== 'number' || typeof output !== 'number') return null;
  return { input, output };
}

/** Read an OpenAI JSON answer. */
export function parseOpenAIJson(body: unknown): CanonAnswer | null {
  if (!isRecord(body)) return null;
  const choice = asArray(body.choices)[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  const content = choice.message.content;
  const text = typeof content === 'string' ? content : asArray(content).filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string').map((b) => (b as Json).text as string).join('');
  const usage = usagePair(body.usage, 'prompt_tokens', 'completion_tokens');
  if (usage === null) return null;
  return { text, inputTokens: usage.input, outputTokens: usage.output };
}

/** Read an OpenAI streamed answer: the deltas joined, the usage from whichever chunk carries it. */
export function parseOpenAIStream(text: string): CanonAnswer | null {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let frames = 0;
  for (const payload of ssePayloads(text)) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    frames += 1;
    const found = usagePair(chunk.usage, 'prompt_tokens', 'completion_tokens');
    if (found !== null) usage = found;
    const choice = asArray(chunk.choices)[0];
    if (isRecord(choice) && isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
  }
  if (frames === 0 || usage === null) return null;
  return { text: out, inputTokens: usage.input, outputTokens: usage.output };
}

/** Read an Anthropic JSON answer. */
export function parseAnthropicJson(body: unknown): CanonAnswer | null {
  if (!isRecord(body)) return null;
  const text = asArray(body.content)
    .filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => (b as Json).text as string)
    .join('');
  const usage = usagePair(body.usage, 'input_tokens', 'output_tokens');
  if (usage === null) return null;
  return { text, inputTokens: usage.input, outputTokens: usage.output };
}

/** Read an Anthropic streamed answer: text deltas joined, input from message_start, output from message_delta. */
export function parseAnthropicStream(text: string): CanonAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let frames = 0;
  for (const payload of ssePayloads(text)) {
    let event: unknown;
    try {
      event = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    frames += 1;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      if (typeof event.message.usage.input_tokens === 'number') input = event.message.usage.input_tokens;
    } else if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta' && typeof event.delta.text === 'string') {
      out += event.delta.text;
    } else if (event.type === 'message_delta' && isRecord(event.usage)) {
      if (typeof event.usage.output_tokens === 'number') output = event.usage.output_tokens;
    }
  }
  if (frames === 0 || input === null || output === null) return null;
  return { text: out, inputTokens: input, outputTokens: output };
}

/** The text and usage of one Gemini answer object (a whole response or one streamed frame). */
function geminiFrame(body: Json): { text: string; usage: { input: number; output: number } | null } {
  const candidate = asArray(body.candidates)[0];
  const parts = isRecord(candidate) && isRecord(candidate.content) ? asArray(candidate.content.parts) : [];
  const text = parts.filter((p) => isRecord(p) && typeof p.text === 'string').map((p) => (p as Json).text as string).join('');
  return { text, usage: usagePair(body.usageMetadata, 'promptTokenCount', 'candidatesTokenCount') };
}

/** Read a Gemini JSON answer. */
export function parseGeminiJson(body: unknown): CanonAnswer | null {
  if (!isRecord(body)) return null;
  const frame = geminiFrame(body);
  if (frame.usage === null) return null;
  return { text: frame.text, inputTokens: frame.usage.input, outputTokens: frame.usage.output };
}

/** Read a Gemini streamed answer: frame texts joined, usage from the last frame that carries it. */
export function parseGeminiStream(text: string): CanonAnswer | null {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let frames = 0;
  for (const payload of ssePayloads(text)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    frames += 1;
    const frame = geminiFrame(parsed);
    out += frame.text;
    if (frame.usage !== null) usage = frame.usage;
  }
  if (frames === 0 || usage === null) return null;
  return { text: out, inputTokens: usage.input, outputTokens: usage.output };
}

/** Split a text into pieces for streaming: whole for short answers, even slices for long ones. */
function pieces(text: string): string[] {
  if (text === '') return [''];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += 512) out.push(text.slice(i, i + 512));
  return out;
}

/** Render the client's OpenAI JSON answer. */
export function renderOpenAIJson(answer: CanonAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: 0,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** Render the client's OpenAI answer as SSE: role, content, finish, usage, done — in that order. */
export function renderOpenAIStream(answer: CanonAnswer, model: string): string {
  const frame = (chunk: Json): string => `data: ${JSON.stringify(chunk)}\n\n`;
  const base = { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: 0, model };
  let out = frame({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  for (const piece of pieces(answer.text)) out += frame({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  out += frame({ ...base, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
  out += frame({ ...base, choices: [], usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens } });
  return `${out}data: [DONE]\n\n`;
}

/** Render the client's Anthropic JSON answer. */
export function renderAnthropicJson(answer: CanonAnswer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** Render the client's Anthropic answer as SSE, in the event order the wire prescribes. */
export function renderAnthropicStream(answer: CanonAnswer, model: string): string {
  const event = (type: string, data: Json): string => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
  let out = event('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  out += event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of pieces(answer.text)) out += event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  out += event('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += event('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } });
  out += event('message_stop', { type: 'message_stop' });
  return out;
}
