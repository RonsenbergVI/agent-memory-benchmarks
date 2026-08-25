// OpenAI's reasoning models (gpt-5-mini) reject `max_tokens` and accept only
// `max_completion_tokens`. agentmemory sends the former and exposes no setting
// for it, so every compression call answers 400 — and the failure is
// destructive rather than degraded: the observation is dropped WITHOUT being
// indexed, so the system then retrieves nothing at all.
//
// The two names are the same parameter, so this renames it in the OpenAI
// provider only, located by its Azure-aware `buildChatUrl` call. The Anthropic,
// OpenRouter and MiniMax providers share the bundle and keep `max_tokens`,
// which is correct for them.
//
// Asserts before and after: a version bump that moves this fails the image
// build rather than silently restoring the 400s.
const fs = require("fs");

const DIR = "/opt/agentmemory/node_modules/@agentmemory/agentmemory/dist";
const MARKER = "buildChatUrl(this.baseUrl, this.isAzure, this.azureApiVersion);";
const FROM = "max_tokens: this.maxTokens";
const TO = "max_completion_tokens: this.maxTokens";
const MAX_DISTANCE = 200;

let patched = 0;
for (const name of fs.readdirSync(DIR).filter((f) => f.endsWith(".mjs"))) {
  const path = `${DIR}/${name}`;
  const src = fs.readFileSync(path, "utf8");
  const marker = src.indexOf(MARKER);
  if (marker === -1) continue;

  const hit = src.indexOf(FROM, marker);
  if (hit === -1) throw new Error(`${name}: no '${FROM}' after the OpenAI marker`);
  if (hit - marker > MAX_DISTANCE) {
    throw new Error(`${name}: '${FROM}' is ${hit - marker} chars past the marker`);
  }

  fs.writeFileSync(path, src.slice(0, hit) + TO + src.slice(hit + FROM.length));
  patched += 1;
}

if (patched === 0) {
  throw new Error("agentmemory's OpenAI provider was not found; this patch needs updating");
}
console.log(`patched max_completion_tokens in ${patched} bundle(s)`);
