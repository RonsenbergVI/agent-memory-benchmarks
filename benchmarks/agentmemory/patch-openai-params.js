// gpt-5-mini accepts only `max_completion_tokens`; agentmemory sends
// `max_tokens` with no setting for it, so every compression call 400s —
// destructively: the observation is dropped WITHOUT being indexed, and the
// system then retrieves nothing at all. Same parameter, so rename it in the
// OpenAI provider only (located by its Azure-aware `buildChatUrl` call); the
// Anthropic/OpenRouter/MiniMax providers keep `max_tokens`, correct for them.
// The asserts make a version bump that moves this fail the image build
// instead of silently restoring the 400s.
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
