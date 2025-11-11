#!/usr/bin/env node
import { request } from 'undici';

async function postJSON(url, payload, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await request(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const bodyText = await res.body.text();
      if (res.statusCode >= 200 && res.statusCode < 300) {
        console.log(`Success attempt=${attempt} status=${res.statusCode}`);
        return bodyText;
      } else {
        console.warn(`Non-2xx status=${res.statusCode} attempt=${attempt} body=${bodyText}`);
      }
    } catch (err) {
      console.error(`Network error attempt=${attempt}`, err);
      if (attempt === retries) throw err;
      await new Promise(r => setTimeout(r, attempt * 1000));
    }
  }
  throw new Error('All retries failed');
}

async function main() {
  console.log("Copilot agent placeholder start");
  try {
    // Placeholder endpoint – değiştirin.
    const resp = await postJSON('https://example.com/api/ping', { repo: 'bearish-alpha-bot' });
    console.log("Response:", resp);
  } catch (e) {
    console.error("Agent run failed:", e);
    process.exit(1);
  }
  console.log("Copilot agent placeholder end");
}

main();
