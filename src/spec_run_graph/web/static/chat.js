// QA chatbot panel: POST /chat/stream and read the SSE response with fetch (tokens, trace, done, error).
const log = () => document.getElementById("chat-log");

function addMsg(cls, text) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  log().appendChild(div);
  log().scrollTop = log().scrollHeight;
  return div;
}

function renderMarkdownLite(el, text) {
  // escape, then minimal markdown: code, bold, tables kept as preformatted text
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  el.innerHTML = esc
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function addTrace(el, trace) {
  if (!trace.length) return;
  const d = document.createElement("details");
  d.className = "trace";
  d.innerHTML = "<summary>how I got this (" + trace.length + " tool calls)</summary>";
  const pre = document.createElement("pre");
  pre.textContent = trace.map(t => t.tool + " " + JSON.stringify(t.args) + (t.edgeql ? "\n  " + t.edgeql : "")).join("\n");
  d.appendChild(pre);
  el.appendChild(d);
}

async function sendChat(ev) {
  ev.preventDefault();
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return false;
  input.value = "";
  addMsg("user", message);
  const bot = addMsg("bot pending", "…");
  let text = "";
  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx).replace(/^\r?\n\r?\n/, "");
        let event = "message", data = "";
        for (const line of raw.split(/\r?\n/)) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).replace(/^ /, "");
        }
        if (event === "token") { text += JSON.parse(data); renderMarkdownLite(bot, text); }
        else if (event === "trace") addTrace(bot, JSON.parse(data));
        else if (event === "error") { bot.classList.add("error"); text += (text ? "\n" : "") + "⚠ " + data; renderMarkdownLite(bot, text); }
        log().scrollTop = log().scrollHeight;
      }
    }
  } catch (e) {
    bot.classList.add("error");
    bot.textContent = "⚠ " + e;
  }
  bot.classList.remove("pending");
  if (!text) bot.textContent = bot.textContent === "…" ? "(no answer)" : bot.textContent;
  return false;
}

async function resetChat() {
  await fetch("/chat/reset", { method: "POST" });
  log().querySelectorAll(".msg.user, .msg.bot:not(:first-child)").forEach(n => n.remove());
}

document.addEventListener("keydown", (e) => {
  if (e.target.id === "chat-input" && e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});
