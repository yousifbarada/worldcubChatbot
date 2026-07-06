(() => {
  const feed = document.getElementById("matchFeed");
  const kickoff = document.getElementById("kickoff");
  const form = document.getElementById("touchlineForm");
  const input = document.getElementById("messageField");
  const sendBtn = document.getElementById("kickBtn");
  const newMatchBtn = document.getElementById("newMatch");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const starterChips = document.querySelectorAll(".starter-chip");

  const SESSION_KEY_MEMORY = { sessionId: null };

  function setStatus(state, label) {
    statusDot.className = "status-dot " + state;
    statusText.textContent = label;
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("bad status");
      const data = await res.json();
      setStatus("live", `Warmed up · ${data.documents_in_store} docs loaded`);
    } catch (err) {
      setStatus("down", "Bench — API unreachable");
    }
  }

  function clearKickoff() {
    if (kickoff && kickoff.parentNode) kickoff.remove();
  }

  function addPlay(role, text, sources) {
    clearKickoff();
    const wrap = document.createElement("div");
    wrap.className = "play " + (role === "user" ? "home" : "away");

    const label = document.createElement("div");
    label.className = "play__label";
    label.textContent = role === "user" ? "You · attacking" : "Commentary";
    wrap.appendChild(label);

    const bubble = document.createElement("div");
    bubble.className = "play__bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);

    if (sources && sources.length) {
      const assists = document.createElement("div");
      assists.className = "assists";
      sources.forEach((s) => {
        const tag = document.createElement("span");
        tag.className = "assist-tag";
        const score = typeof s.similarity_score === "number" ? s.similarity_score.toFixed(2) : "—";
        tag.innerHTML = `<b>assist</b> ${escapeHtml(s.source_file || "unknown")} · ${score}`;
        assists.appendChild(tag);
      });
      wrap.appendChild(assists);
    }

    feed.appendChild(wrap);
    feed.scrollTop = feed.scrollHeight;
    return wrap;
  }

  function addThinking() {
    clearKickoff();
    const wrap = document.createElement("div");
    wrap.className = "play away";
    const label = document.createElement("div");
    label.className = "play__label";
    label.textContent = "Commentary";
    wrap.appendChild(label);
    const bubble = document.createElement("div");
    bubble.className = "play__bubble thinking";
    bubble.innerHTML = "<span></span><span></span><span></span>";
    wrap.appendChild(bubble);
    feed.appendChild(wrap);
    feed.scrollTop = feed.scrollHeight;
    return wrap;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    addPlay("user", trimmed);
    input.value = "";
    autoGrow();
    sendBtn.disabled = true;

    const thinkingEl = addThinking();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          session_id: SESSION_KEY_MEMORY.sessionId,
        }),
      });

      thinkingEl.remove();

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        const wrap = addPlay("model", errBody.detail || "Whistle blown — something went wrong on the pitch. Try again.");
        wrap.querySelector(".play__bubble").classList.add("error");
        return;
      }

      const data = await res.json();
      SESSION_KEY_MEMORY.sessionId = data.session_id;
      addPlay("model", data.answer, data.sources);
    } catch (err) {
      thinkingEl.remove();
      const wrap = addPlay("model", "Can't reach the stadium — check the server is running.");
      wrap.querySelector(".play__bubble").classList.add("error");
      setStatus("down", "Bench — API unreachable");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("input", autoGrow);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  starterChips.forEach((chip) => {
    chip.addEventListener("click", () => sendMessage(chip.dataset.prompt));
  });

  newMatchBtn.addEventListener("click", async () => {
    if (!SESSION_KEY_MEMORY.sessionId) {
      location.reload();
      return;
    }
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_KEY_MEMORY.sessionId }),
      });
    } catch (err) {
      // Reset failing server-side shouldn't block starting fresh client-side.
    } finally {
      location.reload();
    }
  });

  checkHealth();
})();
