const form = document.querySelector("#chatForm");
const input = document.querySelector("#questionInput");
const fileInput = document.querySelector("#fileInput");
const attachButton = document.querySelector("#attachButton");
const attachmentRow = document.querySelector("#attachmentRow");
const attachmentName = document.querySelector("#attachmentName");
const removeAttachment = document.querySelector("#removeAttachment");
const messages = document.querySelector("#messages");
const statusPill = document.querySelector("#indexStatus");
const quickPrompts = document.querySelectorAll("[data-question]");

const history = [];
const historyLimit = 4;
let attachedFile = null;

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;

    const data = await response.json();
    const count = Number(data.document_count || 0);
    statusPill.textContent = `${count} indexed chunks`;
  } catch {
    statusPill.textContent = "Index unavailable";
  }
}

function addMessage(role, text, sources = [], attachment = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  bubble.appendChild(paragraph);

  if (attachment) {
    const attachmentPill = document.createElement("div");
    attachmentPill.className = "message-attachment";
    attachmentPill.textContent = attachment;
    bubble.appendChild(attachmentPill);
  }

  if (sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";

    for (const source of sources) {
      const pill = document.createElement("span");
      pill.className = "source-pill";
      pill.textContent = source;
      sourceList.appendChild(pill);
    }

    bubble.appendChild(sourceList);
  }

  article.appendChild(avatar);
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;

  return article;
}

function addTypingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="avatar" aria-hidden="true">AI</div>
    <div class="bubble" aria-label="Assistant is thinking">
      <div class="typing" aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function setBusy(isBusy) {
  form.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
  input.disabled = isBusy;
  fileInput.disabled = isBusy;
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 132)}px`;
}

function renderAttachment() {
  if (!attachedFile) {
    attachmentRow.hidden = true;
    attachmentName.textContent = "";
    return;
  }

  attachmentName.textContent = attachedFile.name;
  attachmentRow.hidden = false;
}

async function sendChatRequest(question) {
  const recentHistory = history.slice(-historyLimit);

  if (!attachedFile) {
    return fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: recentHistory }),
    });
  }

  const params = new URLSearchParams({
    question,
    history: JSON.stringify(recentHistory),
    file_name: attachedFile.name,
  });

  return fetch(`/api/chat/file?${params.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: attachedFile,
  });
}

async function ask(question) {
  const attachmentLabel = attachedFile ? `Attached: ${attachedFile.name}` : "";
  addMessage("user", question, [], attachmentLabel);
  const typing = addTypingMessage();
  setBusy(true);

  try {
    const response = await sendChatRequest(question);

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The chatbot could not answer right now.");
    }

    typing.remove();
    addMessage("assistant", data.answer, data.sources || []);
    history.push({ question, answer: data.answer });
  } catch (error) {
    typing.remove();
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  resizeInput();
  ask(question);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

attachButton.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  attachedFile = file || null;
  renderAttachment();
});

removeAttachment.addEventListener("click", () => {
  attachedFile = null;
  fileInput.value = "";
  renderAttachment();
  input.focus();
});

for (const prompt of quickPrompts) {
  prompt.addEventListener("click", () => {
    const question = prompt.dataset.question;
    if (question) ask(question);
  });
}

loadStatus();
resizeInput();
renderAttachment();
