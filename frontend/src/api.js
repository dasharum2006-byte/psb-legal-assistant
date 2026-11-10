const API_BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function ask(message, history = []) {
  return request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, filters: {} }),
  });
}

export async function getDocuments() {
  return request("/api/documents");
}
