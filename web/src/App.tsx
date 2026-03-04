import React, { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function sendMessage(query: string, history: Message[]): Promise<Message> {
  const resp = await fetch(`${API_BASE}/v1/mindfulness/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history }),
  });

  if (!resp.ok) {
    throw new Error("Failed to send message");
  }

  const data = await resp.json();
  return { role: "assistant", content: data.message };
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: "user", content: input.trim() };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput("");
    setLoading(true);

    try {
      const reply = await sendMessage(userMsg.content, newHistory);
      setMessages((prev) => [...prev, reply]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong while contacting the server.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-3xl bg-mind-surface/80 backdrop-blur-md rounded-2xl shadow-xl border border-slate-700/60 p-6 flex flex-col gap-4">
        <header className="flex items-center justify-between mb-2">
          <div>
            <h1 className="text-xl font-semibold text-sky-300">
              Mindfulness AI
            </h1>
            <p className="text-sm text-slate-300">
              Share how you feel, and receive a gentle, guided response.
            </p>
          </div>
        </header>

        <div className="flex-1 min-h-[320px] max-h-[420px] overflow-y-auto space-y-3 pr-1">
          {messages.length === 0 && (
            <p className="text-slate-400 text-sm">
              Start by telling the guide what you are going through, for example:
              “I feel anxious about work and cannot relax.”
            </p>
          )}
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex ${
                m.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-mind-accent/90 text-slate-950"
                    : "bg-slate-800/80 text-slate-100 border border-slate-700/70"
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-2 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="How are you feeling?"
            className="flex-1 rounded-xl bg-slate-900/60 border border-slate-700/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-mind-accent focus:border-transparent"
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="px-4 py-2 rounded-xl bg-mind-accent text-slate-950 text-sm font-medium hover:bg-sky-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

