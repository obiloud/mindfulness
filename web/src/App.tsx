import React, { useState, useEffect, useRef } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

// Define a more specific interface for the response to include transcript info
interface AgentResponse {
  message: string;
  transcript?: string; // Optional transcript field in the response
}

// Define a more specific interface for the message type to include transcript
interface MessageWithTranscript extends Message {
  transcript?: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function sendMessage(query: string, history: Message[]): Promise<MessageWithTranscript> {
  const resp = await fetch(`${API_BASE}/v1/mindfulness/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history }),
  });

  if (!resp.ok) {
    throw new Error("Failed to send message");
  }

  const data: AgentResponse = await resp.json();

  // Return a message with the transcript field if available
  return {
    role: "assistant",
    content: data.message,
    transcript: data.transcript
  };
}

export default function App() {
  const [messages, setMessages] = useState<MessageWithTranscript[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: MessageWithTranscript = {
      role: "user",
      content: input.trim(),
      transcript: undefined
    };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput("");
    setLoading(true);
    setIsStreaming(false);
    setTranscript(null);

    try {
      const reply = await sendMessage(userMsg.content, newHistory);

      // Now we directly access the transcript field from the response
      // Instead of parsing the content string
      if (reply.transcript) {
        setTranscript(reply.transcript);
        setIsStreaming(true);
      }

      // Add the response to messages
      setMessages((prev) => [...prev, reply]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong while contacting the server.",
          transcript: undefined
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle audio streaming when transcript is available
  useEffect(() => {
    if (!isStreaming || !transcript) return;

    // Create audio element for streaming
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // Simulate streaming by playing audio in chunks
    const playAudioInChunks = () => {
      const audio = audioRef.current;
      if (!audio || !transcript) return;

      // POST request to initiate audio streaming with transcript in body
      const audioUrl = `${API_BASE}/v1/mindfulness/audio`;

      fetch(audioUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript }),
      })
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`Audio streaming failed: ${response.status}`);
          }

          // Check if response.body is a valid ReadableStream
          if (!response.body || !(response.body instanceof ReadableStream)) {
            console.warn('Response body is not a readable stream. Skipping streaming.');
            return;
          }

          // Stream the audio response as a readable stream
          const reader = response.body.getReader();
          const audio = new Audio();
          audio.src = URL.createObjectURL(new Blob([], { type: 'audio/wav' }));

          // Set up streaming playback
          const playStream = async () => {
            const { done, value } = await reader.read();
            if (done) return;

            // Convert chunk to blob and append to audio source
            const blob = new Blob([value], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            audio.src = url;
            audio.play().catch(() => {
              console.warn('Audio playback failed');
            });

            // Schedule next chunk
            setTimeout(playStream, 100);
          };

          playStream();
        })
        .catch((error) => {
          console.error('Error during audio streaming:', error);
        });
    };

    playAudioInChunks();
  }, [isStreaming, transcript]);

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
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"
                }`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap ${m.role === "user"
                  ? "bg-mind-accent/90 text-slate-950"
                  : "bg-slate-800/80 text-slate-100 border border-slate-700/70"
                  }`}
              >
                {m.content}
              </div>
            </div>
          ))}

          {/* Loading animation */}
          {loading && (
            <div className="flex justify-center items-center py-2">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-sky-400 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-sky-400 rounded-full animate-pulse delay-100"></div>
                <div className="w-2 h-2 bg-sky-400 rounded-full animate-pulse delay-200"></div>
              </div>
            </div>
          )}
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
