import React, { useState, useEffect, useRef } from "react";
import { cartesiaTTSClient } from "./CartesiaTTSClient";

type Message = {
  role: "user" | "assistant";
  content: string;
};

// Define a more specific interface for the response to include transcript info
interface AgentResponse {
  session_id: string;
  message: string;
  transcript?: string; // Optional transcript field in the response
}

// Define a more specific interface for the message type to include transcript
interface MessageWithTranscript extends Message {
  session_id?: string;
  transcript?: string;
}


const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function sendMessage(query: string, session_id?: string): Promise<MessageWithTranscript> {
  const resp = await fetch(`${API_BASE}/v1/mindfulness/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id }),
  });

  if (!resp.ok) {
    throw new Error("Failed to send message");
  }

  const data: AgentResponse = await resp.json();

  // Return a message with the transcript field if available
  return {
    role: "assistant",
    session_id: data.session_id,
    content: data.message,
    transcript: data.transcript
  };
}

export default function App() {
  const [messages, setMessages] = useState<MessageWithTranscript[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [session_id, setSessionId] = useState<string | undefined>(undefined)
  const [isStreaming, setIsStreaming] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: MessageWithTranscript = {
      role: "user",
      content: input.trim(),
      session_id: session_id,
      transcript: undefined
    };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput("");
    setLoading(true);
    setIsStreaming(false);
    setTranscript(null);

    try {
      const reply = await sendMessage(userMsg.content, session_id);

      // Debug: Log when transcript is received
      console.log('Received transcript from assistant:', reply.transcript ? reply.transcript.length : 'null');

      if (reply.transcript) {
        setTranscript(reply.transcript);
        setIsStreaming(true);

        // Debug: Log when streaming is activated
        console.log('Streaming activated with transcript length:', reply.transcript.length);
      }

      setSessionId(reply.session_id)
      setMessages((prev) => [...prev, reply]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong while contacting the server.",
          session_id: undefined,
          transcript: undefined
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('Effect triggered - isStreaming:', isStreaming, 'transcript:', transcript);

    if (!isStreaming || !transcript) return;

    console.log('Starting audio streaming with transcript:', transcript.length, 'characters');

    // Pause and reset audio if needed
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    const playAudioInChunks = async () => {
      const audio = audioRef.current;
      if (!audio || !transcript) return;

      console.log('Making POST request to /v1/mindfulness/audio with transcript length:', transcript.length);

      try {
        // Make the POST request to get the audio stream

        cartesiaTTSClient.sendTranscript(transcript)

        // Play the stream in chunks
        const playStream = async () => {
          if (audioRef.current) {
            audioRef.current.play().catch((err) => {
              console.error('Error playing audio:', err);
            });
          }
        };

        playStream();
      } catch (error) {
        console.error('Error during audio streaming:', error);
      }
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

          {/* Audio playback indicator */}
          {transcript && (
            <div className="flex justify-center mt-2">
              <audio
                ref={audioRef}
                className="hidden" // Hidden by default, but accessible
                style={{ display: 'none' }}
                onCanPlay={() => console.log('Audio can play')}
                onEnded={() => console.log('Audio ended')}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
