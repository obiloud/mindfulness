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
export interface MessageWithTranscript extends Message {
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

export default function App({
  initialMessages = [],
  initialInput = "",
  initialTranscript = null,
  initialSessionId = undefined,
}: {
  initialMessages?: MessageWithTranscript[];
  initialInput?: string;
  initialTranscript?: string | null;
  initialSessionId?: string | undefined;
}) {
  const [messages, setMessages] = useState<MessageWithTranscript[]>(initialMessages);
  const [input, setInput] = useState(initialInput);
  const [loading, setLoading] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(initialTranscript || null);
  const [session_id, setSessionId] = useState<string | undefined>(initialSessionId);

  const messagesContainerRef = useRef<HTMLDivElement | null>(null);

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
    setTranscript(null);

    try {
      const reply = await sendMessage(userMsg.content, session_id);

      console.log('Received transcript from assistant:', reply.transcript ? reply.transcript.length : 'null');

      if (reply.transcript) {
        setTranscript(reply.transcript);
      }

      setSessionId(reply.session_id);
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
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
      const delay = setTimeout(() => {
        messagesContainerRef.current?.scrollTo(0, messagesContainerRef.current.scrollHeight);
      }, 50);

      return () => clearTimeout(delay);
    }
  }, [messages]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      {/* Conditional Rendering: Chat or Transcript View */}
      {transcript ? (
        <TranscriptView
          transcript={transcript}
          onBackToChat={() => {
            setTranscript(null);
          }}
          messages={messages}
        />
      ) : (
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

          <div className="flex-1 min-h-[320px] max-h-[420px] overflow-y-auto space-y-3 pr-1" ref={messagesContainerRef}>
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
      )}
    </div>
  );
}

export const TranscriptView = ({
  transcript,
  onBackToChat,
  messages,
  initialIsPlaying = false,
  initialTranscriptSent = false,
}: {
  transcript: string;
  onBackToChat: () => void;
  messages: MessageWithTranscript[];
  initialIsPlaying?: boolean;
  initialTranscriptSent?: boolean;
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(initialIsPlaying);
  const [transcriptSent, setTranscriptSent] = useState(initialTranscriptSent);

  const playAudio = async () => {
    if (!transcript || !audioRef.current) {
      console.log('Transcript is null or audio ref is null');
      return;
    }

    // Log state before sending transcript
    console.log('Transcript sent state:', transcriptSent);

    // Only send transcript once when the user first clicks the button
    if (!transcriptSent) {
      try {
        cartesiaTTSClient.sendTranscript(transcript);
        console.log('Transcript sent to Cartesia TTS successfully');
        setTranscriptSent(true);
      } catch (error) {
        console.error('Error sending transcript to Cartesia TTS:', error);
      }
    }

    // Toggle play/pause state
    if (isPlaying) {
      audioRef.current.pause();
      console.log('Audio paused');
    } else {
      await audioRef.current.play();
      console.log('Audio started playing');
    }

    setIsPlaying(!isPlaying);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-mind-surface/80 backdrop-blur-md">
      <button
        onClick={onBackToChat}
        className="absolute top-6 left-6 text-slate-300 hover:text-white transition-colors"
        aria-label="Back to chat"
        disabled={isPlaying}
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
      </button>

      <div className="w-full max-w-3xl mt-10 mb-8 text-center">
        <h2 data-testid="mindfulness-response-header"
          className="text-2xl font-bold text-slate-100 mb-4">
          Your Mindfulness Response
        </h2>
        <p className="text-lg text-slate-300 mb-6 leading-relaxed">
          {messages.length > 0 ? messages[messages.length - 1]?.content : 'No message yet.'}
        </p>

        <div className="mb-6">
          <button
            data-testid="play-audio-button"
            onClick={playAudio}
            className={`bg-mind-accent text-slate-950 px-10 py-4 rounded-full text-xl font-semibold hover:bg-sky-400 transition-colors shadow-lg ${isPlaying ? 'bg-mind-accent/80' : 'bg-mind-accent'
              }`}
          >
            {isPlaying ? 'Pause Audio' : 'Play Audio'}
          </button>
        </div>

        <p className="text-sm text-slate-400">
          For the best experience, use headphones.
        </p>

        <div className="flex justify-center mt-2">
          <audio
            ref={audioRef}
            className="hidden"
            style={{ display: 'none' }}
            onCanPlay={() => console.log('Audio can play')}
            onEnded={() => console.log('Audio ended')}
          />
        </div>
      </div>
    </div>
  );
};