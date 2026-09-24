'use client';
import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User, AlertCircle } from 'lucide-react';
import type { Message, ChatResponse } from '@/types';
import { AnswerCard } from './AnswerCard';
import { SourcePanel } from './SourcePanel';

interface ChatPanelProps {
  sessionId: string | null;
  onResponse?: (resp: ChatResponse) => void;
}

/** Safely parse JSON from a fetch Response; falls back to text on failure. */
async function safeJson(resp: Response): Promise<any> {
  const text = await resp.text();
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(
      `Server error (${resp.status}): ${text.slice(0, 200) || 'No response body'}`
    );
  }
}

export function ChatPanel({ sessionId, onResponse }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeResponse, setActiveResponse] = useState<ChatResponse | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || !sessionId || loading) return;
    const question = input.trim();
    setInput('');

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const controller = new AbortController();
      // 3-minute timeout for classify + synthesize (two LLM calls)
      const timeout = setTimeout(() => controller.abort(), 180_000);

      let resp: Response;
      try {
        resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, question }),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeout);
      }

      if (!resp.ok) {
        const errData = await safeJson(resp);
        throw new Error(
          typeof errData === 'object'
            ? errData.detail || errData.message || JSON.stringify(errData)
            : String(errData)
        );
      }

      const data: ChatResponse = await safeJson(resp);
      setActiveResponse(data);
      onResponse?.(data);

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.answer,
        chatResponse: data,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: any) {
      const errText =
        e.name === 'AbortError'
          ? 'Request timed out. The LLM may be slow — please try again.'
          : e.message || 'Unknown error';

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `⚠️ ${errText}`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex gap-4 h-full">
      {/* Chat Column */}
      <div className="flex-1 flex flex-col bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <Bot className="w-12 h-12 text-brand-300 mb-3" />
              <p className="text-slate-500 text-sm">
                {sessionId
                  ? 'Ask any question about the loaded topic!'
                  : 'Load a topic first using the panel on the left.'}
              </p>
              {sessionId && (
                <p className="text-xs text-slate-400 mt-2">
                  ⏱ First question may take ~15s (two LLM calls)
                </p>
              )}
            </div>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  msg.role === 'user' ? 'bg-brand-600' : 'bg-slate-100'
                }`}
              >
                {msg.role === 'user' ? (
                  <User className="w-4 h-4 text-white" />
                ) : (
                  <Bot className="w-4 h-4 text-slate-600" />
                )}
              </div>
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-brand-600 text-white text-sm'
                    : 'bg-slate-50 border border-slate-200'
                }`}
              >
                {msg.role === 'assistant' && msg.chatResponse ? (
                  <AnswerCard answer={msg.content} />
                ) : (
                  <p className="text-sm">{msg.content}</p>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
                <Bot className="w-4 h-4 text-slate-600" />
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3">
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Retrieving → Classifying → Synthesizing…
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-200 p-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              disabled={!sessionId || loading}
              placeholder={sessionId ? 'Ask a question…' : 'Load a topic first'}
              className="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:opacity-50"
            />
            <button
              onClick={sendMessage}
              disabled={!sessionId || loading || !input.trim()}
              className="px-4 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white rounded-xl transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Source Comparison Panel */}
      {activeResponse && (
        <div className="w-80 overflow-y-auto">
          <SourcePanel chatResponse={activeResponse} />
        </div>
      )}
    </div>
  );
}
