'use client';
import React, { useState } from 'react';
import { Brain, Github } from 'lucide-react';
import { IngestPanel } from '@/components/IngestPanel';
import { ChatPanel } from '@/components/ChatPanel';
import type { IngestResponse } from '@/types';

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [topicName, setTopicName] = useState<string>('');

  const handleIngested = (resp: IngestResponse) => {
    setSessionId(resp.session_id);
    setTopicName(resp.topic_name);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center gap-3">
          <Brain className="w-6 h-6 text-brand-600" />
          <span className="font-bold text-slate-900 text-lg">Multi-Source AI Tutor</span>
          {topicName && (
            <span className="ml-2 text-sm text-slate-500">
              — <span className="font-medium text-slate-700">{topicName}</span>
            </span>
          )}
          <div className="ml-auto flex items-center gap-2 text-xs text-slate-400">
            <span>Powered by Llama 3 × Groq</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6 flex gap-4">
        {/* Left sidebar: Ingest */}
        <div className="w-72 flex-shrink-0">
          <IngestPanel onIngested={handleIngested} />

          {/* How it works */}
          <div className="mt-4 bg-white rounded-2xl border border-slate-200 p-4">
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">How It Works</h3>
            <ol className="space-y-2 text-xs text-slate-600">
              <li className="flex gap-2">
                <span className="font-bold text-brand-600 w-4">1.</span>
                <span>Enter a URL → system discovers 2-3 related sources</span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-brand-600 w-4">2.</span>
                <span>All sources are scraped, chunked, and embedded separately</span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-brand-600 w-4">3.</span>
                <span>Each question retrieves top-k chunks <em>per source</em></span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-brand-600 w-4">4.</span>
                <span>Groq classifies source agreement (agree/partial/conflict)</span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-brand-600 w-4">5.</span>
                <span>Answer synthesized with transparent provenance tags</span>
              </li>
            </ol>
          </div>
        </div>

        {/* Main: Chat + Source Panel */}
        <div className="flex-1 min-h-[600px]">
          <ChatPanel sessionId={sessionId} />
        </div>
      </main>
    </div>
  );
}
