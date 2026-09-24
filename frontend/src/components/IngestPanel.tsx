'use client';
import React, { useState } from 'react';
import { Link, Loader2, CheckCircle2, BookOpen, AlertCircle } from 'lucide-react';
import type { IngestResponse } from '@/types';
import { ProvenanceTag } from './ProvenanceTag';

interface IngestPanelProps {
  onIngested: (resp: IngestResponse) => void;
}

/** Safely parse JSON from a fetch Response; falls back to text on failure. */
async function safeJson(resp: Response): Promise<any> {
  const text = await resp.text();
  try {
    return JSON.parse(text);
  } catch {
    // Server returned non-JSON (HTML error page, plain text, etc.)
    throw new Error(
      `Server error (${resp.status}): ${text.slice(0, 200) || 'No response body'}`
    );
  }
}

export function IngestPanel({ onIngested }: IngestPanelProps) {
  const [url, setUrl] = useState('');
  const [topicName, setTopicName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [step, setStep] = useState<string>('');

  const handleIngest = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Step indicators so the user knows it's working (scraping takes ~30s)
      setStep('Discovering related sources…');
      const controller = new AbortController();
      // 3-minute timeout — discovery + scraping + embedding can be slow
      const timeout = setTimeout(() => controller.abort(), 180_000);

      let resp: Response;
      try {
        setStep('Scraping & embedding sources…');
        resp = await fetch('/api/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic_url: url, topic_name: topicName }),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeout);
      }

      setStep('');

      if (!resp.ok) {
        const errData = await safeJson(resp);
        throw new Error(
          typeof errData === 'object'
            ? errData.detail || errData.message || JSON.stringify(errData)
            : String(errData)
        );
      }

      const data: IngestResponse = await safeJson(resp);
      setResult(data);
      onIngested(data);
    } catch (e: any) {
      if (e.name === 'AbortError') {
        setError('Request timed out after 3 minutes. The site may be slow to scrape — try again.');
      } else {
        setError(e.message || 'Unknown error');
      }
    } finally {
      setLoading(false);
      setStep('');
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
      <div className="flex items-center gap-2 mb-4">
        <BookOpen className="w-5 h-5 text-brand-600" />
        <h2 className="text-base font-semibold text-slate-800">Load Topic</h2>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-xs font-medium text-slate-500 mb-1 block">Topic Name (optional)</label>
          <input
            type="text"
            value={topicName}
            onChange={(e) => setTopicName(e.target.value)}
            placeholder="e.g. Retrieval-Augmented Generation"
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
        </div>
        <div>
          <label className="text-xs font-medium text-slate-500 mb-1 block">Primary Source URL *</label>
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://en.wikipedia.org/wiki/..."
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            onKeyDown={(e) => e.key === 'Enter' && !loading && handleIngest()}
          />
        </div>

        {/* Live step indicator */}
        {loading && step && (
          <p className="text-xs text-brand-600 flex items-center gap-1.5">
            <Loader2 className="w-3 h-3 animate-spin" />
            {step}
          </p>
        )}

        {/* Error display */}
        {error && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-2.5">
            <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-red-700 break-words">{error}</p>
          </div>
        )}

        <button
          onClick={handleIngest}
          disabled={loading || !url.trim()}
          className="w-full py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Discovering &amp; Indexing…</>
          ) : (
            <><Link className="w-4 h-4" /> Load &amp; Index Sources</>
          )}
        </button>

        {/* Tip about wait time */}
        {!loading && !result && (
          <p className="text-xs text-slate-400 text-center">
            ⏱ Scraping usually takes 20–60 seconds
          </p>
        )}
      </div>

      {result && (
        <div className="mt-4 pt-4 border-t border-slate-100">
          <div className="flex items-center gap-1.5 text-emerald-700 text-sm font-medium mb-2">
            <CheckCircle2 className="w-4 h-4" />
            Indexed {result.total_chunks} chunks from {result.sources.length} source{result.sources.length !== 1 ? 's' : ''}
          </div>
          <div className="space-y-1">
            {result.sources.map((s) => (
              <div key={s.source_id} className="flex items-start gap-2 text-xs">
                <ProvenanceTag sourceId={s.source_id} />
                <span className="text-slate-600 truncate flex-1" title={s.url}>
                  {s.title || s.url}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
