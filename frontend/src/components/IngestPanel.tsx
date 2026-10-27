'use client';
import React, { useState, useRef } from 'react';
import {
  Link,
  Loader2,
  CheckCircle2,
  BookOpen,
  AlertCircle,
  Upload,
  X,
  FileText,
  Layers,
  Globe,
} from 'lucide-react';
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
    throw new Error(
      `Server error (${resp.status}): ${text.slice(0, 200) || 'No response body'}`
    );
  }
}

type IngestMode = 'hybrid' | 'url_only' | 'files_only';

export function IngestPanel({ onIngested }: IngestPanelProps) {
  const [mode, setMode] = useState<IngestMode>('hybrid');

  // Input states
  const [topicName, setTopicName] = useState('');
  const [url, setUrl] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Status states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [step, setStep] = useState<string>('');

  const isAllowed = (f: File) =>
    f.type === 'application/pdf' ||
    f.name.endsWith('.pdf') ||
    f.name.endsWith('.txt') ||
    f.name.endsWith('.md');

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter(isAllowed);
    setFiles((prev) => [...prev, ...dropped].slice(0, 5));
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const selected = Array.from(e.target.files).filter(isAllowed);
    setFiles((prev) => [...prev, ...selected].slice(0, 5));
    e.target.value = '';
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleIngest = async () => {
    const hasUrl = Boolean(url.trim());
    const hasFiles = files.length > 0;

    if (!hasUrl && !hasFiles) {
      setError('Please provide at least one source (enter a Web URL, attach files, or both).');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      if (hasUrl) {
        formData.append('topic_url', url.trim());
      }
      if (topicName.trim()) {
        formData.append('topic_name', topicName.trim());
      }
      files.forEach((f) => {
        formData.append('files', f);
      });

      if (hasUrl && hasFiles) {
        setStep('Scraping web link & extracting PDF text…');
      } else if (hasUrl) {
        setStep('Scraping web source & discovering related content…');
      } else {
        setStep('Extracting document text & processing chunks…');
      }

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 180_000);

      let resp: Response;
      try {
        resp = await fetch('/api/ingest_hybrid', {
          method: 'POST',
          body: formData,
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
        setError('Request timed out after 3 minutes. The web source or files may be large.');
      } else {
        setError(e.message || 'Unknown ingestion error');
      }
    } finally {
      setLoading(false);
      setStep('');
    }
  };

  const hasUrl = Boolean(url.trim());
  const hasFiles = files.length > 0;

  // Compute dynamic button label
  let buttonLabel = 'Index Sources';
  if (hasUrl && hasFiles) {
    buttonLabel = `Index Web Link + ${files.length} File${files.length > 1 ? 's' : ''}`;
  } else if (hasUrl) {
    buttonLabel = 'Index Web Source';
  } else if (hasFiles) {
    buttonLabel = `Index ${files.length} Document${files.length > 1 ? 's' : ''}`;
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <BookOpen className="w-5 h-5 text-brand-600" />
        <div>
          <h2 className="text-base font-semibold text-slate-800">Knowledge Ingestion</h2>
          <p className="text-xs text-slate-500">Provide a link, upload docs, or both together</p>
        </div>
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex border border-slate-200 rounded-lg overflow-hidden text-xs font-medium">
        <button
          onClick={() => {
            setMode('hybrid');
            setError(null);
          }}
          className={`flex-1 py-1.5 flex items-center justify-center gap-1 transition-colors ${
            mode === 'hybrid' ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Hybrid (Both)</span>
        </button>
        <button
          onClick={() => {
            setMode('url_only');
            setError(null);
          }}
          className={`flex-1 py-1.5 flex items-center justify-center gap-1 transition-colors ${
            mode === 'url_only' ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Globe className="w-3.5 h-3.5" />
          <span>Link Only</span>
        </button>
        <button
          onClick={() => {
            setMode('files_only');
            setError(null);
          }}
          className={`flex-1 py-1.5 flex items-center justify-center gap-1 transition-colors ${
            mode === 'files_only' ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Docs Only</span>
        </button>
      </div>

      <div className="space-y-3">
        {/* Topic Name (Always available) */}
        <div>
          <label className="text-xs font-medium text-slate-500 mb-1 block">
            Topic Name <span className="text-slate-400 font-normal">(optional)</span>
          </label>
          <input
            type="text"
            value={topicName}
            onChange={(e) => setTopicName(e.target.value)}
            placeholder="e.g. Retrieval-Augmented Generation"
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
        </div>

        {/* Web Source URL (Available in Hybrid or URL Only mode) */}
        {(mode === 'hybrid' || mode === 'url_only') && (
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1 flex items-center justify-between">
              <span className="flex items-center gap-1">
                <Globe className="w-3.5 h-3.5 text-brand-500" />
                Web Source URL
              </span>
              {mode === 'hybrid' && (
                <span className="text-[11px] text-slate-400 font-normal">
                  {hasFiles ? 'optional if docs attached' : 'enter URL or attach doc'}
                </span>
              )}
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://en.wikipedia.org/wiki/..."
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
              onKeyDown={(e) => e.key === 'Enter' && !loading && handleIngest()}
            />
          </div>
        )}

        {/* File Upload Zone (Available in Hybrid or Files Only mode) */}
        {(mode === 'hybrid' || mode === 'files_only') && (
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1 flex items-center justify-between">
              <span className="flex items-center gap-1">
                <FileText className="w-3.5 h-3.5 text-brand-500" />
                Upload Documents (PDF, TXT, MD)
              </span>
              {mode === 'hybrid' && (
                <span className="text-[11px] text-slate-400 font-normal">
                  {hasUrl ? 'optional' : 'attach files or enter URL'}
                </span>
              )}
            </label>

            {/* Drop Zone */}
            <div
              onDrop={handleFileDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-200 rounded-xl p-3 text-center cursor-pointer hover:border-brand-400 hover:bg-brand-50 transition-colors"
            >
              <Upload className="w-5 h-5 text-slate-400 mx-auto mb-1" />
              <p className="text-xs text-slate-600">
                Drop PDF / TXT / MD files here or <span className="text-brand-600 font-medium">browse</span>
              </p>
              <p className="text-[10px] text-slate-400 mt-0.5">Up to 5 documents</p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md"
                multiple
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="mt-2 space-y-1.5 max-h-36 overflow-y-auto">
                {files.map((f, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-1.5"
                  >
                    <FileText className="w-3.5 h-3.5 text-brand-600 flex-shrink-0" />
                    <span className="text-xs text-slate-700 flex-1 truncate" title={f.name}>
                      {f.name}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {(f.size / 1024).toFixed(0)} KB
                    </span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(idx);
                      }}
                      className="text-slate-400 hover:text-red-500 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Live Step Indicator */}
        {loading && step && (
          <p className="text-xs text-brand-600 flex items-center gap-1.5">
            <Loader2 className="w-3 h-3 animate-spin" />
            {step}
          </p>
        )}

        {/* Error Message */}
        {error && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-2.5">
            <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-red-700 break-words">{error}</p>
          </div>
        )}

        {/* Action Button */}
        <button
          onClick={handleIngest}
          disabled={loading || (!hasUrl && !hasFiles)}
          className="w-full py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition-colors flex items-center justify-center gap-2 shadow-sm"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Indexing Combined Knowledge…
            </>
          ) : (
            <>
              <Link className="w-4 h-4" />
              {buttonLabel}
            </>
          )}
        </button>

        {!loading && !result && (
          <p className="text-[11px] text-slate-400 text-center">
            ⏱ Indexing usually takes 15–40 seconds depending on document length
          </p>
        )}
      </div>

      {/* Ingestion Results Card */}
      {result && (
        <div className="mt-4 pt-3 border-t border-slate-100 space-y-2">
          <div className="flex items-center gap-1.5 text-emerald-700 text-xs font-semibold">
            <CheckCircle2 className="w-4 h-4" />
            Indexed {result.total_chunks} chunks from {result.sources.length} source
            {result.sources.length !== 1 ? 's' : ''}
          </div>
          <div className="space-y-1">
            {result.sources.map((s) => (
              <div
                key={s.source_id}
                className="flex items-start gap-2 text-xs bg-slate-50 p-1.5 rounded border border-slate-100"
              >
                <ProvenanceTag sourceId={s.source_id} />
                <div className="flex-1 min-w-0">
                  <p className="text-slate-700 font-medium truncate" title={s.title}>
                    {s.title}
                  </p>
                  <p className="text-[10px] text-slate-400 truncate font-mono">
                    {s.url.startsWith('file://') ? `📄 ${s.url.replace('file://', '')}` : s.url}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
