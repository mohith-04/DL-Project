'use client';
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, AlertTriangle, CheckCircle, GitMerge } from 'lucide-react';
import type { ChatResponse } from '@/types';
import { ProvenanceTag } from './ProvenanceTag';

const CLASSIFICATION_CONFIG = {
  agree: {
    icon: CheckCircle,
    label: 'Sources Agree',
    color: 'text-emerald-700',
    bg: 'bg-emerald-50 border-emerald-200',
    badge: 'bg-emerald-100 text-emerald-800',
  },
  partial: {
    icon: GitMerge,
    label: 'Partial Overlap',
    color: 'text-amber-700',
    bg: 'bg-amber-50 border-amber-200',
    badge: 'bg-amber-100 text-amber-800',
  },
  conflict: {
    icon: AlertTriangle,
    label: 'Sources Conflict',
    color: 'text-red-700',
    bg: 'bg-red-50 border-red-200',
    badge: 'bg-red-100 text-red-800',
  },
} as const;

interface SourcePanelProps {
  chatResponse: ChatResponse;
}

export function SourcePanel({ chatResponse }: SourcePanelProps) {
  const [expandedSource, setExpandedSource] = useState<string | null>(null);
  const { classification, source_summaries, retrieved_chunks } = chatResponse;
  const config = CLASSIFICATION_CONFIG[classification.classification];
  const Icon = config.icon;

  return (
    <div className="flex flex-col gap-4">
      {/* Classification Badge */}
      <div className={`rounded-xl border p-4 ${config.bg}`}>
        <div className="flex items-center gap-2 mb-2">
          <Icon className={`w-5 h-5 ${config.color}`} />
          <span className={`font-semibold ${config.color}`}>{config.label}</span>
          <span className={`ml-auto text-xs px-2 py-0.5 rounded-full font-medium ${config.badge}`}>
            {Math.round(classification.confidence * 100)}% confidence
          </span>
        </div>
        <p className="text-sm text-slate-600">{classification.reasoning}</p>
        {classification.key_differences !== 'None' && (
          <div className="mt-2 pt-2 border-t border-current/20">
            <p className="text-xs font-medium text-slate-500 mb-1">Key Differences</p>
            <p className="text-sm text-slate-600">{classification.key_differences}</p>
          </div>
        )}
      </div>

      {/* Per-Source Summaries */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Source Perspectives</h3>
        {source_summaries.map((s) => (
          <div key={s.source_id} className="rounded-lg border border-slate-200 bg-white overflow-hidden">
            <button
              className="w-full flex items-center gap-3 p-3 text-left hover:bg-slate-50 transition-colors"
              onClick={() => setExpandedSource(expandedSource === s.source_id ? null : s.source_id)}
            >
              <ProvenanceTag sourceId={s.source_id} />
              <span className="flex-1 text-sm font-medium text-slate-700 truncate">{s.title}</span>
              {expandedSource === s.source_id ? (
                <ChevronUp className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              )}
            </button>
            {expandedSource === s.source_id && (
              <div className="px-3 pb-3 space-y-2">
                <p className="text-sm text-slate-600">{s.summary}</p>
                <a
                  href={s.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-brand-600 hover:underline"
                >
                  <ExternalLink className="w-3 h-3" />
                  View Source
                </a>
                {/* Retrieved Chunks */}
                {retrieved_chunks[s.source_id]?.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs font-medium text-slate-400 mb-1">Retrieved Chunks</p>
                    <div className="space-y-1">
                      {retrieved_chunks[s.source_id].map((chunk, i) => (
                        <div key={i} className="bg-slate-50 rounded p-2 text-xs text-slate-600">
                          <span className="text-slate-400 font-mono mr-1">score: {chunk.score.toFixed(3)}</span>
                          <span className="line-clamp-2">{chunk.text.slice(0, 150)}...</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
