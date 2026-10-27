import React from 'react';

const SOURCE_COLORS: Record<string, string> = {
  'Source A': 'bg-blue-100 text-blue-800 border border-blue-200',
  'Source B': 'bg-emerald-100 text-emerald-800 border border-emerald-200',
  'Source C': 'bg-amber-100 text-amber-800 border border-amber-200',
  'Source D': 'bg-purple-100 text-purple-800 border border-purple-200',
  'Source E': 'bg-rose-100 text-rose-800 border border-rose-200',
  'Source F': 'bg-cyan-100 text-cyan-800 border border-cyan-200',
};

export function ProvenanceTag({ sourceId }: { sourceId: string }) {
  const colorClass = SOURCE_COLORS[sourceId] || 'bg-slate-100 text-slate-700 border border-slate-200';
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-semibold mx-0.5 ${colorClass}`}>
      {sourceId}
    </span>
  );
}

/**
 * Parse markdown text and convert [Source X] and [Source X, Source Y] patterns
 * into colored ProvenanceTag components.
 */
export function parseProvenanceTags(text: string): React.ReactNode[] {
  const pattern = /\[(Source [A-Z](?:,\s*Source [A-Z])*)\]/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const sources = match[1].split(',').map(s => s.trim());
    sources.forEach((src, i) => {
      parts.push(<ProvenanceTag key={`${match!.index}-${i}`} sourceId={src} />);
    });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}
