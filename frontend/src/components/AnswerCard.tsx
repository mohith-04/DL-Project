'use client';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { parseProvenanceTags } from './ProvenanceTag';

function renderTextWithTags(text: string): React.ReactNode {
  const parts = parseProvenanceTags(text);
  return <>{parts}</>;
}

interface AnswerCardProps {
  answer: string;
}

export function AnswerCard({ answer }: AnswerCardProps) {
  return (
    <div className="prose prose-slate max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => (
            <p className="mb-3 leading-relaxed text-slate-700">
              {typeof children === 'string'
                ? renderTextWithTags(children)
                : children}
            </p>
          ),
          li: ({ children }) => (
            <li className="text-slate-700">
              {typeof children === 'string'
                ? renderTextWithTags(children)
                : children}
            </li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-slate-900">{children}</strong>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-semibold text-slate-900 mt-5 mb-2 border-b border-slate-200 pb-1">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold text-slate-800 mt-4 mb-1">{children}</h3>
          ),
          code: ({ children, className }) => {
            const isBlock = className?.includes('language-');
            return isBlock ? (
              <code className="block bg-slate-900 text-slate-100 rounded-lg p-4 overflow-x-auto text-sm my-3 font-mono">
                {children}
              </code>
            ) : (
              <code className="bg-slate-100 text-slate-800 rounded px-1 py-0.5 text-sm font-mono">
                {children}
              </code>
            );
          },
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-brand-500 pl-4 italic text-slate-600 my-3">
              {children}
            </blockquote>
          ),
        }}
      >
        {answer}
      </ReactMarkdown>
    </div>
  );
}
