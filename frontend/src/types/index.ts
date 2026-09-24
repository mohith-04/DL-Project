export interface SourceInfo {
  source_id: string;
  url: string;
  title: string;
  char_count: number;
}

export interface IngestResponse {
  session_id: string;
  topic_name: string;
  sources: SourceInfo[];
  total_chunks: number;
}

export interface ChunkInfo {
  source_id: string;
  url: string;
  title: string;
  text: string;
  score: number;
}

export interface ClassificationResult {
  classification: 'agree' | 'partial' | 'conflict';
  confidence: number;
  reasoning: string;
  key_differences: string;
}

export interface SourceSummary {
  source_id: string;
  url: string;
  title: string;
  summary: string;
}

export interface ChatResponse {
  answer: string;
  classification: ClassificationResult;
  source_summaries: SourceSummary[];
  retrieved_chunks: Record<string, ChunkInfo[]>;
  session_id: string;
  question: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  chatResponse?: ChatResponse;
  timestamp: Date;
}
