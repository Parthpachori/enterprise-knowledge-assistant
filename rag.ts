// BM25 Search Implementation
export interface Document {
  id: string;
  content: string;
  source: string;
  page?: number;
  metadata: Record<string, any>;
}

export interface Chunk extends Document {
  chunkIndex: number;
  totalChunks: number;
}

export interface SearchResult {
  document: Chunk;
  score: number;
}

// Tokenizer for BM25
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(token => token.length > 0);
}

// BM25 Implementation
export class BM25Index {
  private documents: Chunk[] = [];
  private docLengths: number[] = [];
  private avgDocLength: number = 0;
  private termFreqs: Map<string, Map<number, number>> = new Map();
  private docFreqs: Map<string, number> = new Map();
  private k1 = 1.5;
  private b = 0.75;

  addDocuments(docs: Chunk[]): void {
    this.documents = [...this.documents, ...docs];
    this.buildIndex();
  }

  clear(): void {
    this.documents = [];
    this.docLengths = [];
    this.avgDocLength = 0;
    this.termFreqs.clear();
    this.docFreqs.clear();
  }

  private buildIndex(): void {
    this.docLengths = [];
    this.termFreqs.clear();
    this.docFreqs.clear();

    let totalLength = 0;

    this.documents.forEach((doc, docId) => {
      const tokens = tokenize(doc.content);
      this.docLengths.push(tokens.length);
      totalLength += tokens.length;

      const termFreq = new Map<string, number>();
      tokens.forEach(token => {
        termFreq.set(token, (termFreq.get(token) || 0) + 1);
      });

      termFreq.forEach((freq, term) => {
        if (!this.termFreqs.has(term)) {
          this.termFreqs.set(term, new Map());
        }
        this.termFreqs.get(term)!.set(docId, freq);

        this.docFreqs.set(term, (this.docFreqs.get(term) || 0) + 1);
      });
    });

    this.avgDocLength = totalLength / this.documents.length || 0;
  }

  search(query: string, k: number = 5): SearchResult[] {
    if (this.documents.length === 0) return [];

    const queryTokens = tokenize(query);
    const scores: number[] = new Array(this.documents.length).fill(0);

    const N = this.documents.length;

    queryTokens.forEach(token => {
      const df = this.docFreqs.get(token) || 0;
      if (df === 0) return;

      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
      const termMap = this.termFreqs.get(token);

      if (!termMap) return;

      termMap.forEach((tf, docId) => {
        const docLength = this.docLengths[docId];
        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLength / this.avgDocLength));
        scores[docId] += idf * (numerator / denominator);
      });
    });

    const results: SearchResult[] = scores
      .map((score, idx) => ({ document: this.documents[idx], score }))
      .filter(r => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);

    return results;
  }

  getDocumentCount(): number {
    return this.documents.length;
  }
}

// Vector similarity (simple TF-IDF based for semantic-like search)
export class VectorIndex {
  private documents: Chunk[] = [];
  private docVectors: Map<string, Map<string, number>> = new Map();
  private idfCache: Map<string, number> = new Map();

  addDocuments(docs: Chunk[]): void {
    this.documents = [...this.documents, ...docs];
    this.buildVectors();
  }

  clear(): void {
    this.documents = [];
    this.docVectors.clear();
    this.idfCache.clear();
  }

  private buildVectors(): void {
    this.docVectors.clear();
    this.idfCache.clear();

    const N = this.documents.length;
    const docFreqs = new Map<string, number>();

    // Calculate document frequencies
    this.documents.forEach(doc => {
      const tokens = new Set(tokenize(doc.content));
      tokens.forEach(token => {
        docFreqs.set(token, (docFreqs.get(token) || 0) + 1);
      });
    });

    // Calculate IDF
    docFreqs.forEach((df, term) => {
      this.idfCache.set(term, Math.log(N / df + 1));
    });

    // Build document vectors
    this.documents.forEach(doc => {
      const tokens = tokenize(doc.content);
      const tf = new Map<string, number>();

      tokens.forEach(token => {
        tf.set(token, (tf.get(token) || 0) + 1);
      });

      const vector = new Map<string, number>();
      tf.forEach((freq, term) => {
        const idf = this.idfCache.get(term) || 0;
        vector.set(term, freq * idf);
      });

      this.docVectors.set(doc.id, vector);
    });
  }

  search(query: string, k: number = 5): SearchResult[] {
    if (this.documents.length === 0) return [];

    const queryTokens = tokenize(query);
    const queryVector = new Map<string, number>();

    queryTokens.forEach(token => {
      const idf = this.idfCache.get(token) || 1;
      queryVector.set(token, (queryVector.get(token) || 0) + idf);
    });

    const scores: { document: Chunk; score: number }[] = [];

    this.documents.forEach(doc => {
      const docVector = this.docVectors.get(doc.id);
      if (!docVector) return;

      // Cosine similarity
      let dotProduct = 0;
      let queryNorm = 0;
      let docNorm = 0;

      queryVector.forEach((val, term) => {
        const docVal = docVector.get(term) || 0;
        dotProduct += val * docVal;
        queryNorm += val * val;
      });

      docVector.forEach(val => {
        docNorm += val * val;
      });

      const normProduct = Math.sqrt(queryNorm) * Math.sqrt(docNorm);
      const similarity = normProduct > 0 ? dotProduct / normProduct : 0;

      scores.push({ document: doc, score: similarity });
    });

    return scores
      .filter(r => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  getDocumentCount(): number {
    return this.documents.length;
  }
}

// Hybrid Search combining BM25 and Vector
export class HybridSearch {
  private bm25: BM25Index;
  private vector: VectorIndex;
  private alpha = 0.5; // Weight for vector search

  constructor() {
    this.bm25 = new BM25Index();
    this.vector = new VectorIndex();
  }

  addDocuments(docs: Chunk[]): void {
    this.bm25.addDocuments(docs);
    this.vector.addDocuments(docs);
  }

  clear(): void {
    this.bm25.clear();
    this.vector.clear();
  }

  search(query: string, k: number = 5, mode: 'bm25' | 'vector' | 'hybrid' = 'hybrid'): SearchResult[] {
    if (mode === 'bm25') {
      return this.bm25.search(query, k);
    }

    if (mode === 'vector') {
      return this.vector.search(query, k);
    }

    // Hybrid: combine both
    const bm25Results = this.bm25.search(query, k * 2);
    const vectorResults = this.vector.search(query, k * 2);

    // Normalize scores
    const normalizeScores = (results: SearchResult[]) => {
      if (results.length === 0) return results;
      const maxScore = Math.max(...results.map(r => r.score), 1);
      return results.map(r => ({ ...r, score: r.score / maxScore }));
    };

    const normalizedBM25 = normalizeScores(bm25Results);
    const normalizedVector = normalizeScores(vectorResults);

    // Combine scores
    const combined = new Map<string, SearchResult>();

    normalizedBM25.forEach(result => {
      const existing = combined.get(result.document.id);
      if (existing) {
        existing.score += result.score * (1 - this.alpha);
      } else {
        combined.set(result.document.id, {
          ...result,
          score: result.score * (1 - this.alpha)
        });
      }
    });

    normalizedVector.forEach(result => {
      const existing = combined.get(result.document.id);
      if (existing) {
        existing.score += result.score * this.alpha;
      } else {
        combined.set(result.document.id, {
          ...result,
          score: result.score * this.alpha
        });
      }
    });

    return Array.from(combined.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  getDocumentCount(): number {
    return this.bm25.getDocumentCount();
  }
}

// Text Chunker
export function chunkText(
  text: string,
  source: string,
  page: number | undefined,
  chunkSize: number = 1000,
  overlap: number = 200
): Chunk[] {
  const chunks: Chunk[] = [];

  if (text.length <= chunkSize) {
    chunks.push({
      id: `${source}-0`,
      content: text,
      source,
      page,
      chunkIndex: 0,
      totalChunks: 1,
      metadata: { source, page }
    });
    return chunks;
  }

  let start = 0;
  let chunkIndex = 0;

  while (start < text.length) {
    let end = start + chunkSize;

    // Try to end at a sentence boundary
    if (end < text.length) {
      const lastPeriod = text.lastIndexOf('.', end);
      const lastNewline = text.lastIndexOf('\n', end);
      const boundary = Math.max(lastPeriod, lastNewline);
      if (boundary > start + chunkSize / 2) {
        end = boundary + 1;
      }
    }

    const content = text.slice(start, end).trim();
    if (content.length > 0) {
      chunks.push({
        id: `${source}-${chunkIndex}`,
        content,
        source,
        page,
        chunkIndex,
        totalChunks: 0, // Will be updated
        metadata: { source, page }
      });
    }

    start = end - overlap;
    chunkIndex++;
  }

  // Update total chunks
  chunks.forEach(chunk => {
    chunk.totalChunks = chunks.length;
  });

  return chunks;
}

// Format source citation
export function formatSource(result: SearchResult): string {
  const { source, page } = result.document;
  if (page !== undefined) {
    return `${source} (Page ${page})`;
  }
  return source;
}

// Format multiple sources
export function formatSources(results: SearchResult[]): string[] {
  const seen = new Set<string>();
  return results
    .map(r => formatSource(r))
    .filter(source => {
      if (seen.has(source)) return false;
      seen.add(source);
      return true;
    });
}
