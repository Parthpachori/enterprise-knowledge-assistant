import { SearchResult } from './rag';

const NOT_FOUND_MESSAGE = "Information not found in the indexed documents.";

export interface AnswerResult {
  answer: string;
  sources: string[];
  found: boolean;
  context: string;
}

// Extract relevant sentences from context
function extractRelevantSentences(context: string, query: string, maxSentences: number = 5): string {
  const queryTokens = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  const sentences = context.match(/[^.!?]+[.!?]+/g) || [context];

  const scoredSentences = sentences.map(sentence => {
    const lowerSentence = sentence.toLowerCase();
    let score = 0;
    queryTokens.forEach(token => {
      if (lowerSentence.includes(token)) {
        score += 1;
      }
    });
    return { sentence: sentence.trim(), score };
  });

  const relevant = scoredSentences
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences)
    .map(s => s.sentence);

  return relevant.join(' ');
}

// Generate answer from search results
export function generateAnswer(query: string, results: SearchResult[]): AnswerResult {
  if (results.length === 0) {
    return {
      answer: NOT_FOUND_MESSAGE,
      sources: [],
      found: false,
      context: ''
    };
  }

  // Build context from results
  const contextParts = results.map((r, i) => {
    return `[Document ${i + 1} - ${r.document.source}${r.document.page ? `, Page ${r.document.page}` : ''}]\n${r.document.content}`;
  });
  const context = contextParts.join('\n\n');

  // Extract relevant sentences
  const relevantContent = extractRelevantSentences(context, query);

  if (!relevantContent) {
    return {
      answer: NOT_FOUND_MESSAGE,
      sources: [],
      found: false,
      context
    };
  }

  // Format answer
  let answer = relevantContent;

  // Add citation markers
  const topResult = results[0];
  if (topResult) {
    answer += `\n\n[Source: ${topResult.document.source}${topResult.document.page ? `, Page ${topResult.document.page}` : ''}]`;
  }

  // Format sources
  const sources: string[] = [];
  const seenSources = new Set<string>();

  results.forEach(r => {
    const source = r.document.page
      ? `${r.document.source} (Page ${r.document.page})`
      : r.document.source;

    if (!seenSources.has(source)) {
      seenSources.add(source);
      sources.push(source);
    }
  });

  return {
    answer,
    sources,
    found: true,
    context
  };
}

// Format answer for display
export function formatAnswerForDisplay(result: AnswerResult): string {
  if (!result.found) {
    return result.answer;
  }

  let output = result.answer;

  if (result.sources.length > 0) {
    output += '\n\n**Sources:** ' + result.sources.join(' | ');
  }

  return output;
}
