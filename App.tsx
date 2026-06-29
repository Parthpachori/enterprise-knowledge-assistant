import { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, Send, FileText, Trash2, RefreshCw, Settings, MessageCircle, Book, Search, X, Check, AlertCircle, ChevronDown, ChevronRight } from 'lucide-react';
import { HybridSearch, SearchResult, Chunk } from './lib/rag';
import { processDocument, isFileSupported } from './lib/documents';
import { generateAnswer, AnswerResult } from './lib/answer';

// Sample documents for demo
const SAMPLE_DOCS = [
  {
    name: 'machine_learning.md',
    content: `# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed. The primary aim is to allow computers to learn automatically.

Machine learning algorithms build a mathematical model based on sample data, known as "training data," in order to make predictions or decisions without being explicitly programmed to perform the task.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data and makes predictions. Common applications include:
- Image classification
- Spam detection
- Price prediction

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. Examples include:
- Customer segmentation
- Anomaly detection
- Topic modeling

### Reinforcement Learning
An agent learns by interacting with an environment and receiving rewards or penalties for actions.

## Deep Learning

Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain to learn from large amounts of data.

### Key Components
1. Neural Networks: Computational models inspired by biological neural networks
2. Training: Process of adjusting network weights to minimize error
3. Backpropagation: Algorithm for training neural networks

## Applications

Machine learning and deep learning are used in:
- Natural language processing
- Computer vision
- Healthcare diagnostics
- Financial forecasting
- Autonomous vehicles
- Recommendation systems`
  },
  {
    name: 'python_basics.txt',
    content: `Python Programming Basics

Python is a high-level, interpreted programming language known for its clear syntax and readability. Created by Guido van Rossum and first released in 1991, Python has become one of the most popular programming languages in the world.

Key Features of Python:
1. Easy to learn and read syntax
2. Dynamic typing and automatic memory management
3. Large standard library
4. Cross-platform compatibility
5. Object-oriented, functional, and procedural paradigms

Common Use Cases:
- Web development (Django, Flask)
- Data science and machine learning (NumPy, Pandas, TensorFlow)
- Automation and scripting
- Scientific computing
- Game development

Python's design philosophy emphasizes code readability with the use of significant indentation. Its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages like C++ or Java.`
  },
  {
    name: 'web_development.md',
    content: `# Web Development Guide

Web development encompasses several types of web content creation and maintenance.

## Frontend Development

Frontend development focuses on the user interface and user experience.

### Technologies:
- HTML: Structure of web pages
- CSS: Styling and layout
- JavaScript: Interactivity and dynamic content
- React, Vue, Angular: Modern frontend frameworks

### Best Practices:
1. Responsive design for all devices
2. Accessibility (a11y) standards
3. Performance optimization
4. Cross-browser compatibility

## Backend Development

Backend development handles server-side logic, databases, and APIs.

### Technologies:
- Node.js: JavaScript runtime
- Python (Django, Flask)
- Ruby on Rails
- PHP (Laravel)
- Go

### Database Options:
- SQL: PostgreSQL, MySQL
- NoSQL: MongoDB, Redis
- Graph: Neo4j

## Full-Stack Development

Full-stack developers work on both frontend and backend, understanding the complete web development lifecycle.`
  }
];

type SearchMode = 'bm25' | 'vector' | 'hybrid';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: string[];
  timestamp: Date;
}

interface IndexedDocument {
  name: string;
  chunkCount: number;
  size: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [indexStats, setIndexStats] = useState({ totalChunks: 0, documentCount: 0 });
  const [indexedDocs, setIndexedDocs] = useState<IndexedDocument[]>([]);
  const [searchMode, setSearchMode] = useState<SearchMode>('hybrid');
  const [topK, setTopK] = useState(5);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const searchIndex = useRef(new HybridSearch());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Load sample documents on mount
  useEffect(() => {
    loadSampleDocuments();
  }, []);

  const showStatus = (message: string, duration: number = 3000) => {
    setStatusMessage(message);
    setTimeout(() => setStatusMessage(null), duration);
  };

  const loadSampleDocuments = async () => {
    setIsLoading(true);
    try {
      const allChunks: Chunk[] = [];
      const docStats: Map<string, number> = new Map();

      for (const doc of SAMPLE_DOCS) {
        const chunks = doc.content.length > 1000
          ? chunkContent(doc.content, doc.name)
          : [{ id: `${doc.name}-0`, content: doc.content, source: doc.name, page: undefined, chunkIndex: 0, totalChunks: 1, metadata: {} }];

        chunks.forEach(chunk => {
          allChunks.push(chunk as Chunk);
          docStats.set(doc.name, (docStats.get(doc.name) || 0) + 1);
        });
      }

      searchIndex.current.addDocuments(allChunks);
      updateStats();
      showStatus(`Loaded ${allChunks.length} chunks from ${SAMPLE_DOCS.length} sample documents`);
    } catch (error) {
      console.error('Error loading sample documents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const chunkContent = (content: string, source: string): Chunk[] => {
    const chunkSize = 1000;
    const overlap = 200;
    const chunks: Chunk[] = [];

    if (content.length <= chunkSize) {
      chunks.push({
        id: `${source}-0`,
        content,
        source,
        page: undefined,
        chunkIndex: 0,
        totalChunks: 1,
        metadata: {}
      });
      return chunks;
    }

    let start = 0;
    let chunkIndex = 0;

    while (start < content.length) {
      let end = start + chunkSize;
      if (end < content.length) {
        const lastPeriod = content.lastIndexOf('.', end);
        if (lastPeriod > start + chunkSize / 2) {
          end = lastPeriod + 1;
        }
      }

      chunks.push({
        id: `${source}-${chunkIndex}`,
        content: content.slice(start, end).trim(),
        source,
        page: undefined,
        chunkIndex,
        totalChunks: 0,
        metadata: {}
      });

      start = end - overlap;
      chunkIndex++;
    }

    chunks.forEach(c => c.totalChunks = chunks.length);
    return chunks;
  };

  const updateStats = () => {
    const totalChunks = searchIndex.current.getDocumentCount();
    setIndexStats({ totalChunks, documentCount: indexedDocs.length + SAMPLE_DOCS.length });
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setIsLoading(true);
    let totalChunks = 0;
    const newDocs: IndexedDocument[] = [];
    const errors: string[] = [];

    for (const file of Array.from(files)) {
      if (!isFileSupported(file)) {
        errors.push(`${file.name}: Unsupported file type`);
        continue;
      }

      try {
        showStatus(`Processing ${file.name}...`, 10000);
        const chunks = await processDocument(file);

        if (chunks.length === 0) {
          errors.push(`${file.name}: No text content found`);
          continue;
        }

        searchIndex.current.addDocuments(chunks);
        totalChunks += chunks.length;
        newDocs.push({
          name: file.name,
          chunkCount: chunks.length,
          size: formatFileSize(file.size)
        });
      } catch (error) {
        console.error(`Error processing ${file.name}:`, error);
        const errorMsg = error instanceof Error ? error.message : String(error);
        errors.push(`${file.name}: ${errorMsg}`);
      }
    }

    setIndexedDocs(prev => [...prev, ...newDocs]);
    updateStats();
    setIsLoading(false);

    if (errors.length > 0) {
      showStatus(`Errors: ${errors.join('; ')}`, 5000);
    } else if (totalChunks > 0) {
      showStatus(`Indexed ${totalChunks} chunks from ${newDocs.length} documents`);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const clearIndex = () => {
    searchIndex.current.clear();
    setIndexedDocs([]);
    setMessages([]);
    updateStats();
    showStatus('Index cleared');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Search and generate answer
    setTimeout(() => {
      try {
        const results: SearchResult[] = searchIndex.current.search(input, topK, searchMode);
        const answer: AnswerResult = generateAnswer(input, results);

        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: answer.found ? answer.answer : answer.answer,
          sources: answer.sources,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, assistantMessage]);
      } catch (error) {
        console.error('Error generating answer:', error);
        setMessages(prev => [...prev, {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: 'An error occurred while processing your question.',
          timestamp: new Date()
        }]);
      } finally {
        setIsLoading(false);
      }
    }, 500);
  };

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Status Toast */}
      {statusMessage && (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 bg-slate-800 text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-2">
          <Check className="w-5 h-5 text-green-400" />
          {statusMessage}
        </div>
      )}

      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-16'} bg-slate-900 text-white transition-all duration-300 flex flex-col`}>
        <div className="p-4 flex items-center justify-between border-b border-slate-700">
          {sidebarOpen && (
            <div className="flex items-center gap-2">
              <Book className="w-6 h-6 text-blue-400" />
              <span className="font-semibold">Documents</span>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            {sidebarOpen ? <X className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
          </button>
        </div>

        {sidebarOpen && (
          <>
            {/* Upload Section */}
            <div className="p-4 border-b border-slate-700">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md"
                multiple
                className="hidden"
                onChange={(e) => handleFileUpload(e.target.files)}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg transition-colors"
              >
                <Upload className="w-5 h-5" />
                Upload Documents
              </button>
              <p className="text-xs text-slate-400 mt-2 text-center">
                PDF, TXT, or Markdown files
              </p>
            </div>

            {/* Stats */}
            <div className="p-4 border-b border-slate-700">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-800 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-blue-400">{indexStats.totalChunks}</div>
                  <div className="text-xs text-slate-400">Chunks</div>
                </div>
                <div className="bg-slate-800 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-green-400">{indexStats.documentCount}</div>
                  <div className="text-xs text-slate-400">Documents</div>
                </div>
              </div>
            </div>

            {/* Search Settings */}
            <div className="p-4 border-b border-slate-700">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Search Settings
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="text-sm text-slate-400 block mb-2">Search Mode</label>
                  <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
                    {(['bm25', 'vector', 'hybrid'] as SearchMode[]).map(mode => (
                      <button
                        key={mode}
                        onClick={() => setSearchMode(mode)}
                        className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all ${
                          searchMode === mode
                            ? 'bg-blue-600 text-white'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {mode.charAt(0).toUpperCase() + mode.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm text-slate-400 block mb-2">
                    Sources to retrieve: {topK}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Document List */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Indexed Files
                </h3>
                <button
                  onClick={clearIndex}
                  className="p-2 hover:bg-slate-700 rounded-lg transition-colors text-red-400 hover:text-red-300"
                  title="Clear index"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-2">
                {/* Sample documents */}
                {SAMPLE_DOCS.map((doc, i) => (
                  <div key={i} className="flex items-center gap-3 p-2 bg-slate-800 rounded-lg">
                    <FileText className="w-5 h-5 text-blue-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{doc.name}</div>
                      <div className="text-xs text-slate-400">Sample document</div>
                    </div>
                  </div>
                ))}

                {/* User uploaded documents */}
                {indexedDocs.map((doc, i) => (
                  <div key={i} className="flex items-center gap-3 p-2 bg-slate-800 rounded-lg">
                    <FileText className="w-5 h-5 text-green-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{doc.name}</div>
                      <div className="text-xs text-slate-400">
                        {doc.chunkCount} chunks • {doc.size}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-2 rounded-lg">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800">Enterprise Knowledge Assistant</h1>
              <p className="text-sm text-slate-500">Ask questions about your documents with AI-powered retrieval</p>
            </div>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Welcome message */}
          {messages.length === 0 && (
            <div className="max-w-2xl mx-auto text-center py-12">
              <div className="bg-gradient-to-r from-blue-100 to-indigo-100 rounded-2xl p-8 mb-6">
                <Search className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-slate-800 mb-2">
                  Ask Questions About Your Documents
                </h2>
                <p className="text-slate-600">
                  Upload PDF, TXT, or Markdown files and ask questions. Get answers with source citations.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <Upload className="w-5 h-5 text-blue-600" />
                  </div>
                  <h3 className="font-semibold text-slate-800">1. Upload</h3>
                  <p className="text-slate-500 mt-1">Add your documents</p>
                </div>

                <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                  <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <Book className="w-5 h-5 text-green-600" />
                  </div>
                  <h3 className="font-semibold text-slate-800">2. Index</h3>
                  <p className="text-slate-500 mt-1">Automatic processing</p>
                </div>

                <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <MessageCircle className="w-5 h-5 text-purple-600" />
                  </div>
                  <h3 className="font-semibold text-slate-800">3. Ask</h3>
                  <p className="text-slate-500 mt-1">Get instant answers</p>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map(message => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                  message.type === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white shadow-md border border-slate-200'
                }`}
              >
                <div className="whitespace-pre-wrap">{message.content}</div>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <div className="text-xs font-semibold text-slate-500 mb-1">SOURCES</div>
                    <div className="flex flex-wrap gap-2">
                      {message.sources.map((source, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1 bg-slate-100 text-slate-700 text-xs px-2 py-1 rounded-full"
                        >
                          <FileText className="w-3 h-3" />
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl px-5 py-3 shadow-md border border-slate-200">
                <div className="flex items-center gap-2">
                  <RefreshCw className="w-5 h-5 text-blue-600 animate-spin" />
                  <span className="text-slate-600">Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-slate-200 p-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="flex items-center gap-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question about your documents..."
                  className="w-full px-5 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-800 placeholder-slate-400"
                  disabled={isLoading}
                />
              </div>
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className={`${isLoading || !input.trim() ? 'bg-slate-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white p-3 rounded-xl transition-colors`}
              >
                <Send className="w-5 h-5" />
              </button>
            </div>

            <div className="mt-2 text-center text-xs text-slate-400">
              Uses {searchMode === 'hybrid' ? 'Hybrid (Vector + BM25)' : searchMode === 'vector' ? 'Vector (Semantic)' : 'BM25 (Keyword)'} search • Top {topK} sources
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
