import { chunkText, Chunk } from './rag';

export interface ParsedDocument {
  name: string;
  content: string;
  page?: number;
  pageCount?: number;
}

// Parse text file
export async function parseTextFile(file: File): Promise<ParsedDocument[]> {
  const content = await file.text();
  return [{
    name: file.name,
    content,
    page: 1
  }];
}

// Parse markdown file
export async function parseMarkdownFile(file: File): Promise<ParsedDocument[]> {
  const content = await file.text();
  return [{
    name: file.name,
    content,
    page: 1
  }];
}

// Parse PDF file using pdf.js
export async function parsePDFFile(file: File): Promise<ParsedDocument[]> {
  try {
    const pdfjsLib = await import('pdfjs-dist');

    // Set worker source
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    const documents: ParsedDocument[] = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const text = textContent.items
        .map((item: any) => item.str)
        .join(' ');

      if (text.trim()) {
        documents.push({
          name: file.name,
          content: text,
          page: i,
          pageCount: pdf.numPages
        });
      }
    }

    return documents;
  } catch (error) {
    console.error('Error parsing PDF:', error);
    throw new Error(`Failed to parse PDF: ${error}`);
  }
}

// Parse document based on type
export async function parseDocument(file: File): Promise<ParsedDocument[]> {
  const extension = file.name.split('.').pop()?.toLowerCase();

  switch (extension) {
    case 'txt':
      return parseTextFile(file);
    case 'md':
      return parseMarkdownFile(file);
    case 'pdf':
      return parsePDFFile(file);
    default:
      throw new Error(`Unsupported file type: ${extension}`);
  }
}

// Process document into chunks
export async function processDocument(file: File): Promise<Chunk[]> {
  const documents = await parseDocument(file);
  const allChunks: Chunk[] = [];

  for (const doc of documents) {
    const chunks = chunkText(
      doc.content,
      doc.name,
      doc.page,
      1000, // chunk size
      200   // overlap
    );
    allChunks.push(...chunks);
  }

  return allChunks;
}

// Get supported file extensions
export function getSupportedExtensions(): string[] {
  return ['.pdf', '.txt', '.md'];
}

// Check if file is supported
export function isFileSupported(file: File): boolean {
  const extension = '.' + file.name.split('.').pop()?.toLowerCase();
  return getSupportedExtensions().includes(extension);
}
