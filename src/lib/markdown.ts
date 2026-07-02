import { marked } from "marked";
import DOMPurify from "dompurify";

// GitHub-flavoured Markdown, single newlines → <br> (models format loosely).
marked.setOptions({ gfm: true, breaks: true });

/// Render Markdown coming from the model into sanitized HTML for `{@html}`.
/// Sanitization is mandatory: the text is model output and gets injected as
/// raw HTML, so we strip any <script>/event-handler/`javascript:` payload.
/// Parsing runs on every streamed token — marked is sync and fast enough.
export function renderMarkdown(src: string): string {
  const raw = marked.parse(src ?? "", { async: false }) as string;
  return DOMPurify.sanitize(raw);
}
