//! RAG indexing/retrieval pipeline (UC-04).
//!
//! Ties the [`Embedder`] and [`RagStore`] together behind the [`PrivacyGuard`]:
//! page text, image descriptions, and chats are chunked, embedded, and indexed
//! **only** when the tab has consent and the URL passes the Guard (specs §6).
//! Retrieval embeds a natural-language query and returns the nearest rows.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ai::engine::{Embedder, LlmError};
use crate::ai::privacy::{GuardDecision, PrivacyGuard, TabId};
use crate::ai::rag::store::{ChatRow, Hit, ImageRow, RagStore, WebRow};

/// Epoch-milliseconds now, for stamping indexed rows.
pub fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Indexing + retrieval over the LanceDB history tables.
pub struct RagPipeline {
    store: RagStore,
    embedder: Arc<dyn Embedder>,
    chunk_words: usize,
    overlap: usize,
}

impl RagPipeline {
    /// Default chunking: ~180 words per chunk with 30-word overlap.
    pub fn new(store: RagStore, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            store,
            embedder,
            chunk_words: 180,
            overlap: 30,
        }
    }

    pub fn with_chunking(mut self, chunk_words: usize, overlap: usize) -> Self {
        assert!(chunk_words > overlap, "chunk_words must exceed overlap");
        self.chunk_words = chunk_words;
        self.overlap = overlap;
        self
    }

    /// Borrow the underlying store (e.g. for direct queries).
    pub fn store(&self) -> &RagStore {
        &self.store
    }

    fn check(guard: &PrivacyGuard, tab: TabId, url: &str) -> Result<(), LlmError> {
        match guard.evaluate(tab, url) {
            GuardDecision::Allow => Ok(()),
            GuardDecision::Block { reason } => {
                Err(LlmError::ConsentDenied(format!("{reason:?}")))
            }
        }
    }

    /// Index a page's clean text into `web_history`. Returns the chunk count.
    /// Errors with [`LlmError::ConsentDenied`] if the Guard blocks the URL.
    pub async fn index_page(
        &self,
        guard: &PrivacyGuard,
        tab: TabId,
        url: &str,
        text: &str,
        timestamp: i64,
    ) -> Result<usize, LlmError> {
        Self::check(guard, tab, url)?;
        let version = self.embedder.model_version().to_string();
        let chunks = chunk_text(text, self.chunk_words, self.overlap);

        let mut rows = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let vector = self.embedder.embed(chunk).await?;
            rows.push(WebRow {
                id: format!("{url}::{timestamp}::{i}"),
                vector,
                text_content: chunk.clone(),
                url: url.to_string(),
                embedding_model_version: version.clone(),
                timestamp,
            });
        }
        self.store.insert_web(&rows).await?;
        Ok(rows.len())
    }

    /// Index an AI-generated image description into `image_history`.
    pub async fn index_image(
        &self,
        guard: &PrivacyGuard,
        tab: TabId,
        source_url: &str,
        description: &str,
        thumbnail_path: &str,
        timestamp: i64,
    ) -> Result<(), LlmError> {
        Self::check(guard, tab, source_url)?;
        let vector = self.embedder.embed(description).await?;
        let row = ImageRow {
            id: format!("{source_url}::{timestamp}"),
            vector,
            description: description.to_string(),
            source_url: source_url.to_string(),
            thumbnail_path: thumbnail_path.to_string(),
            embedding_model_version: self.embedder.model_version().to_string(),
            timestamp,
        };
        self.store.insert_image(&[row]).await
    }

    /// Index a chat exchange (prompt+response) into `chat_history`.
    pub async fn index_chat(
        &self,
        guard: &PrivacyGuard,
        tab: TabId,
        context_url: &str,
        conversation_chunk: &str,
        timestamp: i64,
    ) -> Result<(), LlmError> {
        Self::check(guard, tab, context_url)?;
        let vector = self.embedder.embed(conversation_chunk).await?;
        let row = ChatRow {
            id: format!("{context_url}::{timestamp}"),
            vector,
            conversation_chunk: conversation_chunk.to_string(),
            context_url: context_url.to_string(),
            embedding_model_version: self.embedder.model_version().to_string(),
            timestamp,
        };
        self.store.insert_chat(&[row]).await
    }

    /// Retrieve the top-k page chunks semantically closest to `query`.
    ///
    /// The query is user-typed in the command bar (an explicit action over
    /// already-consented content), so it is not itself Guard-gated.
    pub async fn retrieve_pages(&self, query: &str, k: usize) -> Result<Vec<Hit>, LlmError> {
        let vector = self.embedder.embed(query).await?;
        self.store.search_web(vector, k).await
    }

    pub async fn retrieve_images(&self, query: &str, k: usize) -> Result<Vec<Hit>, LlmError> {
        let vector = self.embedder.embed(query).await?;
        self.store.search_image(vector, k).await
    }

    pub async fn retrieve_chats(&self, query: &str, k: usize) -> Result<Vec<Hit>, LlmError> {
        let vector = self.embedder.embed(query).await?;
        self.store.search_chat(vector, k).await
    }
}

/// Split text into overlapping word-windows. Returns one chunk for short text;
/// empty/whitespace-only text yields no chunks.
pub fn chunk_text(text: &str, chunk_words: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    if words.len() <= chunk_words {
        return vec![words.join(" ")];
    }
    let step = chunk_words - overlap;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_words).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ai::embedder::{default_model_dir, MiniLmEmbedder, MiniLmVariant};
    use crate::ai::privacy::PrivacyGuard;

    const TAB: crate::ai::privacy::TabId = 7;

    /// Build a pipeline backed by an in-memory store and the real embedder,
    /// or `None` if the model isn't downloaded.
    async fn pipeline() -> Option<RagPipeline> {
        let dir = default_model_dir();
        if !dir.join(MiniLmVariant::DEFAULT.filename()).exists() {
            return None;
        }
        let embedder = Arc::new(MiniLmEmbedder::from_dir(&dir).unwrap());
        let store = RagStore::open("memory://").await.unwrap();
        Some(RagPipeline::new(store, embedder))
    }

    #[tokio::test]
    async fn index_then_retrieve_finds_semantically_closest_page() {
        let Some(rag) = pipeline().await else { return };
        let mut guard = PrivacyGuard::default();
        guard.grant_consent(TAB);

        rag.index_page(&guard, TAB, "https://pets.example.com/cats",
            "Kittens are young domestic cats. They love to play, chase toys, and nap in warm spots.",
            now_millis()).await.unwrap();
        rag.index_page(&guard, TAB, "https://econ.example.com/rates",
            "The central bank raised the benchmark interest rate to curb inflation this quarter.",
            now_millis()).await.unwrap();

        let hits = rag.retrieve_pages("a small playful cat", 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].url, "https://pets.example.com/cats");
    }

    #[tokio::test]
    async fn guard_blocks_indexing_of_sensitive_url() {
        let Some(rag) = pipeline().await else { return };
        let mut guard = PrivacyGuard::default();
        guard.grant_consent(TAB);

        // Financial host: blocked even with consent → nothing indexed.
        let blocked = rag
            .index_page(&guard, TAB, "https://www.fineco.it/account", "balance details", now_millis())
            .await;
        assert!(matches!(blocked, Err(LlmError::ConsentDenied(_))));
        assert!(rag.retrieve_pages("balance", 5).await.unwrap().is_empty());

        // Non-consented tab is also refused.
        let no_consent = rag
            .index_page(&guard, 999, "https://news.example.com/a", "some article text", now_millis())
            .await;
        assert!(matches!(no_consent, Err(LlmError::ConsentDenied(_))));
    }

    #[test]
    fn chunking_handles_short_and_long_text() {
        assert_eq!(chunk_text("", 10, 2), Vec::<String>::new());
        assert_eq!(chunk_text("a b c", 10, 2), vec!["a b c".to_string()]);

        let text: String = (0..100)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_text(&text, 30, 5);
        assert!(chunks.len() > 1);
        // First chunk is 30 words; consecutive chunks overlap by 5.
        assert_eq!(chunks[0].split_whitespace().count(), 30);
        let first_last5: Vec<&str> = chunks[0].split_whitespace().rev().take(5).collect();
        let second_first5: Vec<&str> = chunks[1].split_whitespace().take(5).collect();
        let first_last5: Vec<&str> = first_last5.into_iter().rev().collect();
        assert_eq!(first_last5, second_first5);
    }
}
