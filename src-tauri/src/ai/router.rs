//! Hybrid inference router (specs §3.3).
//!
//! Picks a local GPU backend vs. a cloud fallback for each request based on:
//! available hardware, the current tab's consent, and the [`PrivacyGuard`]'s
//! URL classification. Callers depend on the router, not on a concrete engine.
//!
//! **Invariant:** the Guard runs first. A blocked URL or a non-consented tab is
//! rejected with [`LlmError::ConsentDenied`] *before* any engine — local or
//! cloud — is touched. Cloud fallback can therefore never leak sensitive
//! content.

use std::sync::Arc;

use crate::ai::engine::{Embedder, LlmEngine, LlmError, TokenStream};
use crate::ai::privacy::{BlockReason, GuardDecision, PrivacyGuard, TabId};

/// Coarse description of the machine's inference capability. Populated by the
/// hardware wizard (UC-05) later; for now a simple capability flag.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardwareProfile {
    /// Whether a usable local GPU/CPU backend is loaded and able to infer.
    pub local_inference_capable: bool,
}

/// Which backend a request was routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Local,
    Cloud,
}

/// Routes generation/embedding requests through the Privacy Guard to a backend.
pub struct InferenceRouter {
    guard: PrivacyGuard,
    local: Option<Arc<dyn LlmEngine>>,
    cloud: Option<Arc<dyn LlmEngine>>,
    embedder: Arc<dyn Embedder>,
    hardware: HardwareProfile,
    /// User policy: may we ever fall back to cloud? Defaults to true, but cloud
    /// still only fires for Guard-approved requests.
    allow_cloud_fallback: bool,
}

impl InferenceRouter {
    pub fn new(guard: PrivacyGuard, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            guard,
            local: None,
            cloud: None,
            embedder,
            hardware: HardwareProfile::default(),
            allow_cloud_fallback: true,
        }
    }

    pub fn with_local(mut self, engine: Arc<dyn LlmEngine>) -> Self {
        self.local = Some(engine);
        self
    }

    pub fn with_cloud(mut self, engine: Arc<dyn LlmEngine>) -> Self {
        self.cloud = Some(engine);
        self
    }

    pub fn with_hardware(mut self, hardware: HardwareProfile) -> Self {
        self.hardware = hardware;
        self
    }

    pub fn set_cloud_fallback(&mut self, allowed: bool) {
        self.allow_cloud_fallback = allowed;
    }

    pub fn guard(&self) -> &PrivacyGuard {
        &self.guard
    }

    pub fn guard_mut(&mut self) -> &mut PrivacyGuard {
        &mut self.guard
    }

    /// Decide which backend to use, independent of the Guard. Prefers local
    /// when the hardware can run it and a local engine is registered;
    /// otherwise falls back to cloud if allowed and registered.
    pub fn select_route(&self) -> Option<Route> {
        if self.hardware.local_inference_capable && self.local.is_some() {
            return Some(Route::Local);
        }
        if self.allow_cloud_fallback && self.cloud.is_some() {
            return Some(Route::Cloud);
        }
        // Last resort: a local engine exists even if hardware wasn't flagged
        // capable (lets tests/headless runs exercise the path).
        if self.local.is_some() {
            return Some(Route::Local);
        }
        None
    }

    fn engine_for(&self, route: Route) -> Option<&Arc<dyn LlmEngine>> {
        match route {
            Route::Local => self.local.as_ref(),
            Route::Cloud => self.cloud.as_ref(),
        }
    }

    /// Map a Guard block into an [`LlmError::ConsentDenied`].
    fn check_guard(&self, tab: TabId, url: &str) -> Result<(), LlmError> {
        match self.guard.evaluate(tab, url) {
            GuardDecision::Allow => Ok(()),
            GuardDecision::Block { reason } => Err(LlmError::ConsentDenied(describe(reason))),
        }
    }

    /// Streaming text generation, gated by the Privacy Guard.
    pub async fn generate_text(
        &self,
        tab: TabId,
        url: &str,
        prompt: &str,
        context: &str,
    ) -> Result<TokenStream, LlmError> {
        self.check_guard(tab, url)?;
        let route = self.select_route().ok_or(LlmError::HardwareUnavailable)?;
        let engine = self
            .engine_for(route)
            .ok_or(LlmError::HardwareUnavailable)?;
        engine.generate_text(prompt, context).await
    }

    /// Streaming image analysis, gated by the Privacy Guard.
    pub async fn analyze_image(
        &self,
        tab: TabId,
        url: &str,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<TokenStream, LlmError> {
        self.check_guard(tab, url)?;
        let route = self.select_route().ok_or(LlmError::HardwareUnavailable)?;
        let engine = self
            .engine_for(route)
            .ok_or(LlmError::HardwareUnavailable)?;
        engine.analyze_image(image_bytes, prompt).await
    }

    /// Embed text for RAG indexing. CPU-only and never routed to cloud, but
    /// still gated by the Guard (embedding happens before storage).
    pub async fn embed(&self, tab: TabId, url: &str, text: &str) -> Result<Vec<f32>, LlmError> {
        self.check_guard(tab, url)?;
        self.embedder.embed(text).await
    }

    pub fn embedder_version(&self) -> &str {
        self.embedder.model_version()
    }
}

fn describe(reason: BlockReason) -> String {
    match reason {
        BlockReason::SensitiveUrl(cat) => format!("sensitive URL ({cat:?})"),
        BlockReason::NoTabConsent => "tab has not granted AI consent".to_string(),
        BlockReason::Unparseable => "URL could not be parsed".to_string(),
        BlockReason::UserBlocked => "URL is on the user blocklist".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::stream;

    /// Minimal engine that echoes which backend produced the text.
    struct MockEngine {
        id: &'static str,
    }

    #[async_trait]
    impl LlmEngine for MockEngine {
        async fn generate_text(&self, prompt: &str, _ctx: &str) -> Result<TokenStream, LlmError> {
            let out = format!("[{}] {}", self.id, prompt);
            Ok(Box::pin(stream::once(async move { Ok(out) })))
        }
        async fn analyze_image(&self, _b: &[u8], prompt: &str) -> Result<TokenStream, LlmError> {
            let out = format!("[{}] img:{}", self.id, prompt);
            Ok(Box::pin(stream::once(async move { Ok(out) })))
        }
        fn backend_id(&self) -> &str {
            self.id
        }
    }

    struct MockEmbedder;
    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, LlmError> {
            Ok(vec![0.0; crate::ai::engine::EMBEDDING_DIM])
        }
        fn model_version(&self) -> &str {
            "mock-v0"
        }
    }

    fn router_with(local: bool, cloud: bool, hw_capable: bool) -> InferenceRouter {
        let mut guard = PrivacyGuard::default();
        guard.grant_consent(1);
        let mut r = InferenceRouter::new(guard, Arc::new(MockEmbedder))
            .with_hardware(HardwareProfile {
                local_inference_capable: hw_capable,
            });
        if local {
            r = r.with_local(Arc::new(MockEngine { id: "local" }));
        }
        if cloud {
            r = r.with_cloud(Arc::new(MockEngine { id: "cloud" }));
        }
        r
    }

    async fn collect(stream: TokenStream) -> String {
        use futures::StreamExt;
        stream
            .filter_map(|r| async move { r.ok() })
            .collect::<Vec<_>>()
            .await
            .concat()
    }

    /// Extract the error from a streaming result without requiring the Ok
    /// (`TokenStream`) type to be `Debug`.
    fn err_of(res: Result<TokenStream, LlmError>) -> LlmError {
        match res {
            Ok(_) => panic!("expected an error, got a stream"),
            Err(e) => e,
        }
    }

    #[tokio::test]
    async fn blocked_url_never_reaches_an_engine() {
        let r = router_with(true, true, true);
        let err = err_of(
            r.generate_text(1, "https://www.fineco.it/login", "hi", "")
                .await,
        );
        assert!(matches!(err, LlmError::ConsentDenied(_)));
    }

    #[tokio::test]
    async fn non_consented_tab_is_denied() {
        let r = router_with(true, true, true); // consent granted only to tab 1
        let err = err_of(r.generate_text(99, "https://example.com/x", "hi", "").await);
        assert!(matches!(err, LlmError::ConsentDenied(_)));
    }

    #[tokio::test]
    async fn prefers_local_when_hardware_capable() {
        let r = router_with(true, true, true);
        assert_eq!(r.select_route(), Some(Route::Local));
        let out = collect(
            r.generate_text(1, "https://example.com/x", "hello", "")
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out, "[local] hello");
    }

    #[tokio::test]
    async fn falls_back_to_cloud_without_local_hardware() {
        let r = router_with(false, true, false);
        assert_eq!(r.select_route(), Some(Route::Cloud));
        let out = collect(
            r.generate_text(1, "https://example.com/x", "hello", "")
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out, "[cloud] hello");
    }

    #[tokio::test]
    async fn cloud_disabled_means_no_route_without_local() {
        let mut r = router_with(false, true, false);
        r.set_cloud_fallback(false);
        assert_eq!(r.select_route(), None);
        let err = err_of(r.generate_text(1, "https://example.com/x", "hi", "").await);
        assert!(matches!(err, LlmError::HardwareUnavailable));
    }

    #[tokio::test]
    async fn embed_is_guarded_and_correct_dim() {
        let r = router_with(true, false, true);
        let v = r.embed(1, "https://example.com/x", "text").await.unwrap();
        assert_eq!(v.len(), crate::ai::engine::EMBEDDING_DIM);
        // blocked url → no embedding
        assert!(r
            .embed(1, "http://localhost/x", "text")
            .await
            .is_err());
    }
}
