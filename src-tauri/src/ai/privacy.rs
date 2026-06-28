//! Privacy Guard (specs §6) — the security gate that runs **before** any
//! embedding, local inference, or cloud call.
//!
//! Two independent protections combine into a single [`PrivacyGuard::evaluate`]
//! decision:
//!
//! 1. **URL classification.** Every URL is sorted into a [`UrlCategory`].
//!    Sensitive categories (financial sites, checkout/payment flows, local
//!    addresses) are excluded from indexing and inference regardless of
//!    consent. We use *categories + maintained lists + user overrides* rather
//!    than a bare regex blocklist.
//! 2. **Per-tab consent.** AI access is granted per browsing tab (privacy-safe
//!    default: no consent). A global "disable the Guard at your own risk"
//!    override bypasses everything.
//!
//! Cloud fallback in particular must never fire for a [`GuardDecision::Block`]
//! — callers check the decision first and only then pick a backend.

use std::collections::HashMap;
use std::net::IpAddr;

use url::{Host, Url};

/// Identifier for a browsing tab. Consent is tracked per tab.
pub type TabId = u64;

/// Why the Privacy Guard classified a URL the way it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlCategory {
    /// Financial institutions, banking, payment processors.
    Financial,
    /// Checkout / cart / payment pages (sensitive regardless of host).
    Checkout,
    /// Loopback, private-network, or `.local`/`.localhost` addresses.
    LocalAddress,
    /// Other explicitly sensitive hosts (health, government, adult, …).
    Sensitive,
    /// Nothing sensitive matched — eligible for AI given tab consent.
    Allowed,
}

impl UrlCategory {
    /// Whether this category, on its own, blocks AI access.
    pub fn is_blocking(self) -> bool {
        !matches!(self, UrlCategory::Allowed)
    }
}

/// Outcome of evaluating a (tab, url) pair against the Guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardDecision {
    /// AI may process this content (local and, if otherwise permitted, cloud).
    Allow,
    /// AI is forbidden. `reason` is safe to log/show to the user.
    Block { reason: BlockReason },
}

impl GuardDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, GuardDecision::Allow)
    }
}

/// Precise reason a request was blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockReason {
    /// The URL fell into a sensitive [`UrlCategory`].
    SensitiveUrl(UrlCategory),
    /// The tab has not granted AI consent (privacy-safe default).
    NoTabConsent,
    /// The URL could not be parsed; fail closed.
    Unparseable,
    /// User-maintained blocklist matched.
    UserBlocked,
}

/// Classifies URLs into [`UrlCategory`] using maintained keyword lists plus
/// user-supplied allow/block overrides.
#[derive(Debug, Clone)]
pub struct UrlClassifier {
    /// Host substrings indicating financial services.
    financial_keywords: Vec<String>,
    /// Host substrings indicating other sensitive categories.
    sensitive_keywords: Vec<String>,
    /// Path substrings indicating a checkout/payment flow.
    checkout_path_markers: Vec<String>,
    /// User override: hosts always treated as [`UrlCategory::Allowed`].
    user_allowlist: Vec<String>,
    /// User override: hosts always blocked ([`BlockReason::UserBlocked`]).
    user_blocklist: Vec<String>,
}

impl Default for UrlClassifier {
    fn default() -> Self {
        Self {
            financial_keywords: [
                "bank", "fineco", "paypal", "stripe", "revolut", "n26", "intesa",
                "unicredit", "binance", "coinbase", "wallet", "creditcard",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            sensitive_keywords: [
                // health, government, identity, adult — illustrative seed list
                "health", "patient", "gov", "irs", "agenziaentrate", "login",
                "account", "signin",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            checkout_path_markers: [
                "/checkout", "/cart", "/payment", "/pay", "/billing", "/order",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            user_allowlist: Vec::new(),
            user_blocklist: Vec::new(),
        }
    }
}

impl UrlClassifier {
    /// Add a host (substring) the user always wants treated as allowed.
    pub fn allow_host(&mut self, host: impl Into<String>) {
        self.user_allowlist.push(host.into().to_lowercase());
    }

    /// Add a host (substring) the user always wants blocked.
    pub fn block_host(&mut self, host: impl Into<String>) {
        self.user_blocklist.push(host.into().to_lowercase());
    }

    /// Classify a parsed URL. The user blocklist wins over everything; the
    /// allowlist wins over the maintained keyword lists (but not over the
    /// blocklist).
    fn classify(&self, url: &Url) -> Result<UrlClassification, ()> {
        let host = url.host().ok_or(())?;

        // Local / private addresses — independent of keyword lists.
        if is_local_host(&host) {
            return Ok(UrlClassification::Category(UrlCategory::LocalAddress));
        }

        let host_str = host.to_string().to_lowercase();
        let path = url.path().to_lowercase();

        if self.user_blocklist.iter().any(|h| host_str.contains(h)) {
            return Ok(UrlClassification::UserBlocked);
        }
        let allowlisted = self.user_allowlist.iter().any(|h| host_str.contains(h));

        // Checkout flows are sensitive even on otherwise-allowed hosts and are
        // not waved through by a host allowlist entry.
        if self
            .checkout_path_markers
            .iter()
            .any(|m| path.contains(m.as_str()))
        {
            return Ok(UrlClassification::Category(UrlCategory::Checkout));
        }

        if allowlisted {
            return Ok(UrlClassification::Category(UrlCategory::Allowed));
        }

        if self.financial_keywords.iter().any(|k| host_str.contains(k)) {
            return Ok(UrlClassification::Category(UrlCategory::Financial));
        }
        if self.sensitive_keywords.iter().any(|k| host_str.contains(k)) {
            return Ok(UrlClassification::Category(UrlCategory::Sensitive));
        }

        Ok(UrlClassification::Category(UrlCategory::Allowed))
    }
}

/// Internal result of classification, distinguishing a category from an
/// explicit user blocklist hit.
enum UrlClassification {
    Category(UrlCategory),
    UserBlocked,
}

/// Returns true for loopback, private-range, `.local`, or `.localhost` hosts.
fn is_local_host(host: &Host<&str>) -> bool {
    match host {
        Host::Ipv4(ip) => IpAddr::V4(*ip).is_loopback() || ip.is_private(),
        Host::Ipv6(ip) => IpAddr::V6(*ip).is_loopback(),
        Host::Domain(name) => {
            let name = name.to_lowercase();
            name == "localhost"
                || name.ends_with(".localhost")
                || name.ends_with(".local")
        }
    }
}

/// The Privacy Guard: classifier + per-tab consent + global override.
#[derive(Debug, Clone)]
pub struct PrivacyGuard {
    classifier: UrlClassifier,
    /// Tabs that have explicitly granted AI consent.
    consented_tabs: HashMap<TabId, bool>,
    /// When true the Guard is disabled entirely ("at your own risk", specs §6).
    globally_disabled: bool,
}

impl Default for PrivacyGuard {
    fn default() -> Self {
        Self {
            classifier: UrlClassifier::default(),
            consented_tabs: HashMap::new(),
            globally_disabled: false,
        }
    }
}

impl PrivacyGuard {
    pub fn new(classifier: UrlClassifier) -> Self {
        Self {
            classifier,
            ..Self::default()
        }
    }

    /// Mutable access to the classifier (to add user allow/block hosts).
    pub fn classifier_mut(&mut self) -> &mut UrlClassifier {
        &mut self.classifier
    }

    /// Grant AI consent for a tab.
    pub fn grant_consent(&mut self, tab: TabId) {
        self.consented_tabs.insert(tab, true);
    }

    /// Revoke AI consent for a tab.
    pub fn revoke_consent(&mut self, tab: TabId) {
        self.consented_tabs.insert(tab, false);
    }

    /// Drop all state for a closed tab.
    pub fn forget_tab(&mut self, tab: TabId) {
        self.consented_tabs.remove(&tab);
    }

    /// Whether a tab currently has consent (default: false).
    pub fn has_consent(&self, tab: TabId) -> bool {
        self.consented_tabs.get(&tab).copied().unwrap_or(false)
    }

    /// Globally disable the Guard. Documented as user-risk; bypasses both URL
    /// classification and per-tab consent.
    pub fn set_globally_disabled(&mut self, disabled: bool) {
        self.globally_disabled = disabled;
    }

    pub fn is_globally_disabled(&self) -> bool {
        self.globally_disabled
    }

    /// The single gate. Call **before** embedding, local inference, or any
    /// cloud request.
    pub fn evaluate(&self, tab: TabId, raw_url: &str) -> GuardDecision {
        if self.globally_disabled {
            return GuardDecision::Allow;
        }

        let url = match Url::parse(raw_url) {
            Ok(u) => u,
            Err(_) => {
                return GuardDecision::Block {
                    reason: BlockReason::Unparseable,
                }
            }
        };

        match self.classifier.classify(&url) {
            Err(()) => GuardDecision::Block {
                reason: BlockReason::Unparseable,
            },
            Ok(UrlClassification::UserBlocked) => GuardDecision::Block {
                reason: BlockReason::UserBlocked,
            },
            Ok(UrlClassification::Category(cat)) if cat.is_blocking() => GuardDecision::Block {
                reason: BlockReason::SensitiveUrl(cat),
            },
            // URL is fine — now require tab consent.
            Ok(UrlClassification::Category(_)) => {
                if self.has_consent(tab) {
                    GuardDecision::Allow
                } else {
                    GuardDecision::Block {
                        reason: BlockReason::NoTabConsent,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAB: TabId = 1;

    fn guard_with_consent() -> PrivacyGuard {
        let mut g = PrivacyGuard::default();
        g.grant_consent(TAB);
        g
    }

    #[test]
    fn default_tab_has_no_consent() {
        let g = PrivacyGuard::default();
        assert!(!g.has_consent(TAB));
        let d = g.evaluate(TAB, "https://example.com/article");
        assert_eq!(
            d,
            GuardDecision::Block {
                reason: BlockReason::NoTabConsent
            }
        );
    }

    #[test]
    fn consented_normal_page_is_allowed() {
        let g = guard_with_consent();
        assert!(g.evaluate(TAB, "https://en.wikipedia.org/wiki/Rust").is_allowed());
    }

    #[test]
    fn financial_host_blocked_even_with_consent() {
        let g = guard_with_consent();
        let d = g.evaluate(TAB, "https://www.fineco.it/home");
        assert_eq!(
            d,
            GuardDecision::Block {
                reason: BlockReason::SensitiveUrl(UrlCategory::Financial)
            }
        );
    }

    #[test]
    fn checkout_path_blocked_on_normal_host() {
        let g = guard_with_consent();
        let d = g.evaluate(TAB, "https://shop.example.com/checkout/step2");
        assert_eq!(
            d,
            GuardDecision::Block {
                reason: BlockReason::SensitiveUrl(UrlCategory::Checkout)
            }
        );
    }

    #[test]
    fn localhost_and_loopback_blocked() {
        let g = guard_with_consent();
        for url in [
            "http://localhost:3000/app",
            "http://127.0.0.1/admin",
            "http://192.168.1.10/router",
            "http://dev.local/x",
        ] {
            assert_eq!(
                g.evaluate(TAB, url),
                GuardDecision::Block {
                    reason: BlockReason::SensitiveUrl(UrlCategory::LocalAddress)
                },
                "expected {url} to be a blocked local address"
            );
        }
    }

    #[test]
    fn unparseable_url_fails_closed() {
        let g = guard_with_consent();
        assert_eq!(
            g.evaluate(TAB, "not a url"),
            GuardDecision::Block {
                reason: BlockReason::Unparseable
            }
        );
    }

    #[test]
    fn global_override_bypasses_everything() {
        let mut g = PrivacyGuard::default(); // no consent at all
        g.set_globally_disabled(true);
        assert!(g.evaluate(TAB, "https://www.fineco.it/login").is_allowed());
        assert!(g.evaluate(TAB, "http://localhost/x").is_allowed());
    }

    #[test]
    fn user_blocklist_overrides_allowed_host() {
        let mut g = guard_with_consent();
        g.classifier_mut().block_host("example.com");
        assert_eq!(
            g.evaluate(TAB, "https://www.example.com/page"),
            GuardDecision::Block {
                reason: BlockReason::UserBlocked
            }
        );
    }

    #[test]
    fn user_allowlist_clears_keyword_match_but_not_checkout() {
        let mut g = guard_with_consent();
        // "account" is a sensitive keyword; user trusts this host.
        g.classifier_mut().allow_host("myaccount.example.com");
        assert!(g
            .evaluate(TAB, "https://myaccount.example.com/profile")
            .is_allowed());
        // …but a checkout path on the same trusted host is still blocked.
        assert_eq!(
            g.evaluate(TAB, "https://myaccount.example.com/checkout"),
            GuardDecision::Block {
                reason: BlockReason::SensitiveUrl(UrlCategory::Checkout)
            }
        );
    }

    #[test]
    fn revoking_consent_blocks_again() {
        let mut g = guard_with_consent();
        assert!(g.evaluate(TAB, "https://example.com/a").is_allowed());
        g.revoke_consent(TAB);
        assert_eq!(
            g.evaluate(TAB, "https://example.com/a"),
            GuardDecision::Block {
                reason: BlockReason::NoTabConsent
            }
        );
    }
}
