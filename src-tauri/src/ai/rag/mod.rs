//! Multimodal semantic history (RAG) over LanceDB — UC-04, specs §4–5.
//!
//! * [`schema`] — Arrow schemas for the three history tables.
//! * [`store`] — connect/create tables, insert embedded rows, top-k search.
//! * [`pipeline`] — chunk → embed → Guard-gated index, and query → retrieve.

pub mod pipeline;
pub mod schema;
pub mod store;

pub use pipeline::{chunk_text, now_millis, RagPipeline};
pub use schema::{ALL_TABLES, CHAT_HISTORY, IMAGE_HISTORY, WEB_HISTORY};
pub use store::{ChatRow, Hit, ImageRow, RagStore, WebRow};
