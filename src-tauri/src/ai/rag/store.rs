//! The LanceDB-backed RAG store: connect, ensure tables exist, insert embedded
//! rows, and run top-k nearest-neighbour search (specs §4–5, UC-04).

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};

use crate::ai::engine::{LlmError, EMBEDDING_DIM};
use crate::ai::rag::schema::{
    self, vector_item_field, ALL_TABLES, CHAT_HISTORY, IMAGE_HISTORY, WEB_HISTORY,
};

/// A row destined for `web_history`.
#[derive(Debug, Clone)]
pub struct WebRow {
    pub id: String,
    pub vector: Vec<f32>,
    pub text_content: String,
    pub url: String,
    pub embedding_model_version: String,
    pub timestamp: i64,
}

/// A row destined for `image_history`.
#[derive(Debug, Clone)]
pub struct ImageRow {
    pub id: String,
    pub vector: Vec<f32>,
    pub description: String,
    pub source_url: String,
    pub thumbnail_path: String,
    pub embedding_model_version: String,
    pub timestamp: i64,
}

/// A row destined for `chat_history`.
#[derive(Debug, Clone)]
pub struct ChatRow {
    pub id: String,
    pub vector: Vec<f32>,
    pub conversation_chunk: String,
    pub context_url: String,
    pub embedding_model_version: String,
    pub timestamp: i64,
}

/// One retrieval result. `text`/`url` map to the table's primary text and URL
/// columns; `distance` is LanceDB's `_distance` (smaller = closer).
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: String,
    pub text: String,
    pub url: String,
    pub distance: f32,
}

/// Handle to the embedded vector database holding the three history tables.
pub struct RagStore {
    conn: Connection,
}

impl RagStore {
    /// Open (creating if needed) the database at `uri` and ensure all three
    /// tables exist with the correct schema. `uri` is a directory path; use
    /// `"memory://"` for an ephemeral in-memory store (tests).
    pub async fn open(uri: &str) -> Result<Self, LlmError> {
        let conn = lancedb::connect(uri).execute().await.map_err(map_lance)?;
        let store = Self { conn };
        store.ensure_tables().await?;
        Ok(store)
    }

    async fn ensure_tables(&self) -> Result<(), LlmError> {
        let existing = self
            .conn
            .table_names()
            .execute()
            .await
            .map_err(map_lance)?;
        for table in ALL_TABLES {
            if !existing.iter().any(|t| t == table) {
                let schema = schema::schema_for(table).expect("known table");
                self.conn
                    .create_empty_table(table, schema)
                    .execute()
                    .await
                    .map_err(map_lance)?;
            }
        }
        Ok(())
    }

    async fn table(&self, name: &str) -> Result<Table, LlmError> {
        self.conn
            .open_table(name)
            .execute()
            .await
            .map_err(map_lance)
    }

    pub async fn insert_web(&self, rows: &[WebRow]) -> Result<(), LlmError> {
        if rows.is_empty() {
            return Ok(());
        }
        let vectors = build_vector_column(rows.iter().map(|r| r.vector.as_slice()))?;
        let batch = RecordBatch::try_new(
            schema::web_history_schema(),
            vec![
                str_col(rows.iter().map(|r| r.id.as_str())),
                vectors,
                str_col(rows.iter().map(|r| r.text_content.as_str())),
                str_col(rows.iter().map(|r| r.url.as_str())),
                str_col(rows.iter().map(|r| r.embedding_model_version.as_str())),
                i64_col(rows.iter().map(|r| r.timestamp)),
            ],
        )
        .map_err(map_arrow)?;
        self.append(WEB_HISTORY, batch).await
    }

    pub async fn insert_image(&self, rows: &[ImageRow]) -> Result<(), LlmError> {
        if rows.is_empty() {
            return Ok(());
        }
        let vectors = build_vector_column(rows.iter().map(|r| r.vector.as_slice()))?;
        let batch = RecordBatch::try_new(
            schema::image_history_schema(),
            vec![
                str_col(rows.iter().map(|r| r.id.as_str())),
                vectors,
                str_col(rows.iter().map(|r| r.description.as_str())),
                str_col(rows.iter().map(|r| r.source_url.as_str())),
                str_col(rows.iter().map(|r| r.thumbnail_path.as_str())),
                str_col(rows.iter().map(|r| r.embedding_model_version.as_str())),
                i64_col(rows.iter().map(|r| r.timestamp)),
            ],
        )
        .map_err(map_arrow)?;
        self.append(IMAGE_HISTORY, batch).await
    }

    pub async fn insert_chat(&self, rows: &[ChatRow]) -> Result<(), LlmError> {
        if rows.is_empty() {
            return Ok(());
        }
        let vectors = build_vector_column(rows.iter().map(|r| r.vector.as_slice()))?;
        let batch = RecordBatch::try_new(
            schema::chat_history_schema(),
            vec![
                str_col(rows.iter().map(|r| r.id.as_str())),
                vectors,
                str_col(rows.iter().map(|r| r.conversation_chunk.as_str())),
                str_col(rows.iter().map(|r| r.context_url.as_str())),
                str_col(rows.iter().map(|r| r.embedding_model_version.as_str())),
                i64_col(rows.iter().map(|r| r.timestamp)),
            ],
        )
        .map_err(map_arrow)?;
        self.append(CHAT_HISTORY, batch).await
    }

    async fn append(&self, table: &str, batch: RecordBatch) -> Result<(), LlmError> {
        let tbl = self.table(table).await?;
        tbl.add(batch).execute().await.map_err(map_lance)?;
        Ok(())
    }

    /// Top-k nearest-neighbour search in `web_history`.
    pub async fn search_web(&self, query: Vec<f32>, k: usize) -> Result<Vec<Hit>, LlmError> {
        self.search(WEB_HISTORY, query, k, "text_content", "url").await
    }

    /// Top-k nearest-neighbour search in `image_history`.
    pub async fn search_image(&self, query: Vec<f32>, k: usize) -> Result<Vec<Hit>, LlmError> {
        self.search(IMAGE_HISTORY, query, k, "description", "source_url")
            .await
    }

    /// Top-k nearest-neighbour search in `chat_history`.
    pub async fn search_chat(&self, query: Vec<f32>, k: usize) -> Result<Vec<Hit>, LlmError> {
        self.search(CHAT_HISTORY, query, k, "conversation_chunk", "context_url")
            .await
    }

    async fn search(
        &self,
        table: &str,
        query: Vec<f32>,
        k: usize,
        text_col: &str,
        url_col: &str,
    ) -> Result<Vec<Hit>, LlmError> {
        let tbl = self.table(table).await?;
        let batches: Vec<RecordBatch> = tbl
            .vector_search(query)
            .map_err(map_lance)?
            .limit(k)
            .execute()
            .await
            .map_err(map_lance)?
            .try_collect()
            .await
            .map_err(map_lance)?;

        let mut hits = Vec::new();
        for batch in &batches {
            let ids = string_column(batch, "id")?;
            let texts = string_column(batch, text_col)?;
            let urls = string_column(batch, url_col)?;
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            for i in 0..batch.num_rows() {
                hits.push(Hit {
                    id: ids.value(i).to_string(),
                    text: texts.value(i).to_string(),
                    url: urls.value(i).to_string(),
                    distance: distances.map(|d| d.value(i)).unwrap_or(f32::NAN),
                });
            }
        }
        Ok(hits)
    }
}

/// Build the `FixedSizeList<Float32, 384>` vector column from per-row slices.
fn build_vector_column<'a>(
    vectors: impl Iterator<Item = &'a [f32]>,
) -> Result<ArrayRef, LlmError> {
    let mut flat: Vec<f32> = Vec::new();
    let mut rows = 0usize;
    for v in vectors {
        if v.len() != EMBEDDING_DIM {
            return Err(LlmError::InferenceFailed(format!(
                "vector has {} dims, expected {EMBEDDING_DIM}",
                v.len()
            )));
        }
        flat.extend_from_slice(v);
        rows += 1;
    }
    let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
    let array =
        FixedSizeListArray::try_new(vector_item_field(), EMBEDDING_DIM as i32, values, None)
            .map_err(map_arrow)?;
    debug_assert_eq!(array.len(), rows);
    Ok(Arc::new(array))
}

fn str_col<'a>(values: impl Iterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values))
}

fn i64_col(values: impl Iterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(values))
}

fn string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray, LlmError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| LlmError::InferenceFailed(format!("column {name} missing or not Utf8")))
}

fn map_lance(e: lancedb::Error) -> LlmError {
    LlmError::InferenceFailed(format!("lancedb: {e}"))
}

fn map_arrow(e: arrow_schema::ArrowError) -> LlmError {
    LlmError::InferenceFailed(format!("arrow: {e}"))
}
