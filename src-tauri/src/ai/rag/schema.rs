//! Arrow schemas for the three LanceDB tables (specs §5).
//!
//! Each table carries a `VECTOR(384)` column (`FixedSizeList<Float32, 384>`)
//! and an `embedding_model_version` field so rows can be re-indexed when the
//! embedding model changes (see [`crate::ai::embedder::MiniLmVariant`]).
//!
//! Timestamps are stored as epoch-milliseconds `Int64` (simple range filters,
//! no timezone handling); the spec's `TIMESTAMP` is illustrative.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::ai::engine::EMBEDDING_DIM;

/// Table of clean page text extracted from the DOM.
pub const WEB_HISTORY: &str = "web_history";
/// Table of AI-generated descriptions of analyzed images.
pub const IMAGE_HISTORY: &str = "image_history";
/// Table of embedded prompt+response chunks.
pub const CHAT_HISTORY: &str = "chat_history";

/// All table names, for iteration when ensuring the database is initialized.
pub const ALL_TABLES: [&str; 3] = [WEB_HISTORY, IMAGE_HISTORY, CHAT_HISTORY];

/// The inner element field of the vector column. Its name/nullability must match
/// between the schema and any [`arrow_array::FixedSizeListArray`] we build.
pub fn vector_item_field() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Float32, true))
}

/// The `VECTOR(384)` column shared by every table.
pub fn vector_field() -> Field {
    Field::new(
        "vector",
        DataType::FixedSizeList(vector_item_field(), EMBEDDING_DIM as i32),
        true,
    )
}

fn text(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

/// `web_history`: id, vector, text_content, url, embedding_model_version, timestamp.
pub fn web_history_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        text("id"),
        vector_field(),
        text("text_content"),
        text("url"),
        text("embedding_model_version"),
        Field::new("timestamp", DataType::Int64, false),
    ]))
}

/// `image_history`: id, vector, description, source_url, thumbnail_path, version, timestamp.
pub fn image_history_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        text("id"),
        vector_field(),
        text("description"),
        text("source_url"),
        text("thumbnail_path"),
        text("embedding_model_version"),
        Field::new("timestamp", DataType::Int64, false),
    ]))
}

/// `chat_history`: id, vector, conversation_chunk, context_url, version, timestamp.
pub fn chat_history_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        text("id"),
        vector_field(),
        text("conversation_chunk"),
        text("context_url"),
        text("embedding_model_version"),
        Field::new("timestamp", DataType::Int64, false),
    ]))
}

/// The schema for a given table name, or `None` if unknown.
pub fn schema_for(table: &str) -> Option<SchemaRef> {
    match table {
        WEB_HISTORY => Some(web_history_schema()),
        IMAGE_HISTORY => Some(image_history_schema()),
        CHAT_HISTORY => Some(chat_history_schema()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_have_a_384_vector_column() {
        for table in ALL_TABLES {
            let schema = schema_for(table).unwrap();
            let field = schema.field_with_name("vector").unwrap();
            match field.data_type() {
                DataType::FixedSizeList(_, dim) => assert_eq!(*dim, 384),
                other => panic!("{table}.vector is {other:?}, expected FixedSizeList"),
            }
            assert!(schema.field_with_name("embedding_model_version").is_ok());
            assert!(schema.field_with_name("id").is_ok());
        }
    }
}
