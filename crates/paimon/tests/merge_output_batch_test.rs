use std::{collections::HashMap, sync::Arc};

use arrow_array::{Array, Int32Array, Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use paimon::{
    catalog::Identifier,
    io::FileIO,
    spec::{
        DataType as PaimonType, Datum, IntType, PredicateBuilder, Schema as TableSchemaBuilder,
        TableSchema, VarCharType,
    },
    table::Table,
};

const ROWS: i32 = 2050;

fn batch(updated: bool) -> RecordBatch {
    let payload = if updated { "new:" } else { "old:" }.to_owned() + &"x".repeat(8192);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
        Field::new("_VALUE_KIND", DataType::Int8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(0..ROWS)),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|_| payload.as_str()),
            )),
            Arc::new(Int8Array::from_iter_values((0..ROWS).map(|id| {
                if updated && id % 7 == 0 {
                    3
                } else {
                    0
                }
            }))),
        ],
    )
    .unwrap()
}

async fn fixture() -> (tempfile::TempDir, Table) {
    let schema = TableSchema::new(
        0,
        &TableSchemaBuilder::builder()
            .column("id", PaimonType::Int(IntType::new()))
            .column("payload", PaimonType::VarChar(VarCharType::string_type()))
            .primary_key(["id"])
            .option("bucket", "1")
            .option("merge-engine", "deduplicate")
            .option("read.batch-size", "16")
            .build()
            .unwrap(),
    );
    let (directory, table) = create_table(schema);
    for updated in [false, true] {
        write(&table, &batch(updated)).await;
    }
    (directory, table)
}

fn create_table(schema: TableSchema) -> (tempfile::TempDir, Table) {
    let directory = tempfile::tempdir().unwrap();
    let location = directory.path().join("table").to_str().unwrap().to_owned();
    std::fs::create_dir_all(format!("{location}/schema")).unwrap();
    std::fs::create_dir_all(format!("{location}/snapshot")).unwrap();
    std::fs::write(
        format!("{location}/schema/schema-0"),
        serde_json::to_vec(&schema).unwrap(),
    )
    .unwrap();
    let table = Table::new(
        FileIO::from_path(&location).unwrap().build().unwrap(),
        Identifier::new("test", "merge_output"),
        location,
        schema,
        None,
    );
    (directory, table)
}

async fn write(table: &Table, batch: &RecordBatch) {
    let builder = table.new_write_builder();
    let mut writer = builder.new_write().unwrap();
    writer.write_arrow_batch(batch).await.unwrap();
    builder
        .new_commit()
        .commit(writer.prepare_commit().await.unwrap())
        .await
        .unwrap();
}

async fn read(table: &Table) -> Vec<RecordBatch> {
    let builder = table.new_read_builder();
    let plan = builder.new_scan().plan().await.unwrap();
    assert!(plan.splits().iter().any(|split| !split.raw_convertible()));
    builder
        .new_read()
        .unwrap()
        .to_arrow(plan.splits())
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

fn rows(batches: &[RecordBatch]) -> Vec<(i32, String)> {
    batches
        .iter()
        .flat_map(|batch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let payloads = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..batch.num_rows()).map(move |row| (ids.value(row), payloads.value(row).to_owned()))
        })
        .collect()
}

#[tokio::test]
async fn merge_output_size_bounds_wide_batches_without_changing_deduplication() {
    let (_directory, table) = fixture().await;
    let legacy = read(&table).await;
    assert_eq!(legacy[0].num_rows(), 1024);
    let expected = rows(&legacy);
    assert_eq!(
        expected.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        (0..ROWS).filter(|id| id % 7 != 0).collect::<Vec<_>>()
    );
    assert!(expected.iter().all(|(_, value)| value.starts_with("new:")));
    let smaller = table.copy_with_options(HashMap::from([(
        "read.merge-batch-size".into(),
        "128".into(),
    )]));
    let bounded = read(&smaller).await;
    assert_eq!(rows(&bounded), expected);
    assert!(
        bounded.iter().all(|batch| batch.num_rows() <= 128),
        "configured merge output sizes: {:?}",
        bounded
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>()
    );
    let peak = |batches: &[RecordBatch]| {
        batches
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .max()
            .unwrap()
    };
    assert!(peak(&bounded) * 4 < peak(&legacy));
    println!(
        "merge outputs: legacy_peak_bytes={} bounded_peak_bytes={} rows={}",
        peak(&legacy),
        peak(&bounded),
        expected.len()
    );

    for value in ["0", "-1", "invalid", "2147483648"] {
        let invalid = table.copy_with_options(HashMap::from([(
            "read.merge-batch-size".into(),
            value.into(),
        )]));
        let builder = invalid.new_read_builder();
        let plan = builder.new_scan().plan().await.unwrap();
        assert!(builder.new_read().unwrap().to_arrow(plan.splits()).is_err());
    }
}

#[tokio::test]
async fn merge_output_boundary_preserves_materialized_merges_and_residuals() {
    for (engine, second, expected) in [("partial-update", None, 10), ("aggregation", Some(20), 30)]
    {
        let mut schema = TableSchemaBuilder::builder()
            .column("id", PaimonType::Int(IntType::new()))
            .column("value", PaimonType::Int(IntType::new()))
            .primary_key(["id"])
            .option("bucket", "1")
            .option("merge-engine", engine)
            .option("read.batch-size", "2");
        if engine == "aggregation" {
            schema = schema.option("fields.value.aggregate-function", "sum");
        }
        let schema = TableSchema::new(0, &schema.build().unwrap());
        let (_directory, table) = create_table(schema);
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]));
        for value in [Some(10), second] {
            let batch = RecordBatch::try_new(
                arrow_schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..5)),
                    Arc::new(Int32Array::from(vec![value; 5])),
                ],
            )
            .unwrap();
            write(&table, &batch).await;
        }
        for size in [1, 3, 1024] {
            let table = table.copy_with_options(HashMap::from([(
                "read.merge-batch-size".into(),
                size.to_string(),
            )]));
            for (filter_value, expected_rows) in [(expected, 5), (20, 0)] {
                let predicate = PredicateBuilder::new(table.schema().fields())
                    .equal("value", Datum::Int(filter_value))
                    .unwrap();
                let mut builder = table.new_read_builder();
                builder
                    .with_filter(predicate)
                    .with_projection(&["id"])
                    .unwrap();
                let plan = builder.new_scan().plan().await.unwrap();
                let batches: Vec<RecordBatch> = builder
                    .new_read()
                    .unwrap()
                    .to_arrow(plan.splits())
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                assert!(batches.iter().all(|batch| batch.num_rows() <= size));
                let ids: Vec<i32> = batches
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .values()
                            .iter()
                            .copied()
                    })
                    .collect();
                assert_eq!(ids, (0..expected_rows).collect::<Vec<_>>(), "{engine}");
            }
        }
    }
}
