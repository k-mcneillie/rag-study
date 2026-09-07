# Database cheat sheet

A practical reference for working with the project's MariaDB database. It
reflects the SQLAlchemy models in `src/rag/storage/orm.py` and the repository
in `src/rag/storage/mariadb_repository.py`.

## Requirements

MariaDB **11.7 or later**, for the native `VECTOR` column type and vector
indexes. Verified against 12.3.3. An earlier server raises a syntax error on
`VECTOR`.

```bash
mariadb -e "SELECT VERSION();"      # must be >= 11.7
```

## Database and user setup

Two databases: the runtime schema and a disposable schema for the integration
test suite, which creates and drops its own tables.

```sql
CREATE DATABASE rag_study      CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE rag_study_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

The application uses a **least-privilege account**. It needs only to read and
write rows; it is deliberately not allowed to change the schema, so a bug or an
injected statement cannot alter it.

```sql
CREATE USER 'rag_app'@'localhost' IDENTIFIED BY 'choose-a-strong-password';
GRANT SELECT, INSERT, UPDATE, DELETE ON rag_study.* TO 'rag_app'@'localhost';

-- The test database is disposable.
GRANT ALL PRIVILEGES ON rag_study_test.* TO 'rag_app'@'localhost';
FLUSH PRIVILEGES;
```

`CREATE command denied to user 'rag_app'` during normal operation is the
security control working, not a misconfiguration. Schema creation is a separate,
privileged operation — see [Schema creation](#schema-creation).

## Connection configuration

The application reads these from the environment (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DB_HOST` | — (required) | host name or address |
| `RAG_DB_PORT` | `3306` | port |
| `RAG_DB_USER` | — (required) | least-privilege application user |
| `RAG_DB_PASSWORD` | — (required) | its password |
| `RAG_DB_NAME` | — (required) | runtime schema |
| `RAG_TEST_DB_NAME` | — | disposable schema for integration tests |

`RAG_ADMIN_DB_URL` is separate: a full SQLAlchemy URL for a privileged account,
used only by `scripts/create_schema.py`, read from the environment so admin
credentials do not enter shell history.

### SQLAlchemy connection

`src/rag/storage/engine.py` builds the URL and engine:

```python
URL.create(
    drivername="mysql+pymysql",
    username=settings.user,
    password=settings.password,
    host=settings.host,
    port=settings.port,
    database=settings.database,
    query={"charset": "utf8mb4"},
)

create_engine(url, echo=False, pool_pre_ping=True, future=True)
sessionmaker(bind=engine, expire_on_commit=False)
```

`URL.create` escapes credentials and masks the password in its `repr`.
`echo=False` is deliberate: echoed statements would place document text and
query parameters into the logs. `pool_pre_ping=True` discards a stale
connection before it is used. The driver is PyMySQL, a pure-Python driver, so
there is no MariaDB Connector/C build dependency.

## Schema

Four tables. All use `InnoDB`, `utf8mb4`, `utf8mb4_unicode_ci`. Primary keys
are generated UUID4 strings (`CHAR(36)`), never derived from a filename.

### `documents`

| Column | Type | Notes |
|---|---|---|
| `id` | `CHAR(36)` | primary key |
| `source_filename` | `VARCHAR(512)` | untrusted, display only |
| `content_hash` | `VARCHAR(64)` | SHA-256 of the file bytes; indexed |
| `metadata` | `JSON` | untrusted PDF metadata (four whitelisted keys) |
| `created_at` | `DATETIME` | `server_default = NOW()` |

### `pages`

| Column | Type | Notes |
|---|---|---|
| `id` | `CHAR(36)` | primary key |
| `document_id` | `CHAR(36)` | FK → `documents.id`, `ON DELETE CASCADE` |
| `page_number` | `INT` | one-based |
| `text` | `TEXT` | cleaned page text |
| `ocr_extracted` | `BOOLEAN` | pipeline-set |
| `metadata` | `JSON` | untrusted |

Unique index `uq_pages_document_page` on `(document_id, page_number)`.

### `chunks`

| Column | Type | Notes |
|---|---|---|
| `id` | `CHAR(36)` | primary key |
| `document_id` | `CHAR(36)` | FK → `documents.id`, `ON DELETE CASCADE`; indexed |
| `page_number` | `INT` | |
| `section` | `VARCHAR(512)` | nullable; heading path, e.g. `2 Methods > 2.1 Sampling` |
| `headers` | `JSON` | the heading trail |
| `text` | `TEXT` | untrusted document content |
| `ocr_extracted` | `BOOLEAN` | pipeline-set |
| `metadata` | `JSON` | untrusted; also carries the pipeline `section_type` tag |
| `created_at` | `DATETIME` | `server_default = NOW()` |

### `embeddings`

| Column | Type | Notes |
|---|---|---|
| `id` | `CHAR(36)` | primary key |
| `chunk_id` | `CHAR(36)` | FK → `chunks.id`, `ON DELETE CASCADE`; indexed |
| `embedding` | `VECTOR(384)` | `NOT NULL`; the ORM attribute is `vector` |
| `model_name` | `VARCHAR(255)` | indexed; e.g. `all-MiniLM-L6-v2` |
| `model_dimension` | `INT` | guards against a silent model change |
| `created_at` | `DATETIME` | `server_default = NOW()` |

Vector index, added by a DDL event listener after the table is created:

```sql
ALTER TABLE embeddings
  ADD VECTOR INDEX ix_embeddings_embedding (embedding) DISTANCE=cosine;
```

### Naming quirks

- The vector **column** is `embedding`, not `vector`: `vector` is a reserved
  word in MariaDB 11.7+ that SQLAlchemy's MySQL dialect does not quote. The ORM
  **attribute** is `vector`.
- The metadata **column** is `metadata`; the ORM **attribute** is
  `extra_metadata`, because `metadata` is reserved on the declarative base.
- Bulk-insert dictionaries use ORM attribute names (`vector`, `extra_metadata`),
  not column names.

## Schema creation

The application user cannot create tables. Use an administrative account:

```bash
RAG_ADMIN_DB_URL='mysql+pymysql://root@localhost/rag_study?unix_socket=/tmp/mysql.sock' \
    python scripts/create_schema.py
```

The script:

- checks `RAG_EMBEDDING_DIMENSION`, if set, against
  `rag.storage.orm.EMBEDDING_DIMENSION` (384). The `VECTOR` width is fixed when
  the ORM class is defined, so a mismatch would build the column at the wrong
  width; the script refuses and names the constant to edit.
- runs `Base.metadata.create_all`, which only ever **adds** missing tables.
- reports **drift**: columns the models declare that the live database lacks
  (`CREATE TABLE` cannot add them to an existing table), printing the `ALTER
  TABLE ... ADD COLUMN` statements needed. It reads column metadata from
  `information_schema.columns` because SQLAlchemy's MySQL reflection cannot
  parse a `VECTOR` column and raises while trying.

On success: `Schema present. Tables: chunks, documents, embeddings, pages`.

There are no migrations. See [future-work.md](future-work.md) §4.

## Vector storage and similarity

Vectors cross the wire in MariaDB's binary format. The `Vector` SQLAlchemy type
(`src/rag/storage/vector.py`) handles conversion inside bound parameters:

- **write**: `func.VEC_FromText(:param)` where `:param` is `json.dumps([...])`;
- **read**: `func.VEC_ToText(embedding)`, parsed with `json.loads`.

No vector value is ever interpolated into SQL, and decoding never uses `eval`
or `pickle`.

Similarity search (`MariaDBRepository._build_search_statement`) is, in effect:

```sql
SELECT c.*, VEC_DISTANCE_COSINE(e.embedding, VEC_FromText(:query)) AS distance
FROM chunks c
JOIN embeddings e ON e.chunk_id = c.id
[ WHERE c.document_id = :document_id ]
ORDER BY distance
LIMIT :top_k;
```

`VEC_DISTANCE_COSINE` returns a distance where smaller is closer. The
application converts it to a relevance score with `score = 1.0 - distance`.
Only the top-k rows return to the application. `top_k` is validated to be
between 1 and `max_top_k` (default 100).

## Common operations

All go through the repository; the SQL below is illustrative.

Check for a stored document by content:

```python
repository.document_exists(content_hash)
# SELECT id FROM documents WHERE content_hash = :h LIMIT 1
```

Store a document atomically (`save_ingested_document`): one transaction that
deletes any prior copy — matched by `id` **or** `content_hash` — then
bulk-inserts the document, pages, chunks, and embeddings. Vector widths are
validated before the transaction opens.

Retrieve candidates: `similarity_search(query_vector, top_k, filters=...)`, as
above.

Manual inspection:

```sql
SELECT COUNT(*) FROM documents;
SELECT id, source_filename, created_at FROM documents ORDER BY created_at DESC;
SELECT COUNT(*) FROM chunks WHERE document_id = 'UUID';
SELECT model_name, COUNT(*) FROM embeddings GROUP BY model_name;
```

Delete a document and everything under it (cascades through the foreign keys):

```sql
-- Destructive. Removes the document's pages, chunks, and embeddings.
DELETE FROM documents WHERE id = 'UUID';
```

## Indexes and performance

Indexed columns: `documents.content_hash`, `chunks.document_id`,
`embeddings.chunk_id`, `embeddings.model_name`, and the cosine `VECTOR INDEX`
on `embeddings.embedding`. The vector index is what keeps similarity search
returning only `top_k` rows rather than scanning every embedding.

```sql
SHOW INDEX FROM embeddings;
EXPLAIN SELECT c.id
        FROM chunks c JOIN embeddings e ON e.chunk_id = c.id
        ORDER BY VEC_DISTANCE_COSINE(e.embedding, VEC_FromText('[...]'))
        LIMIT 5;
```

There is no connection-pool tuning beyond `pool_pre_ping`, and no batching
across documents. This is adequate at the current scale.

## Backups

```bash
mariadb-dump --single-transaction rag_study > rag_study-$(date +%F).sql
```

The corpus itself is not in the database; back up `docs/pool/` separately if it
matters.

## Development versus production

- The integration suite targets `RAG_TEST_DB_NAME` and drops and recreates the
  schema around each test (`tests/conftest.py`). Never point it at the runtime
  database.
- Integration tests skip, rather than fail, when MariaDB is unreachable.
- Production differences are minimal: the same schema, the same least-privilege
  user, `pool_pre_ping` on. Pool sizing and concurrency are unexamined.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error in your SQL syntax ... VECTOR` | MariaDB older than 11.7 |
| `CREATE command denied to user 'rag_app'` | Working as intended; run `scripts/create_schema.py` with `RAG_ADMIN_DB_URL` set |
| `Unknown column ...` during retrieval | The database predates a model change; run `scripts/create_schema.py` and apply the printed `ALTER`, or drop and re-ingest |
| Reflection raises on `embeddings` | Expected; read column metadata from `information_schema` instead |
| Dimension-mismatch error on ingest or retrieval | `RAG_EMBEDDING_DIMENSION`, the `VECTOR` column width, and the model's output disagree; see [model-deployment-cheatsheet.md](model-deployment-cheatsheet.md) |

## Do not

- Build SQL by string interpolation. Use SQLAlchemy expressions; values are
  bound.
- Put credentials in source. They come from the environment.
- Give the application account root or DDL rights.
- Run `DROP` / `DELETE` / `TRUNCATE` against the runtime schema without a
  backup and a clear reason.
- Add a user-controlled `ORDER BY` or column name. Filterable fields are an
  explicit allow-list (`ALLOWED_SEARCH_FILTERS`).
