use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const EMBED_MODEL: &str = "BAAI/bge-small-en-v1.5";

// ============================================================
// Embedding Engine — BERT via candle
// ============================================================

struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEngine {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        let repo = Repo::with_revision(EMBED_MODEL.to_string(), RepoType::Model, "main".to_string());

        let api = Api::new()?;
        let api = api.repo(repo);

        println!("[embed] Baixando tokenizer...");
        let tokenizer_path = api.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        println!("[embed] Baixando config.json...");
        let config_path = api.get("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&config_str)?;
        config.hidden_act = HiddenAct::GeluApproximate;

        println!("[embed] Baixando pesos (model.safetensors)...");
        let weights_path = api.get("model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };

        let model = BertModel::load(vb, &config)?;

        Ok(Self { model, tokenizer, device })
    }

    fn embed(&self, text: &str) -> Result<Tensor> {
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenize error: {e}")))?;
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling (como sentence-transformers)
        let attention_mask_for_pooling = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = attention_mask_for_pooling.sum(1)?;
        let pooled = (embeddings.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;
        let pooled = pooled.broadcast_div(&sum_mask)?;

        // L2 normalize
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        // Remove batch dimension: [1, hidden] -> [hidden]
        let normalized = normalized.squeeze(0)?;

        Ok(normalized)
    }

    fn embed_vec(&self, text: &str) -> Result<Vec<f32>> {
        let tensor = self.embed(text)?;
        Ok(tensor.to_vec1()?)
    }
}

// ============================================================
// Vector DB — rusqlite (busca cosine manual)
// ============================================================

struct VectorDB {
    conn: rusqlite::Connection,
}

impl VectorDB {
    fn new_in_memory() -> Result<Self> {
        let conn = rusqlite::Connection::open_in_memory()?;
        conn.execute_batch(
            "CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                texto TEXT NOT NULL,
                embedding BLOB NOT NULL
            );",
        )?;
        Ok(Self { conn })
    }

    fn insert(&self, texto: &str, embedding: &[f32]) -> Result<i64> {
        let blob = f32_slice_to_bytes(embedding);
        self.conn.execute(
            "INSERT INTO docs (texto, embedding) VALUES (?1, ?2)",
            rusqlite::params![texto, blob],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    fn search(&self, query_emb: &[f32], k: usize) -> Result<Vec<(i64, String, f64)>> {
        let mut stmt = self.conn.prepare("SELECT id, texto, embedding FROM docs")?;

        let rows: Vec<(i64, String, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut results: Vec<(i64, String, f64)> = rows
            .into_iter()
            .map(|(id, texto, blob)| {
                let emb = bytes_to_f32_slice(&blob);
                let dist = cosine_distance(query_emb, &emb);
                (id, texto, dist)
            })
            .collect();

        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    fn count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

fn f32_slice_to_bytes(slice: &[f32]) -> Vec<u8> {
    slice.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<()> {
    let sep = "=".repeat(55);

    println!("{sep}");
    println!("  STACK ULTRA-LEVE: Rust + candle + rusqlite");
    println!("  Embedding: {EMBED_MODEL}");
    println!("{sep}\n");

    // 1. Embedding engine
    let embed_engine = EmbeddingEngine::new()?;
    println!("[ok] Embedding engine pronto\n");

    // 2. Vector DB
    let db = VectorDB::new_in_memory()?;
    println!("[ok] VectorDB pronta (in-memory)\n");

    // 3. Inserir documentos
    println!("Inserindo documentos...");
    let docs = vec![
        "O gato é um animal doméstico popular",
        "O cachorro é o melhor amigo do homem",
        "Python é uma linguagem de programação versátil",
        "Machine learning permite que máquinas aprendam",
        "SQLite é um banco de dados leve e portátil",
    ];

    for doc in &docs {
        let emb = embed_engine.embed_vec(doc)?;
        db.insert(doc, &emb)?;
        println!("  + {}", &doc[..doc.len().min(45)]);
    }
    println!("\nTotal: {} documentos inseridos\n", db.count()?);

    // 4. Busca semântica
    println!("{sep}");
    println!("  TESTE DE BUSCA SEMÂNTICA");
    println!("{sep}");

    let queries = vec![
        "qual animal de estimação é mais leal?",
        "linguagem para programar",
        "banco de dados leve",
    ];

    for query in &queries {
        println!("\n  Busca: '{query}'");
        let qemb = embed_engine.embed_vec(query)?;
        let results = db.search(&qemb, 2)?;
        for (i, (_id, texto, dist)) in results.iter().enumerate() {
            let display = if texto.len() > 50 { &texto[..50] } else { texto };
            println!("    {}. {}... (dist: {dist:.4})", i + 1, display);
        }
    }

    println!("\n{sep}");
    println!("  STACK FUNCIONANDO! Zero dependência Python.");
    println!("{sep}");

    Ok(())
}
