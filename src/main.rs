use std::io::Write;

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tokenizers::Tokenizer;

const EMBED_MODEL: &str = "BAAI/bge-small-en-v1.5";
const CHAT_REPO: &str = "MaziyarPanahi/Qwen3-0.6B-GGUF";
const CHAT_FILE: &str = "Qwen3-0.6B-Q4_K_M.gguf";

struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEngine {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            EMBED_MODEL.to_string(), RepoType::Model, "main".into(),
        ));
        println!("[embed] Baixando tokenizer...");
        let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(E::msg)?;
        println!("[embed] Baixando config...");
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(repo.get("config.json")?)?)?;
        let mut config = config;
        config.hidden_act = HiddenAct::GeluApproximate;
        println!("[embed] Baixando pesos...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[repo.get("model.safetensors")?], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &config)?;
        Ok(Self { model, tokenizer, device })
    }

    fn embed_vec(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;
        let emb = self.model.forward(&token_ids, &type_ids, Some(&mask))?;
        let mask_2d = mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let pooled = (emb.broadcast_mul(&mask_2d)?)
            .sum(1)?
            .broadcast_div(&mask_2d.sum(1)?)?;
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?.squeeze(0)?;
        Ok(normalized.to_vec1()?)
    }
}

struct ChatEngine {
    model: LlamaModel,
    backend: LlamaBackend,
}

impl ChatEngine {
    fn new() -> Result<Self> {
        let backend = LlamaBackend::init()?;
        println!("[chat] Baixando modelo GGUF...");
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            CHAT_REPO.to_string(), RepoType::Model, "main".into(),
        ));
        let model_path = repo.get(CHAT_FILE)?;
        println!("[chat] Carregando modelo...");
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
        println!("[chat] Modelo carregado!");
        Ok(Self { model, backend })
    }

    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let ctx_params = LlamaContextParams::default();
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

        let tokens_list = self.model.str_to_token(prompt, AddBos::Always)?;

        let mut batch = LlamaBatch::new(512, 1);
        let last_index = tokens_list.len() as i32 - 1;
        for (i, token) in (0_i32..).zip(tokens_list.iter().copied()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }
        ctx.decode(&mut batch)?;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler = LlamaSampler::greedy();
        let eos = self.model.token_eos();
        let mut response = String::new();
        let mut n_cur: i32 = batch.n_tokens();

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if token == eos {
                break;
            }
            if let Ok(text) = self.model.token_to_piece(token, &mut decoder, true, None) {
                response.push_str(&text);
                print!("{text}");
                std::io::stdout().flush()?;
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }
        Ok(response)
    }
}

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
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO docs (texto, embedding) VALUES (?1, ?2)",
            rusqlite::params![texto, blob],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    fn search(&self, query_emb: &[f32], k: usize) -> Result<Vec<(i64, String, f64)>> {
        let mut stmt = self.conn.prepare("SELECT id, texto, embedding FROM docs")?;
        let rows: Vec<(i64, String, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, Vec<u8>>(2)?)))?
            .filter_map(|r| r.ok()).collect();

        let mut results: Vec<(i64, String, f64)> = rows.into_iter().map(|(id, texto, blob)| {
            let emb: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let dot: f64 = query_emb.iter().zip(emb.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let na: f64 = query_emb.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = emb.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let dist = if na == 0.0 || nb == 0.0 { 1.0 } else { 1.0 - (dot / (na * nb)) };
            (id, texto, dist)
        }).collect();

        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    fn count(&self) -> Result<usize> {
        let c: i64 = self.conn.query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0))?;
        Ok(c as usize)
    }
}

fn main() -> Result<()> {
    let sep = "=".repeat(55);
    println!("{sep}");
    println!("  STACK ULTRA-LEVE: Rust + candle + llama-cpp + rusqlite");
    println!("  Embedding: {EMBED_MODEL}");
    println!("  Chat: Qwen3-0.6B GGUF Q4_K_M");
    println!("{sep}\n");

    let embed_engine = EmbeddingEngine::new()?;
    println!("[ok] Embedding engine pronto\n");

    let chat_engine = ChatEngine::new()?;
    println!("[ok] Chat engine pronto\n");

    let db = VectorDB::new_in_memory()?;
    println!("[ok] VectorDB pronta\n");

    println!("Inserindo documentos...");
    let docs = vec![
        "O gato e um animal domestico popular",
        "O cachorro e o melhor amigo do homem",
        "Python e uma linguagem de programacao versatil",
        "Machine learning permite que maquinas aprendam",
        "SQLite e um banco de dados leve e portatil",
    ];
    for doc in &docs {
        let emb = embed_engine.embed_vec(doc)?;
        db.insert(doc, &emb)?;
        println!("  + {}", &doc[..doc.len().min(45)]);
    }
    println!("\nTotal: {} documentos inseridos\n", db.count()?);

    println!("{sep}");
    println!("  TESTE DE BUSCA SEMANTICA");
    println!("{sep}");
    let queries = vec![
        "qual animal de estimacao e mais leal?",
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
    println!("  TESTE RAG (CHAT COM CONTEXTO)");
    println!("{sep}");
    let pergunta = "Qual animal e mais leal?";
    println!("\nPergunta: {pergunta}");

    let qemb = embed_engine.embed_vec(pergunta)?;
    let results = db.search(&qemb, 1)?;
    let contexto = if let Some((_, ref texto, _)) = results.first() {
        texto.clone()
    } else {
        String::new()
    };
    println!("Contexto encontrado: {contexto}");

    let prompt = format!(
        "<|im_start|>user\nResponda em portugues com base no contexto.\nContexto: {contexto}\nPergunta: {pergunta}\n<|im_end|>\n<|im_start|>assistant\n"
    );
    println!("\nPrompt enviado ao modelo...\n");
    let resposta = chat_engine.generate(&prompt, 100)?;
    println!("\nResposta: {resposta}");

    println!("\n{sep}");
    println!("  STACK FUNCIONANDO!");
    println!("{sep}");
    Ok(())
}
