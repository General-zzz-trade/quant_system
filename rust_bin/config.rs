// config.rs — YAML configuration for quant_trader binary
//
// Supports two model config styles:
//   1. Directory: model_path points to a directory containing config.json + .json model files
//   2. Explicit: model_path is a direct .json file path, with optional ensemble_paths
//
// Credentials support both direct values and env var references (api_key_env).

use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub trading: TradingConfig,
    #[serde(default)]
    pub strategy: StrategyConfig,
    #[serde(default)]
    pub risk: RiskConfig,
    #[serde(default)]
    pub position_sizing: PositionSizingConfig,
    pub credentials: CredentialsConfig,
    #[serde(default)]
    pub models: HashMap<String, SymbolModelConfig>,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub micro_alpha: MicroAlphaYamlConfig,
}

#[derive(Debug, Deserialize)]
pub struct MicroAlphaYamlConfig {
    /// Enable tick-by-tick aggTrade-driven decisions (default: false)
    #[serde(default)]
    pub tick_by_tick: bool,
    /// Minimum milliseconds between orders per symbol (default: 200)
    #[serde(default = "default_min_order_interval_ms")]
    pub min_order_interval_ms: u64,
    /// Micro signal threshold for tick-by-tick orders (default: 0.4)
    #[serde(default = "default_micro_threshold")]
    pub micro_threshold: f64,
    /// Rolling window in milliseconds (default: 10000)
    #[serde(default = "default_micro_window_ms")]
    pub window_ms: i64,
    /// Large trade multiplier (default: 5.0)
    #[serde(default = "default_large_trade_mult")]
    pub large_trade_mult: f64,
}

impl Default for MicroAlphaYamlConfig {
    fn default() -> Self {
        Self {
            tick_by_tick: false,
            min_order_interval_ms: 200,
            micro_threshold: 0.4,
            window_ms: 10_000,
            large_trade_mult: 5.0,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TradingConfig {
    pub symbols: Vec<String>,
    #[serde(default = "default_exchange")]
    pub exchange: String,
    #[serde(default)]
    pub testnet: bool,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_balance")]
    pub starting_balance: f64,
    #[serde(default = "default_currency")]
    pub currency: String,
    /// Kline interval: "1s", "1m", "3m", "5m", "15m", "1h" etc.
    #[serde(default = "default_interval")]
    pub interval: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct StrategyConfig {
    #[serde(default = "default_deadzone")]
    pub deadzone: f64,
    #[serde(default)]
    pub min_hold: i32,
    #[serde(default)]
    pub long_only: bool,
    #[serde(default)]
    pub trend_follow: bool,
    #[serde(default)]
    pub trend_threshold: f64,
    #[serde(default = "default_trend_indicator")]
    pub trend_indicator: String,
    #[serde(default = "default_max_hold")]
    pub max_hold: i32,
    #[serde(default)]
    pub monthly_gate: bool,
    #[serde(default = "default_monthly_gate_window")]
    pub monthly_gate_window: usize,
    #[serde(default)]
    pub vol_target: Option<f64>,
    #[serde(default = "default_vol_feature")]
    pub vol_feature: String,
    #[serde(default)]
    pub bear_thresholds: Vec<(f64, f64)>,
    /// Per-symbol strategy overrides
    #[serde(default)]
    pub per_symbol: HashMap<String, SymbolStrategyOverride>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct SymbolStrategyOverride {
    pub deadzone: Option<f64>,
    pub min_hold: Option<i32>,
    pub long_only: Option<bool>,
    pub trend_follow: Option<bool>,
    pub monthly_gate: Option<bool>,
    pub monthly_gate_window: Option<usize>,
    pub vol_target: Option<f64>,
    pub vol_feature: Option<String>,
    pub bear_thresholds: Option<Vec<(f64, f64)>>,
}

#[derive(Debug, Deserialize, Default)]
pub struct RiskConfig {
    #[serde(default = "default_max_leverage")]
    pub max_leverage: f64,
    #[serde(default = "default_max_drawdown")]
    pub max_drawdown_pct: f64,
    pub max_position_notional: Option<f64>,
    pub max_daily_orders: Option<u64>,
    pub max_order_notional: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct PositionSizingConfig {
    #[serde(default = "default_fraction")]
    pub fraction: f64,
    #[serde(default)]
    pub lot_sizes: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
pub struct CredentialsConfig {
    /// Direct API key value
    #[serde(default)]
    pub api_key: Option<String>,
    /// Direct API secret value
    #[serde(default)]
    pub api_secret: Option<String>,
    /// Env var name for API key (e.g. "BINANCE_TESTNET_API_KEY")
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Env var name for API secret
    #[serde(default)]
    pub api_secret_env: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SymbolModelConfig {
    /// Path to model directory (containing config.json) or direct .json file
    pub model_path: String,
    /// Optional additional model paths for ensemble (explicit mode)
    #[serde(default)]
    pub ensemble_paths: Vec<String>,
    /// Ensemble weights (explicit mode)
    #[serde(default)]
    pub ensemble_weights: Vec<f64>,
    /// Bear model path (directory or .json file)
    pub bear_model_path: Option<String>,
    /// Short model path (directory or .json file)
    pub short_model_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub json: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json: false,
        }
    }
}

// Defaults
fn default_exchange() -> String { "binance".to_string() }
fn default_mode() -> String { "paper".to_string() }
fn default_balance() -> f64 { 10000.0 }
fn default_currency() -> String { "USDT".to_string() }
fn default_deadzone() -> f64 { 0.5 }
fn default_trend_indicator() -> String { "tf4h_close_vs_ma20".to_string() }
fn default_max_hold() -> i32 { 120 }
fn default_monthly_gate_window() -> usize { 480 }
fn default_vol_feature() -> String { "atr_norm_14".to_string() }
fn default_interval() -> String { "1m".to_string() }
fn default_fraction() -> f64 { 0.02 }
fn default_max_leverage() -> f64 { 5.0 }
fn default_max_drawdown() -> f64 { 0.30 }
fn default_log_level() -> String { "info".to_string() }
fn default_min_order_interval_ms() -> u64 { 200 }
fn default_micro_threshold() -> f64 { 0.4 }
fn default_micro_window_ms() -> i64 { 10_000 }
fn default_large_trade_mult() -> f64 { 5.0 }

/// Resolved model paths + strategy overrides discovered from model config.json.
pub struct ResolvedModelConfig {
    pub json_paths: Vec<String>,
    pub ensemble_weights: Option<Vec<f64>>,
    pub bear_model_path: Option<String>,
    pub short_model_path: Option<String>,
    /// Strategy overrides from the model's config.json
    pub strategy_override: SymbolStrategyOverride,
}

impl Config {
    /// Load config from YAML file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config {}: {}", path.display(), e))?;
        let config: Config = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse config {}: {}", path.display(), e))?;
        Ok(config)
    }

    /// Resolve API key from direct value or env var.
    pub fn resolve_api_key(&self) -> Result<String, String> {
        if let Some(ref key) = self.credentials.api_key {
            return Ok(key.clone());
        }
        if let Some(ref env_name) = self.credentials.api_key_env {
            return std::env::var(env_name)
                .map_err(|_| format!("Env var {} not set", env_name));
        }
        Err("No api_key or api_key_env in credentials".to_string())
    }

    /// Resolve API secret from direct value or env var.
    pub fn resolve_api_secret(&self) -> Result<String, String> {
        if let Some(ref secret) = self.credentials.api_secret {
            return Ok(secret.clone());
        }
        if let Some(ref env_name) = self.credentials.api_secret_env {
            return std::env::var(env_name)
                .map_err(|_| format!("Env var {} not set", env_name));
        }
        Err("No api_secret or api_secret_env in credentials".to_string())
    }

    /// Discover model files for a symbol.
    ///
    /// If model_path is a directory containing config.json, auto-discovers
    /// model .json files, ensemble weights, bear/short models, and strategy params.
    /// If model_path is a direct .json file, uses explicit paths from YAML.
    pub fn resolve_models_for(&self, symbol: &str) -> Result<ResolvedModelConfig, String> {
        let m = self.models.get(symbol)
            .ok_or_else(|| format!("No model config for {}", symbol))?;

        let model_path = PathBuf::from(&m.model_path);

        if model_path.is_dir() {
            // Directory mode: auto-discover from config.json
            self.discover_from_directory(symbol, &model_path, m)
        } else if model_path.extension().map(|e| e == "json").unwrap_or(false) {
            // Explicit .json file mode
            self.resolve_explicit(m)
        } else {
            Err(format!("model_path '{}' is neither a directory nor a .json file", m.model_path))
        }
    }

    fn discover_from_directory(
        &self,
        symbol: &str,
        model_dir: &Path,
        sym_cfg: &SymbolModelConfig,
    ) -> Result<ResolvedModelConfig, String> {
        let config_path = model_dir.join("config.json");
        let content = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;
        let mcfg: JsonValue = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", config_path.display(), e))?;

        // Discover model .json files
        let models = mcfg.get("models")
            .and_then(|m| m.as_array())
            .ok_or_else(|| format!("No 'models' array in {}", config_path.display()))?;

        let mut json_paths = Vec::new();
        for fname in models {
            let fname = fname.as_str()
                .ok_or_else(|| "Model filename is not a string".to_string())?;
            let json_name = fname.replace(".pkl", ".json");
            let json_path = model_dir.join(&json_name);
            if !json_path.exists() {
                return Err(format!("Model file not found: {}", json_path.display()));
            }
            json_paths.push(json_path.to_string_lossy().to_string());
        }

        if json_paths.is_empty() {
            return Err(format!("No model files found in {}", model_dir.display()));
        }

        // Ensemble weights
        let ensemble_weights = mcfg.get("ensemble_weights")
            .and_then(|w| w.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>())
            .filter(|w| w.len() == json_paths.len());

        // Bear model
        let bear_model_path = self.discover_bear_model(&mcfg, model_dir, sym_cfg)?;

        // Short model
        let short_model_path = self.discover_short_model(symbol, model_dir)?;

        // Strategy overrides from config.json
        let pos_mgmt = mcfg.get("position_management");
        let strategy_override = SymbolStrategyOverride {
            deadzone: mcfg.get("deadzone").and_then(|v| v.as_f64()),
            min_hold: mcfg.get("min_hold").and_then(|v| v.as_i64()).map(|v| v as i32),
            long_only: mcfg.get("long_only").and_then(|v| v.as_bool()),
            trend_follow: None,
            monthly_gate: mcfg.get("monthly_gate").and_then(|v| v.as_bool()),
            monthly_gate_window: mcfg.get("monthly_gate_window").and_then(|v| v.as_u64()).map(|v| v as usize),
            vol_target: pos_mgmt.and_then(|p| p.get("vol_target")).and_then(|v| v.as_f64()),
            vol_feature: pos_mgmt.and_then(|p| p.get("vol_feature")).and_then(|v| v.as_str()).map(|s| s.to_string()),
            bear_thresholds: pos_mgmt
                .and_then(|p| p.get("bear_thresholds"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            let a = item.as_array()?;
                            Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?))
                        })
                        .collect()
                }),
        };

        Ok(ResolvedModelConfig {
            json_paths,
            ensemble_weights,
            bear_model_path,
            short_model_path,
            strategy_override,
        })
    }

    fn discover_bear_model(
        &self,
        mcfg: &JsonValue,
        model_dir: &Path,
        sym_cfg: &SymbolModelConfig,
    ) -> Result<Option<String>, String> {
        // Check YAML override first
        if let Some(ref bp) = sym_cfg.bear_model_path {
            let p = PathBuf::from(bp);
            if p.extension().map(|e| e == "json").unwrap_or(false) && p.exists() {
                return Ok(Some(bp.clone()));
            }
        }

        // Auto-discover from config.json's bear_model_path
        let bear_dir_str = mcfg.get("bear_model_path").and_then(|v| v.as_str());
        if bear_dir_str.is_none() {
            // Try convention: {symbol}_bear_c in parent directory
            let parent = model_dir.parent().unwrap_or(model_dir);
            let dir_name = model_dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // e.g. BTCUSDT_gate_v2 -> BTCUSDT_bear_c
            if let Some(sym_prefix) = dir_name.split('_').next() {
                let bear_dir = parent.join(format!("{}_bear_c", sym_prefix));
                if bear_dir.exists() {
                    return self.resolve_first_model_json(&bear_dir);
                }
            }
            return Ok(None);
        }

        let bear_dir = PathBuf::from(bear_dir_str.unwrap());
        if !bear_dir.exists() {
            return Ok(None);
        }
        self.resolve_first_model_json(&bear_dir)
    }

    fn discover_short_model(
        &self,
        symbol: &str,
        model_dir: &Path,
    ) -> Result<Option<String>, String> {
        let parent = model_dir.parent().unwrap_or(model_dir);
        let short_dir = parent.join(format!("{}_short", symbol));
        if !short_dir.exists() {
            return Ok(None);
        }
        self.resolve_first_model_json(&short_dir)
    }

    /// Load config.json from a model directory and return path to first .json model.
    fn resolve_first_model_json(&self, dir: &Path) -> Result<Option<String>, String> {
        let cfg_path = dir.join("config.json");
        if !cfg_path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&cfg_path)
            .map_err(|e| format!("Failed to read {}: {}", cfg_path.display(), e))?;
        let cfg: JsonValue = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", cfg_path.display(), e))?;

        let models = cfg.get("models").and_then(|m| m.as_array());
        if let Some(models) = models {
            for fname in models {
                if let Some(fname) = fname.as_str() {
                    let json_name = fname.replace(".pkl", ".json");
                    let json_path = dir.join(&json_name);
                    if json_path.exists() {
                        return Ok(Some(json_path.to_string_lossy().to_string()));
                    }
                }
            }
        }
        Ok(None)
    }

    fn resolve_explicit(&self, m: &SymbolModelConfig) -> Result<ResolvedModelConfig, String> {
        let mut paths = vec![m.model_path.clone()];
        paths.extend(m.ensemble_paths.iter().cloned());
        let weights = if m.ensemble_weights.is_empty() {
            None
        } else {
            Some(m.ensemble_weights.clone())
        };
        Ok(ResolvedModelConfig {
            json_paths: paths,
            ensemble_weights: weights,
            bear_model_path: m.bear_model_path.clone(),
            short_model_path: m.short_model_path.clone(),
            strategy_override: SymbolStrategyOverride::default(),
        })
    }
}
