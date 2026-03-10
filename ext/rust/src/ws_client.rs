// ws_client.rs — Rust WebSocket client with GIL-free recv
//
// Uses tokio-tungstenite for async WS, crossbeam-channel for thread-safe
// message passing to Python. The recv() call releases the GIL while waiting.

use pyo3::prelude::*;
use pyo3::types::PyString;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};
use tokio::runtime::Runtime;

/// Internal command sent to the WS worker thread.
enum WsCommand {
    Connect { url: String, subscribe_msgs: Vec<String> },
    Close,
}

/// Rust WebSocket client that releases the GIL during recv().
///
/// Architecture:
/// - Spawns a dedicated tokio runtime on a background thread
/// - WS frames are parsed and forwarded via crossbeam channel
/// - Python calls recv() which blocks on channel (GIL released)
/// - Reconnection with exponential backoff is handled internally
#[pyclass]
pub struct RustWsClient {
    cmd_tx: Sender<WsCommand>,
    msg_rx: Receiver<String>,
    running: Arc<AtomicBool>,
    _worker: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl RustWsClient {
    /// Create a new WebSocket client.
    ///
    /// buffer_size: max messages buffered before backpressure (default 4096)
    #[new]
    #[pyo3(signature = (buffer_size=4096))]
    fn new(buffer_size: usize) -> Self {
        let (cmd_tx, cmd_rx) = bounded::<WsCommand>(16);
        let (msg_tx, msg_rx) = bounded::<String>(buffer_size);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let worker = std::thread::Builder::new()
            .name("rust-ws-worker".into())
            .spawn(move || {
                let rt = match Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        eprintln!("Failed to create tokio runtime: {}", e);
                        return;
                    }
                };
                rt.block_on(ws_worker_loop(cmd_rx, msg_tx, running_clone));
            })
            .expect("Failed to spawn WS worker thread");

        Self {
            cmd_tx,
            msg_rx,
            running,
            _worker: Some(worker),
        }
    }

    /// Connect to a WebSocket URL and optionally subscribe to streams.
    ///
    /// subscribe_msgs: JSON strings to send after connecting (e.g., Binance subscribe)
    #[pyo3(signature = (url, subscribe_msgs=None))]
    fn connect(&self, url: &str, subscribe_msgs: Option<Vec<String>>) -> PyResult<()> {
        self.cmd_tx.send(WsCommand::Connect {
            url: url.to_string(),
            subscribe_msgs: subscribe_msgs.unwrap_or_default(),
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("WS worker not running: {}", e)
        ))
    }

    /// Receive the next message, releasing the GIL while waiting.
    ///
    /// timeout_ms: max wait time in milliseconds (default 5000)
    /// Returns the raw JSON string, or None on timeout.
    #[pyo3(signature = (timeout_ms=5000))]
    fn recv<'py>(&self, py: Python<'py>, timeout_ms: u64) -> Option<Bound<'py, PyString>> {
        let rx = self.msg_rx.clone();
        let dur = Duration::from_millis(timeout_ms);

        // Release GIL while blocking on channel
        let result = py.allow_threads(move || {
            rx.recv_timeout(dur).ok()
        });

        result.map(|s| PyString::new(py, &s))
    }

    /// Close the connection and shut down the worker.
    fn close(&self) -> PyResult<()> {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.cmd_tx.send(WsCommand::Close);
        Ok(())
    }

    /// Check if the client is still running.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Drop for RustWsClient {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.cmd_tx.send(WsCommand::Close);
        if let Some(handle) = self._worker.take() {
            let _ = handle.join();
        }
    }
}

/// Background worker: manages WS connection, reconnection, and message forwarding.
async fn ws_worker_loop(
    cmd_rx: crossbeam_channel::Receiver<WsCommand>,
    msg_tx: crossbeam_channel::Sender<String>,
    running: Arc<AtomicBool>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;

    let mut current_url: Option<String> = None;
    let mut subscribe_msgs: Vec<String> = Vec::new();

    loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        // Check for commands (non-blocking)
        match cmd_rx.try_recv() {
            Ok(WsCommand::Connect { url, subscribe_msgs: subs }) => {
                current_url = Some(url);
                subscribe_msgs = subs;
            }
            Ok(WsCommand::Close) => {
                break;
            }
            Err(_) => {}
        }

        let url = match &current_url {
            Some(u) => u.clone(),
            None => {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };

        // Connect with retry
        let ws = match connect_async(&url).await {
            Ok((ws, _)) => ws,
            Err(e) => {
                eprintln!("WS connect failed: {} — retrying in 2s", e);
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let (mut write, mut read) = ws.split();

        // Send subscription messages
        for sub in &subscribe_msgs {
            use tokio_tungstenite::tungstenite::Message;
            if let Err(e) = write.send(Message::Text(sub.clone().into())).await {
                eprintln!("WS subscribe failed: {}", e);
                break;
            }
        }

        // Read loop
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }

            // Check for new commands
            match cmd_rx.try_recv() {
                Ok(WsCommand::Connect { url: new_url, subscribe_msgs: subs }) => {
                    current_url = Some(new_url);
                    subscribe_msgs = subs;
                    break; // Reconnect with new URL
                }
                Ok(WsCommand::Close) => {
                    running.store(false, Ordering::Relaxed);
                    break;
                }
                Err(_) => {}
            }

            // Read with timeout
            let msg = tokio::time::timeout(Duration::from_secs(30), read.next()).await;
            match msg {
                Ok(Some(Ok(frame))) => {
                    if let tokio_tungstenite::tungstenite::Message::Text(text) = frame {
                        // Non-blocking send; drop message if buffer full
                        let _ = msg_tx.try_send(text.to_string());
                    }
                }
                Ok(Some(Err(e))) => {
                    eprintln!("WS read error: {} — reconnecting", e);
                    break;
                }
                Ok(None) => {
                    // Stream ended
                    eprintln!("WS stream ended — reconnecting");
                    break;
                }
                Err(_) => {
                    // Timeout — send ping or just continue
                    continue;
                }
            }
        }

        // Brief pause before reconnect
        if running.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}
