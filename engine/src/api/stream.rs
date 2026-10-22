//! Streaming response bodies that are lossless or dead. Chunks go to the body through a small
//! bounded queue; a client that takes nothing for the stall timeout, an error on the engine side,
//! or an injected harness failure aborts the response and drops the connection, never ends it
//! cleanly, so a reader can tell a cut stream from a finished one. Only a stream that really
//! ended gets Arrow's end-of-stream marker.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use tokio::sync::{Notify, mpsc};

/// Chunks the body holds before the producer waits for the client.
const QUEUE: usize = 8;

/// Drops the connection a request came in on: the accept loop puts one in every request's
/// extensions and stops serving the connection once it fires.
#[derive(Debug, Clone, Default)]
pub struct Kill(Arc<Notify>);

impl Kill {
    pub fn kill(&self) {
        self.0.notify_one();
    }

    /// Resolves once [`Self::kill`] has been called, even before this was awaited.
    pub async fn killed(&self) {
        self.0.notified().await;
    }
}

/// Why a stream stopped early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    /// The client hung up.
    Gone,
    /// The client took nothing for the stall timeout; the connection was dropped.
    Stalled,
    /// The harness asked for a failure here; the connection was dropped.
    Injected,
}

/// The producer's end of a streaming body.
#[derive(Debug)]
pub struct Chunks {
    tx: mpsc::Sender<io::Result<Vec<u8>>>,
    kill: Option<Kill>,
    stall: Duration,
    /// Harness only: fail once this many chunks after the first have gone out.
    fail_after: Option<u64>,
    sent: u64,
}

/// A body fed by the returned [`Chunks`].
pub fn channel(kill: Option<Kill>, stall: Duration, fail_after: Option<u64>) -> (Chunks, Body) {
    let (tx, rx) = mpsc::channel(QUEUE);
    let body = Body::from_stream(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }));
    let chunks = Chunks {
        tx,
        kill,
        stall,
        fail_after,
        sent: 0,
    };
    (chunks, body)
}

impl Chunks {
    /// Sends one chunk, waiting at most the stall timeout for room.
    ///
    /// # Errors
    /// Returns why the stream is over; on [`Cut::Stalled`] and [`Cut::Injected`] the response
    /// has been aborted.
    pub async fn send(&mut self, chunk: Vec<u8>) -> Result<(), Cut> {
        if self.fail_after.is_some_and(|after| self.sent > after) {
            self.abort("injected failure");
            return Err(Cut::Injected);
        }
        match tokio::time::timeout(self.stall, self.tx.send(Ok(chunk))).await {
            Ok(Ok(())) => {
                self.sent += 1;
                Ok(())
            }
            Ok(Err(_)) => Err(Cut::Gone),
            Err(_) => {
                self.abort("the client stopped reading");
                Err(Cut::Stalled)
            }
        }
    }

    /// [`Self::send`] from a blocking thread (`DuckDB` runs on one).
    ///
    /// # Errors
    /// As [`Self::send`].
    pub fn send_blocking(
        &mut self,
        runtime: &tokio::runtime::Handle,
        chunk: Vec<u8>,
    ) -> Result<(), Cut> {
        runtime.block_on(self.send(chunk))
    }

    /// Ends the response with an error, so the client sees a broken stream, and drops its
    /// connection in case the client is not reading and the error would never reach it.
    pub fn abort(&self, why: &str) {
        let _ = self.tx.try_send(Err(io::Error::other(why.to_owned())));
        if let Some(kill) = &self.kill {
            kill.kill();
        }
    }
}
