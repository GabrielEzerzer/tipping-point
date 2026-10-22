//! Streaming response bodies that are lossless or dead. Chunks go to the body through a small
//! bounded queue; a client that takes nothing for the stall timeout, an error on the engine side,
//! a producer that stops without finishing (a panic, an early return) or an injected harness
//! fault aborts the response and drops the connection, never ends it cleanly, so a reader can
//! tell a cut stream from a finished one. Only [`Chunks::finish`] sends Arrow's end-of-stream
//! marker and lets the body end.

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

/// Harness only: a failure every Arrow stream meets once this many chunks after the first (the
/// schema) have gone out. `None` in production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The response is aborted, as an engine-side error aborts it.
    Abort { after: u64 },
    /// The producer panics, as a bug would.
    Panic { after: u64 },
}

/// The producer's end of a streaming body. Dropped without [`Self::finish`], it aborts the
/// response.
#[derive(Debug)]
pub struct Chunks {
    tx: mpsc::Sender<io::Result<Vec<u8>>>,
    kill: Option<Kill>,
    stall: Duration,
    fault: Option<Fault>,
    sent: u64,
    /// The stream finished cleanly or was aborted: nothing is left to do on drop.
    done: bool,
}

/// A body fed by the returned [`Chunks`].
pub fn channel(kill: Option<Kill>, stall: Duration, fault: Option<Fault>) -> (Chunks, Body) {
    let (tx, rx) = mpsc::channel(QUEUE);
    let body = Body::from_stream(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }));
    let chunks = Chunks {
        tx,
        kill,
        stall,
        fault,
        sent: 0,
        done: false,
    };
    (chunks, body)
}

impl Chunks {
    /// Sends one chunk, waiting at most the stall timeout for room.
    ///
    /// # Errors
    /// Returns why the stream is over; on [`Cut::Stalled`] and [`Cut::Injected`] the response
    /// has been aborted.
    ///
    /// # Panics
    /// Only under the harness's [`Fault::Panic`].
    pub async fn send(&mut self, chunk: Vec<u8>) -> Result<(), Cut> {
        match self.fault {
            Some(Fault::Abort { after }) if self.sent > after => {
                self.abort("injected failure");
                return Err(Cut::Injected);
            }
            Some(Fault::Panic { after }) if self.sent > after => {
                panic!("injected panic in an Arrow stream's producer");
            }
            _ => {}
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

    /// Sends the end-of-stream chunk and lets the body end: the one way a stream ends cleanly.
    ///
    /// # Errors
    /// As [`Self::send`]; the response is then aborted.
    pub async fn finish(mut self, end: Vec<u8>) -> Result<(), Cut> {
        self.send(end).await?;
        self.done = true;
        Ok(())
    }

    /// [`Self::finish`] from a blocking thread.
    ///
    /// # Errors
    /// As [`Self::finish`].
    pub fn finish_blocking(
        self,
        runtime: &tokio::runtime::Handle,
        end: Vec<u8>,
    ) -> Result<(), Cut> {
        runtime.block_on(self.finish(end))
    }

    /// Ends the response with an error, so the client sees a broken stream, and drops its
    /// connection in case the client is not reading and the error would never reach it.
    pub fn abort(&mut self, why: &str) {
        let _ = self.tx.try_send(Err(io::Error::other(why.to_owned())));
        if let Some(kill) = &self.kill {
            kill.kill();
        }
        self.done = true;
    }
}

impl Drop for Chunks {
    /// A producer that stops without finishing must not let the body end cleanly: hyper would
    /// write the chunked terminator and a reader would take the prefix for the whole stream.
    fn drop(&mut self) {
        if !self.done {
            self.abort("the stream's producer stopped before its end");
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;

    /// What the body yields until it ends: the chunks, then whether it ended on an error.
    async fn drain(body: Body) -> (Vec<Vec<u8>>, bool) {
        let mut stream = body.into_data_stream();
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => chunks.push(bytes.to_vec()),
                Err(_) => return (chunks, true),
            }
        }
        (chunks, false)
    }

    #[tokio::test]
    async fn only_finish_ends_the_body_cleanly() {
        let (mut chunks, body) = channel(None, Duration::from_secs(5), None);
        let reader = tokio::spawn(drain(body));
        assert_eq!(chunks.send(vec![1]).await, Ok(()));
        assert_eq!(chunks.finish(vec![2]).await, Ok(()));
        let (got, failed) = reader.await.expect("the reader");
        assert_eq!((got, failed), (vec![vec![1], vec![2]], false));
    }

    #[tokio::test]
    async fn a_producer_dropped_before_finishing_aborts_the_body() {
        let (mut chunks, body) = channel(None, Duration::from_secs(5), None);
        let reader = tokio::spawn(drain(body));
        assert_eq!(chunks.send(vec![1]).await, Ok(()));
        drop(chunks);
        let (got, failed) = reader.await.expect("the reader");
        assert_eq!(got, vec![vec![1]]);
        assert!(failed, "the body must end on an error, not cleanly");
    }

    #[tokio::test]
    async fn a_producer_that_panics_aborts_the_body() {
        let (mut chunks, body) = channel(
            None,
            Duration::from_secs(5),
            Some(Fault::Panic { after: 0 }),
        );
        let reader = tokio::spawn(drain(body));
        let producer = tokio::spawn(async move {
            let _ = chunks.send(vec![1]).await;
            let _ = chunks.send(vec![2]).await;
            chunks.finish(vec![3]).await
        });
        assert!(producer.await.is_err(), "the producer panicked");
        let (got, failed) = reader.await.expect("the reader");
        assert_eq!(got, vec![vec![1]]);
        assert!(failed, "a panic must end the body on an error");
    }
}
