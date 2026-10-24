//! Crash injection: the production writer, run as its own process with every frame split in
//! two, is killed 10-500 ms after it opens its log; recovery must then close the log with every
//! good frame kept, seqs contiguous from 0 and every payload the one written at that position.
//! Rounds default to 24 to keep the gate quick; set `TP_CRASH_ROUNDS` for a longer soak.

use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use engine::store::frame::Kind;
use engine::store::segment::{self, Ending, LOG_DIR, SegmentReader, recover_run};
use engine_harness::{KillOnDrop, crash_payload};

#[test]
fn a_killed_writer_loses_no_good_frame_once_recovered() -> Result<(), Box<dyn Error>> {
    let rounds: u32 = std::env::var("TP_CRASH_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(24);
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let mut state = (now.as_secs() << 32) ^ u64::from(now.subsec_nanos()) | 1;
    println!("seed {state}");
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut torn, mut written) = (0, 0);
    for round in 0..rounds {
        let run_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("crash-{}-{round}", std::process::id()));
        let _ = fs::remove_dir_all(&run_dir);
        // Killed on drop if this round fails early; its stdin closes with this process, which
        // ends the writer even if the whole test runner is killed.
        let mut child = KillOnDrop(
            Command::new(env!("CARGO_BIN_EXE_crash-writer"))
                .arg("--end-with-stdin")
                .arg(&run_dir)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()?,
        );
        // Time the kill from when the writer has its log open, not from the spawn: a slow
        // start (a scan of a fresh binary, a loaded machine) must not eat the random delay.
        let started = Instant::now();
        while !run_dir.join(LOG_DIR).exists() && started.elapsed() < Duration::from_secs(20) {
            sleep(Duration::from_millis(2));
        }
        sleep(Duration::from_millis(10 + random() % 491));
        child.kill()?;
        child.wait()?;
        assert!(
            run_dir.join(LOG_DIR).exists(),
            "round {round}: the writer never started"
        );

        let recovery = recover_run(&run_dir)?;
        if recovery.as_ref().is_some_and(|r| r.cut > 0) {
            torn += 1;
        }
        let (mut seq, mut i) = (0, 0);
        for (first_seq, path) in segment::list(&run_dir.join(LOG_DIR))? {
            let mut reader = SegmentReader::open(&path)?;
            while let Some(frame) = reader.next_frame()? {
                assert_eq!(frame.seq, seq, "round {round}: seqs run on from 0");
                if seq == first_seq {
                    // A segment opens with its Meta, or, killed right after the file was made,
                    // holds only the marker recovery gave it.
                    assert!(
                        matches!(frame.kind, Kind::Meta | Kind::SegmentEnd),
                        "round {round}"
                    );
                }
                if frame.kind == Kind::Trade {
                    assert_eq!(frame.payload, crash_payload(i), "round {round}: entry {i}");
                    i += 1;
                }
                seq += 1;
            }
            assert!(
                matches!(reader.ending(), Some(Ending::Closed(_))),
                "round {round}: {path:?}"
            );
        }
        assert_eq!(
            recover_run(&run_dir)?,
            None,
            "round {round}: recovered once is closed"
        );
        written += i;
        fs::remove_dir_all(&run_dir)?;
    }
    println!("{rounds} rounds, {torn} torn tails cut, {written} entries kept");
    if rounds >= 8 {
        assert!(
            torn > 0,
            "no kill landed mid-frame, so the torn-tail path went untested"
        );
    }
    Ok(())
}

/// Started with stdin null and no `--end-with-stdin`, the writer still writes until it is killed,
/// instead of ending at once with nothing written and exit 0, as if it had run (#93). It is
/// killed as soon as it has filled one 16 KiB segment, a fraction of a second.
#[test]
fn a_writer_without_stdin_held_open_still_writes() -> Result<(), Box<dyn Error>> {
    let run_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("crash-no-stdin-{}", std::process::id()));
    let _ = fs::remove_dir_all(&run_dir);
    let mut child = KillOnDrop(
        Command::new(env!("CARGO_BIN_EXE_crash-writer"))
            .arg(&run_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            panic!("the writer ended on its own ({status}) with stdin null");
        }
        let segments = segment::list(&run_dir.join(LOG_DIR)).map_or(0, |list| list.len());
        if segments >= 2 {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{segments} segments after 20 s"
        );
        sleep(Duration::from_millis(5));
    }
    child.kill()?;
    child.wait()?;
    fs::remove_dir_all(&run_dir)?;
    Ok(())
}
