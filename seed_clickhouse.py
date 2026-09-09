import os
import sys
import time
import uuid
import dotenv
import clickhouse_connect
import numpy as np
import pandas as pd

dotenv.load_dotenv()

def get_ch_client():
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", "8443"))
    username = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    secure = os.getenv("CLICKHOUSE_SECURE", "true").lower() in ("true", "1", "yes")

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        secure=secure,
        send_receive_timeout=900,
        connect_timeout=30,
        settings={"async_insert": 0}
    )

def insert_with_retry(client_holder, table, df):
    backoffs = [5, 10, 20, 40, 80]
    for attempt in range(5):
        try:
            client_holder['client'].insert_df(table, df)
            return
        except Exception as e:
            if attempt == 4:
                print("[RETRY] Final attempt 5 failed. Raising exception.")
                raise e
            delay = backoffs[attempt]
            print(f"[RETRY] Insert failed (attempt {attempt + 1}/5): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            try:
                client_holder['client'] = get_ch_client()
            except Exception as conn_err:
                print(f"[RETRY] Failed to reconnect: {conn_err}")


SCENE_LIBRARY_SIZE = 60


def build_scene_library():
    """60 scenes across 6 episodes. The first two are the demo presets. Creative attributes are
    drawn so that several patterns are discoverable rather than random noise:
      A) dialogue-heavy (>0.70), long (>240s), low-motion (<0.30) scenes  -> 18-24 mobile churn
      B) very fast dialogue pacing (>200 wpm)                              -> 55+ tv churn
      C) kinetic (motion >0.60) short scenes                               -> low churn everywhere
    """
    rng = np.random.default_rng(42)
    scenes = [
        ("cut_01", "ep_101", "Safehouse Dialogue Lock (Cut 3)", 272, 0.84, 205, 0.14, "stagnant_two_shot"),
        ("cut_02", "ep_101", "Warehouse Breach Pre-Assault", 135, 0.22, 75, 0.76, "kinetic_suspense"),
    ]
    archetypes = [
        # (weight, title stem, dur range, dialogue range, wpm range, motion range, sentiment)
        (0.28, "Interrogation", (245, 340), (0.72, 0.92), (150, 215), (0.05, 0.28), "tense_dialogue"),
        (0.12, "Monologue", (250, 320), (0.80, 0.95), (205, 240), (0.05, 0.20), "dramatic_speech"),
        (0.20, "Chase", (90, 200), (0.05, 0.25), (30, 90), (0.65, 0.95), "high_octane"),
        (0.20, "Recon", (120, 230), (0.35, 0.65), (90, 160), (0.35, 0.60), "ambient_build"),
        (0.10, "Briefing", (150, 235), (0.60, 0.80), (140, 190), (0.20, 0.45), "procedural_exposition"),
        (0.10, "Standoff", (200, 260), (0.30, 0.55), (80, 140), (0.40, 0.70), "kinetic_suspense"),
    ]
    weights = np.array([a[0] for a in archetypes])
    weights = weights / weights.sum()
    places = ["Rooftop", "Courtroom", "Night Club", "Highway", "Harbour", "Embassy", "Subway", "Vault", "Bridge", "Hangar"]
    idx = 3
    while len(scenes) < SCENE_LIBRARY_SIZE:
        a = archetypes[rng.choice(len(archetypes), p=weights)]
        _, stem, dur, dia, wpm, mot, sent = a
        ep = f"ep_{101 + (idx - 1) % 6}"
        title = f"{places[idx % len(places)]} {stem} {'Climax' if idx % 4 == 0 else 'Sequence' if idx % 3 == 0 else 'Beat'}"
        scenes.append((
            f"cut_{idx:02d}", ep, title,
            int(rng.integers(dur[0], dur[1] + 1)),
            round(float(rng.uniform(*dia)), 2),
            int(rng.integers(wpm[0], wpm[1] + 1)),
            round(float(rng.uniform(*mot)), 2),
            sent,
        ))
        idx += 1
    return scenes

def seed_database():
    print("[CLICKHOUSE SEED] Connecting to ClickHouse...")
    client = get_ch_client()
    client_holder = {'client': client}

    print("[CLICKHOUSE SEED] Creating schema...")
    client.command("""
    CREATE TABLE IF NOT EXISTS scene_dna_features (
        scene_id String,
        episode_id LowCardinality(String),
        scene_title String,
        duration_sec UInt32,
        dialogue_ratio Float32,
        pacing_wpm UInt16,
        motion_intensity Float32,
        sentiment LowCardinality(String)
    ) ENGINE = MergeTree()
    ORDER BY (episode_id, scene_id)
    """)

    client.command("""
    CREATE TABLE IF NOT EXISTS viewer_retention_events (
        event_id UUID,
        viewer_id UInt32,
        episode_id LowCardinality(String),
        scene_id LowCardinality(String),
        age_group LowCardinality(String),
        device LowCardinality(String),
        dropped_out UInt8,
        timestamp_sec UInt32
    ) ENGINE = MergeTree()
    ORDER BY (episode_id, age_group, device, scene_id)
    """)

    if '--reset' in sys.argv:
        print("[CLICKHOUSE SEED] --reset given: truncating viewer_retention_events and scene_dna_features...")
        client.command("TRUNCATE TABLE viewer_retention_events")
        client.command("TRUNCATE TABLE scene_dna_features")

    # Idempotent scene DNA features insertion
    scene_count = client.command("SELECT count() FROM scene_dna_features")
    if scene_count == 0:
        print("[CLICKHOUSE SEED] Seeding baseline scene library...")
        scenes_data = build_scene_library()
        scenes_df = pd.DataFrame(scenes_data, columns=[
            "scene_id", "episode_id", "scene_title", "duration_sec",
            "dialogue_ratio", "pacing_wpm", "motion_intensity", "sentiment"
        ])
        insert_with_retry(client_holder, "scene_dna_features", scenes_df)
        print(f"[CLICKHOUSE SEED] Inserted {len(scenes_df)} reference scenes.")
    else:
        print(f"[CLICKHOUSE SEED] scene_dna_features already has {scene_count} rows. Skipping.")

    total_rows = 4_800_000
    chunk_size = 100_000
    num_chunks = total_rows // chunk_size

    # Resume support (pass --reset to truncate and regenerate from scratch)
    existing_count = client.command("SELECT count() FROM viewer_retention_events")
    if existing_count >= total_rows:
        print(f"[CLICKHOUSE SEED] Seeding is complete. Existing count: {existing_count:,} >= {total_rows:,}. Exiting.")
        return

    chunks_to_skip = existing_count // chunk_size
    if chunks_to_skip > 0:
        print(f"[CLICKHOUSE SEED] Resuming from chunk {chunks_to_skip}. (Skipping first {chunks_to_skip} chunks, approx {chunks_to_skip * chunk_size:,} rows)")
    
    age_groups = np.array(['18-24', '25-34', '35-44', '45-54', '55+'])
    devices = np.array(['mobile', 'desktop', 'tv', 'tablet'])
    library = build_scene_library()
    scene_ids = np.array([sc[0] for sc in library])
    scene_ep_map = {sc[0]: sc[1] for sc in library}
    scene_duration_map = {sc[0]: sc[3] for sc in library}
    pattern_a = {sc[0] for sc in library if sc[4] > 0.70 and sc[3] > 240 and sc[6] < 0.30}
    pattern_b = {sc[0] for sc in library if sc[5] > 200}
    pattern_c = {sc[0] for sc in library if sc[6] > 0.60 and sc[3] < 200}
    high_risk_scenes = pattern_a
    print(f"[CLICKHOUSE SEED] Library: {len(library)} scenes | pattern A (dialogue lock) {len(pattern_a)} | pattern B (fast pacing) {len(pattern_b)} | pattern C (kinetic) {len(pattern_c)}")

    print(f"[CLICKHOUSE SEED] Generating {total_rows:,} viewer retention records across {num_chunks} chunks...")
    start_time = time.time()

    for chunk_idx in range(chunks_to_skip, num_chunks):
        chunk_start = time.time()
        
        chosen_scene_ids = np.random.choice(scene_ids, size=chunk_size)
        chosen_episodes = np.array([scene_ep_map[sid] for sid in chosen_scene_ids])
        chosen_ages = np.random.choice(age_groups, size=chunk_size)
        chosen_devices = np.random.choice(devices, size=chunk_size)
        viewer_ids = np.random.randint(100000, 999999, size=chunk_size, dtype=np.uint32)
        is_high_risk_scene = np.isin(chosen_scene_ids, list(high_risk_scenes))
        is_target_segment = (chosen_ages == '18-24') & (chosen_devices == 'mobile')

        is_pattern_b = np.isin(chosen_scene_ids, list(pattern_b))
        is_pattern_c = np.isin(chosen_scene_ids, list(pattern_c))
        is_senior_tv = (chosen_ages == '55+') & (chosen_devices == 'tv')
        drop_probs = np.full(chunk_size, 0.080)
        drop_probs = np.where(is_pattern_c, 0.062, drop_probs)
        drop_probs = np.where(is_pattern_b & is_senior_tv, 0.150, drop_probs)
        drop_probs = np.where(is_high_risk_scene & is_target_segment, 0.235, drop_probs)
        dropped_out = (np.random.rand(chunk_size) < drop_probs).astype(np.uint8)

        # Timestamps are bounded by each scene's real duration. Baseline drop-offs are
        # spread uniformly; hazard-cohort drop-offs on dialogue-heavy scenes cluster in the
        # mid-scene "dialogue lock" window (peak ~55% of runtime, e.g. 02:15-03:00 on a 272s cut).
        durations = np.vectorize(scene_duration_map.get)(chosen_scene_ids).astype(np.float64)
        uniform_ts = np.random.rand(chunk_size) * durations
        peak_ts = np.random.normal(loc=durations * 0.55, scale=durations * 0.09)
        peak_ts = np.clip(peak_ts, durations * 0.45, durations * 0.75)
        hazard_drop = is_high_risk_scene & is_target_segment & (dropped_out == 1)
        timestamps = np.where(hazard_drop, peak_ts, uniform_ts)
        timestamps = np.clip(timestamps, 0, durations - 1).astype(np.uint32)
        
        # Faster UUID generation using list comprehension of strings
        uuids = [str(uuid.uuid4()) for _ in range(chunk_size)]
        
        chunk_df = pd.DataFrame({
            'event_id': uuids,
            'viewer_id': viewer_ids,
            'episode_id': chosen_episodes,
            'scene_id': chosen_scene_ids,
            'age_group': chosen_ages,
            'device': chosen_devices,
            'dropped_out': dropped_out,
            'timestamp_sec': timestamps
        })
        
        insert_with_retry(client_holder, 'viewer_retention_events', chunk_df)
        
        chunk_duration = time.time() - chunk_start
        cumulative_rows = (chunk_idx + 1) * chunk_size
        rows_per_sec = chunk_size / chunk_duration if chunk_duration > 0 else 0
        print(f"  -> Chunk {chunk_idx + 1}/{num_chunks} inserted ({chunk_size:,} rows) in {chunk_duration:.2f}s | Cumulative: {cumulative_rows:,} rows | Speed: {rows_per_sec:.2f} rows/sec")

    total_duration = time.time() - start_time
    print(f"[CLICKHOUSE SEED] Completed! Successfully seeded {total_rows:,} rows in {total_duration:.2f}s.")

if __name__ == "__main__":
    seed_database()
