"""
3-way motion comparison: Gen-4 Turbo vs Gen-4.5 (standard).
Throwaway script. Budget ~$15.

Gen-4 Turbo clips already exist from tools/motion_comparison.py — copy, don't regen.
No standard Gen-4 model exists for image-to-video (SDK has gen4_turbo and gen4.5 only).
"""

import asyncio
import os
import shutil
import time
import httpx
import runwayml

OUTPUT_DIR = os.path.expanduser("~/Desktop/motion-quality-3way")
EXISTING_DIR = os.path.expanduser("~/Desktop/motion-comparison")
R2_BASE = "https://pub-dca6781384314541a5329feab1da5de4.r2.dev/a1b2c3d4-0001-0001-0001-000000000001/enhanced"

FRAMES = [
    {
        "name": "pool",
        "file": "photo_057_243e2320.jpg",
        "move": "Slow push in toward the pool, water gently rippling. Cinematic approach, inviting.",
    },
    {
        "name": "exterior_b",
        "file": "photo_093_2cfa4018.jpg",
        "move": "Slow pull back from the building to reveal the full exterior and surrounding environment. Smooth, wide establishing shot.",
    },
    {
        "name": "living_room",
        "file": "photo_020_e70156f2.jpg",
        "move": "Gentle lateral drift to the left across the living room. Slow reveal of the full space, warm interior light.",
    },
]

# Pricing: credits per second, $0.01 per credit
MODELS = [
    {"id": "gen4_turbo", "label": "gen4_turbo", "credits_per_sec": 5, "duration": 10, "seed": 42},
    {"id": "gen4.5", "label": "gen4.5", "credits_per_sec": 12, "duration": 10, "seed": 42},
]


async def main():
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        print("ERROR: RUNWAYML_API_SECRET not set")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy existing Turbo clips
    print("Copying existing Gen-4 Turbo clips...")
    for frame in FRAMES:
        src = os.path.join(EXISTING_DIR, f"{frame['name']}_gen4turbo_10s.mp4")
        dst = os.path.join(OUTPUT_DIR, f"{frame['name']}_gen4_turbo.mp4")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {frame['name']}_gen4_turbo.mp4")
        else:
            print(f"  WARNING: {src} not found")

    total_cost = 0.0
    results = []

    # Only generate Gen-4.5 (Turbo already exists)
    model = MODELS[1]  # gen4.5
    print(f"\n{'='*60}")
    print(f"  Generating Gen-4.5 clips (12 credits/sec, ${0.12}/sec)")
    print(f"{'='*60}")

    async with runwayml.AsyncRunwayML(api_key=api_key) as client:
        for frame in FRAMES:
            url = f"{R2_BASE}/{frame['file']}"
            name = frame["name"]
            move = frame["move"]
            duration = model["duration"]
            clip_cost = duration * model["credits_per_sec"] * 0.01

            # Budget check
            if total_cost + clip_cost > 15.0:
                print(f"  {name}: SKIPPED — would exceed $15 budget (total: ${total_cost:.2f})")
                results.append({"name": name, "model": model["label"], "status": "skipped", "cost": 0, "time": 0})
                continue

            print(f"\n  {name}: {frame['file']}")
            print(f"    Move: {move[:60]}...")
            print(f"    Cost: ${clip_cost:.2f} ({duration}s × {model['credits_per_sec']} cred/s)")

            start = time.time()
            try:
                task = await client.image_to_video.create(
                    model=model["id"],
                    prompt_image=url,
                    prompt_text=move,
                    duration=duration,
                    ratio="1280:720",
                    seed=model["seed"],
                )

                print(f"    Task: {task.id}")
                print(f"    Waiting...")

                result = await task.wait_for_task_output(timeout=900)  # Gen-4.5 may be slower

                output_urls = getattr(result, "output", None) or []
                if not output_urls:
                    elapsed = time.time() - start
                    print(f"    FAILED — no output ({elapsed:.0f}s)")
                    results.append({"name": name, "model": model["label"], "status": "failed",
                                    "cost": clip_cost, "time": elapsed, "reason": "no output"})
                    total_cost += clip_cost
                    continue

                async with httpx.AsyncClient(timeout=120) as dl:
                    resp = await dl.get(output_urls[0])
                    resp.raise_for_status()
                    video_bytes = resp.content

                out_path = os.path.join(OUTPUT_DIR, f"{name}_{model['label'].replace('.', '_')}.mp4")
                with open(out_path, "wb") as f:
                    f.write(video_bytes)

                elapsed = time.time() - start
                total_cost += clip_cost

                print(f"    SUCCESS — {len(video_bytes)/1024:.0f}KB, {elapsed:.0f}s, ${clip_cost:.2f}")
                results.append({"name": name, "model": model["label"], "status": "success",
                                "cost": clip_cost, "time": elapsed, "size_kb": len(video_bytes)/1024,
                                "path": out_path})

            except Exception as exc:
                elapsed = time.time() - start
                total_cost += clip_cost
                print(f"    FAILED — {exc} ({elapsed:.0f}s)")
                results.append({"name": name, "model": model["label"], "status": "failed",
                                "cost": clip_cost, "time": elapsed, "reason": str(exc)})

    # Add Turbo results (free — already generated)
    for frame in FRAMES:
        turbo_path = os.path.join(OUTPUT_DIR, f"{frame['name']}_gen4_turbo.mp4")
        if os.path.exists(turbo_path):
            size = os.path.getsize(turbo_path) / 1024
            results.append({"name": frame["name"], "model": "gen4_turbo", "status": "reused",
                            "cost": 0, "time": 0, "size_kb": size})

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  New spend this run: ${total_cost:.2f}")
    print(f"  Gen-4 Turbo: reused from prior run (no additional cost)")
    print()

    # Group by frame for comparison
    for frame in FRAMES:
        print(f"  {frame['name']}:")
        for r in results:
            if r["name"] == frame["name"]:
                status = r["status"].upper()
                cost = f"${r['cost']:.2f}" if r["cost"] > 0 else "reused"
                t = f"{r['time']:.0f}s" if r.get("time", 0) > 0 else "—"
                size = f"{r.get('size_kb', 0):.0f}KB" if r.get("size_kb") else "—"
                print(f"    {r['model']:15s}  {status:8s}  {cost:8s}  {t:5s}  {size}")
        print()

    # Volume cost table
    print("  VOLUME COST TABLE (8 concepts × 5 beats × 10s = 400s/property/month)")
    for m in MODELS:
        monthly = 400 * m["credits_per_sec"] * 0.01
        print(f"    {m['label']:15s}  {m['credits_per_sec']:2d} cred/s  ${monthly:.2f}/property/month")
    print()
    print(f"  NOTE: No standard Gen-4 model exists for image-to-video.")
    print(f"        SDK exposes: gen4_turbo, gen4.5, gen3a_turbo, veo3.1, veo3.1_fast, veo3")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
