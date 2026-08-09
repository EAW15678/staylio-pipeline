"""
Throwaway script: generate 6 Runway Gen-4 Turbo clips for Vista Azule
motion comparison. NOT production code. Budget: ~$10 max.

Usage: python tools/motion_comparison.py
"""

import asyncio
import os
import time
import httpx
import runwayml

OUTPUT_DIR = os.path.expanduser("~/Desktop/motion-comparison")
R2_BASE = "https://pub-dca6781384314541a5329feab1da5de4.r2.dev/a1b2c3d4-0001-0001-0001-000000000001/enhanced"

# The six frames with intended camera moves (matching parallax set)
FRAMES = [
    {
        "name": "exterior_a",
        "file": "photo_029_59501608.jpg",
        "move": "Slow push in through the front columns, revealing the full facade. Steady, cinematic, architectural.",
    },
    {
        "name": "exterior_b",
        "file": "photo_093_2cfa4018.jpg",
        "move": "Slow pull back from the building to reveal the full exterior and surrounding environment. Smooth, wide establishing shot.",
    },
    {
        "name": "outdoor_ent",
        "file": "photo_008_25dcde7b.jpg",
        "move": "Gentle lateral drift to the right across the outdoor entertainment area. Smooth, slow pan revealing depth.",
    },
    {
        "name": "pool",
        "file": "photo_057_243e2320.jpg",
        "move": "Slow push in toward the pool, water gently rippling. Cinematic approach, inviting.",
    },
    {
        "name": "living_room",
        "file": "photo_020_e70156f2.jpg",
        "move": "Gentle lateral drift to the left across the living room. Slow reveal of the full space, warm interior light.",
    },
    {
        "name": "bathroom",
        "file": "photo_037_63f82441.jpg",
        "move": "Very subtle hold with minimal drift. The camera barely moves, just enough to feel alive. Intimate, still.",
    },
]

MODEL = "gen4_turbo"
DURATION = 10  # seconds — SOC-11: 5s reads as fragment, target 8-15s
CREDITS_PER_SECOND = 5
COST_PER_CREDIT = 0.01  # $0.01/credit from docs.dev.runwayml.com


async def main():
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        print("ERROR: RUNWAYML_API_SECRET not set")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_cost = 0.0
    total_time = 0.0
    results = []

    async with runwayml.AsyncRunwayML(api_key=api_key) as client:
        for frame in FRAMES:
            url = f"{R2_BASE}/{frame['file']}"
            name = frame["name"]
            move = frame["move"]

            print(f"\n{'='*60}")
            print(f"  {name}: {frame['file']}")
            print(f"  Move: {move[:60]}...")
            print(f"  URL: {url}")
            print(f"{'='*60}")

            start = time.time()
            clip_cost = DURATION * CREDITS_PER_SECOND * COST_PER_CREDIT

            # Budget check
            if total_cost + clip_cost > 10.0:
                print(f"  SKIPPED — would exceed $10 budget (running total: ${total_cost:.2f})")
                results.append({
                    "name": name,
                    "status": "skipped",
                    "reason": "budget",
                    "cost": 0,
                    "time": 0,
                })
                continue

            try:
                task = await client.image_to_video.create(
                    model=MODEL,
                    prompt_image=url,
                    prompt_text=move,
                    duration=DURATION,
                    ratio="1280:720",
                )

                print(f"  Task started: {task.id}")
                print(f"  Waiting for completion...")

                result = await task.wait_for_task_output(timeout=600)

                output_urls = getattr(result, "output", None) or []
                if not output_urls:
                    elapsed = time.time() - start
                    print(f"  FAILED — no output URLs returned ({elapsed:.0f}s)")
                    results.append({
                        "name": name,
                        "status": "failed",
                        "reason": "no output",
                        "cost": clip_cost,  # credits still consumed
                        "time": elapsed,
                    })
                    total_cost += clip_cost
                    continue

                # Download the clip
                output_url = output_urls[0]
                print(f"  Downloading from: {output_url[:80]}...")

                async with httpx.AsyncClient(timeout=120) as dl:
                    resp = await dl.get(output_url)
                    resp.raise_for_status()
                    video_bytes = resp.content

                # Save to desktop
                out_path = os.path.join(OUTPUT_DIR, f"{name}_gen4turbo_{DURATION}s.mp4")
                with open(out_path, "wb") as f:
                    f.write(video_bytes)

                elapsed = time.time() - start
                total_cost += clip_cost
                total_time += elapsed

                print(f"  SUCCESS — {len(video_bytes)/1024:.0f}KB, {elapsed:.0f}s, ${clip_cost:.2f}")
                print(f"  Saved: {out_path}")

                results.append({
                    "name": name,
                    "status": "success",
                    "cost": clip_cost,
                    "time": elapsed,
                    "size_kb": len(video_bytes) / 1024,
                    "path": out_path,
                })

            except Exception as exc:
                elapsed = time.time() - start
                total_cost += clip_cost  # assume credits consumed on failure
                print(f"  FAILED — {exc} ({elapsed:.0f}s)")
                results.append({
                    "name": name,
                    "status": "failed",
                    "reason": str(exc),
                    "cost": clip_cost,
                    "time": elapsed,
                })

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:     {MODEL}")
    print(f"  Duration:  {DURATION}s per clip")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Total time: {total_time:.0f}s")
    print()
    for r in results:
        status = r["status"].upper()
        cost = f"${r['cost']:.2f}"
        t = f"{r['time']:.0f}s" if r.get("time") else "—"
        print(f"  {r['name']:15s}  {status:8s}  {cost:6s}  {t}")
    print()
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
