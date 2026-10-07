"""What does a run generate, against the worlds it was supposed to recover?

For a FOL-domain run (merger trees) this reads the model's own
canonical keys from all_generated_smiles.in-training.txt and compares them with
the training and validation worlds. It answers three questions a recall number
cannot: which validation worlds were never generated, how often the most common
training worlds were generated, and -- for each missed validation world -- whether
the model produced worlds with the same labels but a different tree. Stdlib only.

    python scripts/inspect_generated_worlds.py mergertrees/experiments/mlj-mergertrees4-sampler=wilson-seed=22-3M data/mergertrees/mergertrees4
"""
import sys, pathlib, collections, re

run, data = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 8

def keys(d):
    return ["".join(sorted(l.strip() for l in f.read_text().splitlines() if l.strip())) for f in d.glob("*.mol")]
def unary_part(k):          # labels only: drop every binary fact
    return "".join(f + ")." for f in re.split(r"\)\.", k) if f and "," not in f)
def edges(k):
    return sorted(set(tuple(sorted(map(int, m))) for m in re.findall(r"\w+\((\d+),(\d+)\)\.", k)))

train, val = collections.Counter(keys(data / "train")), collections.Counter(keys(data / "val"))
gen = collections.Counter(l.strip() for l in open(run / "all_generated_smiles.in-training.txt") if l.strip())
gen_by_unary = collections.defaultdict(collections.Counter)
for k, c in gen.items(): gen_by_unary[unary_part(k)][k] += c
total = sum(gen.values())

print(f"run: {run.name}")
print(f"generated {total:,} samples, {len(gen):,} distinct worlds | train {sum(train.values())} files / {len(train)} distinct | val {sum(val.values())} files / {len(val)} distinct")
hit = [k for k in val if k in gen]
print(f"validation recall: {len(hit)}/{len(val)} distinct worlds = {len(hit)/len(val):.3f};  entry-weighted (MLflow convention) = {sum(val[k] for k in hit)/sum(val.values()):.3f}")

print(f"\n--- top {top} TRAINING worlds: how often the run generated them ---")
for k, c in train.most_common(top):
    print(f"  train x{c:<4} val x{val.get(k,0):<3} generated {gen.get(k,0):>7,}   edges={edges(k)}  labels: {unary_part(k)}")

missed = sorted((k for k in val if k not in gen), key=lambda k: -val[k])
print(f"\n--- MISSED validation worlds ({len(missed)}), most-copied first ---")
for k in missed[:top]:
    same_labels = gen_by_unary.get(unary_part(k), {})
    alt = sum(same_labels.values())
    print(f"  val x{val[k]:<3} train x{train.get(k,0):<4} edges={edges(k)}  labels: {unary_part(k)}")
    if alt:
        best, bc = same_labels.most_common(1)[0]
        print(f"        -> same labels generated {alt:,} times with a DIFFERENT tree; commonest alternative edges={edges(best)} ({bc:,}x)")
    else:
        print(f"        -> this label configuration was never generated at all")

print(f"\n--- top {top} GENERATED worlds ---")
for k, c in gen.most_common(top):
    print(f"  {c:>8,}  train x{train.get(k,0):<4} val x{val.get(k,0):<3} edges={edges(k)}  labels: {unary_part(k)}")
