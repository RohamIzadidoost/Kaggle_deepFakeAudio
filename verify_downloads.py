# %% [markdown]
# ## Verify downloads
# Paste as a cell after the download cell finishes. Checks expected structure,
# file counts, and that a random sample of files actually decodes.

# %%
import glob, os, random
import soundfile as sf

EXPECTED = {
    "asvspoof2019_LA": {
        "protocol": "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "flac_glob": "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac/*.flac",
        "expect_files": 25380,
    },
    "in_the_wild": {
        "wav_glob": "data/in_the_wild/**/*.wav",
        "expect_files": 31779,
    },
    "arabic_arad": {
        "wav_glob": "data/arabic_arad/**/*.wav",
        "expect_files": 19561,
    },
}

def check_count(name, pattern, expect, recursive=False):
    files = glob.glob(pattern, recursive=recursive)
    n = len(files)
    ok = n == expect
    tag = "OK " if ok else ("WARN" if n > 0 else "FAIL")
    print(f"[{tag}] {name}: found {n}, expected {expect}")
    return files

def check_decode(name, files, k=20, seed=0):
    if not files:
        print(f"[FAIL] {name}: no files to sample")
        return
    random.seed(seed)
    sample = random.sample(files, min(k, len(files)))
    bad = []
    for p in sample:
        try:
            sf.read(p)
        except Exception as e:
            bad.append((p, str(e)[:60]))
    if bad:
        print(f"[WARN] {name}: {len(bad)}/{len(sample)} sampled files failed to decode:")
        for p, e in bad[:5]:
            print("   ", p, "->", e)
    else:
        print(f"[OK ] {name}: {len(sample)}/{len(sample)} sampled files decode fine")

print("=== ASVspoof2019-LA ===")
cfg = EXPECTED["asvspoof2019_LA"]
print(f"[{'OK ' if os.path.exists(cfg['protocol']) else 'FAIL'}] protocol file exists")
files = check_count("asvspoof2019_LA flac", cfg["flac_glob"], cfg["expect_files"])
check_decode("asvspoof2019_LA", files)

print("\n=== In-the-Wild ===")
cfg = EXPECTED["in_the_wild"]
files = check_count("in_the_wild wav", cfg["wav_glob"], cfg["expect_files"], recursive=True)
check_decode("in_the_wild", files)

print("\n=== Arabic ArAD ===")
cfg = EXPECTED["arabic_arad"]
files = check_count("arabic_arad wav", cfg["wav_glob"], cfg["expect_files"], recursive=True)
check_decode("arabic_arad", files)

print("\nDone. OK/OK/OK across all three -> safe to proceed to the manifest cell.")
