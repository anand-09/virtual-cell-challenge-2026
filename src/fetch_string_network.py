import json
import time
import urllib.request
import pandas as pd

def fetch_all_string_partners():
    df_perts = pd.read_csv("data/pert_counts.csv")
    target_genes = df_perts["target_gene"].tolist()
    print(f"Total target genes: {len(target_genes)}")

    # Chunk into groups of 30 genes
    chunk_size = 30
    all_partners = {}

    for i in range(0, len(target_genes), chunk_size):
        chunk = target_genes[i:i + chunk_size]
        gene_param = "%0d".join(chunk)
        url = f"https://string-db.org/api/json/interaction_partners?identifiers={gene_param}&species=9606&limit=25&required_score=700"
        print(f"Querying chunk {i//chunk_size + 1} / {(len(target_genes) + chunk_size - 1)//chunk_size} ({len(chunk)} genes)...")
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VCC2026-Research)"})
        success = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                    for item in data:
                        src = str(item.get("preferredName_A"))
                        tgt = str(item.get("preferredName_B"))
                        score = float(item.get("score", 0))
                        if src not in all_partners:
                            all_partners[src] = []
                        all_partners[src].append({"partner": tgt, "score": score})
                    success = True
                    break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}. Retrying in 2s...")
                time.sleep(2)
        
        if not success:
            print(f"  Warning: failed to query chunk {chunk}")
        time.sleep(0.5)

    print(f"\nCompleted fetching. Unique query genes with partners: {len(all_partners)}")
    out_file = "data/string_interactions_300.json"
    with open(out_file, "w") as f:
        json.dump(all_partners, f, indent=2)
    print(f"Saved STRING interactions to {out_file}")

if __name__ == "__main__":
    fetch_all_string_partners()
