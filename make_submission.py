import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="submission.json")
    ap.add_argument("--out", default="submission_upload.json")
    ap.add_argument("--name", default="Beam search solution")
    ap.add_argument("--problem", default=None, help="mbishkruan 'problem' nga skedari hyres")
    ap.add_argument("--challenge", default=None, help="mbishkruan 'challenge' nga skedari hyres")
    args = ap.parse_args()

    with open(args.inp) as fh:
        data = json.load(fh)
    items = data if isinstance(data, list) else [data]

    out = []
    for it in items:
        x = [float(v) for v in it["decisionVector"]]
        desc = "Beam search"
        try:
            import udp as U

            f = U.udp.fitness(x)
            desc = f"Beam search, best fitness={float(f[0])}"
            if not (f[1] == 0 and max(f[2:]) <= 0):
                print("KUJDES: zgjidhja NUK eshte e vlefshme sipas U.udp.fitness!")
            else:
                print(f"Fitness (objektivi): {float(f[0]):.4f}, e vlefshme")
        except Exception as e:  # pa udp.py ose me gabim: vazhdo pa fitness ne pershkrim
            print(f"S'u llogarit fitness-i ({e}); pershkrimi pa vlere.")
        out.append(
            {
                "decisionVector": x,
                "problem": args.problem or it.get("problem", "mine-the-belt"),
                "challenge": args.challenge or it.get("challenge", "spoc-mining"),
                "name": args.name,
                "description": desc,
            }
        )

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=6)
    print(f"U shkrua {args.out} ({len(out)} zgjidhje)")


if __name__ == "__main__":
    main()
