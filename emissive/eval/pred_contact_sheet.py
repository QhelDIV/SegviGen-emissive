"""
Contact sheet: ground truth beside the 72k model's prediction, all 11 paper shapes, same
camera and same treatment in both columns.

Empty predictions are labelled on the panel. A dark panel that says so is evidence; a dark
panel that says nothing is indistinguishable from a broken render, and 4 of these 11 are
empty.

  python code/pred_contact_sheet.py --gt .../final2 --pred .../pred/emis_72k_unfilt_method \
      --summary .../pred_voxels/emis_72k_unfilt/summary.json --out sheet.png
"""
import os
import json
import argparse

from PIL import Image, ImageDraw, ImageFont

TILE = 300
PAD = 8
LABEL_H = 30


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summ = json.load(open(args.summary))
    gal = {e["sid"]: e for e in json.load(open(args.gallery))}
    sids = sorted(summ, key=lambda s: summ[s]["gt_frac"])

    cols, rows = 2, len(sids)
    W = cols * TILE + (cols + 1) * PAD + 210
    H = rows * (TILE + LABEL_H) + PAD
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f = font(13)
    fb = font(15)
    fs = font(11)

    d.text((210 + PAD, 4), "ground truth (oracle mask)", font=fb, fill=(230, 230, 230))
    d.text((210 + PAD * 2 + TILE, 4), "72k model (predicted mask)", font=fb, fill=(230, 230, 230))

    for r, sid in enumerate(sids):
        y = PAD + LABEL_H + r * (TILE + LABEL_H)
        rec = summ[sid]
        pf = rec["pred_frac_by_thr"]["0.5"]
        what = gal.get(sid, {}).get("what", "")
        d.text((6, y + 6), what[:26], font=fb, fill=(235, 235, 235))
        d.text((6, y + 28), f"{sid[:8]}", font=fs, fill=(140, 140, 140))
        d.text((6, y + 48), f"GT coverage   {rec['gt_frac']:.4f}", font=f, fill=(190, 190, 190))
        d.text((6, y + 66), f"predicted     {pf:.4f}", font=f,
               fill=(235, 120, 110) if pf == 0 else (190, 190, 190))
        d.text((6, y + 84), f"IoU@0.5       {rec['iou_by_thr']['0.5']:.4f}", font=f,
               fill=(190, 190, 190))

        for c, root in enumerate((args.gt, args.pred)):
            x = 210 + PAD + c * (TILE + PAD)
            p = os.path.join(root, f"{sid}_glow.png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
                sheet.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + TILE, y + TILE], fill=(40, 30, 30))
                d.text((x + 10, y + TILE // 2), "MISSING", font=fb, fill=(230, 120, 110))
            if c == 1 and pf == 0.0:
                d.rectangle([x, y + TILE - 22, x + TILE, y + TILE], fill=(90, 25, 25))
                d.text((x + 6, y + TILE - 19),
                       "model predicted no emissive voxels", font=fs, fill=(255, 220, 215))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    n_empty = sum(1 for s in sids if summ[s]["pred_frac_by_thr"]["0.5"] == 0.0)
    print(f"WROTE {args.out}  ({len(sids)} shapes, {n_empty} empty predictions)")


if __name__ == "__main__":
    main()
