"""
Explainability/faithfulness eval: best IAD-R1 vs best Arm-C (82.8) on MATCHED correct-anomaly
images (both models correct), 100 product-diverse per benchmark (DS-MVTec, VisA). A Gemini-3-Flash
judge scores REASONING quality (5 axes, blind to which model). Location (3x3 IoU vs GT mask) and
Type (Nomic similarity vs GT defect class, DS-MVTec only) are scored SEPARATELY as their own points.
Run: python scripts/explainability_judge.py [--n 100] [--smoke 6] [--ksamples 1]
"""
import os, io, re, json, time, argparse, collections, random
from pathlib import Path
from PIL import Image
import numpy as np

# ROOT is the workspace holding outputs/, MMAD_repo/ and reasoning_traces_gen/. It defaults
# to the parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WORK_DIR") or REPO_ROOT.parent)
RES = REPO_ROOT / "results"
MMAD_JSON = ROOT / "MMAD_repo/dataset/MMAD/mmad.json"
MMAD_IMG = ROOT / "reasoning_traces_gen/data/MMAD"
OUT = ROOT / "outputs/explainability_judge"; OUT.mkdir(parents=True, exist_ok=True)
# Vertex credentials and project come from the environment. Nothing secret lives in this file.
KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
GMODEL = os.environ.get("EXPLAIN_JUDGE_MODEL", "gemini-3-flash-preview")

RUNS = {  # (model_label, benchmark) -> eval json  [best run of each]
 ("iadr1","DS-MVTec"): "iad_r1_qwen_recanon/eval_dsmvtec_full_iadr1native.json",
 ("iadr1","VisA"):     "iad_r1_qwen_recanon/eval_visa_full_iadr1native.json",
 ("armC","DS-MVTec"):  "sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json",
 ("armC","VisA"):      "sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_visa_full_trainprompt.json",
}

JUDGE_SYS = """You are an expert evaluator of INDUSTRIAL-DEFECT INSPECTION EXPLANATIONS. A vision-language model inspected a product and wrote a reasoning trace ending in a defect verdict. Your job is to score the QUALITY OF ITS REASONING/EXPLANATION -- how well the written reasoning is grounded in and faithful to the actual visual evidence -- NOT whether the final yes/no is right (it is already known to be correct here), and NOT the exact <type>/<location> wording (scored separately).

You are given: (1) the ORIGINAL product image; (2) an OVERLAY with the TRUE defect region in RED; (3) the ground-truth defect category (context); (4) the model's reasoning trace. Judge the trace against WHAT IS ACTUALLY VISIBLE -- never reward fluent, confident, or long writing that is not supported by the image. A convincing description of a defect that is NOT in the red region must score low on faithfulness.

Score each axis 0, 1, or 2 (2=fully, 1=partial, 0=no), writing a one-line justification FIRST:
  A. VISUAL GROUNDING -- claims point to specific, actually-visible regions/cues (vs generic statements that could fit any image).
  B. DEFECT FAITHFULNESS -- the anomaly the trace describes corresponds to the REAL defect shown in the red region (same kind of flaw, same area); it does not hallucinate, invent, or describe a different/absent defect.
  C. EVIDENCE-BEFORE-CONCLUSION -- it describes the visual cues first and then concludes, rather than asserting the defect up front and back-filling.
  D. COHERENCE -- the steps are internally consistent and logically support the conclusion (no contradictions, no non-sequiturs).
  E. CONCISENESS / NO-FILLER -- compact and on-point; no padded, repetitive, or template scan narration. (Length is NEVER a positive signal.)

Return STRICT JSON only:
{"justification":"<=2 sentences citing the image","visual_grounding":0-2,"defect_faithfulness":0-2,"evidence_before_conclusion":0-2,"coherence":0-2,"conciseness":0-2,"overall":0-10}
where overall is the sum of the five axes. Judge on the image, not on eloquence."""

CELLS = {"top-left":(0,0),"top-center":(0,1),"top-right":(0,2),"middle-left":(1,0),"center":(1,1),
         "middle-right":(1,2),"bottom-left":(2,0),"bottom-center":(2,1),"bottom-right":(2,2),
         # aliases the model emits but are off-vocab
         "center-right":(1,2),"center-left":(1,0),"centre":(1,1),"middle-center":(1,1),"center-center":(1,1),
         "top":(0,1),"bottom":(2,1),"left":(1,0),"right":(1,2)}

def gt_type_from_mmad(entry, fallback):
    for q in entry.get("conversation",[]):
        if "type of the defect" in q.get("Question","").lower():
            ans=q.get("Answer",""); opt=q.get("Options",{})
            v=opt.get(ans, ans) if isinstance(opt,dict) else ans
            v=str(v).strip().rstrip(".").strip()
            if v and v.lower() not in ("good","none","no defect"): return v
    return fallback.replace("_"," ")

def load(p): return json.load(open(RES/p))["results"]
def correct_anom(rows): return {r["image_id"]:r for r in rows if r.get("gt_answer")=="yes" and r.get("pred_answer")=="yes"}

def mmad_key(abspath):  # /bulk/.../MMAD/DS-MVTec/tile/image/glue_strip/013.png -> DS-MVTec/tile/image/glue_strip/013.png
    s=str(abspath); i=s.find("/MMAD/"); return s[i+6:] if i>=0 else None

def _fg_mask(mask_path, size=None):
    # foreground = any NON-BLACK pixel (handles white L masks AND colored DS-MVTec rbg_masks)
    im=Image.open(mask_path).convert("RGB")
    if size and im.size!=size: im=im.resize(size, Image.NEAREST)
    return (np.array(im).sum(2) > 30)  # HxW bool

def overlay(img_path, mask_path):
    base=Image.open(img_path).convert("RGBA"); fg=_fg_mask(mask_path, base.size)
    red=np.zeros((base.size[1],base.size[0],4),dtype=np.uint8); red[fg]=(255,0,0,128)
    return Image.alpha_composite(base, Image.fromarray(red,"RGBA")).convert("RGB")

def gt_cells(mask_path):
    fg=_fg_mask(mask_path); H,W=fg.shape; tot=int(fg.sum()); cells=set()
    if tot==0: return cells
    for r in range(3):
        for c in range(3):
            frac=int(fg[r*H//3:(r+1)*H//3, c*W//3:(c+1)*W//3].sum())/tot
            if frac>0.05: cells.add((r,c))   # cell holds >5% of the defect (scale-invariant; works for tiny VisA defects)
    return cells

def pred_cells(loc):
    out=set()
    for tok in re.split(r"[,/;]| and ", (loc or "").lower()):
        tok=tok.strip()
        if tok in CELLS: out.add(CELLS[tok])
    return out

def b64part(im, Part):
    bio=io.BytesIO(); im.convert("RGB").save(bio,format="JPEG",quality=88)
    return Part.from_bytes(data=bio.getvalue(), mime_type="image/jpeg")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100); ap.add_argument("--smoke",type=int,default=0); ap.add_argument("--ksamples",type=int,default=1); a=ap.parse_args()
    if KEY:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", KEY)
    if not PROJECT:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS) for the Vertex judge")
    from google import genai
    from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig
    client=genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    mmad=json.load(open(MMAD_JSON))
    # optional Nomic for type similarity
    nomic=None
    try:
        from sentence_transformers import SentenceTransformer
        nomic=SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        print("Nomic loaded for type similarity")
    except Exception as e:
        print(f"Nomic unavailable ({str(e)[:80]}) -> type uses exact+token match only")

    def type_sim(pred, gt):
        pred=(pred or "").strip().lower(); gt=(gt or "").strip().lower().replace("_"," ")
        if not pred: return 0.0
        if pred==gt or pred in gt or gt in pred: return 1.0
        if nomic is not None:
            import numpy as np
            e=nomic.encode([f"search_query: {pred}", f"search_query: {gt}"], normalize_embeddings=True)
            return float(np.dot(e[0],e[1]))
        ov=set(pred.split())&set(gt.split()); return 0.5 if ov else 0.0

    def judge(img, ov, gt_type, trace):
        scores=[]
        for k in range(a.ksamples):
            try:
                r=client.models.generate_content(model=GMODEL,
                    contents=[Content(role="user", parts=[
                        Part.from_text(text=f"Ground-truth defect category (context): {gt_type}\nORIGINAL image:"),
                        b64part(img,Part), Part.from_text(text="OVERLAY (true defect in RED):"), b64part(ov,Part),
                        Part.from_text(text="Model reasoning trace to score:\n"+trace)])],
                    config=GenerateContentConfig(system_instruction=[Part.from_text(text=JUDGE_SYS)],
                        thinking_config=ThinkingConfig(thinking_budget=512), response_mime_type="application/json"))
                scores.append(json.loads(r.text))
            except Exception as e:
                print(f"   judge retry: {str(e)[:90]}"); time.sleep(3)
        if not scores: return None
        axes=["visual_grounding","defect_faithfulness","evidence_before_conclusion","coherence","conciseness"]
        med=lambda xs: sorted(xs)[len(xs)//2]
        out={ax: med([s.get(ax,0) for s in scores]) for ax in axes}
        out["overall"]=sum(out[ax] for ax in axes); out["justification"]=scores[0].get("justification","")
        return out

    benches=["DS-MVTec","VisA"]
    results=[]; n=a.smoke or a.n
    for bench in benches:
        iad=correct_anom(load(RUNS[("iadr1",bench)])); arm=correct_anom(load(RUNS[("armC",bench)]))
        inter=sorted(set(iad)&set(arm))
        # round-robin by product for diversity
        byprod=collections.defaultdict(list)
        for iid in inter: byprod[iad[iid]["product"]].append(iid)
        rng=random.Random(13); [rng.shuffle(v) for v in byprod.values()]
        order=sorted(byprod, key=lambda p:-len(byprod[p])); picked=[]
        while len(picked)<min(n,len(inter)):
            for p in order:
                if byprod[p]: picked.append(byprod[p].pop())
                if len(picked)>=min(n,len(inter)): break
        print(f"[{bench}] sampling {len(picked)} matched images across {len({iad[i]['product'] for i in picked})} products")
        for idx,iid in enumerate(picked):
            r0=iad[iid]; key=mmad_key(r0["absolute_path"])
            if not key: continue
            entry=mmad.get(key);
            if not entry: continue
            base=MMAD_IMG/key.split("/")[0]/key.split("/")[1]
            mask=base/entry.get("mask_path",""); img=Path(r0["absolute_path"])
            if not (img.exists() and mask.exists()): continue
            gtype=gt_type_from_mmad(entry, key.split("/")[3])  # GT defect type from MMAD MCQ (fallback subfolder)
            ov=overlay(img,mask); im=Image.open(img).convert("RGB"); gcell=gt_cells(mask)
            for model,src in [("iadr1",iad[iid]),("armC",arm[iid])]:
                pt=src.get("pred_tags",{}) or {}
                trace=src.get("pred_full","")
                j=judge(im,ov,gtype,trace)
                if j is None: continue
                pc=pred_cells(pt.get("location",""))
                loc_met=1 if (pc and gcell and pc&gcell) else 0
                tsim=type_sim(pt.get("type",""), gtype)   # both benchmarks now use the MCQ GT type
                results.append({"bench":bench,"model":model,"image_id":iid,"product":r0["product"],
                    "gt_defect":gtype,"pred_type":pt.get("type"),"pred_location":pt.get("location"),
                    "loc_met":loc_met,"type_sim":tsim,**{f"j_{k}":v for k,v in j.items()}})
            if (idx+1)%10==0:
                print(f"   {bench} {idx+1}/{len(picked)} done"); json.dump(results,open(OUT/"raw_results.json","w"),indent=1)
    json.dump(results,open(OUT/"raw_results.json","w"),indent=1)
    # aggregate
    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        for k in ["j_overall","j_visual_grounding","j_defect_faithfulness","j_evidence_before_conclusion","j_coherence","j_conciseness","loc_met"]:
            agg[(r["bench"],r["model"])][k].append(r[k])
        if r["type_sim"] is not None: agg[(r["bench"],r["model"])]["type_sim"].append(r["type_sim"])
    summary={}
    for (bench,model),d in agg.items():
        summary[f"{model}|{bench}"]={k:round(sum(v)/len(v),3) for k,v in d.items()} | {"n":len(d["j_overall"])}
    json.dump(summary,open(OUT/"summary.json","w"),indent=2)
    print("\n=== SUMMARY (mean per model x benchmark) ===")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
