#!/usr/bin/env python
"""§6.8 explainability/faithfulness judge, extended to SIX models (for Amir's 4-model + qwen3 request).

Scores each model INDEPENDENTLY on its own correctly-detected anomalies (gt=yes & pred=yes), so the
weak base models are not penalised by a shrinking cross-model intersection. Uniform methodology across
all models => the table is internally consistent. Same Gemini-3-Flash 5-axis judge + Location(3x3 IoU)
+ Type(Nomic) as the thesis §6.8. Median over --ksamples judge calls to damp the fluctuation Amir noted.

Models: base Qwen2.5-VL, IAD-R1, SFT+GRPO, final SFT (Arm-C), base Qwen3-VL, Qwen3-VL SFT.
(base Qwen2.5-VL has no VisA traces -> DS-MVTec only.)
No GPU for the LLM (uses shipped traces + the Gemini API); Nomic runs on CPU.
Run: python explainability_judge_multi.py [--n 100] [--ksamples 3]
"""
import os, io, re, json, time, argparse, collections, random
from pathlib import Path
from PIL import Image
import numpy as np

# ROOT is the workspace holding outputs/, MMAD_repo/ and reasoning_traces_gen/. It defaults
# to the parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
# The MMAD images themselves are not redistributed here (see README section 3).
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("WORK_DIR") or REPO_ROOT.parent)
MMAD_JSON = ROOT / "MMAD_repo/dataset/MMAD/mmad.json"
MMAD_IMG  = ROOT / "reasoning_traces_gen/data/MMAD"
OUT = ROOT / "outputs/explainability_multi"; OUT.mkdir(parents=True, exist_ok=True)
# Vertex credentials and project come from the environment. Nothing secret lives in this file.
KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
GMODEL = os.environ.get("EXPLAIN_JUDGE_MODEL", "gemini-3-flash-preview")

# model -> {benchmark: eval-json (relative to ROOT)}
RUNS = {
 "base_qwen25": {"DS-MVTec": "outputs/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json",
                 "VisA":     "outputs/qwen25vl_baseline_eval/eval_visa_full_trainprompt.json"},
 # IAD-R1 under its NATIVE grpoprompt: that prompt never asks for reasoning, so ~35-40% of its
 # answers are a bare "Yes" (score ~0) -> this measures explanation RELIABILITY under its own prompt.
 "iadr1":       {"DS-MVTec": "outputs/iad_r1_qwen_recanon/eval_dsmvtec_full_iadr1native.json",
                 "VisA":     "outputs/iad_r1_qwen_recanon/eval_visa_full_iadr1native.json"},
 # IAD-R1 PROMPT-MATCHED to our models (trainprompt asks for detailed reasoning): 100% of its answers
 # contain a trace -> this isolates explanation QUALITY. Report both; they answer different questions.
 "iadr1_trainprompt": {"DS-MVTec": "outputs/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json",
                       "VisA":     "outputs/iad_r1_qwen_recanon/eval_visa_full_trainprompt.json"},
 "sft_grpo":    {"DS-MVTec": "outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json",
                 "VisA":     "outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_visa_full_trainprompt.json"},
 "armC_finalsft": {"DS-MVTec": "outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json",
                   "VisA":     "outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_visa_full_trainprompt.json"},
 "qwen3_base":  {"DS-MVTec": "outputs/qwen3vl_8b_baseline_eval/eval_dsmvtec_full_trainprompt.json",
                 "VisA":     "outputs/qwen3vl_8b_baseline_eval/eval_visa_full_trainprompt.json"},
 "qwen3_sft":   {"DS-MVTec": "outputs/sft_qwen3vl_8b_armC/checkpoint-376/eval_dsmvtec_full_trainprompt.json",
                 "VisA":     "outputs/sft_qwen3vl_8b_armC/checkpoint-376/eval_visa_full_trainprompt.json"},
 # --- appended 2026-09-02 (LLaVA-OneVision cross-architecture rows). Appended at the END on purpose:
 # the shared rng is consumed in RUNS order, so adding keys here leaves the draw of every pre-existing
 # model unchanged in a full run. Eval JSONs are the vLLM path but the schema is byte-identical
 # (prompt_info/metrics/results, rows with gt_answer/pred_answer/pred_full/pred_tags/absolute_path),
 # so no adapter is needed. NOTE: any LLaVA-OV DS-MVTec number carries the vision_flan MVTec
 # contamination caveat.
 "llava_armC":  {"DS-MVTec": "outputs/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/eval_dsmvtec_full_trainprompt_vllm.json",
                 "VisA":     "outputs/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/eval_visa_full_trainprompt_vllm.json"},
 "llava_grpo":  {"DS-MVTec": "outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_dsmvtec_full_trainprompt_vllm.json",
                 "VisA":     "outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_visa_full_trainprompt_vllm.json"},
 # LLaVA SFT only, the 6K Gemini corpus, epoch 1 (85.91/68.26). Completes the three-stage
 # LLaVA ladder: SFT -> SFT+GRPO -> SFT+GRPO+KCR. Appended at the END for the same rng reason.
 "llava_sft":   {"DS-MVTec": "outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/eval_dsmvtec_full_trainprompt.json",
                 "VisA":     "outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/eval_visa_full_trainprompt.json"},
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
def mmad_key(abspath):
    s=str(abspath); i=s.find("/MMAD/"); return s[i+6:] if i>=0 else None
def _fg_mask(mask_path, size=None):
    im=Image.open(mask_path).convert("RGB")
    if size and im.size!=size: im=im.resize(size, Image.NEAREST)
    return (np.array(im).sum(2) > 30)
def overlay(img_path, mask_path):
    base=Image.open(img_path).convert("RGBA"); fg=_fg_mask(mask_path, base.size)
    red=np.zeros((base.size[1],base.size[0],4),dtype=np.uint8); red[fg]=(255,0,0,128)
    return Image.alpha_composite(base, Image.fromarray(red,"RGBA")).convert("RGB")
def gt_cells(mask_path):
    fg=_fg_mask(mask_path); H,W=fg.shape; tot=int(fg.sum()); cells=set()
    if tot==0: return cells
    for r in range(3):
        for c in range(3):
            if int(fg[r*H//3:(r+1)*H//3, c*W//3:(c+1)*W//3].sum())/tot>0.05: cells.add((r,c))
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

def correct_anom(rows):
    return {r["image_id"]:r for r in rows
            if r.get("gt_answer")=="yes" and r.get("pred_answer")=="yes" and r.get("pred_full")}

def diverse(idmap, n, rng):
    byprod=collections.defaultdict(list)
    for iid,r in idmap.items(): byprod[r.get("product","?")].append(iid)
    for v in byprod.values(): rng.shuffle(v)
    order=sorted(byprod, key=lambda p:-len(byprod[p])); out=[]
    while len(out)<min(n,len(idmap)) and any(byprod.values()):
        for p in order:
            if byprod[p]: out.append(byprod[p].pop())
            if len(out)>=min(n,len(idmap)): break
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100); ap.add_argument("--ksamples",type=int,default=3)
    ap.add_argument("--only",type=str,default="",help="comma-separated model keys to run (default: all)")
    ap.add_argument("--tag",type=str,default="",help="suffix for output files, so a partial run does not clobber the full one")
    ap.add_argument("--retries",type=int,default=6,help="attempts per judge call before that call is given up")
    ap.add_argument("--pause",type=float,default=0.0,help="seconds to sleep after each judge call (be polite to the API)")
    a=ap.parse_args()
    sel=[s.strip() for s in a.only.split(",") if s.strip()]
    raw_path=OUT/f"raw_results{('_'+a.tag) if a.tag else ''}.json"
    sum_path=OUT/f"summary{('_'+a.tag) if a.tag else ''}.json"
    if KEY:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", KEY)
    if not PROJECT:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS) for the Vertex judge")
    from google import genai
    from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig
    client=genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    mmad=json.load(open(MMAD_JSON))
    nomic=None
    try:
        from sentence_transformers import SentenceTransformer
        nomic=SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
        print("Nomic on CPU for type similarity", flush=True)
    except Exception as e:
        print(f"Nomic unavailable ({str(e)[:60]}) -> token match only", flush=True)

    def type_sim(pred, gt):
        pred=(pred or "").strip().lower(); gt=(gt or "").strip().lower().replace("_"," ")
        if not pred: return 0.0
        if pred==gt or pred in gt or gt in pred: return 1.0
        if nomic is not None:
            e=nomic.encode([f"search_query: {pred}", f"search_query: {gt}"], normalize_embeddings=True)
            return float(np.dot(e[0],e[1]))
        ov=set(pred.split())&set(gt.split()); return 0.5 if ov else 0.0

    def judge(img, ov, gt_type, trace):
        scores=[]
        for _ in range(a.ksamples):
            # bounded retry with exponential backoff (added 2026-09-02). Before, a single transient
            # 429/500 dropped that judge call silently and turned median-of-3 into median-of-2, which
            # biases the median upward. On a clean call this is byte-identical to the old behaviour.
            for attempt in range(a.retries):
                try:
                    r=client.models.generate_content(model=GMODEL,
                        contents=[Content(role="user", parts=[
                            Part.from_text(text=f"Ground-truth defect category (context): {gt_type}\nORIGINAL image:"),
                            b64part(img,Part), Part.from_text(text="OVERLAY (true defect in RED):"), b64part(ov,Part),
                            Part.from_text(text="Model reasoning trace to score:\n"+trace)])],
                        config=GenerateContentConfig(system_instruction=[Part.from_text(text=JUDGE_SYS)],
                            thinking_config=ThinkingConfig(thinking_budget=512), response_mime_type="application/json"))
                    scores.append(json.loads(r.text)); break
                except Exception as e:
                    print(f"   judge retry: {str(e)[:80]}", flush=True)
                    time.sleep(min(3*(2**attempt), 90))
            time.sleep(a.pause)
        if not scores: return None
        axes=["visual_grounding","defect_faithfulness","evidence_before_conclusion","coherence","conciseness"]
        med=lambda xs: sorted(xs)[len(xs)//2]
        out={ax: med([s.get(ax,0) for s in scores]) for ax in axes}
        out["overall"]=sum(out[ax] for ax in axes); return out

    rng=random.Random(13); results=[]
    for model, benches in RUNS.items():
        if sel and model not in sel: continue
        for bench, rel in benches.items():
            fp=ROOT/rel
            if not fp.exists(): print(f"[{model}/{bench}] MISSING {rel}", flush=True); continue
            ca=correct_anom(json.load(open(fp)).get("results",[]))
            picked=diverse(ca, a.n, rng)
            print(f"[{model}/{bench}] {len(picked)} correct-anomaly traces over {len({ca[i]['product'] for i in picked})} products", flush=True)
            for idx,iid in enumerate(picked):
                r0=ca[iid]; key=mmad_key(r0.get("absolute_path",""))
                entry=mmad.get(key) if key else None
                if not entry: continue
                base=MMAD_IMG/key.split("/")[0]/key.split("/")[1]
                mask=base/entry.get("mask_path",""); img=Path(r0["absolute_path"])
                if not (img.exists() and entry.get("mask_path") and mask.exists()): continue
                gtype=gt_type_from_mmad(entry, key.split("/")[3] if len(key.split("/"))>3 else "")
                try:
                    ov=overlay(img,mask); im=Image.open(img).convert("RGB"); gcell=gt_cells(mask)
                except Exception as e:
                    print(f"   img err {iid}: {str(e)[:60]}", flush=True); continue
                pt=r0.get("pred_tags",{}) or {}; trace=r0.get("pred_full","")
                j=judge(im,ov,gtype,trace)
                if j is None: continue
                pc=pred_cells(pt.get("location","")); loc_met=1 if (pc and gcell and pc&gcell) else 0
                tsim=type_sim(pt.get("type",""), gtype)
                results.append({"model":model,"bench":bench,"image_id":iid,"product":r0.get("product"),
                    "gt_defect":gtype,"loc_met":loc_met,"type_sim":tsim,**{f"j_{k}":v for k,v in j.items()}})
                if (idx+1)%20==0:
                    print(f"   {model}/{bench} {idx+1}/{len(picked)}", flush=True)
                    json.dump(results,open(raw_path,"w"),indent=1)
    json.dump(results,open(raw_path,"w"),indent=1)

    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        for k in ["j_overall","j_visual_grounding","j_defect_faithfulness","j_evidence_before_conclusion","j_coherence","j_conciseness","loc_met","type_sim"]:
            agg[(r["model"],r["bench"])][k].append(r[k])
    summary={f"{m}|{b}":{k:round(sum(v)/len(v),3) for k,v in d.items()}|{"n":len(d["j_overall"])} for (m,b),d in agg.items()}
    json.dump(summary,open(sum_path,"w"),indent=2)
    print("\n=== EXPLAINABILITY SUMMARY (mean; overall out of 10) ===", flush=True)
    print(f"  {'model/bench':28s} {'overall':>7} {'ground':>7} {'faith':>6} {'loc':>5} {'type':>5}  n", flush=True)
    for k in sorted(summary):
        s=summary[k]; print(f"  {k:28s} {s['j_overall']:7.2f} {s['j_visual_grounding']:7.2f} {s['j_defect_faithfulness']:6.2f} {s['loc_met']:5.2f} {s['type_sim']:5.2f}  {s['n']}", flush=True)

if __name__=="__main__": main()
