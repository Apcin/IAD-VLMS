"""
Verify the 15K (C1) SFT training corpus with GPT-5-mini (reference-guided, literature-updated rubric).
Reads combined_sft_c1_train.json directly; for anomalies sends original + RED GT-mask overlay,
for normals sends the original image only. Batches 15/request. Resumable (skip-existing).
Run:  /users/aacudad/miniconda3/envs/FINETUNE/bin/python verify_15k.py            # full
      ... verify_15k.py --limit 10                                                # smoke
"""
import os, io, re, json, time, base64, argparse, logging
from pathlib import Path
from PIL import Image
from openai import OpenAI

ROOT = Path("/bulk/aacudad/reasoning_traces")
ENV_PATH = ROOT / "repository_tu_delft_vlms/.env"
CORPUS = ROOT / "Training/15k_dataset_regenerated/combined_sft_train.json"  # the REAL 15K (14,472) the headline 15K SFT run used (iad_sft_15k_regen_combined)
OUT = ROOT / "outputs/verify_15k_regen_results"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.txt"
MODEL = "gpt-5-mini"
BATCH = 15

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(OUT/"run.log"), logging.StreamHandler()], force=True)
log = logging.getLogger()

VALID_ISSUES = ["missing_think_tags","missing_answer_tag","missing_location_tag","missing_type_tag",
                "extra_location_tag","extra_type_tag","has_batch_reference","illogical_reasoning",
                "incoherent_content","wrong_product_reference"]

SYS = f"""You are an expert verifier auditing industrial-defect reasoning traces for a TRAINING-DATA QUALITY FILTER. You grade each trace against the IMAGES, never against how fluent or confident it sounds.

For EACH trace you receive the ORIGINAL product image, then (for anomalies) an OVERLAY image with the ground-truth defect region in RED, then the reasoning text. Normal samples have no overlay.

Grade each CRITERION as MET/UNMET (judge on the image, not the prose), then assign a status.
FORMAT: F1 has <think>...</think>; F2 exactly one <answer>Yes/No</answer>; F3 Yes=>has <location>&<type>, No=>has neither.
CONTENT (vs the images): C1 correct product; C2 (anomaly) described defect matches what is visible in the RED region (not hallucinated/different); C3 (anomaly) <location> matches where the red region actually is; C4 visible cues described BEFORE naming the defect/verdict; C5 steps coherent and actually support the conclusion; C6 no batch/sequence words ("another/next/previous/final/tenth/across samples/in this batch") and no mention of labels/masks/overlay/ground-truth; C7 causes are tentative not asserted as fact; C8 (normal) inspects key regions and justifies "No" against absence of cues, not a generic "looks fine".
NOT criteria (never flag): trace length/verbosity, stylistic wording, exact location phrasing if right area, exact type naming if it matches what's visible.

A fluent, confident trace that names a defect not visible in the red region, or in the wrong place, MUST fail C2/C3. Leakage (C6), wrong product (C1), or illogical reasoning (C5) is AT MOST "wrong" even if otherwise excellent.

STATUS: completely_correct = all FORMAT + all applicable CONTENT met. correct = all FORMAT met, C1/C2/C3/C5/C6 met, at most a minor C4/C7/C8 slip. wrong = any FORMAT violation OR any of C1/C2/C3/C5/C6 UNMET. completely_wrong = multiple/severe failures.

ISSUE CODES (pick ONLY from this list; EMPTY for completely_correct/correct):
{json.dumps(VALID_ISSUES)}

Return strict JSON: {{"verifications":[{{"image_id":"exact_id","issues":[...],"issue_elaboration":["Image shows X but reasoning says Y", ...],"status":"completely_correct|correct|wrong|completely_wrong","confidence_score":0.0-1.0}}]}}. Write issues+elaboration FIRST then status. One entry per trace. confidence is an audit signal only."""

def load_key():
    for ln in ENV_PATH.read_text().splitlines():
        if ln.strip().startswith("OPENAI_API_KEY"):
            return ln.split("=",1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key")

def b64(im):
    b=io.BytesIO(); im.convert("RGB").save(b,format="JPEG",quality=85); return base64.b64encode(b.getvalue()).decode()

def overlay(img_path, mask_path):
    base=Image.open(img_path).convert("RGBA"); m=Image.open(mask_path).convert("L")
    if m.size!=base.size: m=m.resize(base.size, Image.NEAREST)
    red=Image.new("RGBA",base.size,(0,0,0,0)); px=red.load(); mp=m.load()
    for x in range(base.width):
        for y in range(base.height):
            if mp[x,y]>127: px[x,y]=(255,0,0,128)
    return Image.alpha_composite(base,red).convert("RGB")

def load_items():
    d=json.load(open(CORPUS)); items=[]
    for x in d:
        img=x["images"][0]
        asst=[m for m in x["messages"] if m["role"]=="assistant"][0]["content"]
        iid=Path(img).stem
        items.append({"image_id":iid,"image":img,"reasoning":asst,
                      "mask":img.rsplit(".",1)[0]+".png"})
    return items

def verify_batch(client, batch):
    content=[{"type":"input_text","text":f"Verify these {len(batch)} traces. For each: original image, (anomaly) red-overlay image, then the reasoning."}]
    for i,it in enumerate(batch,1):
        content.append({"type":"input_text","text":f"\n--- TRACE {i} ---\nID: {it['image_id']}"})
        try:
            if os.path.exists(it["image"]):
                content.append({"type":"input_image","image_url":f"data:image/jpeg;base64,{b64(Image.open(it['image']))}"})
                if os.path.exists(it["mask"]):
                    content.append({"type":"input_image","image_url":f"data:image/jpeg;base64,{b64(overlay(it['image'],it['mask']))}"})
        except Exception as e:
            content.append({"type":"input_text","text":"[image load failed]"})
        content.append({"type":"input_text","text":"\nReasoning:\n"+it["reasoning"]})
    r=client.responses.create(model=MODEL,
        input=[{"role":"system","content":SYS},{"role":"user","content":content}],
        reasoning={"effort":"low"}, text={"format":{"type":"json_object"}})
    txt=None
    for o in r.output:
        if getattr(o,"type",None)=="message":
            for c in o.content:
                if getattr(c,"type",None)=="output_text": txt=c.text
    js=json.loads(txt)
    if isinstance(js,list): js={"verifications":js}
    out={}
    for v in js.get("verifications",[]):
        out[v["image_id"]]=v
    return out, r.usage

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=None); a=ap.parse_args()
    client=OpenAI(api_key=load_key())
    items=load_items()
    if a.limit: items=items[:a.limit]
    # resume: count a trace done only if it has a NON-error verdict, so errors are retried on re-run
    done=set()
    for p in OUT.glob("verify_*.json"):
        try:
            if json.load(open(p)).get("status")!="error": done.add(p.stem.replace("verify_",""))
        except Exception: pass
    todo=[it for it in items if it["image_id"] not in done]
    log.info(f"corpus={len(items)} done={len(done)} todo={len(todo)} model={MODEL} batch={BATCH}")
    tin=tout=0; counts={}
    batches=[todo[i:i+BATCH] for i in range(0,len(todo),BATCH)]
    for bi,batch in enumerate(batches):
        try:
            res=u=None
            for attempt in range(3):
                try:
                    res,u=verify_batch(client,batch); break
                except Exception as ie:
                    if attempt==2: raise
                    log.info(f"  batch {bi} attempt {attempt+1} retry: {str(ie)[:120]}"); time.sleep(8*(attempt+1))
            tin+=u.input_tokens; tout+=u.output_tokens
            for it in batch:
                v=res.get(it["image_id"], {"image_id":it["image_id"],"status":"error","issues":[],"issue_elaboration":[],"confidence_score":0.0})
                v["has_overlay"]=os.path.exists(it["mask"])
                json.dump(v, open(OUT/f"verify_{it['image_id']}.json","w"), indent=1)
                counts[v.get("status","error")]=counts.get(v.get("status","error"),0)+1
        except Exception as e:
            log.error(f"batch {bi} failed: {str(e)[:200]}")
            for it in batch:
                json.dump({"image_id":it["image_id"],"status":"error","issues":[],"issue_elaboration":[],"confidence_score":0.0,"has_overlay":os.path.exists(it["mask"])}, open(OUT/f"verify_{it['image_id']}.json","w"))
                counts["error"]=counts.get("error",0)+1
        ndone=len(done)+sum(len(b) for b in batches[:bi+1])
        cost=tin/1e6*0.25+tout/1e6*2.0
        msg=f"batch {bi+1}/{len(batches)} done~{ndone}/{len(items)} tok(in={tin},out={tout}) ~${cost:.2f} {counts}"
        PROGRESS.write_text(msg)
        log.info(msg)  # log EVERY batch for crash-recovery visibility
        if bi%10==0:    # flush a consolidated summary every ~10 batches (~150 traces)
            json.dump({"counts":counts,"tokens":{"in":tin,"out":tout},"batches_done":bi+1,"of":len(batches)}, open(OUT/"summary.json","w"), indent=2)
        time.sleep(0.4)
    keep=counts.get("completely_correct",0)+counts.get("correct",0)
    tot=sum(counts.values()) or 1
    log.info(f"DONE counts={counts} KEEP(cc+c)={keep} ({100*keep/tot:.1f}%) cost~${tin/1e6*0.25+tout/1e6*2.0:.2f}")
    json.dump({"counts":counts,"tokens":{"in":tin,"out":tout}}, open(OUT/"summary.json","w"), indent=2)
    PROGRESS.write_text(f"COMPLETE counts={counts} keep={keep}/{tot}")

if __name__=="__main__": main()
