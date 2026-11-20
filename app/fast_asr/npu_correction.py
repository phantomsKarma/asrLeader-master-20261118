"""Process-local Qwen correction override. Install on a loaded TranscriptPostProcessor."""
import copy
import difflib
import functools
import hashlib
import inspect
import json
import math
import re
import threading
import time
from contextlib import contextmanager

import torch


PROTECTED = set("〇零一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億")
NAMED = re.compile(r"[\u4e00-\u9fff]{2,4}(?=同志|书记|局长|科长|处长|主任|老师|先生|女士|经理)|"
                   r"(?:姓名|联系人|负责人|主持人)(?:是|为|叫|[：:])\s*([\u4e00-\u9fff]{2,4})")


def name_ranges(text):
    return [m.span(1) if m.lastindex else m.span() for m in NAMED.finditer(text)]


def project(source, proposed):
    if source == proposed:
        return source
    replacements = [(a,b,i,j) for op,a,b,i,j in difflib.SequenceMatcher(
        None,source,proposed,autojunk=False).get_opcodes() if op=="replace" and b==a+1 and j==i+1]
    if len(replacements) != 1:
        return source
    a,b,i,j = replacements[0]
    old,new = source[a:b],proposed[i:j]
    if not ("\u4e00" <= old <= "\u9fff" and "\u4e00" <= new <= "\u9fff") or {old,new} & PROTECTED:
        return source
    if any(start <= a < end for start,end in name_ranges(source)):
        return source
    return source[:a]+new+source[b:]


def payload(texts):
    return [{"id":i,"text":text} for i,text in enumerate(texts)]


def pack(tokenizer, texts):
    """Partition in input order using actual tokenizer output cost, preserving singleton outliers."""
    def estimate(group):
        return len(tokenizer.encode(json.dumps(payload(group),ensure_ascii=False),add_special_tokens=False))
    groups, lengths = [], []
    group, length = [], 0
    for text in texts:
        trial = group+[text]
        size = estimate(trial)
        if group and size > 62:
            groups.append(group)
            lengths.append(length)
            group, length = [text], estimate([text])
        else:
            group, length = trial, size
    if group:
        groups.append(group)
        lengths.append(length)
    return groups, [min(512,max(144,math.ceil(1.25*n)+16)) for n in lengths]


def sequence_hash(texts):
    return hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()


class Override:
    def __init__(self, post, require_npu):
        if getattr(post,"worker_count",1)!=1 or not post.enabled or not post.load():
            raise RuntimeError("one loaded and enabled production Qwen processor is required")
        self.post = post
        self.model = post._model
        self.tokenizer = post._tokenizer
        self.device = self.model.device
        if require_npu and str(self.device).split(":")[0] != "npu":
            raise RuntimeError("expected Qwen weights on an Ascend NPU")
        self.original = post.correct_segments
        self.lock = threading.RLock()
        self.stats_lock = threading.Lock()
        self.latest = {"status":"ready","device":str(self.device)}
        self.enabled = True
        self.calls = 0
        parameters = inspect.signature(self.model.forward).parameters
        self.keep_keyword = next((k for k in ("logits_to_keep","num_logits_to_keep") if k in parameters),None)
        self.option_ids = [self.tokenizer.encode(x,add_special_tokens=False) for x in ("A","B")]
        if any(len(ids)!=1 for ids in self.option_ids):
            raise RuntimeError("the verifier requires single-token A/B decisions")
        self.option_ids = [ids[0] for ids in self.option_ids]

        @functools.wraps(self.original)
        def wrapped(segments):
            with self.lock:
                if not self.enabled:
                    return self.original(segments)
                return self._call(segments)
        self.wrapped = wrapped
        post.correct_segments = wrapped

    def enable(self, value=True):
        with self.lock:
            self.enabled = bool(value)

    def snapshot(self):
        with self.stats_lock:
            return copy.deepcopy(self.latest)

    def _publish(self, stats):
        with self.stats_lock:
            self.latest = copy.deepcopy(stats)

    def uninstall(self):
        with self.lock:
            if self.post.correct_segments is self.wrapped:
                self.post.correct_segments = self.original
            self.enabled = False

    @contextmanager
    def _stage(self, stats, name):
        phase = dict(name=name,start_epoch=time.time())
        stats["stages"].append(phase)
        self._publish(stats)
        try:
            yield
        finally:
            phase["end_epoch"] = time.time()

    def _verify(self, pairs):
        prompts = []
        t = self.tokenizer
        for src,proposed in pairs:
            for a,b in ((src,proposed),(proposed,src)):
                msg=[{"role":"system","content":"你是中文会议转写校对员。判断哪一句合乎上下文并且用字正确。除了明显的错别字，应保留原始句。只回答 A 或 B。"},
                     {"role":"user","content":f"原始句：{src}\nA：{a}\nB：{b}\n正确选项是"}]
                prompts.append(t.apply_chat_template(msg,tokenize=False,add_generation_prompt=True))
        batch = t(prompts,return_tensors="pt",padding=True).to(self.device)
        pos = batch.attention_mask.long().cumsum(-1)-1
        pos.masked_fill_(batch.attention_mask==0,0)
        kw = dict(position_ids=pos,use_cache=False,return_dict=True)
        if self.keep_keyword:
            logits = self.model(**batch,**kw,**{self.keep_keyword:1}).logits[:,-1,:]
        elif getattr(self.model.config,"model_type",None) == "qwen2":
            # In Transformers 4.44 Qwen2ForCausalLM projects all positions. Project only
            # the last hidden state with its actual output head, avoiding a full vocabulary tensor.
            hidden = self.model.model(**batch,**kw).last_hidden_state[:,-1:,:]
            logits = self.model.lm_head(hidden)[:,-1,:].float()
        else:
            raise RuntimeError("unsupported causal LM verifier")
        a,b = self.option_ids
        margins = (logits[:,b]-logits[:,a]).float().detach().cpu().tolist()
        if len(margins)!=2*len(pairs):
            raise RuntimeError("incorrect number of Qwen verifier decisions")
        return [min(margins[i],-margins[i+1])>=1.0 for i in range(0,len(margins),2)]

    def _call(self, segments):
        self.calls += 1
        stats = dict(status="running",call=self.calls,device=str(self.device),start_epoch=time.time(),
                     segments=len(segments),stages=[],primary_width=16,retry_width=4,
                     generate_calls=0,generated_prompts=0,generated_tokens=0,token_cap_hits=0,
                     verify_calls=0,verified_pairs=0,accepted_edits=0)
        original_side = self.tokenizer.padding_side
        self._publish(stats)
        try:
            if str(self.device).split(":")[0]=="npu":
                torch.npu.set_device(self.device)
            self.tokenizer.padding_side = "left"
            with torch.inference_mode():
                result = self._correct(segments,stats)
            stats["status"]="completed"
            return result
        except BaseException as exc:
            stats.update(status="error",error_type=type(exc).__name__)
            raise
        finally:
            self.tokenizer.padding_side = original_side
            stats["end_epoch"] = time.time()
            stats["seconds"] = stats["end_epoch"]-stats["start_epoch"]
            self._publish(stats)

    def _correct(self, segments, stats):
        tok, post, model = self.tokenizer, self.post, self.model
        texts = [str(s.get("text","")) for s in segments]
        groups,caps = pack(tok,texts)
        prompts = [post._format_prompt(post._messages(payload(g))) for g in groups]
        sizes = [len(ids) for ids in tok(prompts)["input_ids"]] if prompts else []
        stats.update(prompts=len(groups),expected_batches=len(groups),input_sequence_sha256=sequence_hash(texts),
                     min_planned_cap=min(caps,default=0),max_planned_cap=max(caps,default=0))
        valid = [False]*len(groups)
        suggested = [g.copy() for g in groups]
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos,int) else set(eos)

        def generate(ids, cap):
            batch = tok([prompts[i] for i in ids],return_tensors="pt",padding=True).to(self.device)
            output = model.generate(**batch,do_sample=False,max_new_tokens=cap,pad_token_id=tok.eos_token_id)
            decoded = output[:,batch.input_ids.shape[1]:].detach().cpu().tolist()
            if len(decoded)!=len(ids):
                raise RuntimeError("Qwen output batch size mismatch")
            stats["generate_calls"] += 1
            stats["generated_prompts"] += len(decoded)
            for idx,seq in zip(ids,decoded):
                n = next((k+1 for k,value in enumerate(seq) if value in eos),len(seq))
                stats["generated_tokens"] += n
                stats["token_cap_hits"] += int(n==cap and seq[n-1] not in eos)
                original = groups[idx]
                parsed = post._parse_json_array(tok.decode(seq[:n],skip_special_tokens=True).strip())
                ok = isinstance(parsed,list) and len(parsed)==len(original) and all(
                    isinstance(v,dict) and type(v.get("id")) is int and v["id"]==j and
                    isinstance(v.get("text"),str) and v["text"].strip() for j,v in enumerate(parsed))
                valid[idx] = bool(ok)
                suggested[idx] = post._validated_texts(parsed,original) if ok else original.copy()

        primary = sorted(range(len(groups)),key=sizes.__getitem__)
        with self._stage(stats,"primary"):
            for offset in range(0,len(primary),16):
                ids = primary[offset:offset+16]
                generate(ids,max(caps[i] for i in ids))
        failed = [i for i,ok in enumerate(valid) if not ok]
        stats.update(first_pass_valid_prompts=sum(valid),retry_prompts=len(failed),
                     first_failed_indices=failed,primary_max_width=min(16,len(groups)),
                     primary_generate_calls=stats["generate_calls"])
        with self._stage(stats,"retry"):
            retry = sorted(failed,key=sizes.__getitem__)
            for offset in range(0,len(retry),4):
                generate(retry[offset:offset+4],512)
        stats.update(valid_prompts=sum(valid),valid_batches=sum(valid),retry_recovered=sum(valid[i] for i in failed),
                     final_failed_indices=[i for i,ok in enumerate(valid) if not ok],
                     fallback_prompts=len(valid)-sum(valid),retry_max_width=min(4,len(failed)),
                     valid_group_coverage_segments=sum(len(g) for g,ok in zip(groups,valid) if ok),
                     structural_fallback_segments=sum(len(g) for g,ok in zip(groups,valid) if not ok))
        raw = [s for group in suggested for s in group]
        if len(raw)!=len(texts):
            raise RuntimeError("generated row count mismatch")
        projected = [project(s,p) for s,p in zip(texts,raw)]
        changes = [i for i,(s,p) in enumerate(zip(texts,projected)) if s!=p]
        stats.update(raw_changed_segments=sum(s!=p for s,p in zip(texts,raw)),projected_candidates=len(changes))
        corrected = [dict(s) for s in segments]
        with self._stage(stats,"verification"):
            for offset in range(0,len(changes),8):
                idx = changes[offset:offset+8]
                decisions = self._verify([(texts[i],projected[i]) for i in idx])
                stats["verify_calls"] += 1
                stats["verified_pairs"] += len(decisions)
                for i,accept in zip(idx,decisions):
                    if accept:
                        corrected[i].setdefault("raw_text",texts[i])
                        corrected[i].update(text=projected[i],post_processed=True)
                        stats["accepted_edits"] += 1
        stats["result_sequence_sha256"] = sequence_hash([str(s.get("text","")) for s in corrected])
        return corrected


def install(post, require_npu=True):
    """Call once after app startup. enable(False) restores scalar behavior for a baseline."""
    return Override(post,require_npu)
