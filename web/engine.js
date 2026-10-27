// 오미크론 추론 엔진: model.py 의 TinyGPT 를 자바스크립트로 그대로 옮긴 것.
// 한 글자씩 넣으면서 앞 글자들의 K, V 를 저장(캐시)해 두어 빠르게 계산한다.
function createOmicron(W) {
  const cfg = W.config, C = cfg.n_embd, H = cfg.n_head, D = C / H, L = cfg.n_layer, BS = cfg.block_size;
  const bin = atob(W.weights), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const all = new Float32Array(buf);
  const P = name => { const [off, shape] = W.manifest[name]; return all.subarray(off, off + shape.reduce((a, b) => a * b, 1)); };

  const vocab = W.vocab, stoi = new Map(vocab.map((c, i) => [c, i])), V = vocab.length;
  const tokEmb = P("tok_emb.weight"), posEmb = P("pos_emb.weight");
  const layers = [];
  for (let l = 0; l < L; l++) {
    const p = n => P(`blocks.${l}.${n}`);
    layers.push({
      ln1w: p("ln1.weight"), ln1b: p("ln1.bias"), qkvW: p("attn.qkv.weight"), qkvB: p("attn.qkv.bias"),
      projW: p("attn.proj.weight"), projB: p("attn.proj.bias"), ln2w: p("ln2.weight"), ln2b: p("ln2.bias"),
      fcW: p("mlp.fc.weight"), fcB: p("mlp.fc.bias"), outW: p("mlp.out.weight"), outB: p("mlp.out.bias"),
    });
  }
  const lnfW = P("ln_f.weight"), lnfB = P("ln_f.bias");

  function layerNorm(x, w, b) {
    let m = 0; for (let i = 0; i < x.length; i++) m += x[i]; m /= x.length;
    let v = 0; for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2; v /= x.length;
    const r = 1 / Math.sqrt(v + 1e-5), y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = (x[i] - m) * r * w[i] + b[i];
    return y;
  }
  function linear(x, Wt, b, out) {
    const n = x.length, y = new Float32Array(out);
    for (let o = 0; o < out; o++) { let s = b ? b[o] : 0; const r = o * n; for (let i = 0; i < n; i++) s += Wt[r + i] * x[i]; y[o] = s; }
    return y;
  }
  function erf(x) { // Abramowitz-Stegun 근사
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

  let kCache, vCache, pos;
  function reset() { kCache = layers.map(() => []); vCache = layers.map(() => []); pos = 0; }

  // 글자 하나를 넣고, 다음 글자 점수(logits)를 돌려준다.
  function step(tok) {
    let x = new Float32Array(C);
    for (let i = 0; i < C; i++) x[i] = tokEmb[tok * C + i] + posEmb[pos * C + i];
    for (let l = 0; l < L; l++) {
      const ly = layers[l];
      const qkv = linear(layerNorm(x, ly.ln1w, ly.ln1b), ly.qkvW, ly.qkvB, 3 * C);
      kCache[l].push(qkv.slice(C, 2 * C)); vCache[l].push(qkv.slice(2 * C));
      const T = kCache[l].length, att = new Float32Array(C), sc = new Float32Array(T);
      for (let h = 0; h < H; h++) {
        const o = h * D; let mx = -Infinity;
        for (let t = 0; t < T; t++) { let s = 0; const k = kCache[l][t]; for (let d = 0; d < D; d++) s += qkv[o + d] * k[o + d]; s /= Math.sqrt(D); sc[t] = s; if (s > mx) mx = s; }
        let sum = 0; for (let t = 0; t < T; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
        for (let t = 0; t < T; t++) { const w = sc[t] / sum, v = vCache[l][t]; for (let d = 0; d < D; d++) att[o + d] += w * v[o + d]; }
      }
      const pr = linear(att, ly.projW, ly.projB, C);
      for (let i = 0; i < C; i++) x[i] += pr[i];
      const hd = linear(layerNorm(x, ly.ln2w, ly.ln2b), ly.fcW, ly.fcB, 4 * C);
      for (let i = 0; i < hd.length; i++) hd[i] = gelu(hd[i]);
      const mo = linear(hd, ly.outW, ly.outB, C);
      for (let i = 0; i < C; i++) x[i] += mo[i];
    }
    pos++;
    return linear(layerNorm(x, lnfW, lnfB), tokEmb, null, V);
  }

  function sample(logits, temperature, topK) {
    const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
    const mx = logits[idx[0]] / temperature;
    const ps = idx.map(i => Math.exp(logits[i] / temperature - mx));
    let r = Math.random() * ps.reduce((a, b) => a + b, 0);
    for (let j = 0; j < idx.length; j++) { r -= ps[j]; if (r <= 0) return idx[j]; }
    return idx[idx.length - 1];
  }

  const encode = s => Array.from(s).map(c => stoi.get(c) ?? 0);

  // 프롬프트를 넣고 대답을 한 글자씩 onChar 로 흘려보낸다.
  async function* generate(prompt, { maxNew = 100, temperature = 0.7, topK = 20 } = {}) {
    let toks = encode(prompt).slice(-(BS - 40));
    reset();
    let logits; for (const t of toks) logits = step(t);
    const nl = stoi.get("\n");
    for (let n = 0; n < maxNew; n++) {
      const t = sample(logits, temperature, topK);
      if (t === nl) return;
      toks.push(t);
      yield vocab[t];
      if (pos >= BS) { // 기억 길이가 꽉 차면 최근 글자들로 다시 계산
        const keep = toks.slice(-64); reset();
        for (const k of keep.slice(0, -1)) step(k);
        logits = step(keep[keep.length - 1]);
      } else logits = step(t);
    }
  }

  return { generate, encode, step, reset, vocab };
}
if (typeof module !== "undefined") module.exports = { createOmicron };
