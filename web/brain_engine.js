// ============================================================
//  brain_engine.js — 오미크론 두뇌를 브라우저에서 돌리는 엔진
// ------------------------------------------------------------
//  model.py 의 TinyGPT 를 자바스크립트로 그대로 옮긴 것.
//  brain/model.bin (두뇌 숫자) + brain/tokenizer.json (사전) 을 읽는다.
//  글자 단위(작은 오미크론), 단어 조각(큰 오미크론) 둘 다 지원.
// ============================================================
(function (root) {
  // ---------- 두뇌 파일 읽기 ----------
  function parseBrain(buf) {
    const dv = new DataView(buf);
    const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
    if (magic !== "OMI1") throw new Error("두뇌 파일 형식이 아니에요");
    const hlen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
    const base = 8 + hlen, cache = {};
    function P(name) {
      if (cache[name]) return cache[name];
      const t = header.tensors[name];
      if (!t) throw new Error("없는 숫자 묶음: " + name);
      const n = t.shape.reduce((a, b) => a * b, 1), out = new Float32Array(n);
      if (t.dtype === "f32") {
        out.set(new Float32Array(buf, base + t.offset, n));
      } else { // q8: 행마다 배율(float32) + 8비트 정수
        const rows = t.shape[0], cols = n / rows;
        const sc = new Float32Array(buf, base + t.offset, rows);
        const q = new Int8Array(buf, base + t.offset + rows * 4, n);
        for (let r = 0; r < rows; r++) { const s = sc[r], o = r * cols; for (let c = 0; c < cols; c++) out[o + c] = q[o + c] * s; }
      }
      return (cache[name] = out);
    }
    return { header, config: header.config, P };
  }

  // ---------- 사전 ----------
  function makeTokenizer(T) {
    const vocab = T.vocab, stoi = new Map(vocab.map((t, i) => [t, i]));
    if (T.type === "char") {
      return {
        type: "char", size: vocab.length, newline: stoi.get("\n"),
        encode: s => Array.from(s).map(c => stoi.get(c) ?? 0),
        decode: ids => ids.map(i => vocab[i]).join(""),
      };
    }
    // 단어 조각(BPE): 글자로 쪼갠 뒤, 사전에 적힌 순서대로 붙일 수 있는 짝을 계속 붙인다
    const rank = new Map(T.merges.map(([a, b], i) => [a + "\u0000" + b, i]));
    const unk = T.unk ?? 0, memo = new Map();
    function bpe(piece) {
      if (memo.has(piece)) return memo.get(piece);
      let sym = Array.from(piece).map(c => (stoi.has(c) ? c : null));
      for (;;) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i + 1 < sym.length; i++) {
          if (sym[i] === null || sym[i + 1] === null) continue;
          const r = rank.get(sym[i] + "\u0000" + sym[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        sym.splice(best, 2, sym[best] + sym[best + 1]);
      }
      const ids = sym.map(s => (s === null ? unk : stoi.get(s)));
      if (memo.size < 50000) memo.set(piece, ids);
      return ids;
    }
    return {
      type: "bpe", size: vocab.length, newline: stoi.get("\n"),
      encode(s) {
        s = s.replace(/[^\S\n]/g, " ").replace(/ /g, "▁");
        const out = [];
        for (const p of s.match(/\n|▁?[^▁\n]+|▁+/gu) || []) out.push(...bpe(p));
        return out;
      },
      decode: ids => ids.map(i => vocab[i] ?? "").join("").replace(/▁/g, " "),
    };
  }

  // ---------- 두뇌 계산 ----------
  function createModel(brain) {
    const cfg = brain.config, P = brain.P;
    const C = cfg.n_embd, H = cfg.n_head, D = C / H, L = cfg.n_layer, V = cfg.vocab_size;
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
      let v = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - m; v += d * d; } v /= x.length;
      const r = 1 / Math.sqrt(v + 1e-5), y = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) y[i] = (x[i] - m) * r * w[i] + b[i];
      return y;
    }
    function linear(x, W, b, out) {
      const n = x.length, y = new Float32Array(out);
      for (let o = 0; o < out; o++) {
        let s = b ? b[o] : 0, r = o * n, i = 0;
        for (; i + 3 < n; i += 4) s += W[r + i] * x[i] + W[r + i + 1] * x[i + 1] + W[r + i + 2] * x[i + 2] + W[r + i + 3] * x[i + 3];
        for (; i < n; i++) s += W[r + i] * x[i];
        y[o] = s;
      }
      return y;
    }
    function erf(x) {
      const s = Math.sign(x); x = Math.abs(x);
      const t = 1 / (1 + 0.3275911 * x);
      return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
    }

    let K, Vc, pos;
    function reset() { K = layers.map(() => []); Vc = layers.map(() => []); pos = 0; }
    reset();

    // 조각 하나를 넣으면 다음 조각 점수를 돌려준다 (앞 조각들의 K,V 는 기억해 둔다)
    function step(tok) {
      const x = new Float32Array(C);
      for (let i = 0; i < C; i++) x[i] = tokEmb[tok * C + i] + posEmb[pos * C + i];
      for (let l = 0; l < L; l++) {
        const ly = layers[l];
        const qkv = linear(layerNorm(x, ly.ln1w, ly.ln1b), ly.qkvW, ly.qkvB, 3 * C);
        K[l].push(qkv.slice(C, 2 * C)); Vc[l].push(qkv.slice(2 * C));
        const T = K[l].length, att = new Float32Array(C), sc = new Float32Array(T), inv = 1 / Math.sqrt(D);
        for (let h = 0; h < H; h++) {
          const o = h * D; let mx = -Infinity;
          for (let t = 0; t < T; t++) { let s = 0; const k = K[l][t]; for (let d = 0; d < D; d++) s += qkv[o + d] * k[o + d]; s *= inv; sc[t] = s; if (s > mx) mx = s; }
          let sum = 0; for (let t = 0; t < T; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
          for (let t = 0; t < T; t++) { const w = sc[t] / sum, v = Vc[l][t]; for (let d = 0; d < D; d++) att[o + d] += w * v[o + d]; }
        }
        const pr = linear(att, ly.projW, ly.projB, C);
        for (let i = 0; i < C; i++) x[i] += pr[i];
        const hd = linear(layerNorm(x, ly.ln2w, ly.ln2b), ly.fcW, ly.fcB, 4 * C);
        for (let i = 0; i < hd.length; i++) hd[i] = 0.5 * hd[i] * (1 + erf(hd[i] / Math.SQRT2));
        const mo = linear(hd, ly.outW, ly.outB, C);
        for (let i = 0; i < C; i++) x[i] += mo[i];
      }
      pos++;
      return linear(layerNorm(x, lnfW, lnfB), tokEmb, null, V);
    }
    return { step, reset, get pos() { return pos; }, cfg };
  }

  // ---------- 대화 ----------
  function createChat(brain, tokJson, opts = {}) {
    const model = createModel(brain), tok = makeTokenizer(tokJson);
    const NAME = brain.header.name || "오미크론", USER = opts.user || "나";
    const BS = brain.config.block_size, NL = tok.newline;
    const maxNew = opts.maxNew || (tok.type === "bpe" ? 60 : 120);
    let history = [], logits = null;
    const tick = () => new Promise(r => setTimeout(r, 0));

    function sample(lg, recent) {
      const temp = opts.temperature || 0.8, topK = opts.topK || 40, pen = opts.repeat || 1.15;
      const z = Float32Array.from(lg);
      for (const t of recent) z[t] = z[t] > 0 ? z[t] / pen : z[t] * pen;   // 방금 한 말 반복 줄이기
      const idx = Array.from(z.keys()).sort((a, b) => z[b] - z[a]).slice(0, topK);
      const mx = z[idx[0]] / temp, ps = idx.map(i => Math.exp(z[i] / temp - mx));
      let r = Math.random() * ps.reduce((a, b) => a + b, 0);
      for (let j = 0; j < idx.length; j++) { r -= ps[j]; if (r <= 0) return idx[j]; }
      return idx[idx.length - 1];
    }

    async function feed(ids) {
      for (let i = 0; i < ids.length; i++) {
        if (model.pos >= BS) await rebuild();
        logits = model.step(ids[i]); history.push(ids[i]);
        if (i % 4 === 3) await tick();
      }
    }
    async function rebuild() { // 기억 길이가 꽉 차면 최근 대화만 남기고 다시 읽는다
      let start = Math.max(0, history.length - Math.floor(BS / 2));
      const nl = history.indexOf(NL, start);
      const keep = nl >= 0 && nl + 1 < history.length ? history.slice(nl + 1) : history.slice(start);
      model.reset(); history = [];
      for (const t of keep) { logits = model.step(t); history.push(t); }
    }

    return {
      name: NAME, tokenizer: tok, config: brain.config,
      reset() { model.reset(); history = []; logits = null; },
      async *reply(text) {
        const prompt = tok.encode(`${USER}: ${text}\n${NAME}: `);
        if (history.length + prompt.length + maxNew > BS) {
          if (prompt.length + maxNew > BS) { model.reset(); history = []; }
          else {
            while (history.length + prompt.length + maxNew > BS && history.length) {
              const before = history.length;
              await rebuild();
              if (history.length >= before) { model.reset(); history = []; }
            }
          }
        }
        await feed(prompt);
        const out = [];
        for (let n = 0; n < maxNew; n++) {
          const t = sample(logits, out);
          if (t === NL) break;
          out.push(t);
          yield tok.decode(out);
          await feed([t]);
        }
        await feed([NL]);
      },
    };
  }

  const api = { parseBrain, makeTokenizer, createModel, createChat };
  if (typeof module !== "undefined") module.exports = api; else root.OmicronBrain = api;
})(typeof window !== "undefined" ? window : globalThis);
