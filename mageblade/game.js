// ============================================================
//  game.js — 「은월의 마검사」 액션 로그라이트 미리보기
//  - 스프라이트: assets/mageblade/mageblade_sheet.png + .json (8열 × 6행, 칸 64×96, 기준점 32,92)
//  - 이펙트: 캐릭터와 다른 레이어에서 따로 재생 (투사체·장판·폭발)
//  - 상태: 쓰러짐 > 피격 > 공격·스킬 > 달리기·대기  (config.priority)
// ============================================================
(async function () {
  const C = window.MAGEBLADE_CONFIG;
  const W = C.view.width, H = C.view.height, GROUND = C.view.groundY;
  const cv = document.getElementById("game"), ctx = cv.getContext("2d");
  const $ = id => document.getElementById(id);

  // ---------- 리소스 ----------
  const loadImg = src => new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error(src + " 를 불러오지 못했어요")); i.src = src; });
  const base = "assets/mageblade/";
  const [sheetMeta, fxMeta] = await Promise.all([fetch(base + "mageblade_sheet.json").then(r => r.json()), fetch(base + "effects.json").then(r => r.json())]);
  const sheetImg = await loadImg(base + sheetMeta.image);
  const fxImg = {};
  for (const [k, v] of Object.entries(fxMeta.effects)) fxImg[k] = await loadImg(base + v.image);
  const [AX, AY] = sheetMeta.anchor, CW = sheetMeta.cell[0], CH = sheetMeta.cell[1];

  // 재생 시간: config 가 있으면 config, 없으면 JSON
  const ANIM = {};
  for (const [k, a] of Object.entries(sheetMeta.animations)) {
    const over = C.animations[k] || {};
    const ms = over.frameMs === undefined ? a.frameMs : (Array.isArray(over.frameMs) ? over.frameMs : a.frames.map(() => over.frameMs));
    ANIM[k] = { frames: a.frames, ms, loop: a.loop, holdLast: a.holdLast };
  }

  // ---------- 화면: 정수 배율 ----------
  cv.width = W; cv.height = H;
  function fit() {
    const box = $("stage").getBoundingClientRect();
    const s = Math.max(1, Math.floor(Math.min(box.width / W, box.height / H)));
    cv.style.width = W * s + "px"; cv.style.height = H * s + "px";
    $("scaleTag").textContent = "×" + s;
  }
  addEventListener("resize", fit); fit();
  ctx.imageSmoothingEnabled = false;

  // ---------- 입력 (키보드 + 터치 버튼) ----------
  const input = { left: false, right: false, attack: false, skill: false };
  const buffer = { attack: 0, skill: 0 };          // 짧게 눌러도 놓치지 않도록 0.2초 동안 입력을 기억
  const press = k => { if (k in buffer) buffer[k] = 200; };
  const KEYS = { ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right", KeyJ: "attack", KeyZ: "attack", KeyK: "skill", KeyX: "skill" };
  addEventListener("keydown", e => {
    if (KEYS[e.code]) { if (!e.repeat) press(KEYS[e.code]); input[KEYS[e.code]] = true; e.preventDefault(); }
    if (e.code === "KeyH") toggleDebug();
    if (e.code === "KeyR" && game.over) restart();
  });
  addEventListener("keyup", e => { if (KEYS[e.code]) input[KEYS[e.code]] = false; });
  document.querySelectorAll("[data-key]").forEach(b => {
    const k = b.dataset.key;
    const on = e => { e.preventDefault(); press(k); input[k] = true; b.classList.add("on"); };
    const off = e => { e.preventDefault(); input[k] = false; b.classList.remove("on"); };
    b.addEventListener("pointerdown", on); b.addEventListener("pointerup", off);
    b.addEventListener("pointerleave", off); b.addEventListener("pointercancel", off);
  });
  let debug = false;
  function toggleDebug() { debug = !debug; $("dbg").classList.toggle("on", debug); }
  $("dbg").addEventListener("click", toggleDebug);

  // ---------- 애니메이션 재생기 ----------
  function animator(name) { return { name, i: 0, t: 0, done: false }; }
  function stepAnim(a, dt) {
    const A = ANIM[a.name];
    a.t += dt;
    while (!a.done && a.t >= A.ms[a.i]) {
      a.t -= A.ms[a.i];
      if (a.i < A.frames.length - 1) a.i++;
      else if (A.loop) a.i = 0;
      else { a.done = true; a.t = 0; }          // 한 번 재생 끝 (쓰러짐은 마지막 프레임 유지)
    }
  }

  // ---------- 판정 도우미 ----------
  function box(ent, b) {                         // 기준점 + 상자, 왼쪽을 보면 좌우 반전
    const x = ent.face > 0 ? ent.x + b.x : ent.x - b.x - b.w;
    return { x, y: ent.y + b.y, w: b.w, h: b.h };
  }
  const overlap = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

  // ---------- 게임 상태 ----------
  const game = { room: 1, over: false, choosing: false, effects: [], shots: [], enemies: [], marks: [], shake: 0 };
  const P = {};
  function resetPlayer() {
    Object.assign(P, {
      x: 70, y: GROUND, face: 1, hp: C.player.maxHp, maxHp: C.player.maxHp,
      state: "idle", anim: animator("idle"), invuln: 0, vx: 0, skillCd: 0, hitDone: new Set(),
      mods: { atk: 1, spd: 1, cd: 1 },
    });
  }
  function setState(s, force) {                 // 우선순위가 같거나 높을 때만 바뀜
    const cur = C.priority[P.state], nxt = C.priority[s];
    if (!force && nxt < cur && !(P.anim.done && P.state !== "down")) return false;
    if (P.state === s && (s === "idle" || s === "run")) return true;
    P.state = s; P.anim = animator(s); P.hitDone = new Set(); P.castDone = false;
    return true;
  }

  function spawnRoom() {
    const n = C.enemy.perRoom[Math.min(game.room - 1, C.enemy.perRoom.length - 1)] + Math.max(0, game.room - C.enemy.perRoom.length);
    for (let i = 0; i < n; i++) {
      const fromRight = i % 2 === 0;
      game.enemies.push({
        x: fromRight ? W + 20 + i * 22 : -20 - i * 22, y: GROUND, face: fromRight ? -1 : 1,
        hp: C.enemy.baseHp + C.enemy.hpPerRoom * (game.room - 1), maxHp: C.enemy.baseHp + C.enemy.hpPerRoom * (game.room - 1),
        speed: C.enemy.speed[0] + Math.random() * (C.enemy.speed[1] - C.enemy.speed[0]), kb: 0, flash: 0, wob: Math.random() * 6,
      });
    }
    $("room").textContent = game.room;
  }

  function addFx(kind, x, y, face) {
    const m = fxMeta.effects[kind];
    const e = { kind, x, y, face, i: 0, t: 0, done: false, m, age: 0 };
    game.effects.push(e); return e;
  }

  function damageEnemy(en, dmg, kb, dir) {
    en.hp -= dmg; en.flash = 120; en.kb = kb * dir;
    game.marks.push({ x: en.x, y: en.y - 18, txt: String(Math.round(dmg)), t: 600 });
  }

  // ---------- 한 프레임 진행 ----------
  function update(dt) {
    const sec = dt / 1000;
    if (game.choosing) return;
    P.invuln = Math.max(0, P.invuln - dt);
    P.skillCd = Math.max(0, P.skillCd - dt);

    // 1) 입력 → 상태 (쓰러짐·피격이 우선, 공격·스킬 중엔 이동 애니메이션이 덮어쓰지 않음)
    const busy = ["attack", "skill", "hit", "down"].includes(P.state) && !P.anim.done;
    if (!game.over) {
      if (!busy) {
        if (buffer.attack > 0 || input.attack) { if (setState("attack")) buffer.attack = 0; }
        else if ((buffer.skill > 0 || input.skill) && P.skillCd <= 0) {
          if (setState("skill")) { P.skillCd = C.skill.cooldownMs * P.mods.cd; buffer.skill = 0; }
        }
      }
      buffer.attack = Math.max(0, buffer.attack - dt); buffer.skill = Math.max(0, buffer.skill - dt);
      const moveDir = (input.right ? 1 : 0) - (input.left ? 1 : 0);
      const locked = (P.state === "attack" && C.attack.lockMove) || (P.state === "skill" && C.skill.lockMove) ||
                     (P.state === "hit" && C.hit.lockMove) || P.state === "down";
      if (!locked || P.anim.done) {
        if (P.state !== "down") {
          if (moveDir) { P.face = moveDir; P.x += moveDir * C.player.moveSpeed * P.mods.spd * sec; setState("run", true); }
          else setState("idle", true);
        }
      }
    }
    P.x += P.vx * sec; P.vx *= Math.pow(0.002, sec);
    P.x = Math.max(16, Math.min(W - 16, P.x));
    stepAnim(P.anim, dt);

    // 2) 공격 판정 (설정한 프레임에서만)
    if (P.state === "attack" && C.attack.activeFrames.includes(P.anim.i) && !P.anim.done) {
      const hb = box(P, C.attack.hitbox);
      for (const en of game.enemies) {
        if (!P.hitDone.has(en) && overlap(hb, box(en, C.enemy.hurtbox))) {
          P.hitDone.add(en); damageEnemy(en, C.attack.damage * P.mods.atk, C.attack.knockback, P.face); game.shake = 80;
        }
      }
    }
    // 3) 스킬: 시전 프레임에 장판(표시만) + 투사체
    if (P.state === "skill" && P.anim.i === C.skill.castFrame && !P.castDone) {
      P.castDone = true;
      addFx("field", P.x + C.skill.field.offsetX * P.face, P.y + C.skill.field.offsetY, P.face);
      const sp = C.skill.projectile.spawn;
      game.shots.push({ x: P.x + sp.x * P.face, y: P.y + sp.y, face: P.face, age: 0, fx: addFx("projectile", 0, 0, P.face) });
    }
    // 4) 투사체 이동 → 맞거나 시간 끝나면 폭발
    for (const s of game.shots) {
      s.x += s.face * C.skill.projectile.speed * sec; s.age += dt;
      s.fx.x = s.x; s.fx.y = s.y;
      const hb = box(s, C.skill.projectile.hitbox);
      const target = game.enemies.find(en => overlap(hb, box(en, C.enemy.hurtbox)));
      if (target) damageEnemy(target, C.skill.projectile.damage * P.mods.atk, 30, s.face);
      if (target || s.age > C.skill.projectile.maxMs || s.x < -30 || s.x > W + 30) {
        s.dead = true; s.fx.done = true;
        const ex = addFx("explosion", Math.max(10, Math.min(W - 10, s.x)), GROUND, s.face);
        ex.hits = new Set();
      }
    }
    game.shots = game.shots.filter(s => !s.dead);
    // 5) 이펙트 재생 (캐릭터와 다른 레이어) + 폭발 판정
    for (const e of game.effects) {
      e.t += dt; e.age += dt;
      while (!e.done && e.t >= e.m.frameMs) { e.t -= e.m.frameMs; e.i++; if (e.i >= e.m.frames) { if (e.m.loop) e.i = 0; else { e.done = true; e.i = e.m.frames - 1; } } }
      if (e.kind === "explosion" && e.i === C.skill.explosion.damageFrame && e.hits) {
        const hb = box({ x: e.x, y: e.y, face: 1 }, C.skill.explosion.hitbox);
        for (const en of game.enemies) if (!e.hits.has(en) && overlap(hb, box(en, C.enemy.hurtbox))) {
          e.hits.add(en); damageEnemy(en, C.skill.explosion.damage * P.mods.atk, C.skill.explosion.knockback, Math.sign(en.x - e.x) || 1); game.shake = 140;
        }
      }
    }
    game.effects = game.effects.filter(e => !e.done);
    // 6) 적
    for (const en of game.enemies) {
      en.flash = Math.max(0, en.flash - dt); en.wob += sec * 6;
      if (en.kb) { en.x += en.kb * sec; en.kb *= Math.pow(0.001, sec); if (Math.abs(en.kb) < 4) en.kb = 0; }
      else if (!game.over) { const d = Math.sign(P.x - en.x); en.face = d || en.face; en.x += d * en.speed * sec; }
      if (!game.over && P.invuln <= 0 && P.state !== "down" && overlap(box(P, C.player.hurtbox), box(en, C.enemy.hurtbox))) {
        P.hp -= C.enemy.contactDamage; P.invuln = C.player.invulnMs; P.vx = Math.sign(P.x - en.x || 1) * C.player.knockback;
        game.shake = 120;
        if (P.hp <= 0) { P.hp = 0; setState("down", true); game.over = true; $("over").hidden = false; }
        else setState("hit", true);                // 피격은 공격·스킬을 끊는다
      }
    }
    game.enemies = game.enemies.filter(en => en.hp > 0);
    for (const m of game.marks) { m.t -= dt; m.y -= sec * 18; }
    game.marks = game.marks.filter(m => m.t > 0);
    game.shake = Math.max(0, game.shake - dt);
    // 7) 방 클리어 → 강화 선택
    if (!game.over && game.enemies.length === 0 && !game.choosing) offerUpgrade();
    $("hp").style.width = (P.hp / P.maxHp * 100) + "%"; $("hpText").textContent = `${Math.ceil(P.hp)} / ${P.maxHp}`;
    $("cd").style.width = (100 - P.skillCd / (C.skill.cooldownMs * P.mods.cd) * 100) + "%";
    $("stateTag").textContent = P.state + " " + (P.anim.i + 1) + "/" + ANIM[P.state].frames.length;
  }

  function offerUpgrade() {
    game.choosing = true;
    const pool = [...C.upgrades].sort(() => Math.random() - 0.5).slice(0, 3);
    const wrap = $("cards"); wrap.innerHTML = "";
    for (const u of pool) {
      const b = document.createElement("button");
      b.className = "card"; b.innerHTML = `<b>${u.name}</b><span>${u.desc}</span>`;
      b.addEventListener("click", () => {
        if (u.id === "atk") P.mods.atk *= 1.2;
        if (u.id === "spd") P.mods.spd *= 1.15;
        if (u.id === "cd") P.mods.cd *= 0.8;
        if (u.id === "hp") { P.maxHp += 20; P.hp = P.maxHp; }
        $("pick").hidden = true; game.choosing = false; game.room++; spawnRoom();
      });
      wrap.appendChild(b);
    }
    $("pickTitle").textContent = `${game.room}번째 방 돌파! 강화를 고르세요`;
    $("pick").hidden = false;
  }

  function restart() {
    game.room = 1; game.over = false; game.choosing = false; game.effects = []; game.shots = []; game.enemies = []; game.marks = [];
    $("over").hidden = true; $("pick").hidden = true;
    resetPlayer(); spawnRoom();
  }
  $("again").addEventListener("click", restart);

  // ---------- 그리기 ----------
  function drawBackground() {
    ctx.fillStyle = "#0d1a29"; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#13233a"; ctx.fillRect(0, 60, W, GROUND - 60);
    ctx.fillStyle = "#dfe0f0";                                      // 은빛 달 (픽셀 원)
    for (let y = -12; y <= 12; y++) for (let x = -12; x <= 12; x++) if (x * x + y * y <= 144) ctx.fillRect(250 + x, 34 + y, 1, 1);
    ctx.fillStyle = "#0d1a29";
    for (let y = -12; y <= 12; y++) for (let x = -12; x <= 12; x++) if ((x + 5) * (x + 5) + (y - 2) * (y - 2) <= 110) ctx.fillRect(250 + x, 34 + y, 1, 1);
    ctx.fillStyle = "#1a2c45";                                      // 먼 성벽
    for (let x = 0; x < W; x += 16) { const h = 24 + ((x * 37) % 19); ctx.fillRect(x, GROUND - h, 14, h); }
    ctx.fillStyle = "#2a2f45"; ctx.fillRect(0, GROUND, W, H - GROUND);
    ctx.fillStyle = "#3d4462"; ctx.fillRect(0, GROUND, W, 1);
    ctx.fillStyle = "#353b57"; for (let x = 4; x < W; x += 24) ctx.fillRect(x, GROUND + 6, 10, 1);
  }
  function drawSprite(ent, anim) {                // 기준점(32,92)을 엔티티 위치에 맞춤, 왼쪽은 좌우 반전
    const f = ANIM[anim.name].frames[anim.i];
    const x = Math.round(ent.x), y = Math.round(ent.y);
    ctx.save();
    if (ent.face < 0) { ctx.translate(x, 0); ctx.scale(-1, 1); ctx.translate(-x, 0); }
    ctx.drawImage(sheetImg, f.x, f.y, f.w, f.h, x - AX, y - AY, CW, CH);
    ctx.restore();
  }
  function drawFx(e) {
    const m = e.m, x = Math.round(e.x), y = Math.round(e.y);
    ctx.save();
    if (e.face < 0) { ctx.translate(x, 0); ctx.scale(-1, 1); ctx.translate(-x, 0); }
    if (e.kind === "field" && e.done === false && e.i === m.frames - 1) ctx.globalAlpha = 0.7;
    ctx.drawImage(fxImg[e.kind], e.i * m.frameW, 0, m.frameW, m.frameH, x - m.anchor[0], y - m.anchor[1], m.frameW, m.frameH);
    ctx.restore();
    if (debug) { ctx.strokeStyle = "#4aa3ff"; ctx.strokeRect(x - m.anchor[0] + 0.5, y - m.anchor[1] + 0.5, m.frameW - 1, m.frameH - 1); }
  }
  function drawEnemy(en) {                        // 그림자 망령 (코드로 그린 간단한 적, 원작 그림 아님)
    const x = Math.round(en.x), y = Math.round(en.y) - 2 - Math.round(Math.sin(en.wob) * 1.5);
    const body = en.flash > 0 ? "#f0e9f7" : "#3b3655", hi = en.flash > 0 ? "#ffffff" : "#5a4f86";
    ctx.fillStyle = "#06060f"; ctx.fillRect(x - 8, y - 28, 16, 25); ctx.fillRect(x - 6, y - 30, 12, 2);
    ctx.fillStyle = body; ctx.fillRect(x - 7, y - 27, 14, 23); ctx.fillRect(x - 5, y - 29, 10, 2);
    ctx.fillStyle = hi; ctx.fillRect(x - 5, y - 26, 3, 14);
    for (let i = 0; i < 4; i++) {                 // 너울거리는 아랫단
      const h = 2 + ((i + Math.floor(en.wob * 2)) % 2) * 2;
      ctx.fillStyle = body; ctx.fillRect(x - 7 + i * 4, y - 4, 3, h);
    }
    ctx.fillStyle = "#b98cfa"; ctx.fillRect(x + en.face * 3 - 1, y - 22, 2, 2); ctx.fillRect(x + en.face * 3 - 4 * en.face - 1, y - 22, 2, 2);
    ctx.fillStyle = "#06060f"; ctx.fillRect(x - 8, y - 35, 16, 2);
    ctx.fillStyle = "#8f3a4f"; ctx.fillRect(x - 8, y - 35, Math.round(16 * en.hp / en.maxHp), 2);
  }
  function rect(b, color) { ctx.strokeStyle = color; ctx.strokeRect(Math.round(b.x) + 0.5, Math.round(b.y) + 0.5, b.w - 1, b.h - 1); }

  function draw() {
    ctx.save();
    if (game.shake > 0) ctx.translate(Math.round((Math.random() - 0.5) * 2), 0);
    drawBackground();
    // 레이어: 바닥 이펙트(장판) → 적 → 캐릭터 → 공중 이펙트(투사체·폭발) → 숫자
    for (const e of game.effects) if (e.kind === "field") drawFx(e);
    for (const en of game.enemies) drawEnemy(en);
    if (!(P.invuln > 0 && P.state !== "down" && Math.floor(P.invuln / 70) % 2)) drawSprite(P, P.anim);
    for (const e of game.effects) if (e.kind !== "field") drawFx(e);
    ctx.font = "8px monospace"; ctx.textAlign = "center";
    for (const m of game.marks) { ctx.fillStyle = "#06060f"; ctx.fillText(m.txt, m.x + 1, m.y + 1); ctx.fillStyle = "#e4b385"; ctx.fillText(m.txt, m.x, m.y); }
    if (debug) {
      rect(box(P, C.player.hurtbox), "#3cd27a");
      if (P.state === "attack") rect(box(P, C.attack.hitbox), C.attack.activeFrames.includes(P.anim.i) ? "#ff3b3b" : "#7a3030");
      for (const s of game.shots) rect(box(s, C.skill.projectile.hitbox), "#ff3b3b");
      for (const e of game.effects) if (e.kind === "explosion") rect(box({ x: e.x, y: e.y, face: 1 }, C.skill.explosion.hitbox), e.i === C.skill.explosion.damageFrame ? "#ff3b3b" : "#7a3030");
      for (const en of game.enemies) rect(box(en, C.enemy.hurtbox), "#3cd27a");
      ctx.fillStyle = "#ffffff"; ctx.fillRect(Math.round(P.x), Math.round(P.y), 1, 1);   // 기준점
    }
    ctx.restore();
  }

  // ---------- 시작 ----------
  resetPlayer(); spawnRoom();
  $("loading").hidden = true;
  let last = performance.now();
  function frame(t) {
    const dt = Math.min(50, t - last); last = t;
    update(dt); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  window.__mageblade = { game, P, C };          // 디버그용
})().catch(err => {
  const el = document.getElementById("loading");
  el.hidden = false; el.textContent = "불러오기 실패: " + err.message;
});
