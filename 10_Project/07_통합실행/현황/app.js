/* 알박기 관제 · 실시간 (통합실행)
 * MVP 관제 대시보드와 같은 모양으로, 알박기_실행.py가 1초마다 판정한 결과(status.json)를 보여 준다.
 * 선택한 룸의 카메라 화면은 stream/<룸>.mjpg 로 실시간으로 받는다 (박스·감지 범위는 마지막 판정 결과).
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const CSSV = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const ST = ["사용 중", "자리 비움", "이용 가능"];
  const S = { sel: null, rooms: [], tiles: new Map(), prev: new Map(), roomKey: "", eventsKey: "", alertKey: "", data: null };

  const pad = (n) => String(n).padStart(2, "0");
  // 받침 있으면 a, 없으면 b (30분이 / 1분 30초가)
  const josa = (w, a, b) => { const c = w.charCodeAt(w.length - 1) - 0xac00; return c >= 0 && c < 11172 && c % 28 ? a : b; };
  const limitTxt = (d) => d.limit_text;
  function timer(sec) {
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }
  function dur(sec) {
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
    return h ? `${h}시간 ${m}분` : m ? `${m}분` : `${sec}초`;
  }

  // ---------- 룸 지도 ----------
  function buildFloors(rooms) {
    const box = $("floors");
    box.innerHTML = "";
    S.tiles.clear();
    const groups = [];
    rooms.forEach((r) => { let g = groups.find((x) => x.name === r.group); if (!g) groups.push(g = { name: r.group, rooms: [] }); g.rooms.push(r); });
    groups.forEach((g) => {
      const f = el("div", "floor");
      f.append(el("div", "floor-head", `<h3>${g.name}</h3><span>룸 ${g.rooms.length}개 · ${[...new Set(g.rooms.map((r) => r.kind))].join(", ")}</span>`));
      const grid = el("div", "rooms");
      g.rooms.forEach((r) => {
        const b = el("button", "room");
        b.type = "button";
        b.innerHTML = `<div class="room-top"><span class="room-no">${r.no}</span><span class="room-tag live">${r.kind}</span></div>
          <div class="room-state"><span class="dot"></span><span class="st"></span></div>
          <div class="room-time"></div><div class="room-bar"><i></i></div><canvas width="200" height="12" aria-hidden="true"></canvas>`;
        b.addEventListener("click", () => select(r.id));
        S.tiles.set(r.id, b);
        grid.append(b);
      });
      f.append(grid);
      box.append(f);
    });
  }

  function select(id) {
    S.sel = id;
    S.tiles.forEach((t, k) => t.classList.toggle("sel", k === id));
    const r = S.rooms.find((x) => x.id === id);
    if (!r) return;
    $("dType").textContent = `${r.group} · ${r.kind}`;
    $("dName").textContent = r.id;
    $("camWait").hidden = false;
    $("camWait").textContent = "카메라 연결 중…";
    $("cam").src = `stream/${encodeURIComponent(id)}.mjpg`;
    S.eventsKey = "";
    if (S.data) render(S.data);
  }
  $("cam").addEventListener("load", () => { $("camWait").hidden = true; });
  $("cam").addEventListener("error", () => { $("camWait").hidden = false; $("camWait").textContent = "카메라 화면을 받을 수 없습니다."; });

  function tileText(r) {
    if (r.code < 0) return { main: "준비 중", sub: "첫 판정 대기" };
    if (r.code === 0) return { main: "사용 중", sub: `${r.since}부터` };
    if (r.code === 1) return { main: "자리 비움", sub: `비운 지 ${timer(r.away)} · 초록까지 ${timer(r.left)}` };
    if (r.empty) return { main: "이용 가능", sub: "빈 책상 · 짐 없음" };
    return { main: "이용 가능", sub: `비운 지 ${dur(r.away)} · 바구니 이동 가능` };
  }

  function drawRuns(cv, r, elapsed, heightFrac = 1, y0 = 0) {
    const ctx = cv.getContext("2d");
    const col = [CSSV("--red"), CSSV("--yellow"), CSSV("--green")];
    const W = cv.width, H = cv.height * heightFrac, span = Math.max(1, elapsed);
    r.runs.forEach(([a, s], i) => {
      const b = i + 1 < r.runs.length ? r.runs[i + 1][0] : elapsed;
      ctx.fillStyle = col[s];
      ctx.fillRect((a / span) * W, y0, Math.max(1, ((b - a) / span) * W), H);
    });
  }

  function renderTiles(d) {
    const line = CSSV("--line");
    d.rooms.forEach((r) => {
      const tile = S.tiles.get(r.id);
      if (!tile) return;
      const was = S.prev.get(r.id);
      tile.classList.remove("s0", "s1", "s2");
      if (r.code >= 0) tile.classList.add("s" + r.code);
      if (was !== undefined && was !== r.code) { tile.classList.remove("flash"); void tile.offsetWidth; tile.classList.add("flash"); }
      S.prev.set(r.id, r.code);
      const tx = tileText(r);
      tile.querySelector(".st").textContent = tx.main;
      tile.querySelector(".room-time").textContent = tx.sub;
      const frac = r.code === 1 || (r.code === 2 && !r.empty) ? Math.min(1, r.away / r.limit) : 0;
      tile.querySelector(".room-bar i").style.width = (frac * 100).toFixed(1) + "%";
      const cv = tile.querySelector("canvas");
      const ctx = cv.getContext("2d");
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.fillStyle = line; ctx.globalAlpha = .45; ctx.fillRect(0, 0, cv.width, cv.height); ctx.globalAlpha = 1;
      drawRuns(cv, r, d.elapsed);
    });
  }

  // ---------- KPI ----------
  function renderKpis(d) {
    const n = d.rooms.length;
    const red = d.rooms.filter((r) => r.code === 0).length;
    const yel = d.rooms.filter((r) => r.code === 1).length;
    const grnEmpty = d.rooms.filter((r) => r.code === 2 && r.empty).length;
    const stale = d.rooms.filter((r) => r.code === 2 && !r.empty).length;
    const act = n ? Math.round((red / n) * 100) : 0;
    const ok = d.cycle_ms && d.cycle_ms < d.interval_ms;
    $("kpis").innerHTML = `
      <div class="kpi st" style="--c:var(--red)"><span class="k">🔴 사용 중</span><span class="v">${red}<small> / ${n}</small></span><span class="s">감지 범위 안에 사람 있음</span></div>
      <div class="kpi st" style="--c:var(--yellow)"><span class="k">🟡 자리 비움</span><span class="v">${yel}</span><span class="s">${limitTxt(d)} 미만 · 짐에 손대지 않음</span></div>
      <div class="kpi st" style="--c:var(--green)"><span class="k">🟢 이용 가능</span><span class="v">${grnEmpty + stale}</span><span class="s">빈 책상 ${grnEmpty} · ${limitTxt(d)} 넘은 짐 ${stale}</span></div>
      <div class="kpi"><span class="k">지금 룸 활성화율</span><span class="v">${act}<small>%</small></span>
        <div class="meter"><i style="width:${act}%"></i><b style="left:80%"></b></div><span class="s">사람이 실제로 있는 룸 ÷ ${n} · 목표 80%</span></div>
      <div class="kpi ${stale ? "bad" : "good"}"><span class="k">${limitTxt(d)} 넘게 짐만 있는 룸</span><span class="v">${stale}</span><span class="s">목표 0 · 초록불로 안내 중</span></div>
      <div class="kpi ${ok ? "good" : "bad"}"><span class="k">한 바퀴 판정 시간</span><span class="v">${d.cycle_ms}<small>ms</small></span><span class="s">룸 ${n}개 · ${d.interval_ms}ms 안에 ${ok ? "끝남" : "못 끝냄"}</span></div>`;
  }

  // ---------- 선택한 룸 ----------
  function renderDetail(d) {
    const r = d.rooms.find((x) => x.id === S.sel);
    if (!r) return;
    const chip = $("dState");
    chip.className = "state-chip" + (r.code >= 0 ? " s" + r.code : "");
    chip.textContent = r.code >= 0 ? ST[r.code] : "준비 중";
    $("camJudge").textContent = r.code >= 0 ? `판정 ${r.judged_ago.toFixed(1)}초 전 · ${d.interval_ms / 1000}초마다` : "판정 대기";
    $("jIn").textContent = `${r.inside}명` + (r.persons > r.inside ? ` (범위 밖 ${r.persons - r.inside})` : "");
    $("jItems").textContent = r.items ? `${r.items}개` : "없음";
    $("jItems").title = r.item_names.join(", ");
    $("jAway").textContent = r.code === 0 || r.code < 0 ? "—" : timer(r.away);
    $("jLeft").textContent = r.code === 1 ? timer(r.left) : r.code === 2 ? "켜짐" : "—";
    const fill = $("limitFill");
    fill.style.width = (r.code === 0 || r.empty ? 0 : Math.min(1, r.away / r.limit) * 100).toFixed(1) + "%";
    fill.classList.toggle("full", r.code === 2);
    const itemsTxt = r.items ? `짐 ${r.items}개(${r.item_names.join(", ")})가 남아 있습니다.` : "";
    const p = r.pend;
    $("jText").innerHTML = r.code < 0 ? "첫 판정을 기다리는 중입니다."
      : (r.code === 0 && p > 0)
      ? `사람이 <b>${Math.floor(p)}초째</b> 보이지 않지만, 잠깐 가려졌거나 검출을 놓친 것일 수 있어 <b>${d.rules.off}초</b>가 될 때까지는 사용 중으로 유지합니다.`
      : (r.code !== 0 && p < 0)
      ? `사람이 다시 보이기 시작했습니다 (${Math.floor(-p)}/${d.rules.on}초). 오검출을 막기 위해 <b>${d.rules.on}초</b> 연속으로 보이면 사용 중으로 바꾸고 부재 타이머를 0으로 돌립니다.`
      : r.code === 0
      ? `감지 범위 안에 사람 ${r.inside}명이 있어 <b>사용 중</b>입니다. ${r.since}부터 계속 사람이 있었습니다. 문 앞 표시등은 빨강입니다.`
      : r.code === 1
      ? `<b>${r.away_wall}</b>에 마지막으로 사람이 보였습니다. ${itemsTxt} ${limitTxt(d)}${josa(limitTxt(d), "이", "가")} 되는 <b>${r.green_wall}</b>에 초록불로 바뀝니다. 그전까지는 짐에 손대지 않습니다.`
      : r.empty
      ? `사람도 짐도 ${d.rules.empty}초 넘게 보이지 않아 <b>빈 책상</b>으로 판정했습니다. ${limitTxt(d)}${josa(limitTxt(d), "을", "를")} 기다리지 않고 바로 초록불입니다.`
      : `<b>${r.away_wall}</b>부터 ${limitTxt(d)} 넘게 사람이 없습니다. ${itemsTxt} ${d.limit_sec === 1800 ? "학교 30분 규칙" : "초록불 규칙"}에 따라 <b>짐을 룸 앞 회색 바구니로 옮기고 사용할 수 있습니다.</b>`;
    const box = $("camConfirm");
    if (r.code === 0 && p > 0) {
      box.hidden = false;
      const k = Math.min(d.rules.off, Math.floor(p));
      box.innerHTML = `사람이 안 보임 · 부재 확인 중 <b>${k}/${d.rules.off}초</b><span class="pips">${Array.from({ length: d.rules.off }, (_, i) => `<i class="${i < k ? "on" : ""}"></i>`).join("")}</span><span style="opacity:.7">가림·놓침 방지</span>`;
    } else if (r.code !== 0 && p < 0) {
      box.hidden = false;
      box.innerHTML = `사람 감지 · 재석 확인 중 <b>${Math.floor(-p)}/${d.rules.on}초</b><span style="opacity:.7">오검출 방지</span>`;
    } else box.hidden = true;
    renderTimeline(d, r);
    renderEvents(r);
  }

  function renderTimeline(d, r) {
    const cv = $("timeline");
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(cv.clientWidth * dpr), H = Math.round(56 * dpr);
    if (cv.width !== W) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    const barY = 10 * dpr, barH = 32 * dpr;
    ctx.fillStyle = CSSV("--surface-2"); ctx.fillRect(0, barY, W, barH);
    ctx.save(); ctx.translate(0, barY);
    drawRuns({ getContext: () => ctx, width: W, height: barH }, r, d.elapsed);
    ctx.restore();
    ctx.fillStyle = CSSV("--ink");
    ctx.fillRect(W - 2 * dpr, 2 * dpr, 2 * dpr, H - 10 * dpr);
    $("tlAxis").innerHTML = `<span>${d.started} 시작</span><span>${d.clock} 지금</span>`;
  }

  function renderEvents(r) {
    const key = r.id + ":" + r.events.length + ":" + (r.events[0] ? r.events[0].ts : 0);
    if (key === S.eventsKey) return;
    S.eventsKey = key;
    const ol = $("events");
    ol.innerHTML = "";
    if (!r.events.length) { ol.append(el("li", "empty", "아직 상태 변화가 없습니다.")); return; }
    r.events.slice(0, 12).forEach((e) => {
      const li = el("li", "ev");
      const img = e.snap ? `<img src="snap/${encodeURIComponent(e.snap)}" alt="${r.id} ${e.t} 스냅샷 (사람 모자이크)" loading="lazy">` : `<div class="noimg">스냅샷 없음</div>`;
      const first = e.from < 0;
      li.innerHTML = `${img}<div class="ev-main">
        <div class="ev-title"><span class="mono">${e.t}</span>${first ? `<span class="pill s${e.to}">${ST[e.to]}</span> 기록 시작` : `<span class="pill s${e.from}">${ST[e.from]}</span><span class="arrow">→</span><span class="pill s${e.to}">${ST[e.to]}</span>`}</div>
        <div class="ev-sub">${first ? "프로그램을 켠 시점의 상태" : e.why}</div>
        <div class="ev-sub">${e.snap ? "스냅샷은 1일 뒤 자동 삭제 · 이후 데이터만 남음" : "스냅샷 저장 꺼짐"}</div></div>`;
      if (e.snap) li.addEventListener("click", () => window.open(`snap/${encodeURIComponent(e.snap)}`, "_blank"));
      ol.append(li);
    });
  }

  // ---------- 알림 ----------
  function renderAlerts(d) {
    const items = [];
    d.rooms.forEach((r) => r.events.forEach((e) => {
      if (e.from >= 0 && e.to === 2) items.push({ ts: e.ts, t: e.t, room: r.id, cls: "g",
        text: e.empty ? "초록불 · 빈 책상, 바로 사용 가능" : `초록불 · ${e.away_wall}부터 ${limitTxt(d)} 비어 짐을 바구니로 옮기고 사용 가능` });
      if (e.from === 2 && e.to === 0) items.push({ ts: e.ts, t: e.t, room: r.id, cls: "r", text: "초록불 뒤 입실 · 사용 시작" });
    }));
    items.sort((a, b) => b.ts - a.ts);
    const key = items.length + ":" + (items[0] ? items[0].ts : 0);
    if (key === S.alertKey) return;
    S.alertKey = key;
    const ol = $("alerts");
    ol.innerHTML = "";
    if (!items.length) { ol.append(el("li", "empty", "아직 알림이 없습니다.")); return; }
    items.slice(0, 6).forEach((a) => {
      const li = el("li", "alert " + a.cls, `<span class="mono">${a.t}</span><span class="no">${a.room}</span><span>${a.text}</span>`);
      li.addEventListener("click", () => select(a.room));
      ol.append(li);
    });
  }

  // ---------- 갱신 ----------
  function render(d) {
    $("clock").textContent = d.clock;
    $("dayLabel").textContent = `${d.date} · 실시간 판정 · ${d.started}부터`;
    $("perf").innerHTML = `<span class="badge live">룸 ${d.rooms.length}개 판정 중</span><span class="badge">한 바퀴 ${d.cycle_ms}ms / ${d.interval_ms}ms</span><span class="badge">${d.device}</span>`;
    $("tabNote").textContent = d.limit_sec !== 1800 ? `시험 설정 · 초록불 규칙 ${d.limit_text} (학교 규칙은 30분)` : "";
    $("ruleLimit").textContent = d.limit_text;
    renderTiles(d);
    renderKpis(d);
    renderDetail(d);
    renderAlerts(d);
  }

  async function tick() {
    try {
      const d = await (await fetch("status.json", { cache: "no-store" })).json();
      S.data = d;
      const key = d.rooms.map((r) => r.id).join("|");
      if (key !== S.roomKey) {
        S.roomKey = key; S.rooms = d.rooms; buildFloors(d.rooms);
        if (!d.rooms.some((r) => r.id === S.sel) && d.rooms.length) { S.data = d; select(d.rooms[0].id); }
        else select(S.sel);
      }
      S.rooms = d.rooms;
      render(d);
    } catch (err) {
      $("dayLabel").textContent = "판정 프로그램과 연결이 끊겼습니다";
    }
  }
  tick();
  setInterval(tick, 1000);
  document.documentElement.style.setProperty("--top-h", document.querySelector(".topbar").offsetHeight + 48 + "px");
})();
