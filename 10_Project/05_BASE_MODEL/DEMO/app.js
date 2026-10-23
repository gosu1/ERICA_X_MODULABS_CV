/* 스터디룸 관제 (관리자용 가상 운영 대시보드)
 *
 * 데이터: data/detections.js (build_data.py 결과)
 *   1차.mp4(천장 고정 카메라, 22분)를 1초마다 파인튜닝 YOLO26s v2로 판정한 결과와 사람 모자이크 화면.
 * 가상 운영: 이 실측 판정을 장면별 구간(공부, 나감, 짐만 남음, 복귀, 엎드림)으로 나눠 이어 붙여
 *   14개 룸의 14:00~23:00(오후 수업·저녁 식사·저녁 자습)을 만든다. 현장 관찰대로 짐이 없는 룸은 없고(팀 교체는 5분 안),
 *   저녁 8시 이후 활성화율은 9할을 넘는다.
 *   301호는 1차.mp4를 그대로 재생하는 구간이 있다.
 * 판정 규칙: mvp/room_logic.py 와 같다. 사람 2초 연속 → 있음, 10초 연속 없음 → 없음(부재 시각은 마지막으로 보인 순간),
 *   30분 비면 초록.
 */
(() => {
  "use strict";

  const DATA = window.DETECTIONS;
  const ROWS = DATA.rows;
  const INIT_QS = new URLSearchParams(location.search);   // 시나리오를 고르면 주소가 바뀌므로 처음 주소를 따로 둔다
  const START = 14 * 3600;          // 14:00
  const N = 9 * 3600;               // 14:00:00 ~ 22:59:59
  const RULE = { on: 2, off: 10, limit: 30 * 60, empty: 10 };   // empty: 사람도 짐도 이만큼 연속으로 없으면 빈 책상 → 바로 초록
  const KEEP = 24 * 3600;           // 스냅샷 보관 1일
  const ST = ["사용 중", "자리 비움", "이용 가능"];
  const ST_LONG = ["🔴 사용 중", "🟡 자리 비움", "🟢 이용 가능"];
  const CSSV = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const M = 60;

  // ---------- 시간대 ----------
  const BANDS = [
    { a: 0, b: 3 * 3600, key: "afternoon", name: "오후 수업", short: "오후 수업" },
    { a: 3 * 3600, b: 5.5 * 3600, key: "dinner", name: "저녁 식사", short: "저녁 식사" },
    { a: 5.5 * 3600, b: 9 * 3600, key: "evening", name: "저녁 자습", short: "저녁 자습" },
  ];
  const KPI_WIN = [5 * 3600, 8 * 3600];   // 기획서 KPI: 저녁 7~10시
  const bandAt = (t) => BANDS.find((b) => t >= b.a && t < b.b) || BANDS[BANDS.length - 1];

  // ---------- 실측 영상 구간 (1차.mp4 초 단위, 판정 결과를 보고 정함) ----------
  // 0~211초 사람 있음 / 212~1147초 사람 없음(짐만) / 1148~1317초 사람 있음(마지막 몇 초 엎드림)
  const CLIPS = {
    study: [5, 185, true],        // 공부 중 (반복)
    study2: [1180, 1300, true],   // 복귀 후 공부 중 (반복)
    sleep: [1306, 1318, true],    // 책상에 엎드림 (반복)
    leave: [185, 235, false],     // 짐을 두고 나가는 장면 (211초에 마지막으로 보임)
    away: [240, 1125, true],      // 짐만 남은 룸 (반복)
    ret: [1125, 1180, false],     // 짐 있는 룸으로 들어오는 장면 (1148초부터 보임)
    real: [0, 1318, false],       // 1차.mp4 그대로
  };

  // ---------- 룸 구성 (기획서: 3층 6인 6개, 4층 오픈형 4인 6개, 4층 독립형 6인 2개) ----------
  const CAM3 = "문 위 천장 카메라", CAM4 = "천장 카메라", CAMI = "룸 안 천장 카메라 · 촬영 안내문 부착";
  const ROOMS = [
    ...["301", "302", "303", "304", "305", "306"].map((no) => ({ no, floor: "3층", kind: "6인 룸", cam: CAM3, live: no === "301" })),
    ...["401", "402", "403", "404", "405", "406"].map((no) => ({ no, floor: "4층 오픈형", kind: "4인 오픈형", cam: CAM4 })),
    ...["407", "408"].map((no) => ({ no, floor: "4층 독립형", kind: "6인 독립형", cam: CAMI })),
  ];

  // ---------- 시나리오 → 초별 원본 프레임 ----------
  function buildSource(prog, phaseSeed) {
    const out = new Int16Array(N);
    const real = new Uint8Array(N);
    let pos = 0;
    for (const [type, durRaw] of prog) {
      const [a, b, loop] = CLIPS[type];
      const len = b - a;
      const dur = durRaw === "rest" ? N - pos : (loop ? durRaw : len);
      const phase = loop ? (phaseSeed * 137 + pos) % len : 0;
      for (let i = 0; i < dur && pos < N; i++, pos++) {
        out[pos] = loop ? a + (phase + i) % len : a + i;
        real[pos] = type === "real" ? 1 : 0;
      }
    }
    return { src: out, real };
  }

  // ---------- 판정 (mvp/room_logic.py 와 같은 규칙) ----------
  function simulate(room) {
    const st = new Uint8Array(N);
    const awayS = new Int32Array(N);
    const occS = new Int32Array(N);
    const pend = new Int8Array(N);   // >0: 부재 확인 중 경과 초, <0: 재석 확인 중 경과 초
    const empty = new Uint8Array(N);  // 1: 사람도 짐도 없는 빈 책상
    const events = [];
    let present, lastSeen, awaySince, occSince, runStart = null, prev = -1, noItemsSince = null;
    if (room.initAway != null) { present = false; lastSeen = null; awaySince = -room.initAway; occSince = 0; }
    else { present = true; lastSeen = 0; awaySince = 0; occSince = -((room.initOcc ?? 40) * M); }

    for (let t = 0; t < N; t++) {
      const row = ROWS[room.src[t]];
      const raw = row.present;
      if (raw) {
        lastSeen = t;
        if (runStart === null) runStart = t;
        if (!present && t - runStart >= RULE.on) { present = true; occSince = runStart; }
      } else {
        runStart = null;
        if (present && lastSeen !== null && t - lastSeen >= RULE.off) { present = false; awaySince = lastSeen; }
      }
      if (row.items.length) noItemsSince = null; else if (noItemsSince === null) noItemsSince = t;
      const isEmpty = !present && !raw && noItemsSince !== null && t - noItemsSince >= RULE.empty;
      const s = present ? 0 : (isEmpty || t - awaySince >= RULE.limit ? 2 : 1);
      st[t] = s; awayS[t] = awaySince; occS[t] = occSince; empty[t] = isEmpty ? 1 : 0;
      pend[t] = (!raw && present && lastSeen !== null) ? (t - lastSeen) : (raw && !present ? -(t - runStart + 1) : 0);
      if (s !== prev) {
        events.push({ t, from: prev, to: s, awaySince, occSince, room: room.no, items: row.items.length, empty: isEmpty });
        prev = s;
      }
    }
    return { st, awayS, occS, pend, empty, events };
  }

  // ---------- 시나리오 생성 ----------
  // 실측 장면으로 만들 수 있는 상황을 시간대에 맞춰 섞어 룸마다 다른 오후·저녁을 만든다. 시드가 같으면 같은 결과.
  // 현장 관찰(도입 전): 짐을 치우는 사람이 없다. 짐만 남은 룸은 주인이 돌아올 때까지 그대로다.
  //   짐을 챙겨 나가면 다음 팀이 길어야 5분 안에 들어온다. 14~17시 50% 안팎, 17:00~19:30 40% 미만, 8시 이후 9할 이상.
  // 도입 후: 초록불이 켜지면 기다리던 학생이 짐을 바구니로 옮기고 5분 안에 들어온다 (저녁 식사 시간엔 채울 학생이 없음).
  // 같은 시드면 두 모드에서 주인들의 행동은 같고, 초록불 뒤 누가 먼저 오느냐만 다르다 (룸마다 따로 난수를 씀).
  // 비운 시간 = away 길이 + 나가는 장면 뒤 24초 + 들어오는 장면 앞 22초 (약 46초)
  const BREAKS_DAY = [["short", 30], ["meal", 22], ["sneaky", 12], ["edge", 8], ["long", 20]];
  const BREAKS_LATE = [["late", 86], ["meal", 8], ["sneaky", 4], ["edge", 2]];   // 20시 이후: 거의 화장실·물 정도
  const TAG = {
    real: "실측 영상 구간", startAway: "14시 전부터 짐만", class: "수업 가며 짐만 두고 감", dinner: "저녁 식사",
    short: "잠깐 비움", late: "잠깐 비움(8시 이후)", meal: "간식·통화(30분 미만)", sneaky: "30분 직전 재방문", edge: "30분 직후 주인 복귀",
    long: "오래 비움(30분 넘음)", newcomer: "초록불 뒤 짐 치우고 새 학생", turnover: "퇴실 후 5분 안에 다음 팀", gone: "짐 두고 오래 자리 비움",
    sleep: "엎드려 휴식", endaway: "마감 무렵 자리 비움",
  };
  const TAKE_MIN = [0.3, 4.4];   // 도입 후: 초록불 뒤 새 학생이 들어오기까지 (들어오는 장면 22초 + 재석 확인 2초를 더해 길어야 5분)
  const DINNER = [3 * 3600, 5.5 * 3600];   // 17:00~19:30 저녁 식사 시간 (짐 두고 간 주인은 19:30 무렵 돌아옴)
  const SYS = { after: false };            // false = 도입 전(아무도 안 치움), true = 도입 후

  function mulberry32(a) {
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generate(room, rng, after) {
    const R = (a, b) => a + rng() * (b - a);
    const hour = (t) => (START + t) / 3600;
    const prog = [], tags = {}, newAt = [];   // newAt: 새 학생이 들어오는 장면의 시작 시각
    const tag = (k) => (tags[k] = (tags[k] || 0) + 1);
    let pos = 0, alt = rng() < 0.5, teams = 0;
    const len = (type, d) => (CLIPS[type][2] ? (d === "rest" ? N - pos : d) : CLIPS[type][1] - CLIPS[type][0]);
    const push = (type, d = 0) => { prog.push([type, d]); pos += len(type, d); };
    const nextStudy = () => ((alt = !alt) ? "study" : "study2");
    const study = (d) => {
      if (d > 25 * M && rng() < 0.15) {
        const a = Math.round(d * R(0.3, 0.6));
        push(nextStudy(), a); push("sleep", Math.round(R(8, 30) * M)); tag("sleep");
        push(nextStudy(), Math.max(60, d - a));
      } else push(nextStudy(), d);
    };
    // 주인이 ownerAway 뒤에 돌아올 예정일 때, 실제 away 길이와 누가 오는지 정한다.
    // greenAway: 지금부터 이만큼 지나면 초록불. 난수는 두 모드에서 똑같이 뽑아 같은 날이 되게 한다
    const decide = (ownerAway, greenAway) => {
      const fill = greenAway + Math.round(R(TAKE_MIN[0], TAKE_MIN[1]) * M);   // 도입 후: 언제든 초록불 뒤 5분 안에 새 학생
      if (after && ownerAway > greenAway && fill < ownerAway) return { away: fill, newcomer: true };
      return { away: ownerAway, newcomer: false };
    };
    const pick = (list) => {
      let x = rng() * list.reduce((s, b) => s + b[1], 0);
      for (const [k, w] of list) { if ((x -= w) < 0) return k; }
      return list[0][0];
    };
    const takeOver = (d) => { if (d.newcomer) { tag("newcomer"); teams++; newAt.push(pos); } };

    // 14:00 시작 상태: 사용 중이거나, 수업 가느라 짐만 남아 있음 (짐이 하나도 없는 룸은 없음)
    const init = {};
    let state, ownerLeft = 0;
    if (!room.live && rng() < 0.5) {
      init.initAway = Math.round(R(0, 29) * M); state = "items"; tag("startAway");
      ownerLeft = Math.max(3 * M, Math.round(R(40, 90) * M) - init.initAway);   // 수업이 끝나 돌아오기까지 남은 시간
    } else { init.initOcc = Math.round(R(10, 120)); state = "occ"; teams++; }

    let realAt = room.live ? Math.round(R(5.5 * 3600, 7.2 * 3600)) : Infinity;   // 19:30~21:10 무렵 (쉬는 시간과 겹치면 조금 늦어짐)
    let realDone = !room.live, realStart = null, dinnerDone = false;
    const dinnerAt = 3 * 3600 + Math.round(R(0, 80) * M);   // 이 룸 사람들이 저녁 먹으러 나가는 시각 (17:00~18:20)

    while (pos < N - 120) {
      if (!realDone && pos >= realAt) {
        if (state === "items") push("ret");
        realStart = pos; push("real"); tag("real"); realDone = true; state = "occ";
        continue;
      }
      if (state === "items") {
        // 짐만 남은 룸: 도입 전엔 주인이 올 때까지 그대로, 도입 후엔 초록불 뒤 새 학생이 먼저 올 수 있음
        const awaySoFar = prog.length ? 24 : (init.initAway || 0);
        const d = decide(ownerLeft, Math.max(0, RULE.limit - awaySoFar - 22));
        let w = d.away;
        if (!realDone && pos + w > realAt) { w = Math.max(60, realAt - pos); d.newcomer = false; }   // 301호 실측 구간 시각에 맞춰 주인이 들어옴
        push("away", Math.max(60, w));
        if (pos >= N - 120) break;
        takeOver(d); push("ret"); state = "occ";
        continue;
      }
      // 사용 중: 이번 팀이 머무는 시간 (시작 시각에 따라 다름). 8시 이후에 온 팀은 대부분 마감까지 있다
      const h0 = hour(pos);
      const stay = (h0 < 17 ? R(40, 110) : h0 < 19.5 ? R(30, 90) : h0 < 20 ? R(90, 200) : R(150, 300)) * M;
      const endAt = Math.min(N - 120, pos + stay);
      let ended = false;
      while (pos < endAt) {
        // 한 번에 앉아 있는 시간: 오후·식사 시간에는 짧게 끊기고, 저녁 8시 이후에는 길다
        const hs = hour(pos);
        let blk = Math.round((hs < 17 ? R(25, 60) : hs < 19.5 ? R(15, 45) : hs < 20 ? R(40, 100) : R(80, 180)) * M);
        const toDinner = !dinnerDone && pos < dinnerAt && pos + blk > dinnerAt;
        if (toDinner) blk = Math.max(60, dinnerAt - pos);
        // 301호 실측 구간이 정한 시각에 시작하도록 공부 구간을 거기서 끊는다
        if (!realDone && pos < realAt && pos + blk > realAt) blk = Math.max(60, realAt - pos);
        study(blk);
        if (pos >= endAt || pos >= N - 120) break;
        if (!realDone && pos >= realAt) break;
        const h = hour(pos);
        let k, plan;
        if (!dinnerDone && (toDinner || (h >= 17 && h < 19.5)) && rng() < 0.9) { k = "dinner"; plan = R(40, 90); dinnerDone = true; }
        else if (h < 17 && rng() < 0.45) { k = "class"; plan = R(45, 75); }
        else { k = pick(h >= 20 ? BREAKS_LATE : BREAKS_DAY); plan = { short: R(2, 9), late: R(2, 6), meal: R(12, 24), edge: R(29.4, 31.6), long: R(34, 80), sneaky: 62 }[k]; }
        plan *= M;
        // 쉬는 동안 301호 실측 구간 시각이 지나가 버리면, 쉬지 않고 바로 실측 구간을 시작한다
        if (!realDone && pos >= 5.5 * 3600 && pos + plan * 1.5 + 120 > realAt) { realAt = pos; break; }
        push("leave");
        if (pos + plan + 60 > N) { tag("endaway"); push("away", "rest"); ended = true; break; }
        tag(k);
        if (k === "sneaky") {
          const times = rng() < 0.5 ? 2 : 1;
          for (let i = 0; i < times; i++) {
            push("away", Math.round(R(25, 28.3) * M)); push("ret");
            push(nextStudy(), Math.round(R(1, 4) * M)); push("leave");
          }
          push("away", Math.round(R(10, 20) * M));
        } else if (k === "edge") {
          push("away", Math.round(plan));   // 30분 직후 주인이 바로 돌아옴
        } else {
          const d = decide(Math.round(plan), RULE.limit - 46);
          push("away", d.away);
          takeOver(d);
        }
        push("ret");
      }
      if (ended || pos >= N - 120) break;
      if (!realDone && pos >= realAt) continue;
      // 이번 팀이 일어남: 대부분 짐을 챙겨 나가고 다음 팀이 5분 안에 들어옴(짐 없는 시간은 찰나).
      // 일부는 짐을 두고 오래 자리를 비움 (도입 전엔 주인이 올 때까지 짐만 남음)
      if (rng() < (hour(pos) < 19.5 ? 0.5 : hour(pos) < 20 ? 0.85 : 0.97)) {
        push("leave"); push("away", Math.round(R(0.3, 4.2) * M)); push("ret");
        tag("turnover"); teams++;
      } else {
        push("leave"); tag("gone"); state = "items";
        // 주인이 돌아오기까지: 오후엔 수업 한두 개, 식사 시간엔 19:30~19:50 사이, 저녁 자습 시간엔 짧게
        const hg = hour(pos);
        ownerLeft = Math.round(hg < 17 ? R(40, 120) * M
          : hg < 19.5 ? Math.max(30 * M, DINNER[1] + R(0, 20) * M - pos)
          : R(20, 50) * M);
      }
    }
    if (!realDone) { if (state === "items") push("ret"); realStart = pos; push("real"); tag("real"); state = "occ"; }
    push(state === "items" ? "away" : nextStudy(), "rest");

    const words = Object.entries(tags).filter(([k]) => k !== "real").map(([k, n]) => TAG[k] + (n > 1 ? ` ${n}회` : ""));
    let story = `${teams}팀 이용` + (words.length ? " · " + words.join(" · ") : "");
    if (room.live && realStart != null && realStart < N) {
      story = `1차.mp4 실측 구간 ${clock(realStart, false)}~${clock(Math.min(N, realStart + 1318), false)} 포함 · ${story}`;
    }
    return { prog, story, newAt, ...init };
  }

  // ---------- 시나리오 적용 ----------
  const CURATED_SEED = 378;   // 추천 시나리오: 여러 상황이 고르게 나오는 시드 (점검으로 고름)
  let ALL_EVENTS = [];
  const SCN = { mode: "curated", seed: CURATED_SEED };
  const roomRng = (s, i) => mulberry32((s * 2654435761) ^ ((i + 1) * 40503));

  // 시드와 도입 전/후로 룸 14개의 하루를 만든다 (화면에 쓰지 않는 비교 계산에도 씀)
  function buildDay(s, after) {
    return ROOMS.map((r, i) => {
      const g = generate(r, roomRng(s, i), after);
      const day = { no: r.no, live: r.live, prog: g.prog, story: g.story, newAt: g.newAt, initOcc: g.initOcc, initAway: g.initAway };
      Object.assign(day, buildSource(day.prog, i + 3 + s));
      Object.assign(day, simulate(day));
      day.runs = toRuns(day.st);
      return day;
    });
  }

  function applyScenario(mode, seed) {
    SCN.mode = mode; SCN.seed = seed;
    const s = mode === "curated" ? CURATED_SEED : seed;
    const day = buildDay(s, SYS.after);
    ROOMS.forEach((r, i) => Object.assign(r, day[i]));
    ALL_EVENTS = ROOMS.flatMap((r) => r.events.filter((e) => e.t > 0)).sort((a, b) => a.t - b.t);
  }
  function resimulate(r) { Object.assign(r, simulate(r)); r.runs = toRuns(r.st); }

  function toRuns(st) {
    const runs = [];
    let s = 0;
    for (let t = 1; t <= N; t++) if (t === N || st[t] !== st[s]) { runs.push({ a: s, b: t, s: st[s] }); s = t; }
    return runs;
  }

  // ---------- 형식 ----------
  const pad = (n) => String(n).padStart(2, "0");
  function clock(t, withSec = true) {
    let x = Math.floor(START + t);
    const neg = x < 0; x = Math.abs(x);
    const h = Math.floor(x / 3600) % 24, m = Math.floor(x / 60) % 60, s = x % 60;
    return (neg ? "-" : "") + pad(h) + ":" + pad(m) + (withSec ? ":" + pad(s) : "");
  }
  function timer(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60;
    return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }
  function dur(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60;
    if (h) return `${h}시간 ${pad(m)}분`;
    if (m) return `${m}분 ${pad(s)}초`;
    return `${s}초`;
  }
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  // ---------- 상태 ----------
  const S = { t: 0, playing: false, speed: 60, sel: 0, tab: "ops", layers: { zone: true, table: true, person: true, items: false } };
  const tick = () => Math.min(N - 1, Math.max(0, Math.floor(S.t)));

  // ---------- 이미지 ----------
  const cache = new Map();
  function img(srcIdx) {
    let im = cache.get(srcIdx);
    if (!im) {
      im = new Image();
      im.src = `data/frames/${String(srcIdx).padStart(6, "0")}.jpg`;
      cache.set(srcIdx, im);
      if (cache.size > 900) cache.delete(cache.keys().next().value);
    }
    return im;
  }
  function whenLoaded(im, fn) { if (im.complete && im.naturalWidth) fn(); else im.addEventListener("load", fn, { once: true }); }

  function drawFrame(cv, srcIdx, opts = {}) {
    const ctx = cv.getContext("2d");
    const im = img(srcIdx);
    whenLoaded(im, () => {
      const W = cv.width, H = cv.height;
      ctx.drawImage(im, 0, 0, W, H);
      drawOverlay(ctx, ROWS[srcIdx], W, H, opts.layers || { zone: true, table: false, person: true, items: false }, opts.small);
      if (opts.stamp) {
        ctx.font = `600 ${Math.round(H / 14)}px ${CSSV("--mono") || "monospace"}`;
        const w = ctx.measureText(opts.stamp).width;
        ctx.fillStyle = "rgba(8,12,16,.75)";
        ctx.fillRect(8, H - H / 9 - 8, w + 16, H / 9);
        ctx.fillStyle = "#fff";
        ctx.fillText(opts.stamp, 16, H - 8 - H / 36);
      }
    });
  }

  function drawOverlay(ctx, row, W, H, ly, small) {
    const lw = small ? 2 : 3;
    const font = small ? 0 : Math.round(H / 34);
    const label = (x, y, text, color) => {
      if (!font) return;
      ctx.font = `600 ${font}px ${CSSV("--sans") || "sans-serif"}`;
      const w = ctx.measureText(text).width + 10;
      const yy = Math.max(0, y - font - 6);
      ctx.fillStyle = color; ctx.fillRect(x, yy, w, font + 6);
      ctx.fillStyle = "#fff"; ctx.fillText(text, x + 5, yy + font);
    };
    const R = (b) => [b[0] * W, b[1] * H, (b[2] - b[0]) * W, (b[3] - b[1]) * H];
    if (ly.zone && row.zone) {
      ctx.save(); ctx.setLineDash([12, 8]); ctx.lineWidth = lw; ctx.strokeStyle = CSSV("--zone");
      const z = R(row.zone);
      // 감지 영역은 화면 밖까지 넓어질 수 있어 안쪽으로 살짝 들여 그린다
      const zx = Math.max(2, z[0]), zy = Math.max(2, z[1]);
      const zw = Math.min(W - 2, z[0] + z[2]) - zx, zh = Math.min(H - 2, z[1] + z[3]) - zy;
      ctx.strokeRect(zx, zy, zw, zh);
      ctx.restore();
      // 이름표는 왼쪽 아래 안쪽 (왼쪽 위는 카메라 배지 자리)
      label(zx + 4, zy + zh - 6, "감지 영역 = 테이블 +30%", CSSV("--zone"));
    }
    if (ly.table && row.table) {
      ctx.lineWidth = lw; ctx.strokeStyle = CSSV("--table");
      const b = R(row.table); ctx.strokeRect(...b);
      label(b[0] + 4, Math.max(font + 12, b[1]) + font + 10, "테이블 (자동 검출)", CSSV("--table"));
    }
    if (ly.items) {
      for (const it of row.items) {
        ctx.lineWidth = Math.max(1, lw - 1); ctx.strokeStyle = CSSV("--items");
        const b = R(it.box); ctx.strokeRect(...b);
        if (!small) label(b[0], b[1], it.name, CSSV("--items"));
      }
    }
    if (ly.person) {
      for (const p of row.persons) {
        const b = R(p.box);
        ctx.save();
        ctx.lineWidth = lw + 1;
        if (p.inside) { ctx.strokeStyle = CSSV("--red"); }
        else { ctx.strokeStyle = "rgba(200,210,220,.9)"; ctx.setLineDash([8, 6]); }
        ctx.strokeRect(...b);
        ctx.restore();
        label(b[0], b[1], `사람 ${p.conf.toFixed(2)}${p.inside ? "" : " · 영역 밖"}`, p.inside ? CSSV("--red") : "#66727e");
      }
    }
  }

  // ---------- 화면 만들기 ----------
  function buildFloors() {
    const groups = [["3층", "3층 · 6인 룸 6개", "문 위 천장 카메라"], ["4층 오픈형", "4층 · 오픈형 4인 6개", "천장 카메라"], ["4층 독립형", "4층 · 독립형 6인 2개", "벽으로 막혀 룸 안 천장 카메라, 촬영 안내문 부착"]];
    const box = $("floors");
    groups.forEach(([key, title, note]) => {
      const f = el("div", "floor" + (key === "4층 독립형" ? " indep" : ""));
      f.append(el("div", "floor-head", `<h3>${title}</h3><span>${note}</span>`));
      const grid = el("div", "rooms");
      ROOMS.forEach((r, i) => {
        if (r.floor !== key) return;
        const b = el("button", "room");
        b.type = "button";
        b.innerHTML = `<div class="room-top"><span class="room-no">${r.no}</span>${r.live ? '<span class="room-tag live">실측</span>' : `<span class="room-tag">${r.kind}</span>`}</div>
          <div class="room-state"><span class="dot"></span><span class="st"></span></div>
          <div class="room-time"></div><div class="room-bar"><i></i></div><canvas width="200" height="12" aria-hidden="true"></canvas>`;
        b.addEventListener("click", () => select(i));
        r.tile = b;
        grid.append(b);
      });
      f.append(grid);
      box.append(f);
    });
  }

  function select(i) {
    S.sel = i;
    ROOMS.forEach((r, j) => r.tile.classList.toggle("sel", j === i));
    const r = ROOMS[i];
    $("dType").textContent = `${r.floor} · ${r.kind} · ${r.cam}`;
    $("dName").textContent = `${r.no}호 · ${r.story}`;
    lastEventsKey = "";
    render(true);
  }

  // ---------- 렌더 ----------
  let lastSec = -1, lastHeavy = 0, lastEventsKey = "";
  const prevState = new Map();

  function render(force = false) {
    const t = tick();
    $("clock").textContent = clock(t);
    $("dayLabel").textContent = `2026-12-07 (월) · ${bandAt(t).name} · 가상 운영`;
    if (!scrubbing) $("scrub").value = t;
    drawCam(t);
    const now = performance.now();
    if (!force && t === lastSec && now - lastHeavy < 400) return;
    lastSec = t; lastHeavy = now;
    renderTiles(t);
    renderKpis(t);
    renderDetail(t);
    renderAlerts(t);
    if (S.tab === "stats") renderStats(t);
    if (S.tab === "board") renderBoard(t);
    if (S.tab === "records" && force) renderRecords(t);
  }

  function drawCam(t) {
    const r = ROOMS[S.sel];
    const cv = $("cam");
    drawFrame(cv, r.src[t], { layers: S.layers });
    const real = r.real[t];
    const b = $("camSrc");
    b.className = "badge " + (real ? "live" : "sim");
    b.textContent = real ? `실측 카메라 · 1차.mp4 ${timer(r.src[t])}` : "가상 구성 · 실측 장면 반복";
    $("camTime").textContent = clock(t);
    const p = r.pend[t];
    const box = $("camConfirm");
    if (p > 0) {
      box.hidden = false;
      box.innerHTML = `사람이 안 보임 · 부재 확인 중 <b>${p}/${RULE.off}초</b><span class="pips">${Array.from({ length: RULE.off }, (_, k) => `<i class="${k < p ? "on" : ""}"></i>`).join("")}</span><span style="opacity:.7">가림·놓침 방지</span>`;
    } else if (p < 0) {
      box.hidden = false;
      box.innerHTML = `사람 감지 · 재석 확인 중 <b>${-p}/${RULE.on}초</b><span style="opacity:.7">오검출 방지</span>`;
    } else box.hidden = true;
  }

  function stateText(r, t) {
    const s = r.st[t];
    if (s === 0) return { main: "사용 중", sub: `${clock(r.occS[t], false)}부터 · ${dur(t - r.occS[t])}째` };
    const away = t - r.awayS[t];
    if (s === 1) return { main: "자리 비움", sub: `비운 지 ${timer(away)} · 초록까지 ${timer(RULE.limit - away)}` };
    if (r.empty[t]) return { main: "이용 가능", sub: "빈 책상 · 짐 없음" };
    return { main: "이용 가능", sub: `비운 지 ${dur(away)} · 바구니 이동 가능` };
  }

  function renderTiles(t) {
    const red = CSSV("--red"), yel = CSSV("--yellow"), grn = CSSV("--green"), line = CSSV("--line");
    const col = [red, yel, grn];
    ROOMS.forEach((r) => {
      const s = r.st[t];
      const tile = r.tile;
      const was = prevState.get(r.no);
      tile.classList.remove("s0", "s1", "s2");
      tile.classList.add("s" + s);
      if (was !== undefined && was !== s && S.playing) { tile.classList.remove("flash"); void tile.offsetWidth; tile.classList.add("flash"); }
      prevState.set(r.no, s);
      const tx = stateText(r, t);
      tile.querySelector(".st").textContent = tx.main;
      tile.querySelector(".room-time").textContent = tx.sub;
      const frac = s === 0 ? 0 : Math.min(1, (t - r.awayS[t]) / RULE.limit);
      tile.querySelector(".room-bar i").style.width = (frac * 100).toFixed(1) + "%";
      // 기록 띠 (14:00 → 지금)
      const cv = tile.querySelector("canvas");
      const ctx = cv.getContext("2d");
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.fillStyle = line; ctx.globalAlpha = .45; ctx.fillRect(0, 0, cv.width, cv.height); ctx.globalAlpha = 1;
      const pxPer = cv.width / N;
      for (const run of r.runs) {
        if (run.a > t) break;
        ctx.fillStyle = col[run.s];
        ctx.fillRect(run.a * pxPer, 0, Math.max(1, (Math.min(run.b, t + 1) - run.a) * pxPer), cv.height);
      }
    });
  }

  function renderKpis(t) {
    const cnt = [0, 0, 0];
    let stale = 0;
    ROOMS.forEach((r) => {
      cnt[r.st[t]]++;
      if (r.st[t] === 2 && ROWS[r.src[t]].items.length > 0) stale++;
    });
    const act = Math.round((cnt[0] / ROOMS.length) * 100);
    const sneaky = findSneaky(t).length;
    const changes = ALL_EVENTS.filter((e) => e.t <= t).length;
    $("kpis").innerHTML = `
      <div class="kpi st" style="--c:var(--red)"><span class="k">🔴 사용 중</span><span class="v">${cnt[0]}<small> / 14</small></span><span class="s">감지 영역 안에 사람 있음</span></div>
      <div class="kpi st" style="--c:var(--yellow)"><span class="k">🟡 자리 비움</span><span class="v">${cnt[1]}</span><span class="s">30분 미만 · 짐에 손대지 않음</span></div>
      <div class="kpi st" style="--c:var(--green)"><span class="k">🟢 이용 가능</span><span class="v">${cnt[2]}</span><span class="s">30분 이상 · 짐을 바구니로</span></div>      <div class="kpi"><span class="k">지금 룸 활성화율</span><span class="v">${act}<small>%</small></span>
        <div class="meter"><i style="width:${act}%"></i><b style="left:80%" title="목표 80%"></b></div><span class="s">목표 80% · 도입 전 50% 이하</span></div>
      <div class="kpi ${stale ? "bad" : "good"}"><span class="k">30분 넘게 짐만 있는 룸</span><span class="v">${stale}</span><span class="s">목표 0 · 초록불로 안내 중</span></div>
      <div class="kpi ${sneaky ? "bad" : ""}"><span class="k">얌체 이용 의심 · 상태 변화</span><span class="v">${sneaky}<small> 건 · ${changes}회</small></span><span class="s">30분 직전 짧은 재방문</span></div>`;
  }

  function renderDetail(t) {
    const r = ROOMS[S.sel];
    const s = r.st[t];
    const chip = $("dState");
    chip.className = "state-chip s" + s;
    chip.textContent = ST[s];
    const row = ROWS[r.src[t]];
    const inside = row.persons.filter((p) => p.inside).length;
    $("jIn").textContent = `${inside}명` + (row.persons.length > inside ? ` (영역 밖 ${row.persons.length - inside})` : "");
    const names = {};
    row.items.forEach((it) => (names[it.name] = (names[it.name] || 0) + 1));
    $("jItems").textContent = row.items.length ? `${row.items.length}개` : "없음";
    $("jItems").title = Object.entries(names).map(([k, v]) => `${k} ${v}`).join(", ");
    const away = s === 0 ? 0 : t - r.awayS[t];
    $("jAway").textContent = s === 0 ? "—" : timer(away);
    $("jLeft").textContent = s === 1 ? timer(RULE.limit - away) : (s === 2 ? "켜짐" : "—");
    const fill = $("limitFill");
    fill.style.width = (Math.min(1, away / RULE.limit) * 100).toFixed(1) + "%";
    fill.classList.toggle("full", s === 2);
    const since = clock(r.awayS[t]);
    const itemsTxt = row.items.length ? `짐 ${row.items.length}개(${Object.entries(names).map(([k, v]) => k + " " + v).join(", ")})가 남아 있습니다.` : "";
    const p = r.pend[t];
    $("jText").innerHTML = (s === 0 && p > 0)
      ? `사람이 <b>${p}초째</b> 보이지 않지만, 잠깐 가려졌거나 검출을 놓친 것일 수 있어 <b>${RULE.off}초</b>가 될 때까지는 사용 중으로 유지합니다. ${RULE.off}초가 되면 마지막으로 보인 <b>${clock(t - p)}</b>부터 부재로 기록합니다.`
      : (s !== 0 && p < 0)
      ? `사람이 다시 보이기 시작했습니다 (${-p}/${RULE.on}초). 오검출을 막기 위해 <b>${RULE.on}초</b> 연속으로 보이면 사용 중으로 바꾸고 부재 타이머를 0으로 돌립니다.`
      : s === 0
      ? `감지 영역 안에 사람 ${inside}명이 있어 <b>사용 중</b>입니다. ${clock(r.occS[t])}부터 계속 사람이 있었습니다. 문 앞 표시등은 빨강입니다.`
      : s === 1
        ? `<b>${since}</b>에 마지막으로 사람이 보였습니다. ${itemsTxt} 30분이 되는 <b>${clock(r.awayS[t] + RULE.limit)}</b>에 초록불로 바뀝니다. 그전까지는 짐에 손대지 않습니다.`
        : r.empty[t] ? `사람도 짐도 ${RULE.empty}초 넘게 보이지 않아 <b>빈 책상</b>으로 판정했습니다. 30분을 기다리지 않고 바로 초록불입니다.`
        : `<b>${since}</b>부터 ${dur(away)} 동안 사람이 없습니다. ${itemsTxt} 학교 30분 규칙에 따라 <b>짐을 룸 앞 회색 바구니로 옮기고 사용할 수 있습니다.</b> (귀중품은 룸 안 한쪽에 둡니다)`;
    renderTimeline(t);
    renderEvents(t);
  }

  function renderTimeline(t) {
    const r = ROOMS[S.sel];
    const cv = $("timeline");
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(cv.clientWidth * dpr), H = Math.round(56 * dpr);
    if (cv.width !== W) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    const col = [CSSV("--red"), CSSV("--yellow"), CSSV("--green")];
    const barY = 14 * dpr, barH = 26 * dpr;
    ctx.fillStyle = CSSV("--surface-2"); ctx.fillRect(0, barY, W, barH);
    // 아직 오지 않은 시간은 빗금
    ctx.save(); ctx.beginPath(); ctx.rect((t / N) * W, barY, W, barH); ctx.clip();
    ctx.strokeStyle = CSSV("--line"); ctx.lineWidth = 1 * dpr;
    for (let x = -H; x < W; x += 8 * dpr) { ctx.beginPath(); ctx.moveTo(x, barY + barH); ctx.lineTo(x + barH, barY); ctx.stroke(); }
    ctx.restore();
    for (const run of r.runs) {
      if (run.a > t) break;
      ctx.fillStyle = col[run.s];
      ctx.fillRect((run.a / N) * W, barY, Math.max(1, ((Math.min(run.b, t + 1) - run.a) / N) * W), barH);
    }
    // 1시간 눈금과 시간대 경계
    ctx.fillStyle = CSSV("--faint");
    for (let h = 1; h < N / 3600; h++) ctx.fillRect((h * 3600 / N) * W, barY + barH - 6 * dpr, 1 * dpr, 6 * dpr);
    ctx.font = `${10 * dpr}px ${CSSV("--sans")}`;
    BANDS.forEach((b, i) => {
      if (i > 0) { ctx.fillStyle = CSSV("--ink"); ctx.globalAlpha = .5; ctx.fillRect((b.a / N) * W, barY - 4 * dpr, 1.5 * dpr, barH + 8 * dpr); ctx.globalAlpha = 1; }
      ctx.fillStyle = CSSV("--muted");
      ctx.fillText(b.short, (b.a / N) * W + 4 * dpr, H - 3 * dpr);
    });
    // 실측 구간 표시
    if (r.live) {
      let a = -1, b = -1;
      for (let i = 0; i < N; i++) if (r.real[i]) { if (a < 0) a = i; b = i; }
      ctx.fillStyle = CSSV("--accent");
      ctx.fillRect((a / N) * W, 4 * dpr, ((b - a) / N) * W, 4 * dpr);
      ctx.font = `${11 * dpr}px ${CSSV("--sans")}`;
      ctx.fillText("실측 구간", (b / N) * W + 6 * dpr, 10 * dpr);
    }
    // 현재 위치
    const x = (t / N) * W;
    ctx.fillStyle = CSSV("--ink");
    ctx.fillRect(x - 1 * dpr, 2 * dpr, 2 * dpr, H - 4 * dpr);
    ctx.beginPath(); ctx.moveTo(x - 5 * dpr, 2 * dpr); ctx.lineTo(x + 5 * dpr, 2 * dpr); ctx.lineTo(x, 9 * dpr); ctx.fill();
  }

  function renderEvents(t) {
    const r = ROOMS[S.sel];
    const evs = r.events.filter((e) => e.t <= t).slice().reverse();
    const key = r.no + ":" + evs.length + ":" + Math.floor(t / 60);
    if (key === lastEventsKey) return;
    lastEventsKey = key;
    const ol = $("events");
    ol.innerHTML = "";
    if (!evs.length) { ol.append(el("li", "empty", "아직 상태 변화가 없습니다.")); return; }
    evs.slice(0, 12).forEach((e) => {
      const li = el("li", "ev");
      const cv = el("canvas"); cv.width = 224; cv.height = 126;
      const left = KEEP - (t - e.t);
      const first = e.from < 0;
      const why = e.to === 0 ? `사람이 ${RULE.on}초 연속 보여 재석으로 판정`
        : e.to === 1 ? `마지막으로 보인 ${clock(e.awaySince)}부터 부재 (${RULE.off}초 확인 후 판정)`
          : e.empty ? `사람도 짐도 ${RULE.empty}초 연속 없어 빈 책상으로 판정 (바로 초록)` : `${clock(e.awaySince)}부터 30분 경과`;
      const main = el("div", "ev-main");
      main.innerHTML = `<div class="ev-title"><span class="mono">${clock(e.t)}</span>${first ? '<span class="pill s' + e.to + '">' + ST[e.to] + "</span> 기록 시작" : `<span class="pill s${e.from}">${ST[e.from]}</span><span class="arrow">→</span><span class="pill s${e.to}">${ST[e.to]}</span>`}</div>
        <div class="ev-sub">${first ? "14:00 시스템 기록 시작 시점 상태" : why}</div>
        <div class="ev-sub">스냅샷 자동 삭제까지 ${timer(left)} · 이후 데이터만 남음</div>`;
      li.append(cv, main);
      li.addEventListener("click", () => openModal(r, e.t, `${r.no}호 · ${clock(e.t)} 상태 변화 스냅샷`,
        `사람은 모자이크된 상태로 저장됩니다. 저장 시각 ${clock(e.t)} → ${clock(e.t)}(다음 날) 자동 삭제. ${why}.`));
      ol.append(li);
      drawFrame(cv, r.src[e.t], { small: true, layers: { zone: false, table: false, person: true, items: false } });
    });
  }

  let lastAlertKey = "";
  function renderAlerts(t) {
    const items = [];
    ROOMS.forEach((r) => {
      r.events.forEach((e, i) => {
        if (e.t > t) return;
        if (e.to === 2 && e.items > 0) items.push({ t: e.t, room: r.no, cls: "g", text: SYS.after
          ? `초록불 · ${clock(e.awaySince, false)}부터 30분 비어 짐을 바구니로 옮기고 사용 가능`
          : `초록불 · ${clock(e.awaySince, false)}부터 30분 넘게 짐만 있음 (도입 전이라 아무도 안 치움)` });
        if (e.to === 0 && e.from === 2) {
          const green = e.awaySince + RULE.limit;
          const fresh = (r.newAt || []).some((x) => e.t >= x && e.t <= x + 70);
          items.push({ t: e.t, room: r.no, cls: "r", text: fresh
            ? `초록불 뒤 새 학생 · 초록불 ${dur(e.t - green)} 만에 짐을 바구니로 옮기고 사용 시작`
            : `초록불 뒤 주인 복귀 · 초록불 ${dur(e.t - green)} 동안 ${SYS.after ? "아무도 안 들어옴" : "짐만 있었음 (아무도 안 치움)"}` });
        }
      });
    });
    findSneaky(t).forEach((x) => items.push({ t: x.at + x.stay + RULE.off, room: x.room, cls: "w", text: `30분 직전 재방문 의심 · ${dur(x.awayLen)} 비움 → ${dur(x.stay)} 머묾 → 다시 나감` }));
    items.sort((a, b) => b.t - a.t);
    const key = items.length + ":" + (items[0] ? items[0].t : 0);
    if (key === lastAlertKey) return;
    lastAlertKey = key;
    const ol = $("alerts");
    ol.innerHTML = "";
    if (!items.length) { ol.append(el("li", "empty", "아직 알림이 없습니다.")); return; }
    items.slice(0, 6).forEach((a) => {
      const li = el("li", "alert " + a.cls, `<span class="mono">${clock(a.t)}</span><span class="no">${a.room}</span><span>${a.text}</span>`);
      li.addEventListener("click", () => select(ROOMS.findIndex((r) => r.no === a.room)));
      ol.append(li);
    });
  }

  function openModal(r, t, title, note) {
    $("modalTitle").textContent = title;
    $("modalNote").textContent = note;
    $("modal").hidden = false;
    drawFrame($("modalCanvas"), r.src[t], { layers: { zone: true, table: true, person: true, items: true }, stamp: `${r.no}호 ${clock(t)}` });
  }

  // ---------- 통계 ----------
  function samples(t) {
    const out = [];
    for (let k = 0; k * 600 <= t && k * 600 < N; k++) {
      const tk = k * 600;
      out.push({ tk, pct: (ROOMS.filter((r) => r.st[tk] === 0).length / ROOMS.length) * 100,
        away: ROOMS.filter((r) => r.st[tk] !== 0).length,
        stale: ROOMS.filter((r) => r.st[tk] === 2 && ROWS[r.src[tk]].items.length > 0).length });
    }
    return out;
  }

  function findSneaky(t) {
    // 25분 넘게(30분 전) 비웠다가 잠깐 들른 뒤 10분 안에 다시 나간 경우
    const out = [];
    ROOMS.forEach((r) => {
      const runs = r.runs.filter((x) => x.a <= t);
      for (let i = 1; i + 1 < runs.length; i++) {
        const occ = runs[i];
        if (occ.s !== 0 || occ.b > t) continue;
        const prevAway = runs[i - 1];
        const awayLen = occ.a - r.awayS[prevAway.b - 1];
        const nextIsAway = runs[i + 1] && runs[i + 1].s !== 0;
        if (prevAway.s === 1 && awayLen >= 25 * M && awayLen < RULE.limit && occ.b - occ.a < 10 * M && nextIsAway) {
          out.push({ room: r.no, at: occ.a, awayLen, stay: occ.b - occ.a });
        }
      }
    });
    return out.sort((a, b) => a.at - b.at);
  }

  function handovers(t) {
    const out = [];
    ROOMS.forEach((r) => {
      r.runs.forEach((run, i) => {
        if (run.s !== 2 || run.a > t) return;
        const next = r.runs[i + 1];
        // 14:00 전에 이미 초록이었던 룸도 있어 초록불 시각은 '비운 시각 + 30분'으로 계산한다
        const green = r.awayS[run.a] + RULE.limit;
        if (next && next.s === 0 && next.a <= t) out.push({ room: r.no, green, enter: next.a, wait: next.a - green });
        else out.push({ room: r.no, green, enter: null, wait: Math.min(run.b, t) - green });
      });
    });
    return out.sort((a, b) => a.green - b.green);
  }

  function renderStats(t) {
    const sm = samples(t);
    const win = sm.filter((d) => d.tk >= KPI_WIN[0] && d.tk < KPI_WIN[1]);
    const avg = win.length ? win.reduce((a, b) => a + b.pct, 0) / win.length : null;
    const hos = handovers(t);
    const done = hos.filter((h) => h.enter != null);
    const avgWait = done.length ? done.reduce((a, b) => a + b.wait, 0) / done.length : 0;
    const sneaky = findSneaky(t);
    let stale = 0;
    ROOMS.forEach((r) => { if (r.st[t] === 2 && ROWS[r.src[t]].items.length) stale++; });
    $("kpiBig").innerHTML = `
      <div class="kpi"><span class="k">저녁 7~10시 평균 룸 활성화율 (10분 표본 ${win.length}/18개)</span><span class="v">${avg == null ? "—" : avg.toFixed(0)}<small>%</small></span>
        <div class="meter"><i style="width:${avg || 0}%"></i><b style="left:80%"></b></div><span class="s">${avg == null ? "19:00부터 계산합니다" : "Primary KPI · 목표 80% 이상 (기획서 정의)"}</span></div>
      <div class="kpi ${stale ? "bad" : "good"}"><span class="k">지금 30분 넘게 짐만 있는 룸</span><span class="v">${stale}</span><span class="s">Primary KPI · 목표 0개 (도입 전 7개 이상)</span></div>
      <div class="kpi"><span class="k">초록불 → 다시 사용까지 평균</span><span class="v">${done.length ? Math.round(avgWait / 60) : "—"}<small>분</small></span><span class="s">초록불 ${hos.length}회 중 다시 사용 ${done.length}회 · ${SYS.after ? "도입 후" : "도입 전(주인이 올 때까지)"}</span></div>
      <div class="kpi ${sneaky.length ? "bad" : ""}"><span class="k">30분 직전 재방문 의심</span><span class="v">${sneaky.length}<small>건</small></span><span class="s">2027년 누적 부재 기준 도입 판단 자료</span></div>`;
    drawActChart(sm);
    renderBands(t, sm);
    renderCompare();
    // 룸별 상태 비율
    const sb = $("shareBars"); sb.innerHTML = "";
    const col = ["var(--red)", "var(--yellow)", "var(--green)"];
    ROOMS.forEach((r) => {
      const c = [0, 0, 0];
      for (const run of r.runs) { if (run.a > t) break; c[run.s] += Math.min(run.b, t + 1) - run.a; }
      const tot = c[0] + c[1] + c[2] || 1;
      const row = el("div", "share-row");
      row.innerHTML = `<span class="no">${r.no}</span><span class="bar">${c.map((v, i) => `<i style="width:${(v / tot) * 100}%;background:${col[i]}" title="${ST[i]} ${dur(v)}"></i>`).join("")}</span><span class="pct">${Math.round((c[0] / tot) * 100)}%</span>`;
      sb.append(row);
    });
    const sn = $("sneaky");
    sn.innerHTML = sneaky.length ? "" : '<p class="empty">아직 없습니다.</p>';
    const list = el("div", "list");
    sneaky.forEach((x) => list.append(el("div", "list-row", `<span class="no">${x.room}</span><span>${dur(x.awayLen)} 비움 → ${dur(x.stay)} 머묾 → 다시 나감</span><span class="mono">${clock(x.at)}</span>`)));
    sn.append(list);
    const ho = $("handover");
    ho.innerHTML = hos.length ? "" : '<p class="empty">아직 초록불이 켜진 룸이 없습니다.</p>';
    const l2 = el("div", "list");
    hos.forEach((h) => l2.append(el("div", "list-row", `<span class="no">${h.room}</span><span>${clock(h.green)} 초록불 → ${h.enter != null ? clock(h.enter) + " 입실" : "아직 비어 있음"}</span><span class="mono">${dur(h.wait)}</span>`)));
    ho.append(l2);
  }

  function drawActChart(sm) {
    const svg = $("actChart");
    const W = 720, H = 260, L = 44, Rr = 16, T = 22, B = 30;
    const K = N / 600;
    const iw = W - L - Rr, ih = H - T - B;
    const x = (k) => L + (k + 0.5) * (iw / K);
    const y = (p) => T + ih - (p / 100) * ih;
    let s = "";
    const bx = (sec) => L + (sec / N) * iw;
    BANDS.forEach((b, i) => {
      s += `<rect x="${bx(b.a)}" y="${T}" width="${bx(b.b) - bx(b.a)}" height="${ih}" fill="${i % 2 ? "var(--surface-2)" : "transparent"}"/>`;
      s += `<text x="${bx(b.a) + 4}" y="${T - 8}" style="fill:var(--ink)">${b.short}</text>`;
    });
    s += `<rect x="${bx(KPI_WIN[0])}" y="${T + ih + 2}" width="${bx(KPI_WIN[1]) - bx(KPI_WIN[0])}" height="4" fill="var(--accent)"/>`;
    for (const p of [0, 25, 50, 75, 100]) {
      s += `<line x1="${L}" x2="${W - Rr}" y1="${y(p)}" y2="${y(p)}" stroke="var(--line)" stroke-width="1"/>`;
      s += `<text x="${L - 8}" y="${y(p) + 4}" text-anchor="end">${p}%</text>`;
    }
    const bw = (iw / K) * 0.66;
    sm.forEach((d, k) => {
      const c = d.pct >= 80 ? "var(--green)" : d.pct >= 50 ? "var(--accent)" : "var(--yellow)";
      s += `<rect x="${x(k) - bw / 2}" y="${y(d.pct)}" width="${bw}" height="${y(0) - y(d.pct)}" rx="2" fill="${c}"><title>${clock(d.tk, false)} · ${d.pct.toFixed(0)}%</title></rect>`;
    });
    for (let k = 0; k <= K; k += 6) s += `<text x="${L + k * (iw / K)}" y="${H - 6}" text-anchor="middle">${clock(k * 600, false)}</text>`;
    s += `<line x1="${L}" x2="${W - Rr}" y1="${y(80)}" y2="${y(80)}" stroke="var(--green)" stroke-dasharray="6 4" stroke-width="1.5"/>`;
    s += `<text x="${W - Rr}" y="${y(80) - 6}" text-anchor="end" style="fill:var(--green)">목표 80%</text>`;
    s += `<line x1="${L}" x2="${W - Rr}" y1="${y(50)}" y2="${y(50)}" stroke="var(--faint)" stroke-dasharray="3 4" stroke-width="1.5"/>`;
    s += `<text x="${W - Rr}" y="${y(50) - 6}" text-anchor="end">도입 전 50% 이하</text>`;
    svg.innerHTML = s;
  }

  function renderBands(t, sm) {
    const rows = BANDS.map((b) => {
      const ss = sm.filter((d) => d.tk >= b.a && d.tk < b.b);
      const greens = ALL_EVENTS.filter((e) => e.to === 2 && e.items > 0 && e.t >= b.a && e.t < b.b && e.t <= t).length;
      const leaves = ALL_EVENTS.filter((e) => e.to === 1 && e.from === 0 && e.t >= b.a && e.t < b.b && e.t <= t).length;
      const avg = (k) => (ss.length ? ss.reduce((a, d) => a + d[k], 0) / ss.length : null);
      return { b, n: ss.length, act: avg("pct"), away: avg("away"), stale: avg("stale"), greens, leaves };
    });
    $("bands").innerHTML = `<table class="band-table"><thead><tr><th>시간대</th><th>평균 활성화율</th><th>평균 '짐만 남은 룸'</th><th>짐만 30분+ 룸</th><th>자리 비움 시작</th><th>짐 남은 룸 초록불</th></tr></thead><tbody>${
      rows.map((r) => `<tr><td><b>${r.b.name}</b><span class="hint"> ${clock(r.b.a, false)}~${clock(r.b.b, false)}</span></td>${r.n ? `<td class="mono">${r.act.toFixed(0)}%</td><td class="mono">${r.away.toFixed(1)}개</td><td class="mono">${r.stale.toFixed(1)}개</td><td class="mono">${r.leaves}회</td><td class="mono">${r.greens}회</td>` : '<td colspan="5" class="hint">아직 재생 전</td>'}</tr>`).join("")
    }</tbody></table>`;
  }

  // ---------- 기록 ----------
  function recordRows(t, filter) {
    const rows = [];
    for (let m = 0; m * 60 <= t && m * 60 < N; m++) {
      const tm = m * 60;
      ROOMS.forEach((r) => {
        if (filter !== "all" && r.no !== filter) return;
        const s = r.st[tm];
        rows.push({ room: r.no, t: tm, person: ROWS[r.src[tm]].present, s, away: s === 0 ? 0 : tm - r.awayS[tm], change: m > 0 && r.st[tm - 60] !== s });
      });
    }
    return rows;
  }
  function renderRecords(t) {
    const f = $("recRoom").value;
    const rows = recordRows(t, f).reverse().slice(0, 1500);
    $("recBody").innerHTML = rows.map((x) => `<tr class="${x.change ? "change" : ""}"><td class="mono">${x.room}</td><td class="mono">2026-12-07 ${clock(x.t)}</td><td>${x.person ? "있음" : "없음"}</td><td><span class="pill s${x.s}">${ST[x.s]}</span></td><td class="mono">${x.s === 0 ? "—" : dur(x.away)}</td></tr>`).join("");
  }
  function csv() {
    const rows = recordRows(tick(), $("recRoom").value);
    const lines = ["룸,시각,사람,상태,비운시간(초)"].concat(rows.map((x) => `${x.room},2026-12-07 ${clock(x.t)},${x.person ? "있음" : "없음"},${ST[x.s]},${x.away}`));
    const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `스터디룸_부재기록_${clock(tick()).replace(/:/g, "")}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------- 부재 기록 조회 ----------
  function lookup() {
    const r = ROOMS[+$("lkRoom").value];
    const [hh, mm, ss] = ($("lkTime").value || "20:30:00").split(":").map(Number);
    let q = hh * 3600 + mm * 60 + (ss || 0) - START;
    const out = $("lkResult");
    if (q < 0 || q >= N) { out.innerHTML = '<p class="empty">14:00:00 ~ 22:59:59 사이 시각을 넣어 주세요.</p>'; return; }
    const future = q > tick();
    const s = r.st[q];
    const away = s === 0 ? 0 : q - r.awayS[q];
    const lastSeen = s === 0 ? q : r.awayS[q];
    const greenAt = r.awayS[q] + RULE.limit;
    const head = s === 0
      ? `<div class="big">사용 중이었습니다</div><p>${clock(r.occS[q])}부터 감지 영역 안에 사람이 계속 있었습니다. 이 시각에는 짐을 옮길 수 없습니다.</p>`
      : s === 1
        ? `<div class="big">자리 비움 ${dur(away)}째 · 아직 30분 전</div><p>마지막으로 사람이 보인 시각은 ${clock(r.awayS[q])}입니다. 초록불은 ${clock(greenAt)}에 켜질 예정이었으므로, 이 시각에는 짐에 손대면 안 됩니다.</p>`
        : `<div class="big">이용 가능 · ${dur(away)} 동안 비어 있었습니다</div><p>${clock(r.awayS[q])} 이후 사람이 보이지 않았고, ${clock(greenAt)}에 30분이 지나 초록불이 켜졌습니다. 학교 30분 규칙에 따라 짐을 바구니로 옮길 수 있는 시각이었습니다.</p>`;
    const cert = `[부재 기록 확인]
룸        : 융합교육관 ${r.no}호 (${r.floor} ${r.kind})
조회 시각 : 2026-12-07 ${clock(q)}
상태      : ${ST_LONG[s]}${s !== 0 ? ` (짐 ${ROWS[r.src[q]].items.length}개 남음)` : ""}
${s === 0 ? `재석 시작 : ${clock(r.occS[q])}` : `마지막 재석: ${clock(r.awayS[q])}
비운 시간 : ${dur(away)}
초록불    : ${s === 2 ? clock(greenAt) + " 점등" : clock(greenAt) + " 점등 예정이었음"}`}
판정 기준 : 학교 30분 자리 비움 규칙 (사람 ${RULE.off}초 연속 미검출 시 부재)
근거      : 시스템 부재 기록 (사람 모자이크 스냅샷은 1일 후 삭제, 이후 데이터만 보관)`;
    out.innerHTML = `
      ${future ? '<p class="empty">※ 아직 재생하지 않은 시각입니다. 실제 운영에서는 지난 기록만 조회됩니다.</p>' : ""}
      <div class="verdict s${s}">${head}</div>
      <div class="facts">
        <div class="fact"><div class="k">조회 시각</div><div class="v">${clock(q)}</div></div>
        <div class="fact"><div class="k">${s === 0 ? "재석 시작" : "마지막으로 사람이 보인 시각"}</div><div class="v">${clock(s === 0 ? r.occS[q] : r.awayS[q])}</div></div>
        <div class="fact"><div class="k">비운 시간</div><div class="v">${s === 0 ? "—" : timer(away)}</div></div>
        <div class="fact"><div class="k">초록불</div><div class="v">${s === 0 ? "—" : clock(greenAt)}</div></div>
      </div>
      <div class="evidence">
        <figure><canvas width="640" height="360" id="evA"></canvas><figcaption>${s === 0 ? "조회 시각 화면" : `마지막으로 사람이 보인 순간 (${clock(lastSeen)})`}</figcaption></figure>
        <figure><canvas width="640" height="360" id="evB"></canvas><figcaption>조회 시각 화면 (${clock(q)}) · 감지 영역과 짐 표시</figcaption></figure>
      </div>
      <div><div class="block-title"><span>기록 요약</span><button class="btn" id="certCopy">복사</button></div><pre class="cert" id="certText"></pre></div>`;
    $("certText").textContent = cert;
    $("certCopy").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(cert); $("certCopy").textContent = "복사됨"; }
      catch { const rg = document.createRange(); rg.selectNodeContents($("certText")); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(rg); }
    });
    const lsT = Math.max(0, Math.min(N - 1, lastSeen));
    drawFrame($("evA"), r.src[lsT], { layers: { zone: true, table: false, person: true, items: false }, stamp: clock(lsT) });
    drawFrame($("evB"), r.src[q], { layers: { zone: true, table: false, person: true, items: true }, stamp: clock(q) });
  }

  // ---------- 1층 현황판 ----------
  function renderBoard(t) {
    const free = ROOMS.filter((r) => r.st[t] === 2).length;
    const groups = ["3층", "4층 오픈형", "4층 독립형"];
    let h = `<div class="board-head"><div><div class="sub">융합교육관 1층 · 스터디룸 현황 (2027-02-15 설치 예정 화면) · ${bandAt(t).name}</div><h2>지금 바로 쓸 수 있는 룸</h2></div>
      <div class="count"><b>${free}</b>개 · ${clock(t, false)} 기준</div></div>`;
    groups.forEach((g) => {
      h += `<div class="board-floor"><h3>${g}</h3><div class="board-rooms">`;
      ROOMS.filter((r) => r.floor === g).forEach((r) => {
        const s = r.st[t];
        const away = t - r.awayS[t];
        const ex = s === 0 ? "사용 중" : s === 1 ? `약 ${Math.ceil((RULE.limit - away) / 60)}분 뒤 이용 가능` : "짐은 앞 회색 바구니로";
        h += `<div class="board-room s${s}"><span class="no">${r.no}</span><span class="st">${ST[s]}</span><span class="ex">${s === 0 ? r.kind : ex}</span></div>`;
      });
      h += `</div></div>`;
    });
    h += `<div class="board-legend"><span class="r">사용 중</span><span class="y">자리 비움 (30분 미만 · 짐에 손대지 않기)</span><span class="g">이용 가능 (30분 이상 · 짐을 바구니로 옮기고 사용)</span></div>`;
    $("board").innerHTML = h;
  }

  // ---------- 재생 ----------
  let lastFrame = null;
  function loop(ts) {
    if (S.playing) {
      if (lastFrame != null) S.t = Math.min(N - 1, S.t + ((ts - lastFrame) / 1000) * S.speed);
      if (S.t >= N - 1) setPlaying(false);
    }
    lastFrame = ts;
    render();
    // 다음 몇 초 화면 미리 읽기
    const r = ROOMS[S.sel], t = tick();
    for (let k = 1; k <= Math.min(6, S.speed); k++) if (t + k < N) img(r.src[t + k]);
    requestAnimationFrame(loop);
  }
  function setPlaying(p) {
    S.playing = p;
    $("btnPlay").textContent = p ? "일시정지" : "재생";
    $("btnPlay").setAttribute("aria-label", p ? "일시정지" : "재생");
  }
  function seek(t) { S.t = Math.max(0, Math.min(N - 1, t)); lastEventsKey = ""; render(true); }

  // ---------- 위쪽 시간 막대 ----------
  let scrubbing = false;
  function drawScrubBg() {
    // 분마다 전체 14개 룸의 상태 비율을 세로로 쌓아 저녁 흐름을 한눈에 보여 준다
    const cv = $("scrubBg");
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(cv.clientWidth * dpr)), H = Math.round(12 * dpr);
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    const col = [CSSV("--red"), CSSV("--yellow"), CSSV("--green")];
    const mins = N / 60;
    for (let m = 0; m < mins; m++) {
      const c = [0, 0, 0];
      ROOMS.forEach((r) => c[r.st[m * 60]]++);
      const x0 = (m / mins) * W, w = Math.ceil(W / mins);
      let y = H;
      for (const s of [0, 1, 2]) {
        const h = (c[s] / ROOMS.length) * H;
        ctx.fillStyle = col[s]; ctx.globalAlpha = 0.85;
        ctx.fillRect(x0, y - h, w, h);
        y -= h;
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = CSSV("--ink");
    BANDS.slice(1).forEach((b) => ctx.fillRect((b.a / N) * W - dpr, 0, 2 * dpr, H));
  }
  function parseClock(v) {
    const [h, m, s] = (v || "").split(":").map(Number);
    if (Number.isNaN(h) || Number.isNaN(m)) return null;
    return h * 3600 + m * 60 + (s || 0) - START;
  }

  // ---------- 시나리오 바꾸기 ----------
  function setScenario(mode, seed) {
    applyScenario(mode, seed);
    prevState.clear();
    lastEventsKey = ""; lastAlertKey = "";
    $("scnMode").value = mode;
    $("seedLabel").textContent = mode === "random" ? `시드 #${seed}` : `시드 #${CURATED_SEED}`;
    $("btnReroll").textContent = mode === "random" ? "새로 만들기" : "무작위로 만들기";
    $("scenarioNote").textContent = (mode === "random" ? `무작위 시나리오 #${seed}` : "추천 시나리오")
      + ` · ${SYS.after ? "도입 후: 초록불 뒤 짐 치우고 사용" : "도입 전: 아무도 짐을 안 치움"} · ${DATA.meta.video} 실측 장면으로 만든 가상 운영`;
    document.querySelectorAll(".sys button").forEach((b) => b.classList.toggle("on", (b.dataset.sys === "after") === SYS.after));
    document.body.classList.toggle("sys-after", SYS.after);
    try {
      const qs = new URLSearchParams();
      if (mode === "random") qs.set("seed", seed);
      if (SYS.after) qs.set("sys", "after");
      history.replaceState(null, "", qs.toString() ? `?${qs}` : location.pathname);
    } catch { /* 파일로 열면 무시 */ }
    compareCache = null;
    drawScrubBg();
    select(S.sel);
  }
  const newSeed = () => 1000 + Math.floor(Math.random() * 899000);

  // 같은 시드로 도입 전/후 하루를 각각 만들어 비교한다 (화면의 룸과 따로 계산)
  let compareCache = null;
  function compareDays() {
    const s = SCN.mode === "curated" ? CURATED_SEED : SCN.seed;
    if (compareCache && compareCache.s === s) return compareCache;
    const days = { before: buildDay(s, false), after: buildDay(s, true) };
    const stat = (R, a, b) => {
      let act = 0, stale = 0, c = 0;
      for (let t = a; t < b; t += 300) {
        act += R.filter((r) => r.st[t] === 0).length;
        stale += R.filter((r) => r.st[t] === 2 && ROWS[r.src[t]].items.length > 0).length;
        c++;
      }
      // 이 시간대에 켜진 초록불이 다시 사용되기까지 걸린 시간 (마감까지 안 쓰인 것은 뺌)
      const waits = [];
      R.forEach((r) => {
        for (let t = Math.max(1, a); t < b; t++) {
          if (r.st[t] !== 2 || r.st[t - 1] === 2) continue;
          let e = t; while (e < N && r.st[e] === 2) e++;
          if (e < N) waits.push(e - t);
        }
      });
      const wait = waits.length ? waits.reduce((x, y) => x + y, 0) / waits.length : null;
      return { act: (act / c / R.length) * 100, stale: stale / c, wait, greens: waits.length };
    };
    const rows = [...BANDS.map((b) => ({ name: b.name, range: `${clock(b.a, false)}~${clock(b.b, false)}`, a: b.a, b: b.b })),
      { name: "기획서 KPI 구간", range: "19:00~22:00", a: KPI_WIN[0], b: KPI_WIN[1] }]
      .map((x) => ({ ...x, before: stat(days.before, x.a, x.b), after: stat(days.after, x.a, x.b) }));
    const takeovers = days.after.reduce((n, r) => n + (r.story.match(/초록불 뒤 짐 치우고 새 학생(?: (\d+)회)?/) ? +(r.story.match(/초록불 뒤 짐 치우고 새 학생(?: (\d+)회)?/)[1] || 1) : 0), 0);
    compareCache = { s, rows, takeovers };
    return compareCache;
  }
  function renderCompare() {
    const c = compareDays();
    const arrow = (b, a, unit, digits = 0, goodUp = true) => {
      const d = a - b;
      const cls = Math.abs(d) < (unit === "%" ? 1 : 0.1) ? "" : ((d > 0) === goodUp ? "up-good" : "up-bad");
      return `<span class="mono">${b.toFixed(digits)}${unit}</span> → <b class="mono ${cls}">${a.toFixed(digits)}${unit}</b>`;
    };
    const waitCell = (b, a) => {
      const m = (x) => (x == null ? "없음" : `${Math.round(x / 60)}분`);
      const cls = b.wait != null && a.wait != null && a.wait < b.wait - 60 ? "up-good" : "";
      return `<span class="mono">${m(b.wait)}</span> → <b class="mono ${cls}">${m(a.wait)}</b>`;
    };
    $("compare").innerHTML = `<table class="band-table"><thead><tr><th>시간대</th><th>활성화율 (도입 전 → 후)</th><th>30분 넘게 짐만 있는 룸 평균</th><th>초록불 → 다시 사용 평균</th></tr></thead><tbody>${
      c.rows.map((r) => `<tr><td><b>${r.name}</b><span class="hint"> ${r.range}</span></td><td>${arrow(r.before.act, r.after.act, "%")}</td><td>${arrow(r.before.stale, r.after.stale, "개", 1, false)}</td><td>${waitCell(r.before, r.after)}</td></tr>`).join("")
    }</tbody></table><p class="note">같은 시드라 주인들이 나가고 돌아오는 시각은 같고, 도입 후에는 초록불이 켜진 룸에 다른 학생이 짐을 바구니로 옮기고 5분 안에 들어옵니다 (오늘 ${c.takeovers}번). 짐 두고 오래 비우는 룸이 가장 많은 저녁 식사 시간(17:00~19:30)에 차이가 가장 큽니다.</p>`;
  }

  // ---------- 이벤트 연결 ----------
  function wire() {
    $("btnPlay").addEventListener("click", () => setPlaying(!S.playing));
    $("btnBack").addEventListener("click", () => seek(S.t - 60));
    $("btnNext").addEventListener("click", () => {
      const t = tick();
      const own = ROOMS[S.sel].events.find((e) => e.t > t + 1);
      const any = ALL_EVENTS.find((e) => e.t > t + 1);
      const target = own || any;
      if (!target) return;
      if (!own && any) select(ROOMS.findIndex((r) => r.no === any.room));
      seek(Math.max(0, target.t - (target.to === 1 ? RULE.off + 3 : 3)));
    });
    document.querySelectorAll(".speed button").forEach((b) => b.addEventListener("click", () => {
      S.speed = +b.dataset.speed;
      document.querySelectorAll(".speed button").forEach((x) => x.classList.toggle("on", x === b));
    }));
    document.querySelectorAll(".speed button").forEach((x) => x.classList.toggle("on", +x.dataset.speed === S.speed));
    document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
      S.tab = b.dataset.tab;
      document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
      document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== "tab-" + S.tab));
      if (S.tab === "lookup") { $("lkRoom").value = S.sel; $("lkTime").value = clock(tick()); lookup(); }
      render(true);
      if (S.tab === "ops") renderTimeline(tick());
    }));
    $("timeline").addEventListener("click", (ev) => {
      const rect = ev.currentTarget.getBoundingClientRect();
      seek(((ev.clientX - rect.left) / rect.width) * N);
    });
    for (const [id, k] of [["lyZone", "zone"], ["lyTable", "table"], ["lyPerson", "person"], ["lyItems", "items"]]) {
      $(id).checked = S.layers[k];
      $(id).addEventListener("change", (e) => { S.layers[k] = e.target.checked; render(true); });
    }
    const opts = ROOMS.map((r, i) => `<option value="${i}">${r.no}호 · ${r.floor}</option>`).join("");
    $("lkRoom").innerHTML = opts;
    $("lkGo").addEventListener("click", lookup);
    $("lkNow").addEventListener("click", () => { $("lkTime").value = clock(tick()); lookup(); });
    $("recRoom").innerHTML = `<option value="all">전체 룸</option>` + ROOMS.map((r) => `<option value="${r.no}">${r.no}호</option>`).join("");
    $("recRoom").addEventListener("change", () => renderRecords(tick()));
    $("recCsv").addEventListener("click", csv);
    $("modalClose").addEventListener("click", () => ($("modal").hidden = true));
    $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("modal").hidden = true; });
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea")) return;
      if (e.code === "Space") { e.preventDefault(); setPlaying(!S.playing); }
      if (e.key === "ArrowLeft") seek(S.t - 60);
      if (e.key === "ArrowRight") seek(S.t + 60);
      if (e.key === "Escape") $("modal").hidden = true;
    });
    window.addEventListener("resize", () => { renderTimeline(tick()); drawScrubBg(); fitSticky(); });

    // 시간 이동: 막대 끌기, 시각 입력, 빠른 버튼
    const scrub = $("scrub");
    scrub.addEventListener("input", () => { scrubbing = true; seek(+scrub.value); });
    scrub.addEventListener("change", () => { scrubbing = false; });
    $("jumpGo").addEventListener("click", () => { const t = parseClock($("jumpTime").value); if (t != null) seek(t); });
    $("jumpTime").addEventListener("keydown", (e) => { if (e.key === "Enter") $("jumpGo").click(); });
    const quick = [[0, "14:00 수업"], [180, "17:00 저녁 식사"], [330, "19:30 저녁 자습"], [360, "20:00"], [420, "21:00"], [480, "22:00"]];
    $("quick").innerHTML = quick.map(([m, label]) => `<button type="button" data-m="${m}">${label}</button>`).join("");
    $("quick").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => seek(+b.dataset.m * 60)));


    // 도입 전/후 비교 스위치
    document.querySelectorAll(".sys button").forEach((b) => b.addEventListener("click", () => {
      SYS.after = b.dataset.sys === "after";
      setScenario(SCN.mode, SCN.seed);
      render(true);
    }));

    // 시나리오
    $("scnMode").addEventListener("change", (e) => setScenario(e.target.value, e.target.value === "random" ? newSeed() : CURATED_SEED));
    $("btnReroll").addEventListener("click", () => setScenario("random", newSeed()));
  }

  // 브라우저 콘솔에서 시나리오를 점검할 때 쓰는 손잡이
  window.__demo = { rooms: ROOMS, setScenario, events: () => ALL_EVENTS, buildDay, SYS };

  function fitSticky() {
    // 위 고정 바 높이에 맞춰 상세 패널이 붙는 위치를 정한다
    document.documentElement.style.setProperty("--top-h", document.querySelector(".topbar").offsetHeight + "px");
  }

  buildFloors();
  wire();
  fitSticky();
  {
    const qs = new URLSearchParams(location.search);
    SYS.after = qs.get("sys") === "after";
    const q = qs.get("seed");
    if (q && /^\d+$/.test(q)) setScenario("random", +q);
    else setScenario("curated", CURATED_SEED);
  }
  select(0);
  // 주소로 바로 열기: ?t=18:30 (시각), &tab=stats (탭), &play=0 (멈춘 채로)
  const qs0 = INIT_QS;
  const tq = qs0.get("t");
  if (tq && /^\d{1,2}:\d{2}$/.test(tq)) { const [h, m] = tq.split(":").map(Number); seek(h * 3600 + m * 60 - START); }
  const tabq = qs0.get("tab");
  if (tabq) document.querySelector(`.tabs button[data-tab="${tabq}"]`)?.click();
  setPlaying(qs0.get("play") !== "0");
  requestAnimationFrame(loop);
})();
