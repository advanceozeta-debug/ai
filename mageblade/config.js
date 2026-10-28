// ============================================================
//  config.js — 「은월의 마검사」 미리보기 설정값
//  재생 속도, 공격 판정 시점·범위, 능력치를 코드와 분리해 여기서만 고친다.
//  좌표 단위는 게임 픽셀. 판정 상자는 "오른쪽을 볼 때" 기준점(발 가운데)에서 잰 값이고,
//  왼쪽을 보면 자동으로 좌우 반전된다.
// ============================================================
window.MAGEBLADE_CONFIG = {
  view: { width: 320, height: 180, groundY: 158 },   // 화면은 정수 배율로만 확대

  // 동작별 재생 속도 (프레임당 ms). 비워 두면 mageblade_sheet.json 값을 그대로 씀
  animations: {
    idle:   { frameMs: 200 },
    run:    { frameMs: 80 },
    attack: { frameMs: [90, 150, 110, 170] },  // 자세 잡기 → 치켜들기(멈칫) → 내려베기 → 마무리
    skill:  { frameMs: [120, 180, 220] },
    hit:    { frameMs: [110, 180] },
    down:   { frameMs: [160, 210, 280] },
  },

  // 그림 위에 더하는 실제 몸 움직임(px). x 는 바라보는 방향 기준 (+ = 앞)
  //  dx/dy : 프레임마다 몸을 그리는 위치를 옮김 (발 기준점은 그대로, 판정에는 영향 없음)
  //  step  : 그 프레임에 들어설 때 캐릭터가 실제로 이동하는 거리 (판정도 같이 이동)
  motion: {
    idle:   { dy: [0, 0, 0, 0, 0] },
    run:    { dy: [0, -1, -2, -1, 0, -1, -2, -1] },      // 한 걸음마다 몸이 오르내림
    attack: { dx: [0, -2, 3, 1], dy: [0, -1, 0, 0], step: [0, 0, 7, 2] },  // 뒤로 젖혔다가 앞으로 내딛으며 베기
    skill:  { dx: [0, -2, -1], step: [0, 0, 0] },        // 쏘는 순간 반동
    hit:    { dx: [-3, -1], dy: [-1, 0] },               // 맞고 뒤로 젖혀짐
    down:   { dx: [-1, -3, -4], step: [0, -2, -2] },
  },
  hitFlashMs: 90,             // 맞는 순간 하얗게 번쩍 (깜빡이는 잔상 효과는 쓰지 않음)

  // 상태 우선순위: 높은 상태가 낮은 상태를 끊는다 (같은 순위는 못 끊음)
  priority: { idle: 0, run: 0, attack: 2, skill: 2, hit: 3, down: 4 },

  player: {
    name: "은월의 마검사", grade: 6, role: "근접 공격형",
    maxHp: 100,
    moveSpeed: 92,            // 픽셀/초
    hurtbox: { x: -9, y: -54, w: 18, h: 54 },   // 맞는 영역 (몸통)
    invulnMs: 700,            // 맞은 뒤 무적 시간
    knockback: 70,
  },

  attack: {
    damage: 14,
    activeFrames: [2],        // 0부터 셈: 3번째 프레임(내려베기)에서만 판정
    hitbox: { x: 2, y: -46, w: 30, h: 40 },     // 공격 판정 (그림 크기와 별개)
    knockback: 90,
    lockMove: true,
  },

  skill: {
    cooldownMs: 3500,
    castFrame: 1,             // 2번째 프레임에서 장판 + 투사체 발사
    field: { offsetX: 0, offsetY: -6 },          // 장판: 발밑에 표시만 (판정 없음)
    projectile: {
      speed: 230, maxMs: 1100,
      spawn: { x: 18, y: -34 },
      hitbox: { x: -10, y: -7, w: 20, h: 14 },   // 투사체 판정 (그림 78×26 보다 작게)
      damage: 8,
    },
    explosion: {
      damageFrame: 1,         // 폭발 2번째 프레임에 판정
      hitbox: { x: -22, y: -34, w: 44, h: 34 },  // 폭발 판정 (그림 76×60 보다 작게)
      damage: 26,
      knockback: 120,
    },
    lockMove: true,
  },

  hit: { lockMove: true },

  enemy: {
    baseHp: 30, hpPerRoom: 8,
    speed: [28, 44],
    contactDamage: 10,
    hurtbox: { x: -8, y: -30, w: 16, h: 30 },   // 그림자 망령 몸통
    perRoom: [3, 4, 5, 6, 7, 8],
  },

  upgrades: [
    { id: "atk", name: "월영검 연마", desc: "공격·스킬 피해 +20%" },
    { id: "spd", name: "은빛 발걸음", desc: "이동 속도 +15%" },
    { id: "cd", name: "마력 순환", desc: "스킬 재사용 대기 -20%" },
    { id: "hp", name: "남색 장포 수선", desc: "최대 체력 +20, 체력 모두 회복" },
  ],
};
