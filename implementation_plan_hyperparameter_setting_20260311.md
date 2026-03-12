# Hyperparameter Optimization for EmotionHeart+ Finetuning

## Background

[finetune.py](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/models/emotionheart/finetune.py) 파인튜닝 성능(`best_test_f1`)을 극대화하기 위해 [iemocap.yaml](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/config/iemocap.yaml)의 주요 하이퍼파라미터를 체계적으로 탐색한다.
모든 나머지 파라미터(모델 구조, 스케줄러, batch_size 등)는 고정한다.

## 탐색 파라미터 & 우선순위

탐색 순위는 실증적인 연구 근거와 파라미터 간 의존성에 따라 결정한다.

| Tier | 파라미터 | 근거 |
|------|----------|------|
| 1 (가장 중요) | `learning_rate`, `dropout` | 수렴 속도·일반화에 가장 직접적 영향 |
| 2 (손실 구성) | `do_NACL`, `unimodal_lambda`, `NACL_lambda` | 멀티모달 손실 결합 비율 결정 |
| 3 (대조 학습 세부) | `temperature`, `topk` | do_NACL=True일 때만 의미 있음 |

### Phase 1 — Coarse Random Search (N=20 trials)

```python
COARSE_SPACE = {
    "learning_rate":   [1e-4, 7e-5, 5e-5, 3e-5, 1e-5],
    "dropout":         [0.1, 0.2, 0.3, 0.4],
    "do_NACL":         [True, False],
    "unimodal_lambda": [0.05, 0.1, 0.3, 0.5],
    "NACL_lambda":     [0.1, 0.3, 0.5],
    "temperature":     [0.05, 0.1, 0.2],
    "topk":            [5, 10, 15],
}
```

각 trial은 랜덤 샘플링 (seed 고정). 20개로 대부분의 주요 영역 커버.

### Phase 2 — Fine Grid Search (N≈12 trials)

Phase 1 베스트 config 주변을 좁은 격자로 재탐색. `do_NACL`은 Phase 1 결과로 고정.

```python
# 예: best lr=5e-5 이면 → [3e-5, 5e-5, 7e-5]
# 예: best dropout=0.2 → [0.15, 0.2, 0.25]
```

### Phase 3 — Micro-tuning (선택적)

Phase 2 최선 config에서 `weight_decay`, `topk` 1개씩 fine-grain 재탐색. F1 향상이 0.3% 미만이면 중단.

## Proposed Changes

---

### Coach

#### [MODIFY] [Coach.py](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/models/Coach.py)

- [train()](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/models/Coach.py#66-255) 내 `finetuned_model_checkpoints` 경로 생성 로직 추가:
  ```python
  _trial_id = getattr(self.args, 'hp_trial_id', None)
  if _trial_id:
      _hp_dir = f"./{self.args.save_model_checkpoint}_{self.args2.dataset}/hp_search"
      os.makedirs(_hp_dir, exist_ok=True)
      finetuned_model_checkpoints = f"{_hp_dir}/{_trial_id}.pt"
  else:
      finetuned_model_checkpoints = ...  # 기존 경로
  ```
- [train()](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/models/Coach.py#66-255) return에 `btf1` 추가 (12번째 반환값):
  ```python
  return ..., test_losses, btf1
  ```

---

### New Script

#### [NEW] [hparam_search.py](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/hparam_search.py)

**핵심 설계:**
1. **데이터 1회 로드** → 모든 trial 재사용 (속도 최적화)
2. **trial 루프**: args2 deepcopy → HP 설정 → `args1.hp_trial_id = run_id` → Coach 실행
3. **결과 즉시 JSON 저장** (`hparam_results.json`): 오류 발생 시에도 이전 결과 보존
4. **체크포인트 폴더**: `model_checkpoints/iemocap_iemocap/hp_search/{run_id}_f1={f1:.4f}.pt`
5. Phase 1 완료 → 자동으로 Phase 2 공간 생성 → Phase 2 실행
6. 최종 최선 HP를 [iemocap.yaml](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/config/iemocap.yaml)에 자동 반영

> [!IMPORTANT]
> 사용 전 [model_checkpoints/iemocap_iemocap/pretrain_atv_best_model.pt](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/model_checkpoints/iemocap_iemocap/pretrain_atv_best_model.pt)가 존재해야 한다.
> [iemocap_pretrain.yaml](file:///home/neuroai/users/dhkim/merc/emotionheart_plus/config/iemocap_pretrain.yaml)의 `from_begin: False`를 확인해야 한다.

## Verification Plan

### Automated (Dry-run)
```bash
conda activate emotionheart
cd /home/neuroai/users/dhkim/merc/emotionheart_plus
python hparam_search.py --dry_run --n_trials 2
```
- `--dry_run` 옵션: 1 epoch만 실행, 데이터 로딩·Coach 연결만 검증

### 결과 검증
```bash
cat hparam_results.json | python -m json.tool | head -60
ls model_checkpoints/iemocap_iemocap/hp_search/
```
체크포인트 파일이 `{run_id}_f1=X.XXXX.pt` 형태로 생성되는지 확인.

### 전체 탐색 실행
```bash
python hparam_search.py --phase all 2>&1 | tee hparam_search.log
```
