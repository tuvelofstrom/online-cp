# Tasks: Optimize KNN Compute Efficiency

**Input**: Design documents from `/specs/001-optimize-knn-efficiency/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included as requested in the feature specification for validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Ensure scipy and sklearn dependencies are available
- [ ] T003 [P] Configure performance benchmarking tools

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Analyze current KNN implementation in src/online_cp/classifiers.py
- [ ] T005 [P] Create SpatialIndexManager class skeleton in src/online_cp/classifiers.py
- [ ] T006 [P] Implement distance computation validation functions
- [ ] T007 Setup dual implementation framework (original + optimized)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Faster KNN Predictions (Priority: P1) 🎯 MVP

**Goal**: Implement spatial indexing to achieve sub-quadratic prediction time scaling

**Independent Test**: Measure prediction time on datasets of increasing size and verify O(log n) scaling vs O(n²)

### Tests for User Story 1 ⚠️

Note: Write these tests FIRST, ensure they FAIL before implementation

- [ ] T008 [P] [US1] Performance benchmark test in tests/test_classifiers_performance.py
- [ ] T009 [P] [US1] Scaling validation test in tests/test_classifiers_performance.py

### Implementation for User Story 1

- [ ] T010 [US1] Implement SpatialIndexManager core functionality in src/online_cp/classifiers.py
- [ ] T011 [US1] Add KDTree/BallTree selection logic based on distance metric
- [ ] T012 [US1] Implement per-label spatial index management
- [ ] T013 [US1] Add lazy index building with threshold activation
- [ ] T014 [US1] Implement efficient k-nearest neighbor queries for same/different labels
- [ ] T015 [US1] Integrate spatial indexing into prediction workflow
- [ ] T016 [US1] Add performance monitoring and timing

**Checkpoint**: At this point, User Story 1 should deliver faster predictions with spatial indexing

---

## Phase 4: User Story 2 - Maintain Exact Conformal Guarantees (Priority: P1)

**Goal**: Ensure optimized implementation produces identical results to original

**Independent Test**: Compare prediction outputs between optimized and original implementations on same data

### Tests for User Story 2 ⚠️

- [ ] T017 [P] [US2] Correctness validation test in tests/test_classifiers_correctness.py
- [ ] T018 [P] [US2] Distance computation equivalence test in tests/test_classifiers_correctness.py

### Implementation for User Story 2

- [ ] T019 [US2] Implement exact distance computation validation
- [ ] T020 [US2] Add nonconformity score verification against original implementation
- [ ] T021 [US2] Ensure p-value calculation matches exactly
- [ ] T022 [US2] Add prediction set equivalence checking
- [ ] T023 [US2] Implement fallback to original method for edge cases
- [ ] T024 [US2] Add comprehensive correctness testing framework

**Checkpoint**: At this point, User Stories 1 AND 2 should both work with guaranteed correctness

---

## Phase 5: User Story 3 - Backward Compatibility (Priority: P2)

**Goal**: Provide same API and behavior as current implementation

**Independent Test**: Run existing code and notebooks with new implementation, verify no breaking changes

### Tests for User Story 3 ⚠️

- [ ] T025 [P] [US3] API compatibility test in tests/test_classifiers_api.py
- [ ] T026 [P] [US3] Notebook integration test in tests/test_notebooks.py

### Implementation for User Story 3

- [ ] T027 [US3] Ensure constructor parameters match original ConformalNearestNeighboursClassifier
- [ ] T028 [US3] Implement use_spatial_index parameter with default behavior
- [ ] T029 [US3] Add automatic fallback for unsupported configurations
- [ ] T030 [US3] Preserve all method signatures and return types
- [ ] T031 [US3] Maintain error handling and edge case behavior
- [ ] T032 [US3] Update documentation and examples

**Checkpoint**: All user stories should now be independently functional with full backward compatibility

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T033 [P] Documentation updates in README.md and notebooks
- [ ] T034 Code cleanup and performance optimization
- [ ] T035 [P] Additional unit tests in tests/test_classifiers.py
- [ ] T036 Memory usage optimization
- [ ] T037 [P] Run quickstart.md validation
- [ ] T038 Update CITATION.cff and package metadata

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P1 → P2)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Should work independently but validates US1
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Core spatial indexing before integration
- Correctness validation before performance optimization
- API compatibility before documentation updates

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Performance benchmark test in tests/test_classifiers_performance.py"
Task: "Scaling validation test in tests/test_classifiers_performance.py"

# Launch core implementation tasks:
Task: "Implement SpatialIndexManager core functionality in src/online_cp/classifiers.py"
Task: "Add KDTree/BallTree selection logic based on distance metric"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently with performance benchmarks
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP with faster predictions!)
3. Add User Story 2 → Test independently → Deploy/Demo (guaranteed correctness)
4. Add User Story 3 → Test independently → Deploy/Demo (backward compatible)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (performance optimization)
   - Developer B: User Story 2 (correctness validation)
   - Developer C: User Story 3 (API compatibility)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
