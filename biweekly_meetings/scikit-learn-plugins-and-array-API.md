# scikit-learn plugins and array API

Meeting about all things related to plugins for scikit-learn.

*Note: These meeting notes have been archived. For the latest meeting notes, please see: https://hackmd.io/--MJTgQzSFSYaaAgcJWQZg*


## When and where

The meeting happens every second Thursday. We cycle between an Asia friendly meeting time (11am Paris time, [in your timezone](https://arewemeetingyet.com/Zurich/2025-07-10/11:00/q)) and a Americas friendly time (3pm Paris time, [in your timezone](https://arewemeetingyet.com/Zurich/2025-06-26/15:00/q)).

We will meet in the #table-1-voice channel https://discord.com/channels/731163543038197871/731163543646502986


## 2026-08-17

- [name=betatim] Always store fitted attributes as NumPy (https://github.com/scikit-learn/scikit-learn/issues/34604)
  - Make a benchmark with scoring callback and array API: Ridge and LogisticRegression supports both.

- [name=Olivier] What needs to be done to move Array API support out of experimental?

- [name=virchan] Using scoring and array API with Pandas object: https://github.com/scikit-learn/scikit-learn/issues/33822


## 2026-08-03

- [name=betatim] Always store fitted attributes as Numpy arrays?
    - motivation:
        - user's code doesn't need to deal with changing types
            - inspection, plotting, etc
            - same code would run on laptop and GPU server machine
        - no change with respect to array API on/off
            - would make https://github.com/scikit-learn/scikit-learn/pull/34324 simpler
        - unpickling would just work on any machine
    - against it:
        - performance hit in `predict`, `predict_proba`, `transform`, etc
            - constantly converting prediction related attributes from numpy is slow?
                - can we solve this with caching?
            - needs benchmarks!!
                - what happens in a grid search setting? constantly moving from CPU<>GPU between `fit()` and `score()`
                - what about scoring callbacks?
                - benchmark case: PCA with large number of components
                    - fit on n_samples >> n_features and n_components ~= n_features, then predict on small n_samples or n_samples=1
                - benchmark case: cross val Pipeline with PCA with n_features ~= n_components and then logistic regression
                - benchmark case: logistic regression with n_features ~= n_classes
        - big change from current policy
        - make RFC more precise with respect to the difference between private and public fitted attributes
            - for example k nearest neighbors (potentially) has tree structures as private fitted attributes
    - https://github.com/scikit-learn/scikit-learn/issues/34604


## 2026-07-20
- [name=betatim] https://github.com/data-apis/array-api-strict/pull/221


## 2026-07-06

- [name=Olivier] Recently merged:
    - newton_cg for LogisticRegression:
        - https://github.com/scikit-learn/scikit-learn/pull/34412

    - common test for mixed namespace X and string y
        - TODO: follow-up with XFAIL resolutions
        - https://github.com/scikit-learn/scikit-learn/pull/34142


## 2026-06-22

- [name=betatim] running the unit tests with array API dispatch enabled
    - this found a few issues, working on resolving them
    - https://github.com/scikit-learn/scikit-learn/pull/34324
        - has some logic to coerce inputs to numpy when dispatch is anabled
        - goal is to maintain "inputs that used to work still work"
            - for example torch-cpu inputs work today with all estimators
        - better name "is cheap to convert"?
        - what about calling np.asarray on jax-gpu? is this something that should work?
            - if jax-gpu works today, it should also keep working in the future
    - https://github.com/scikit-learn/scikit-learn/pull/34317

- [name=betatim] `move_to` with negative strides:
    - https://github.com/scikit-learn/scikit-learn/issues/34307
    - check that the crash is really in pytorch, if yes report it upstream


## 2026-05-28

- [name=Olivier]
    - mixed namespace with str values:
        - https://github.com/scikit-learn/scikit-learn/pull/34142
    - mixed namespace with pandas.Series:
        - https://github.com/scikit-learn/scikit-learn/issues/33822
        - tricky to support
- [name=Olivier]
    - TODO: better test that array-like inputs (e.g. lists, Series, torch with CPU backend) are treated the same way when `array_api_dispatch` is enabled or not.
        - we need both docs and a new or extended common test
        - https://github.com/scikit-learn/scikit-learn/pull/34144 (small fix and some tests)
        - we need this as a pre-condition for being able to make array API dispatching "on by default"
- [name=tim]
    - ledoitwolf?
    - https://github.com/scikit-learn/scikit-learn/pull/33573
- [name=Olivier]
    - Create an RFC issue for device based auto-dispatching?
    - select the device in the style of sentence-transformer
        - tries CUDA first, then MPS, then foo, then bar, ..., then CPU
    - might reduce the need for the `MoveTo` transformer
    - needs to be not "too magic"
    - user needs to opt-in for this at least for now
        - or we convert back to the original input array namespace?
    - check the behavior of `SentenceTransformer` and `TabICL` for inspiration.
    - we need to specify an order of known "fast devices"
        - e.g. prefer pytorch+CUDA over cupy (or the other way around)
        - not include all devices in this (e.g. MPS can often be much slower)
    - use a global switch to enable/disable
    - each estimator decides by itself what to do if global switch is enabled
    - do we need an estimator level way to set policy?
    - If we implement device-based dispatching policy, then we might not need the `MoveTo` transformer discussed in: https://github.com/scikit-learn/scikit-learn/issues/33737
- [name=Olivier] Public API: `move_estimator_to`
    - https://github.com/scikit-learn/scikit-learn/issues/34135
- [name=tim] what do we do next?
    - get lucy's test for mixed y string and X input merged
        - mark estimators as xfail
    - start fixing the xfails
    - better test that array-like inputs work
    - write RFC about automatic dispatching
- [name=arthur]
    - non scipy-backed sparse support?
        - https://github.com/scikit-learn/scikit-learn/issues/34087


## 2026-04-30

- [name=Olivier] Reviewed and merged:
    - https://github.com/scikit-learn/scikit-learn/issues/33907
- [name=Olivier] array-api-strict handling of cross-device `__setitem__`
    - https://github.com/data-apis/array-api-strict/issues/207
    - to be discussed at the SPEC level: https://github.com/data-apis/array-api/issues/998#issuecomment-4322685877
- [name=Olivier] Having a look at LogisticRegressionCV:
    - https://github.com/scikit-learn/scikit-learn/pull/33906
- [name=Tim] Idea: allow estimators to dispatch the expensive computation to a different library namespace/device if available automatically by default?
    - what do the scipy devs think of this?

## 2026-04-16

- [name=Olivier] floating point dtype for result arrays `LedoitWolf` on `float32` inputs
    - https://github.com/scikit-learn/scikit-learn/pull/33573
    - Caused by `np.cov` with `dtype=None`.
    - Investigate to check if the `float64` upcast is actually needed/useful?
- [name=Olivier] chunking policy for computation with large intermediate arrays (e.g. `LedoitWolf` / `pairwise_distances`)
    - measure the memory usage/speed tradeoff of various chunksizes for various namespace/device combinations.
    - find a way to adjust the default value of `working_memory` in `pairwise_distances_chunked` based on the `device`.
- [name=Olivier] Board triage / quick reviews.
- [name=Tim] are host to device transfers as bad as everyone thinks?
    - https://gist.github.com/betatim/f4128b961525eda49df9b76c7c339388
    - fairly large datasets can be "transferred" in very little time compared to the computation of the algorithm
    - maybe we don't need to be so afraid of doing them?
    - maybe use a "random order" for the size of arrays that you are transferring to avoid having "perfectly" pre-warmed allocations that you can reuse
    - assess whether host to device and device to host transfer have a symmetric cost.
    - try to measure transfer times towards a pre-allocated target array
        - `cp.empty_like` and then inplace assign from an array on a different device??

## 2026-04-02

- [name=Tim] move meeting to make it easier to attend this meeting for Virgil and/or Omar
- [name=Lucy] Just FYI Evgeni has started adding DLPack tests to array-api-tests (https://github.com/data-apis/array-api-tests/pull/433) and has encountered several issues (some of which we have noticed as well) - https://gist.github.com/ev-br/2972ea3695b8a7fb8b447ef224a2d1d1
- [name=Lucas] `sparse_cg` questions: https://discord.com/channels/731163543038197871/1065913520547430400/1488918231442788713
- [name=Olivier] how to call `move_estimator_to` on pipelines with a `FunctionTransformer(partial(torch.asarray, device="cuda"))` step?
    - Instead of `FunctionTransformer` we could have a `MoveTo` transformer and make this `move_estimator_to`-aware.
    - there is a `__sklearn_array_api_convert__` (or something like it) that allows estimators to define custom conversion logic.


## 2026-03-19

- [name=Tim] board update, we haven't done that for a long time
- [name=Tim] https://github.com/scikit-learn/scikit-learn/pull/33564 some examples that show off array API
    - some are faster on the GPU, some try to demonstrate that your workflows become less complicated
    - not sure if we want to merge them, but wanted to share them
    - didn't find any bugs as part of making these :(
    - maybe use numerical columns + categorical (string) columns. TargetEncoder the categoricals, then do work on the GPU
        - maybe use skrub? TableVectorizer
- [name=Tim] fit and predict namespace mismatch https://github.com/scikit-learn/scikit-learn/pull/33076 would enjoy a review
    - Tim will resolve the conflicts
- [name=Olivier] Mixed namespace for estimators to review: https://github.com/scikit-learn/scikit-learn/pull/33525
- [name=Olivier] TODOS: want to find time next week to continue with:
    - https://github.com/scikit-learn/scikit-learn/pull/32873 and
    - https://github.com/scikit-learn/scikit-learn/pull/33473
    - extract the test infra refactoring out of the JAX PR: https://github.com/scikit-learn/scikit-learn/pull/29647/changes/176efc4573449086fd432fdf2082d0e265252be4


## 2026-03-05

- blog post
    - https://github.com/scikit-learn/blog/pull/223
- mixed namespaces metrics:
    - https://github.com/scikit-learn/scikit-learn/pull/32755
- fit / predict mixed namespace / conversion:
    - https://github.com/scikit-learn/scikit-learn/pull/33076
- CUDA CI:
    - https://github.com/scikit-learn/scikit-learn/pull/33445
- linear models follow-up:
    - https://github.com/scikit-learn/scikit-learn/pull/33345 merged
    - `PoissonRegressor`: https://github.com/scikit-learn/scikit-learn/pull/33348
- JAX PR pros/cons summary:
    - https://github.com/scikit-learn/scikit-learn/pull/29647
    - maybe take the test infra refactor and the changes to parallel processing
    - leave the rest as a PR that we can come back to
        - if it goes stale Claude and friends should be able to revive it
- A plan for moving out of experimental:
    - https://github.com/scikit-learn/scikit-learn/issues/33444
    - making sure that enabling `array_api_dispatch=True` does not break with numpy inputs on estimators that do not support array API in general.
    - What about pytorch/CPU input with `array_api_dispatch=False`?
        - https://github.com/scikit-learn/scikit-learn/pull/32676#issuecomment-3976964930
        - latest PR that fixes a problem with a display and PyTorch inputs: https://github.com/scikit-learn/scikit-learn/pull/33405
        - add a common test with pytorch-cpu input and `array_api_dispatch=False`
            - establish a reference point for what works and doesn't work
            - with `array_api_dispatch=True` what used to work should continue to work, there might be slight behavioural changes with `array_api_dispatch=True` because we now delegate to pytorch instead of converting to numpy silently when passing a PyTorch CPU tensor.

## 2026-02-19

- `LogisticRegression` was merged!
- Communication:
    - Notebook for an array API demo for an `NVIDIA` webinar: https://colab.research.google.com/drive/1YrCt5iBPT6gnmp7geahRn_9OqCPrfoLb?usp=sharing
    - Lucy's blog post: https://github.com/Quansight/Quansight-website/pull/946
        - investigate https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Attributes/rel#canonical for the scikit-learn blog post to avoid search engine anger
- use [nanobanana](https://nanobananana.com) to generate a dataset of 100 documents per class of passport, drivers license, invoice, etc
- Fit and predict PR: https://github.com/scikit-learn/scikit-learn/pull/33076
    - maybe consolidate `move_to` and estimator conversion function
    - what about `dtype=` arg that we might add to `move_to`

## 2026-02-05

- lock file for the array API CI:
    - [name=Olivier] I can no longer update it from macOS... :sob: 
    - Tim has mamba 1.5.1, conda 4.14.0
    - live debugging and fixed:
        - https://github.com/scikit-learn/scikit-learn/pull/33212
- jax update: still work in progress but progressing smoothly
- blog post(s)
- project board review

## 2026-01-22

- LogisticRegression update
    - https://github.com/scikit-learn/scikit-learn/pull/32644
    - WIP: testing numerical stability of the code both on float64 and float32 values
- JAX support update
    - https://github.com/scikit-learn/scikit-learn/pull/29647
    - WIP: mostly `xpx.at(array)[indices].set(value)`
- metrics common tests update
    - comparing dtypes from different namespaces - https://github.com/data-apis/array-api-strict/issues/109#issuecomment-2539831099
    - example https://github.com/data-apis/array-api-tests/pull/409/files#diff-7f0fb4609256f3fa325745a3edba07fd7fa59d4c7058537137c02c6ff06cc192R209
- scipy mixed int float promotion: https://github.com/scikit-learn/scikit-learn/issues/32552
    - scipy rule: https://github.com/scipy/scipy/pull/22695/files#r1997905891
    - https://github.com/scikit-learn/scikit-learn/pull/33022
- Pytorch tensor/array-like support with dispatch disabled: https://github.com/scikit-learn/scikit-learn/pull/33028
    - numpy interoperability: https://numpy.org/doc/stable/user/basics.interoperability.html#using-arbitrary-objects-in-numpy 
- update the board

## 2026-01-08

- [name=Tim] trying to learn "vibe coding", array API support for `GaussianProcessRegressor` as playground:
    - let's see how this goes
- [name=Olivier] Mixed namespace metrics PR: implicit `float32` to `float64` upcasts
    - https://github.com/scikit-learn/scikit-learn/pull/32755#discussion_r2611300838
- What to do next for `LogisticRegression`?
    - [name=Olivier] need to open an informational issue to review estimators that have the same problem.
- Project board review / update

## 2025-12-11

- how can we prepare the LogisticRegression array API support on Monday?
    - https://github.com/scikit-learn/scikit-learn/pull/32644
    - expensive part of `fit` is computing the gradients
    - move matrix of shape (n_classes, n_features) from GPU to CPU to feed into optimiser
    - having the losses implemented in cython and "array API Python" creates duplication
    - Big Question: How do deal with this duplication? Can it be avoided? Maintenance burden?
    - tim: how does pytorch cpu compare to the cython version?
        - someone needs to benchmark it
    - tim: why are the losses implemented in cython?
        - probably historical, but likely more memory efficient and maybe faster
- mixed namespace metrics PR:
    - https://github.com/scikit-learn/scikit-learn/pull/32755
- integration testing with mixed input testing for metrics, estimators and scorers API:
    - https://github.com/scikit-learn/scikit-learn/pull/32873
    - highlighted a few cascading issues, will stop working on it for now
- ready for second review, enable array API + pass numpy inputs to estimator that has no array API support it should still work:
    - https://github.com/scikit-learn/scikit-learn/pull/32846
- keep board up to date https://github.com/orgs/scikit-learn/projects/12
- JAX support and xlearn/jax-sklearn: https://github.com/chenxingqiang/jax-sklearn

## 2025-11-27
- keep board up to date https://github.com/orgs/scikit-learn/projects/12
- To review:
    - Common test for mixed array input for metrics: https://github.com/scikit-learn/scikit-learn/pull/32755
        - Related - how should we check estimators: https://github.com/scikit-learn/scikit-learn/issues/28668#issuecomment-3584031855
- scipy working on a PR about quantile+sample weight https://github.com/scipy/scipy/pull/23941. Reviewed, tested by Olivier to replace our implementation. Single test failure with edge-case behaviour in scikit-learn (0 weight everywhere? - fix in https://github.com/scikit-learn/scikit-learn/pull/32212 - second review and decision around how to amend common test required). Working well with PyTorch + benchmarked.
  - impact on effort to add quantile in array-api-extra? Useful for projects that don't want to depend on scipy, are there many?
- middle-management existential question: would it be somewhat useful to split "in progress" in more columns: no approval, one approval, almost there? 
    - Tim prefers one column until it gets too full (in which case we should finish more instead of starting new things)
    - alternative proposal: two columns "in progress" vs "needs review"

## 2025-11-13

- To review:
    - `LogisticRegression`
        - would be nice to get this done for 1.8. Tricky because the RC should already be out there
    - `move_to`
- Board updates https://github.com/orgs/scikit-learn/projects/12

## 2025-10-30

- a few PRs that add supports for estimators were merged: Ridge, RidgeCV, CalibratedClassifierCV
- [name=Olivier] start to work on an interesting example task LLM response problematic or not (to block it). HuggingFace transformers for encoding -> move it to GPU -> RidgeCV using array API. complex: need to take into account the context (polars cumulative eval windowing functions). WIP Colab notebook: https://colab.research.google.com/drive/1S03Ry3726urs9I46iS4NowcVcD9V-3Oh#scrollTo=tI_E6Pw0oltY. Suggestion: text encoder in skrub. Cheap baseline with scikit-learn and skrub and compare to an LLM.
    - alternative: use an image embedding with Ridge on top as blog post
- https://github.com/orgs/scikit-learn/projects/12 exists, let's try and use it!


- important metrics are the proper scoring rules brier_score, log_loss, `d2_*` versions. Others metrics not so much of a priority.
- LabelBinarizer for array API then remove the convert to numpy logic https://github.com/scikit-learn/scikit-learn/pull/32582
    - follow up to this is to clean up uses of this for the `int` usecase. Some old users use `_convert_to_numpy`
- LogisticRegression is an important one https://github.com/scikit-learn/scikit-learn/issues/32611



## 2025-09-18

- [name=Lucy] PR adding array API support to `contingency_matrix` ready for second review: https://github.com/scikit-learn/scikit-learn/pull/29251 :D
- [name=Emily] Using `assert_allclose` vs `assert_array_almost_equal` in tests

## 2025-09-04

- [name=Lucy] What combination of arrays do we want to test for mixed array inputs - https://github.com/scikit-learn/scikit-learn/pull/31829#issuecomment-3251931985
   - torch cuda -> cupy
   - torch cuda -> numpy CPU
    
- `CalibratedClassifierCV` - make this array API compatible.
    - useful to calibrate torch models
    - https://github.com/scikit-learn/scikit-learn/issues/31869
- `quantile` - addition to array api extra https://github.com/data-apis/array-api-extra/pull/341
    - pick a quantile method that is symmetric and supports weights
        - namely https://github.com/scikit-learn/scikit-learn/pull/31775
    - array API support in that scipy `quantile` supports (search for 'quantile'): 
        - CPU: https://scipy.github.io/devdocs/dev/api-dev/array_api_modules_tables/stats.html#array-api-support-stats-cpu
        - GPU: https://scipy.github.io/devdocs/dev/api-dev/array_api_modules_tables/stats.html#array-api-support-stats-gpu
        - Seems to support torch on CPU and GPU but not jax or dask

## 2025-07-10

🚨🚨🚨🚨 Using https://meet.google.com/gjn-fyyb-vra 🚨🚨🚨🚨

- [name=Lucy] Contingency matrix PR, which will unblock several other metrics: https://github.com/scikit-learn/scikit-learn/pull/29251
    - We convert to numpy within the function as we use `coo_matrix` for the histogram calculation - I don't *think* it's worth investigating alternate array API implementation.
    - What should the return array type be?
    - I don't think any of the downstream metrics that use `contigency_matrix` will need conversion to numpy (though `mutual_info_score` looks like it may be tricky to implement)

## 2025-06-26

- [name=Tim] How to implement a function that can convert an estimator from one namespace to another?
    - is it even possible to have one converter that can convert from any namespace to any other namespace?
    - should we punt on this? Only implement the check from https://github.com/scikit-learn/scikit-learn/pull/29313 (`fit` and `predict` have to use same namespace/device)?
        - eventually DLPack will help, but not now?
        - Tim thinks we should implement the check but without giving instructions on how to convert
        - go via numpy as a fallback (with a warning if needed)
        - conclusion: Tim will try to revive #29313, it looks more advanced than the discussion suggests
            - raise warning if going via Numpy
- [name=Olivier]
    - `stable_cumsum` RFC was "accepted" two weeks ago
    - https://github.com/scikit-learn/scikit-learn/issues/31533
    - it's only used in 4 places left, only one or two which could benefit to array API support
    - PRs for individual removal are welcome, related to array API or not.
- [name=Olivier]
    - `PolynomialFeatures` should be ready for a final review (CI is now green!)
    - https://github.com/scikit-learn/scikit-learn/pull/31580
- [name=Olivier]
    - Merged https://github.com/scikit-learn/scikit-learn/pull/31650 to add Intel GPU tests.
- [name=Olivier]
    - We probably need a new common test for https://github.com/scikit-learn/scikit-learn/pull/31452 and a meta issue to track progress on this.

## 2025-06-12

Today's meeting is at 11am Paris/Zurich time ([In your timezone](https://arewemeetingyet.com/Zurich/2025-06-12/11:00/b/Scikit-learn%20Engines)).



- [name=Olivier]
    - Did a quick google colab benchmark of the GMM PR:
        - https://github.com/scikit-learn/scikit-learn/pull/30777#pullrequestreview-2916452772
        - 20x speed-up of GPU over CPU.
        - [name=Loïc] my previous [quick benchmark](https://github.com/scikit-learn/scikit-learn/pull/30777#issuecomment-2866643963) was more 3-7x

- [name=Loïc]
    - what should we do about `_fill_or_add_to_diagonal` bug with non C-contiguous arrays https://github.com/scikit-learn/scikit-learn/pull/31445? I would rather have it merged before the API support for pairwise kernels https://github.com/scikit-learn/scikit-learn/pull/29822 ...
    - agreement to do in-place with `.flat` for numpy + for loop for non-numpy. Since `_fill_or_add_to_diagonal` is a private helper we can always change our mind about returning a value or improve performance later.



## 2025-05-15

TODAYs meeting is here instead: https://meet.google.com/fqp-xezm-apc?authuser=1


- [name=Olivier]
    - Move next meeting(s) to a AUS friendly timezone?
        - 29th May is a holiday, we will skip that meeting
        - 12th June is the next meeting, which will be at a AUS friendly time, maybe 9am Europe time?
    - Currently reviewing:
        - https://github.com/scikit-learn/scikit-learn/pull/29822 (tests are green besides coverage)
    - Always converting metrics nd outputs to numpy arrays:
      https://github.com/scikit-learn/scikit-learn/issues/31286
        - Particular case (exception) of `confusion_matrix` is here: https://github.com/scikit-learn/scikit-learn/pull/30562
        - maybe the answer depends on what people typically do with the return value
            - typical things: printing values, plotting (bar plots or boxplot), pandas DF
            - unlikely that these values will be passed back into "machinery" where they'll encounter ohter array API arrays
            - arrays will typically be small
            - how much code will we have to add to always convert to numpy arrays?
            - how (in)convenient is it for the user when they do typical things?
    - Let's make people's approval explicit on the `y_pred` follows `y_true` https://github.com/scikit-learn/scikit-learn/issues/31274

- [name=Loïc]
    - GaussianMixture soon ready for first round of review https://github.com/scikit-learn/scikit-learn/pull/30777
        - scipy array API bug in logsumexp https://github.com/scipy/scipy/issues/22680
        - I think indeed this will be fixed in the next version of scipy (1.16 I think) that uses a Python float instead of `xp.asarray(0, dtype=a_max.dtype)` (i.e. device unspecified) https://github.com/scipy/scipy/pull/22683/files#diff-5a4b78539c1d53d65d7b54249a7374cbe0b947357b7a90eee262a561029f3445
    

- [name=Emily]
    - Synced with Lucy: she will finish the [pairwise_kernel](https://github.com/scikit-learn/scikit-learn/pull/29822) PR, then I will continue the [Nystroem approximation](https://github.com/scikit-learn/scikit-learn/pull/29661) PR 
    - Lucy found this [stalled PR](https://github.com/scikit-learn/scikit-learn/pull/7996) that I will look at in the meanwhile to see if it's still relevant


# 2024-04-17

- [name=Loïc]
    - from `GaussianMixture` PR with Stefanie:
        - all arrays follow X, scipy took a simpler approach it seems (not scipy's problem if arrays are not in the same namespace) https://github.com/scipy/scipy/issues/22680#issuecomment-2786873708
            - we need "all arrays follow X" because there is no way for a user to move `y`, but they can move `X`
            - the scikit-learn discussion is in https://github.com/scikit-learn/scikit-learn/issues/28668
            - what's the use case with a Pipeline that needs y follows X? I forgot ...
            - dlpack latest spec (from 6 months ago?) supports moving across namespaces but not implemented yet in array libraries, in particular cross-device support
            - fine for scipy because they have a functional API but in scikit-learn estimators, with array constructor parameters, fit accept arrays. User not in full control, especially for predict and score. 
            - Pipeline with pandas input, pandas feature transformation using CPU gives numpy array, move to GPU with FunctionTransformer, Ridge on GPU, predict output of Ridge PyTorch array. Compute a metric e.g. score internally in grid-search, y_true pandas, y_pred PyTorch. This is even worse if classification with dtype object for classes. TargetEncoder outputs categorical values needs to be on CPU (no GPU support for dtype object), can be moved to GPU, Ridge consumes y and can not. I lost track ... Olivier wants to write a meta issue about this kind of use cases eventually.
            - actionable point: for writing tests in GaussianMixture only test numpy array constructor params and call `fit(X_xp)`. For now don't test constructor parameters and `X` in separate namespaces (e.g. `means_init=means_array_api_strict` and `X_torch`).
            - Loïc: for now under the assumption that numpy arrays will be moved to any array namespace (and the other way around probably) but generic cross array namespaces (cupy to torch, array_api_strict to torch) is unspecified. General consensus on this.
            - Tim: ideally "we will attempt to move 'other arrays' to the namespace (and device) of `X`. If this is not possible an explicit error will be raised."
            - it would still be interesting to get a feeling of how user-hostile the cross array namespaces conversions are `xp.asarray(array_other_xp, device=other_device)`. `array_api_strict` errors don't matter that much because no "real user" will use `array_api_strict`
            - Thomas Also xref previous discussion from scikit-learn https://github.com/scikit-learn/scikit-learn/issues/28668 and https://github.com/scikit-learn/scikit-learn/pull/31190#issuecomment-2807974066
        - `xp.linalg.solve` vs. `scipy.linalg.solve_triangular`
        - `xp.linalg.cholesky` vs `scipy.linalg.cholesky`
            - actionable point: if it is short enough not to mess up readability of the code, we could have numpy-specific branch, otherwise not
            - in the past numpy specific branch to avoid introducing performance regression or different results. Sometimes it makes the code too complex, and hurts maintainability. Keeping numpy-specific branch seems OKish especially if it does not happen too often. Whether it changes results for `numpy` arrays is something to look at. Need also quick benchmark to evaluate that performance is not impacted too much.
- [name=Tim] Interesting new(?) repo https://github.com/jefferythewind/warpgbm/tree/main
    - gradient boosted trees implemented mostly in Python, with a bit of CUDA kernels
    - maybe interesting if you are interested in how far a pytorch (plus a bit more) gradient boosted tree model can go, e.g. for plugins
- [name=Emily Chen]
    - Would greatly appreciate a summary on what has Lucy been doing on [Pairwise Kernels](https://github.com/scikit-learn/scikit-learn/pull/29822) (a lot seem to have happened)

## 2025-04-03

- [name=Olivier]
    - `_weighted_percentile` is ready for final review
      https://github.com/scikit-learn/scikit-learn/pull/29431
    - check what could be upstreamed to scipy `main`

- [name=Olivier]
    - memory layout issues
    - chunking for pairwise distances: https://github.com/scikit-learn/scikit-learn/pull/29822/files#r2026776152
    - similar problem in `_weighted_percentile`: https://github.com/scikit-learn/scikit-learn/pull/29431/files#diff-c7d6538a14900fc61a71b07411b38f654ed7e35f54e1150ca5e73ae31cab7c0eR75-R101

- [name=Loïc] WIP `GaussianMixture` array API support with Stefanie https://github.com/scikit-learn/scikit-learn/pull/30777 
  - a bit annoying to pass `xp` along, why not computing it once (at fit-time) and store it as a class variable `self.xp_`? Same with `self.device_`?
      - picklability of device may not be guaranteed?
      - private attributes would be more suitable?
      - [`_estimator_with_converted_arrays`](https://github.com/scikit-learn/scikit-learn/blob/434010c883a21ecf354385ddb3d730b5c3bf12f4/sklearn/utils/_array_api.py#L790) would need adjusting
  - expectation for when `X` and array-like parameters like `weights_init` are not in same array namespace or device? error or not?
      - generic advice https://github.com/scikit-learn/scikit-learn/issues/28668 (y follows X)
      - not entirely clear, it would be nice to define an estimator with numpy arrays and fit it on different `X` on different array namespaces or device. grid-searchCV with a `class` dtype object use case that I didn't have time to write ...
      - consensus seems to be: other arrays follow X (including `y`, `sample_weights`, constructor parameters, etc ...)
  - inplace assignment via a tuple of integer array indices for now we have a work-around with a `for` loop ...
    ```py  
    # TODO: instead of for-loop, find something more efficient; previous code:
    # resp[indices, xp.arange(self.n_components)] = 1
    for count, index in enumerate(indices):
        resp[index, count] = 1
    ```
    
    From https://hackmd.io/zn5bvdZTQIeJmb3RW1B-8g?stext=5759%3A80%3A1%3A1743688847%3ApsI_ew&edit= (array API meeting 2025-02-20) it looks like multi-dimensional integer arrays will become supported.
    
    `a3[0, da.array([0]), da.array([0, 1])].compute()` was the example in the array API meeting

- [name=Olivier]
    - Dask support status update:
        - https://github.com/scikit-learn/scikit-learn/pull/28588#issuecomment-2764774210

## 2025-03-20

- [name=Lucas] co-vendor `array-api-{compat, extra}` has been refreshed to get green tests
    - https://github.com/scikit-learn/scikit-learn/pull/30340

- [name=Olivier] question about the status of JAX / Dask support in SciPy:
    - https://github.com/data-apis/array-api-extra/issues/122
    - example PRs in scipy:
        - https://github.com/scipy/scipy/pull/22342 (introduces `lazy_apply` in the linkage function of `scipy.cluster`)
        - https://github.com/scipy/scipy/pull/22686 (generate API docs to report which backends actually work for which public function in scipy)

- [name=Tim] reboot plugins?
    - need to define a way to tests that 2 implementations are "approximately" equivalent (e.g. based on a statistical test on score values on a resampled dataset).

- [name=Tim] check out https://developer.nvidia.com/blog/nvidia-cuml-brings-zero-code-change-acceleration-to-scikit-learn/ and report issues


## 2025-02-20

- [name=Olivier] testing with multiple devices on `array-api-strict`
    - started to update https://github.com/scikit-learn/scikit-learn/pull/30090
    - but there are still failures to resolve
    - the `array-api-strict` device handling is probably still too strict when doing operations with Python scalars.
- [name=Tim] I think it is a red panda, not a cat. Also sometimes hackmd has snow?!

## 2025-02-06

- [name=Olivier] testing with multiple devices on array-api-strict
    - https://github.com/scikit-learn/scikit-learn/pull/30090
    - need to update the lock file and try again to see if it's now mergeable

- [name=Loïc] looked with Stefanie at `GaussianMixture` array API support. Stuck on `scipy.linalg.solve_triangular`. Will try to use `xp.linalg.solve` as recommended by Olivier. WIP PR: https://github.com/scikit-learn/scikit-learn/pull/30777 

- [name=Jérôme] [ridgecv](https://github.com/scikit-learn/scikit-learn/pull/27961) last review should have been addressed, still missing a test to complete the coverage
 
## 2025-01-09
- [name=Emily] Happy new year! 
    - Fixed some tests for [weighted_percentile](https://github.com/scikit-learn/scikit-learn/pull/29431). I _think_ the actual code changes should be done, just figuring out the best way to write the Array API tests (deciphering how it's done in PCA right now)
    - Going to work on [pairwise kernels](https://github.com/scikit-learn/scikit-learn/pull/29822) next, after which [Nystroem](https://github.com/scikit-learn/scikit-learn/pull/29661) can be worked on 
    - I will most likely be pushing tiny commits irregularly in the next month, but around mid-February I should be able to push more stuff
- [name=Lucas] Vendoring array-api-compat and array-api-extra: https://github.com/scikit-learn/scikit-learn/pull/30340
    - allows shared use of helpers between array consuming libraries - i.e. `sklearn.utils._array_api` should only have to contain things specific to sklearn, and other libraries should be able to reuse helpers sklearn has written, like https://data-apis.org/array-api-extra/generated/array_api_extra.setdiff1d.html
    - to make progress on JAX support, looks like we need https://data-apis.org/array-api-extra/generated/array_api_extra.at.html for now - see https://github.com/scipy/scipy/issues/22246
    - sklearn can still accept private helpers in PRs - they can be moved to array-api-extra whenever is convenient
- [name=Olivier] been working on improving dtype consistency on `GaussianMixture` and this could be a good candidate for array API-ification, once the following PR is merged.
    - https://github.com/scikit-learn/scikit-learn/pull/30415
- [name=Lucas] materializing lazy arrays
    - https://github.com/data-apis/array-api/issues/839
- [name=Lucas] device transfers: error out systematically in scipy
    - `confusion_matrix`: https://github.com/scikit-learn/scikit-learn/pull/30440

## 2024-11-28

- [name=Tim] CirusCI has a potentially intesting offering: https://github.com/scipy/scipy/issues/21740#issuecomment-2504583921
    - $150/month for unlimited GPU minutes, would remove complexity in our setup, would cost more than we currently spend
    - maybe something to keep at the back of our mind
    - integrates with GitHub Actions as a "self-hosted runner".
- [name=Tim] scalar arguments for `xp.where` / `xp.clip` / ...
    - Needs to be written in the standard
    - Possibly changes in `array-api-compat`? Maybe not.
- [name=Stefanie] question on how to handle pandas import error for CI test
    - https://github.com/scikit-learn/scikit-learn/pull/29519




## 2024-11-14
- [name=Tim] Trying out what "full class" plugin could look like 
    - https://github.com/scikit-learn/scikit-learn/pull/30250
    - how to solve (in)compatibility problems between versions of the plugin and scikit-learn
    - input validation: should the work to do it be done in scikit-learn (for all input formats/types) or should we give it to plugin implementors?

```
from sklearn.cluster import KMeans


try:
    import cuml
    cuml.register_sklearn_plugins(kmeans=True)
except ImportError:
    ...
    

sklearn.enable_plugin("cuml")

def resister_sklearn_plugins():
    sklearn.plugins["kmeans"]["init"] = my_callable

KMeans(init=cuml.kmeans_init(...), algorithm=cuml.kmeans_algo(...)).fit(X)

def get_kmeans_init(init: str):
    return sklearn.plugins.get("kmeans", {}).get("init", None)

class KMeans(...):
    def __init__(self, init="kmeans++-auto"):
    """
    init: str = "kmeans++-auto"
        It's kmeans++ by default. If a plugin is registered for it, it'll be used.
        
        Other options: "kmeans++": it won't use the plugin, ever.
    """
    
    def fit(self, X, y):
        init = get_kmeans_init(self.init)

```

## 2024-10-31

- [name=Olivier] [plugin API revival](https://github.com/scikit-learn/scikit-learn/issues/22438)?
    - Thinking about it more, it might be much simpler to do full estimator
      dispatch (instead of only in a few well chosen inner estimator specific private methods) and we would trust the plugin to do the right thing to not break scikit-learn when enabled.
    - The "right thing" would be encoded in plugin aware common tests.
    - Developer API refactoring is under way.
    - Some private methods could be reused accross plugins via subclassing the estimator class.
    - scikit-image is interested in doing pluggable docstring / API doc at the sphinx level: this could be dynamic with javascript calls that fetches extra up to date info from the plugin websites.
- [name=Emily] 
    - Continued to work on `_weighted_percentile` and added comments on an [index error](https://github.com/scikit-learn/scikit-learn/pull/29431/files#r1824515972) 
    - test cases for Array API: should we modify existing tests at all then? Since we should create tests that contain `array_api` in the name and compare that output to the numpy outputs? 
        - revert all modified tests and create new tests 
- [name=Tim] small tweaks to `array-api-strict` to solve https://github.com/scikit-learn/scikit-learn/pull/30090
- [name=Olivier] quick update on the items of the past meeting


## 2024-10-17

- [name=Olivier] sparse output types:
    - https://github.com/scikit-learn/scikit-learn/pull/29251#pullrequestreview-2375172780
- [name=Olivier] accept array API inputs even when the computation happens on the CPU via an implicit numpy conversion for convenience:
    - https://github.com/scikit-learn/scikit-learn/pull/29881
- [name=Tim]'s PR on multi-device support for `array-api-strict` was merged: https://github.com/data-apis/array-api-strict/pull/59

## 2024-10-02

- [name=Tim] Been working on https://github.com/data-apis/array-api-strict/pull/59 - multi device support for array-api-strict
    - maybe make a follow up PR that has different max precision for different devices
- [name=Tim] https://github.com/data-apis/array-api-extra/ has been started by Lucas
    - what do we want to do?
- Array API mutability and Jax https://github.com/data-apis/array-api/issues/845
- https://www.youtube.com/playlist?list=PLc_vA1r0qoiTjlrINKUuFrI8Ptoopm8Vz - videos from the/a recent OpenAI Triton conference, maybe something interesting in it?
- https://github.com/scikit-image/scikit-image/pull/7520 - plugin/dispatching/backend mechanism for scikit-image
    - possible source of alternative implementation https://github.com/rapidsai/cucim
- Maybe we can move the meeting to Thursday 15:00h Europe time?


## 2024-09-04

Topics:
- Can we move this meeting to later in the day (maybe by 5-6hours)? That way Emily could attend as well.
- [name=Tim] working on adding option to have more than one device to array-api-compat https://github.com/data-apis/array-api-strict/pull/59
- [name=Tim] did some work with scikit-image for backend dispatching https://github.com/scikit-image/scikit-image/pull/7520
- [name=Emily] Nystroem approximation is blocked by pairwise kernels, going to start working on that 



## 2024-08-07

Topics:

- [name=Tim] sparse handling in `get_namespace`:
    - https://github.com/scikit-learn/scikit-learn/pull/29476
        - maybe `remove_sparse=True` as default?
        - probably can't use `remove_types` because the test for sparse'ness 
        - if mix of cupy/pytorch/numpy and sparse -> filter out sparse and apply usual logic to determine namespace
        - if only sparse -> `False` for "is array API compatible"
        - Think about potential future handling of https://sparse.pydata.org/en/stable/
           - without considering potential array API compat:
               - https://github.com/scikit-learn/scikit-learn/pull/29031
           - considering potential array API compat in the future:
               - https://github.com/scikit-learn/scikit-learn/discussions/29064
- [name=Tim] status update on the label-triggered CUDA CI
    - workflows triggering other workflows?
    - use of commandline tool?
    - github account permissions?
- [name=Emily] The weird CPU device error on [weight percentile](https://github.com/scikit-learn/scikit-learn/pull/29431#discussion_r1704008545)
- [name=Tim] should we make an issue with a list of more complicated estimators for people to tackle?
    - if yes, what?
        - TargetEncoder?
            - Not sure we can do it because of lack of categorical dtype support in array API.
            - input is typically a dataframe
        - LogisticRegression
            - maybe too complicated because it needs a cython/python branching?

## 2024-07-24

Topics:
- [name=Tim] testing saga continues
    - we should be using array-api-strict in (all) our normal CI runs so that we can find problems early
    - it seems like we aren't doing that, or only partially because people find new test failures when
      running things on their laptop :-/
    - https://github.com/scikit-learn/scikit-learn/pull/29227
- [name=Emily] 
    - _weight_percentile things
        - https://github.com/scikit-learn/scikit-learn/pull/29431
    - Adding more code coverage so things don't break in weird ways 

- [name=Jerome]
    - forcing array api config to be the same in fit and predict: preferred way of doing it for all estimators?
        - in the long term array API will be enabled by default. Right now it is configurable so users can opt in
        - maybe we can assume that the state won't change between `fit` and `predict`
        - document that it is a mistake to change the state of the config during runtime?

- [name=Loïc]
    - `SCIPY_ARRAY_API` one time import-time setting vs scikit-learn run-time thing that can be set on and off? [`SCIPY_ARRAY_API` doc](https://docs.scipy.org/doc/scipy//dev/api-dev/array_api.html#using-array-api-standard-support)
        - how do we deal with the problem of scikit-learn using array API and scipy not?
            - raise an error when `set_config(array_api_dispatch=True)` is called?
        - should we switch to using a environment variable like scipy?
            - would solve the problem of people turning on/off array API mid execution
            - makes it harder to have some examples with array API on and some with off
    - Follow-up with `mean_poisson_deviance` https://github.com/scikit-learn/scikit-learn/issues/29549

Make sure that when you do not set the config that enables array API, nothing breaks! This is something we need to
work on before the next release is made.


## 2024-07-10

Topics:
- update on the CUDA CI
    - label based workflow / pending runners
    - update on the dependencies (numpy 2 / array-api-strict)
- same namespace check:
    - https://github.com/scikit-learn/scikit-learn/pull/29313#issuecomment-2180497459
    - nested estimators
    - handling of config: is this the behavior we want (it is what is currently done in the PR):
```python
>>> from sklearn.datasets import make_classification
>>> from sklearn import config_context
>>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
>>> import torch

>>> X_np, y_np = make_classification(random_state=0)
>>> X_torch = torch.asarray(X_np)
>>> y_torch = torch.asarray(y_np)

>>> lda = LinearDiscriminantAnalysis()
>>> with config_context(array_api_dispatch=True):
...     lda.fit(X_torch, y_torch)
LinearDiscriminantAnalysis()
>>> with config_context(array_api_dispatch=False):
...     lda.predict(X_torch)
Traceback (most recent call last):
    ...
ValueError: Inputs passed to LinearDiscriminantAnalysis.decision_function() must use the same array library and the same device as those passed to fit(). Array api namespaces used during fit (array_api_compat.torch) and decision_function (numpy) differ. You can convert the estimator to the same library and device as X with: 'from sklearn.utils._array_api import convert_estimator; estimator = convert_estimator(estimator, X)'
```
decisions:
    - fitting and predicting with different config raises an error. may be later relaxed to tolerate it when all arrays are numpy
    - nested estimator converion should be handled, estimators can expose a __sklearn_array_api_convert__ or similar method
- which PRs to prioritise reviewing?
    - https://github.com/scikit-learn/scikit-learn/pull/27369 `f1_score` and `multilabel_confusion_matrix` is blocking many classification metrics
    - https://github.com/scikit-learn/scikit-learn/pull/29433 euclidean_distances and rbf_kernel
        - interesting to look at
        - what to do about numerical precision? for devices that only support float32
            - StandardScaler inspect device+namespace to find maximum precision float dtype
            - then use this dtype
            - for some devices this will be less precise than others
            - we have to live with this
            - RidgeCV PR also has this problem and adopts the same strategy
            - StandardScaler PR is still open, contributor hasn't replied in a while
            - for euclidean_distances we can use alternative formulation, which is faster but less precise
                - this is currently what is used for numpy
                - and what is implemented in the PR
                - does it make sense to offer an alternative method that is slower but more stable?
                    - https://github.com/scikit-learn/scikit-learn/pull/29433#pullrequestreview-2168416516
            - go for "fast" option for now, document that numerical stability is device dependent
                - deal with stability improvement later, in a new PR
    - https://github.com/scikit-learn/scikit-learn/pull/29431 `_weighted_percentile` requires feedback.
    - https://github.com/scikit-learn/scikit-learn/pull/29389 - easy merge!
- How to handle expected exceptions such as `np.linalg.LinalgError` alternatives:
    - https://github.com/scikit-learn/scikit-learn/pull/29318
- Discussion of `take_along_axis` https://github.com/data-apis/array-api/issues/808
- 



## 2024-06-12

Attendees:
- Tim Head
- Jerome Dockes
- Olivier Grisel

Topics:
- CI ?!?!!1 \o\ \o/ /o/
    - https://github.com/scikit-learn/scikit-learn/issues/29221 - post follow up work there
- RidgeCV status
    - how to handle `X` at predict time which is in a different namespace/device than `X` at fit time?
        - raise an error!
            - following the lead of PyTorch
        - error message should contain instructions on how to convert the estimator
        - LinearDiscriminantAnalysis probably also has this problem
        - make a separate PR that adds this and fixes LDA
            - don't mix with existing RidgeCV PR
            - update all the classes that already have array API support
                - `predict` for estimators and `transform` for transformers
            - maybe handle this in `check_fitted` by passing in the namespace of `X`
- Other array API PR reviews
    - `f1_score` and `multilabel_confusion_matrix` https://github.com/scikit-learn/scikit-learn/pull/27369
    - `max_error` should be super easy to review & merge


## 2024-05-29

Attendees:

  - Tim Head
  - Jerome Dockes
  - Olivier Grisel

Topics:

- CI !!!
    - https://github.com/scikit-learn/scikit-learn/pull/29130/files
    - https://github.com/organizations/scikit-learn/settings/actions/github-hosted-runners/4
- `*SearchCV`
- RidgeCV progress report
    - y follows X
- EuroSciPy
    - Tim submitted a proposal for a workshop
        - Numpy + array API to do noise reduction
        - maybe extend workshop to include pytorch classifier + RidgeCV (compare to Ridge in GridSearchCV)


## 2024-05-15

Attendees:

  - Tim Head
  - Jerome Dockes
  - Olivier Grisel

Topics:

- `LabelBinarizer` 
    - how often is `LabelBinarizer` used in estimators inside scikit-learn?
        - RidgeClassifierCV, MLP, logistic w/ liblinear solver, chi2 feature selection, OneVsRestClassifier, naivebayes, 
    - fitted attributes are currently stored as numpy arrays
    - this is different from other estimators
    - however it goes against the convention for "array api support" in estimators, they normally store
      fitted attributes in the Array API container type
    - ... much discussion ...
    - conclusion: we leave `LabelBinarizer` unchanged, so that it does not support Array API. Users of it will have to take care of converting the input to Numpy. They also need to look out for keeping attributes that they might expose (e.g. classes_) in the right array namespace (the one of `X`).
    - keep in mind that refactoring might required in the future if other scikit-learn estimators need a similar treatment to avoid redundant complex namespace conversion logic in our code base.
- EuroSciPy
    - answer CFP on Array API related things (ends 25th May in Europe!)
        - maintainers track again?
        - or a talk / tutorial to raise awareness / reach a wider audience?
            - simple bit of code in numpy, then transform to arry api, then show how it works with numpy, pytorch, cupy, etc
            - show off a PyTorch pre-trained model wrapped for a pipeline with Ridge as classifier
    - sprint? - yes!
- tolerance setting for float32/float64 support in RidgeClassifier/-RegressorCV and 
    - is upcasting `X` to float64 (in Numpy) a usability bug?
        - if the user passes in float32 they expect lower precision and higher performance
        - silently converting to float64 for a large array like `X` uses a lot more memory
        - scikit-learn shouldn't do this?


## 2024-02-07

- how should we progress with r2_score PR https://github.com/scikit-learn/scikit-learn/pull/27904
    - Ridge PR https://github.com/scikit-learn/scikit-learn/pull/27800
        - revive after r2_score is done
- GPU CI
    - cirun is probably the only option for now
    - https://github.com/scikit-learn/scikit-learn/issues/24491
- GPU FAQ update but pending comment not too sure how to tackle:
    - https://github.com/scikit-learn/scikit-learn/pull/28328#discussion_r1476299586
- Tim plans to investigate why tests failed with cupy, I think its related to using `device="cpu"` and such.
    - Might be fixed in #27904
- Agree on priority for next estimators / tools:
    - what is our goal?
        - short term: a nice example of a whole workflow with pipeline and a supervised estimator with cross-validation;
        - example image classification pipeline on generic image features:
            - https://huggingface.co/facebook/dinov2-base as the feature extractor
            - and a large enough image datasets such as https://paperswithcode.com/dataset/stanford-cars
        - long long term: array api everything that naturally fits its contraints, starting with most used estimators.
    - `train_test_split` (Tim)
    - LabelBinarizer (dependency for other more interesting estimators)
    - Ridge/RidgeCV
    - RidgeClassifier/RidgeClassifierCV
    - LogisticRegression (start by profiling the CPU version on large-ish data to see where are the critical sections)
    - FunctionTransformer or dedicated transformer to move data to the GPU
        - useful for example if you load data with pandas
        - consider the problem that `y` can't be moved from on device to another in an estimator. This means that a function transformer in a pipeline can move `X` but not `y`. Which leads to weird situations at the input to a regressor where `X` and `y` are not on the same device
            - what to do about this?
            - consider a pipeline that starts with CPU specific code with pandas in a column transformer at the beginning, followed by target encoder then Ridge.
            - one possible way out: allow estimators to move `y` to the device of `X` automatically when needed.
    - Other candidates:
        - `MLPClassifier`/`MLPRegressor`
        - `Nystroem`


## 2023-08-31

* [Franck] asking for [name=ogrisel] or [name=betatim] to force push
* [Franck] introducing a new plugin https://github.com/soda-inria/sklearn-pytorch-engine with pytorch backend

## 2023-07-13
* triton + torch stack exploration
  + [Franck] Found surprisingly good performance for triton matmul on A100: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html (NB: example adapted for float32)
* is it realistic to try and merge PCA and MinMaxScaler before everyone goes on holiday?

* [Franck] K-Means in pure pytorch example: https://gist.github.com/fcharras/ce1f1df7d15675268827e1fb9b65265b
* [Franck] KNN in pure pytorch example: https://gist.github.com/fcharras/82772cf7651e087b3b91b99105a860ddh



## 2023-06-29


### Topics
* testing with array API (being worked in the PCA pull request)
* PCA pull request need (manual) testing on GPUs
* Create META issue to encourage people to convert `sklearn.preprocessing`
    * wait for testing infra from the PCA PR
    * create some easy to use tooling to compare performance of Numpy and Array API based solutions
        * e.g. a dataset that makes the estimator run for ~a few seconds
* Lazy evaluation:
    * https://github.com/scikit-learn/scikit-learn/issues/26724
* Array-API support for K-Means [issue](https://github.com/scikit-learn/scikit-learn/issues/26585)
* Array-API support for KNN [issue](https://github.com/scikit-learn/scikit-learn/issues/26586)
* Exploring triton + torch stack


## 2023-05-21
### Who is coming
* Meekail / @micky774
* Raghuveer Devulapalli / @r-devulap
* Julien / @jjerphan

### Topics we talked about:

 - Raghuveer's work on https://github.com/intel/x86-simd-sort
 - https://github.com/numpy/SVML
 - Cython's implementations
 - Reasons for not using C++ in scikit-learn
 - Pluging systems and current trade-offs
 - Distance Metrics SIMD-implementation by Meekail:
     - https://github.com/Micky774/distmetris-xsimd/
     - https://github.com/scikit-learn/scikit-learn/issues/26329
 - **Actionable items**:
     - assess the cost of [`qsort` in HGBT](https://github.com/scikit-learn/scikit-learn/blob/c8f79e234e99733bd2e9c52b97684cd924aa73e4/sklearn/ensemble/_hist_gradient_boosting/splitting.pyx#L946-L947)
     - see if we can use `PyArray_*Sort` interfaces from NumPy Cython API or lower-level C API


### Pointers

#### During discussions

 - An example of using Cythont to simply wrap and import an external C++ function:
https://github.com/Micky774/distmetric-xsimd/blob/main/distance_metrics/_dist_metrics.pxd
 - NumPy's cython API: https://github.com/cython/cython/blob/8af954f812f78d5258cb8440b13cc6944eccf940/Cython/Includes/numpy/__init__.pxd
 - https://github.com/scikit-learn/scikit-learn/blob/c8f79e234e99733bd2e9c52b97684cd924aa73e4/sklearn/neighbors/_partition_nodes.pyx
 - https://github.com/scikit-learn/scikit-learn/blob/c8f79e234e99733bd2e9c52b97684cd924aa73e4/sklearn/neighbors/_partition_nodes.pyx#L4
 - https://github.com/scikit-learn/scikit-learn/blob/c8f79e234e99733bd2e9c52b97684cd924aa73e4/sklearn/neighbors/_partition_nodes.pyx#L4

https://github.com/mbatoul/sklearn_benchmarks
https://mbatoul.github.io/sklearn_benchmarks/results/github_ci/add_dbscan/20220315T181132/scikit_learn_intelex_vs_scikit_learn.html

### Cython 

 - https://scikit-learn.org/dev/developers/cython.html 
 - https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html

### A recent related interesting piece of work

Irrelevant for Python, but nice to share for the design:

https://github.com/openjdk/jdk/pull/14227

## 2023-04-20
### Who is coming
* Tim / @betatim

### Topics to talk about

* Array API and plugins https://github.com/scikit-learn/scikit-learn/issues/26024
* Triton autotune, how does it exactly work?
* 

## 2023-03-09
### Who is coming
* Tim / @betatim
* Olivier / @ogrisel
* Julien / @jjerphan

### Topics to talk about

- ANN
- Breaking changes
- Other things



## 2023-02-24
### Who is coming
* Tim / @betatim
* Franck / @fcharras

### Topics to talk about
- `check_array` with custom `asarray`, https://github.com/scikit-learn/scikit-learn/pull/25617
    - For tim: answer thomas' questions, fix conflicts, see what happens
- what to do about tests that don't pass? Should all tests pass?
    - not all tests should pass. For example "dense fit, sparse predict" won't work if plugin doesn't support sparse input
    - [Tim] I think it is up to plugins to decide what tests they want to exclude, maintain list within the plugin
        - [Franck] doesn't that make versionning really hard for plugin developers ?
            - Tim is not sure, some false positives are Ok no?
- What timelines/schedules do people have for plugin work?
    - [Tim] Knowing what other people's schedules are is useful to estimate an end date, etc
    - [Tim] we should have a plan for merging "something" (maybe subset of features, maybe all features), so that we don't end up with a branch that just continues to grow forever and ever. Work isn't done when the first engine PR is merged, but it lets us tick something off/feeling of progress. Also giant PRs are impossible to review

- scikit-learn code must break if a plugin failed to implement the API correctly (risk for developers to use have a broken plugin without noticing because of fallback mechanism): case of `engine_name` attribute
    - see https://github.com/scikit-learn/scikit-learn/pull/25535#discussion_r1106055439

## 2023-02-09
### Who is coming
* Franck / @fcharras
* Olivier / @ogrisel
* Tim / @betatim
* Julien / @jjerphan

### Topics to talk about

- Recommended way of testing an engine with `sklearn` unit tests (see https://github.com/scikit-learn/scikit-learn/pull/25535#issuecomment-1422865990 )
- KNeighbors engine API
  - development paused because of blockers in numba_dpex (will hopefully resume soon ;)
- How to do CI with GPU?
  - No new development: still difficult, requires private infrastructure
  - Or very expensive (e.g. circle ci)

### Testing

- Agree that scikit learn can make available to plugin a framework to re-use unit tests with an engine
- The fallback mechanism can be kept in the test ?
    - Olivier: I think it's fine to have the fallback enabled when running the scikit-learn tests with the pytest fixture.
- nice to have: pytest fixture checks that for a list of known tests that the engine under test was selected. The list of tests for this is maintained by each plugin in the `pyproject.toml`
- sklearn tests that use `predict` and `transform` should only make assertions that are array API compatible
  - how many tests currently fail with engine types ?


## Some leftover notes for discussions on `check_array` (Jan. 2023)

https://github.com/scikit-learn/scikit-learn/issues/25433

> [name=Tim] ~~I'd post the above. See what people think. And~~ keep the below for follow up discussions.

This has several advantages:

- re-using all the data validation pipeline from `scikit-learn` saves lines of code
- choices that have been made in `scikit-learn` don't need to be discussed again and maintained separately in plugins - apply mindlessly those of `scikit-learn`, saves time
- for the user, it ensures some level of consistency across the vanilla `scikit-learn` implementations and the behavior with the plugins
- for the developers, it's also a closer first shot to having a plugin interface that is compatible with `scikit-learn` unit test pipelines.
- also, one of the nice things `scikit-learn` validation does, is that the user can feed inputs without caring too much about the type, and the validation pipeline will do at most one copy in memory, and only if it's necessary (e.g for casting dtype), this is an efficient trick that plugins would want to have anyway

But there's currently an issue that prevent having that without some hack-ish code (see it [in action here](https://github.com/soda-inria/sklearn-numba-dpex/blob/main/sklearn_numba_dpex/kmeans/engine.py#L294 )), it's that the `asarray` method that is part of the `Array API` only exposes a subset of the parameters that a given array library might be able to support, and that might not be enough for what the plugin needs.

An example is the `order` argument: [currently in `sklearn` the `order` argument is only used for `numpy` arrays](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_array_api.py#L168), else it's ignored. But other array libraries might also support order and the plugins might need it. We're in this case with [`dpctl.tensor`](https://intelpython.github.io/dpctl/latest/docfiles/dpctl/dpctl.tensor_functions_api.html#dpctl.tensor.asarray) and would like to enforce `F` order. 

#### Describe your proposed solution

To my understanding, this `asarray` call is the only step in this pipeline that will cause this limitation. So rather than having the currently hardcoded `asarray_with_order` call, we propose adding a new parameter to `_validate_data` and `check_array`, named `asarray_fn`, that expects a callable, which will be used in place of  the `asarray_with_order` call. This will let just enough room for plugins to benefit from `check_array`. 

#### Describe alternatives you've considered, if relevant

The proposal in https://github.com/scikit-learn/scikit-learn/issues/25000 is related, since it would offer an entry point that enable passing a customized `xp`, including a customized `asarray`, effectively solving this issue, but it has a broader scope than this proposal, and uses global config.

I've also inquired the reasons for not supporting `order` in the `Array API` and have had a comprehensive answer from the maintainers, see https://github.com/data-apis/array-api/issues/571 , points that stand out being:

> [...] array API standard wants to only standardize things that are implemented or can be implemented in all array libraries. This is not the case for `order=`. 
>
> [...]
>
> It can be implemented as a superset of what's in the array API standard, so there is no problem here.

That makes me think of the Array API like a nice thing to rely on, but where array libraries have reasons to implement specific additional parameters such as in `asarray`, a more flexible approach is beneficial, and here parameterizing a callable makes sense.

---


## 2023-01-26
### Who is coming
* Tim / @betatim
* Olivier / @ogrisel
* Julien / @jjerphan

### Topics to talk about
* Merge [the "mixin PR"](https://github.com/ogrisel/scikit-learn/pull/13)
* KNeighbors engine API?

## 2023-01-12
### Who is coming
* Tim / @betatim
* Franck / @fcharras
* Olivier / @ogrisel
* Julien / @jjerphan

### Topics to talk about

* [name=Olivier] https://github.com/ogrisel/scikit-learn/pull/14 - array api for cluster count
* [name=Tim] lets talk about `accepts()`:
    * [Tim's last remark](https://github.com/soda-inria/sklearn-numba-dpex/pull/74#issuecomment-1378558887)
    * set engine provider during fit time
    * when converting an estimator to numpy unset the fitted attribute that records the engine provider
    * when engine provider attribute is not set, it will be set as part of the resolution

```python=
# Fit with engine provider A and convert attributes to numpy
# The goal is to use the default scikit-learn implementation at predict time
with config_context(engine_providers="a", engine_attributes="sklearn_types"):
    estimator.fit(X, y)

with config_context(engine_providers="b"):
    predicted = estimator.predict(X)  # should not work

predicted = estimator.predict(X)  # should work with scikit-learn's default engine

# maybe later, not a priority for this iteration
estimator.set_engine_provider("b")  # updates the fitted attributes
estimator.predict(X)  # could work with engine b?

# alternatively:
estimator.set_engine_provider("b")  # could raise NotImplementedError on fitted estimators and tell the user to clone and refit with engine b first.
```

### Discussion on accepts

```python=
estimator.fit([[0], [1]], [0, 1])  # this works in scikit-learn

with config_context(engine_providers="A"):
    # this might never get used in a realistic
    # use case, but it would also be weird if this
    # didn't work
    estimator.fit([[0], [1]], [0, 1])  # engine A should accept this
```

TODO: make an issue to make it possible to pass a custom `asarray` implementation
to the scikit-learn `_validate_data` / `check_array` methods / functions.

## 2022-12-15
### Who is coming
* Tim / @betatim
* Franck / @fcharras
* Olivier / @ogrisel
* Julien / @jjerphan

### Topics to talk about

- [name=Franck] Early conda packaging of the `wip-engines` branch for use withg `sklearn-numba-dpex` found to be too complicated to be worth considering at this point
- [name=Franck] [Bumped](https://github.com/soda-inria/sklearn-numba-dpex/pull/74) `wip-engines` dependency commmit for `sklearn-numba-dpex`. `accepts` interface found to be functional, considering the engine can store data internally during `accepts` and reuse later.
- [name=Tim] engine naming, regressor vs classifier? Use class name?

### Discussion

- Need an option to automatically convert back to numpy
- Need an option to disable automated fallback to the default engine (or other engines for later providers) and instead raise an exception that can be handled by the pytest plugin in scikit-learn to XFAIL (or skip) the test.
    - implement this in the engine selector code in `wip-engines`?
    - NB: current pytest plugin is here https://github.com/scikit-learn/scikit-learn/blob/ff191e296fa87d57ade7e2a3fb573870eded2f26/sklearn/_engine/testing.py
- Investigate if pytest-plugin can take care of conversion to numpy (given that an engine provides a method to perform conversion)
- Investigate having a "return numpy" flag for engines. Would useful for re-using the scikit-learn unit tests
    - add a method to the engine that takes care of converting an estimator from "native array" mode
- Add a unit test with a pipeline where one step is a KMeans in GPU mode, but other steps of the pipeline are in "numpy mode"
    - TODO: write those tests in `sklearn_numba_dpex` first, decide later what part should be in sklearn ?

Proposed API to enable those flags (for instance in the pytest plugin):

```python
set_config(
    engine_provider="sklearn_numba_dpex",
    # Should more than one engine be tried?
    engine_fallback=False,
    # Choose the type of estimator attributes
    engine_attributes="sklearn_types", # vs "engine_types" (default)
    # for output conversion see comment below
)
```

We could rely on `set_output` to let the user control the output types and just extend it with an new "engine_types" option that would depend on the currently active engine.

```python
KMeans().set_output(transform="engine_types")
```

We need to introduce an `EngineCapable` mixin class to:
- convert or not the fitted attributes based on the `engine_attributes` configuration flag at the end of `fit`
- convert or not the values returned by `predict` and `transform` depending on the estimator state configured by the user using `set_output`.

We need a standard API on the engine to convert an engine type container to the matching sklearn type that could then be passed to a generic conversion loop implemented in at the mixin class. This would be similar to the conversion loop already provided by the Array API spec converter:

- similar to [`sklearn._array_api._estimator_with_converted_arrays`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_array_api.py#L207), but without the `if hasattr()` check


But the difference would be that the callable of the engine would also receive the attribute name (in addition to the attribute value to convert) to make it easier to decide if they need to inspect nested datastructures such as Python lists or dicts recursively.


Alternative would be to expose a method that modifies the estimator attached to the engine inplace.


## 2022-12-01
### Who is coming
* Tim / @betatim
* Franck / @fcharras
* Olivier / @ogrisel
* Julien / @jjerphan

### Topics to talk about

We would like to propose a roadmap, decide yet undecided points ("bespoke" VS standard API), and agree on how the tasks are shared. 

Here's the roadmap proposal, that builds on the experience of developing [sklearn-numba-dpex](https://github.com/soda-inria/sklearn-numba-dpex):

Proposal of points to prioritize for the plugin API on the sklearn side:

- Synchronize development: let's all of us use the same branch from now on
- Decide bespoke API vs standard API to go ahead with short-term development
- Package the experimental engine branch on a custom conda channel ([name=Franck] less of a priority, will look into it)
- On-the-fly registration of custom engine (outside of entry-points) for testing, benchmarking and debugging purposes, see [this comment](https://github.com/soda-inria/sklearn-numba-dpex/pull/65#issuecomment-1329068853).
    - make a PR with a proposal of how to do it
- API for conversion of a fitted estimator to the default sklearn engine
    - [name=Tim] +1
    - an engine should provide two functions (class methods?):
        - one to convert from engine to numpy; and
        - one to convert from numpy to engine
- API for conversion of a fitted estimator to another engine ([name=Olivier] no a priority)
- API for setting the engine without the context manager (e.g estimator.set_engine)
- Automatic fallback to default engine when the input data or hyperparams are not supported by the estimator
- Building on the Array API:
    - through default Array API library setting [discussion](https://github.com/scikit-learn/scikit-learn/issues/25000)
    - or pluginify `_get_namespace` and `asarray_with_order`
- Decide if explicit activation by users is mandatory
    - Proposal (Franck): have a white list of plugins for which activation is not mandatory 
    - [name=Tim] maybe not on day one, but in the future I think having automatic activation is nice

[Our KMeans engine](https://github.com/soda-inria/sklearn-numba-dpex/blob/ef49d7ebe1b02b74903c8246e964fa0553cfef08/sklearn_numba_dpex/kmeans/engine.py) might be a good read for highlighting current strength and weaknesses of the branch `wip-engines`.


### Comments from the meeting:
- one branch to work on the scikit-learn side of changes, needs to be outside the mai nrepo unfortuantely because of permissions
- combine input validation and "engine acceptance" methods into one
- TODO: adapt `sklearn_numba_dpex` to use wip-engines + engine negotiation from Tim's branch (via dedicated PR) and adapt the documentation accordingly
- Bespoke VS standard ?
    - use `wip-engines` for now
- store provider name on the estimator instance so we can re-use it for `predict`
- engine instances are short lived, they get destroyed at the end of `fit`, `predict`, etc
    - this means engines can keep references to input data if they wish
    - but they can't remember things between separate calls to `fit`, `predict`
- existing sklearn tests for an estimator [should pass](https://github.com/soda-inria/sklearn-numba-dpex/blob/ef49d7ebe1b02b74903c8246e964fa0553cfef08/.github/workflows/run_tests.yaml#L31) when an engine is activated
    - maybe tests need adjusting because they are too specific (e.g. [1](https://github.com/scikit-learn/scikit-learn/pull/24779))


## 2022-11-16
### Who is coming
Indicate if you are coming or not by listing your name/github handle
* Tim / @betatim
* Franck / @fcharras
* Julien / @jjerphan

### Topics to talk about
Add topics to talk about at the next meeting. This gives people a chance to think about them before, which increases the chances of them saying something useful.

General organization of those meetings:

- Mention what we have been doing in the past week
- What we plan to do next

---

* [name=Julien] Mainly have been reviewing Franck's PRs
* [name=Tim] https://github.com/scikit-learn/scikit-learn/pull/24826
    * what do people think of a "generic" engine API?
    * TL;DR:
        * each public estimator method has three engine parts: `pre_fit`, `fit`, `post_fit`
        * store provider name when `fit` is called, raise exception when provider changes during `predict`, etc
        * can we find a way to combine/copy/inspire by the callback PR https://github.com/scikit-learn/scikit-learn/pull/22000
        * we can go from finegrained API to a gernic API without breaking all existing plugins
            * we can't go the other way
        * 
    * open questions, independent of implementation:
        * should engine provider mismatch between `fit` and `predict` be an error?
            * estimator could ignore user and select the correct one instead of erroring
            * yes, lets make it an error. Simple to understand, no magic.
    * Next: main focus on figuring out sklearn API side of things.
        * Next?: start writing a SLEP
        * Next?: move towards getting this merged. Need opinions from others on what is missing/nice/ugly

* [name=Franck] 
    * KMeans++ about to be merged in [`sklearn_numba_dpex`](https://github.com/soda-inria/sklearn-numba-dpex/pull/37), all core methods for KMeans merged
    * Working next on interoperability:
        * stabilizing an input validation model for KMeans++
        * accept on-device arrays without trigger useless memory transfers between host and device
        * harder than thought, provides interesting practical issues
    * Thoughts on having device attributes stored as engine-specific library (e.g cupy, dpctl.tensor,...):
        * better as the default behavior
        * re-using sklearn tests require numpy-like outputs and fitted attributes -> need either an interface for parameterizing the engine for tests, or a distinct engine used specifically for testing (bad practice but easier to setup for sklearn unit tests)
            * First idea: create a second engine within the package that outputs numpy arrays
            * related idea: fit with an engine, and predict/transform with another engine (and in particular sklearn default)
        * what should happen when `fit` is called with engine X but `predict` is called with default engine
            * should it be an error?
            * should the estimator use the engine X function to convert itself to default?
            * feeling is that it should be an error. Simple. No magic. Estimators can store the engine used at fit time as an attribute ?
    * Tasks in mind for extending wip-engines branch:
        * interface for activation of engines through an estimator method rather than a context manager
- Thoughts on choosing engine APIs:
    - maybe get inspiration from the callback APIs ?
    - Let's keep the Callback API as a requirement in mind: https://github.com/scikit-learn/scikit-learn/pull/22000
    - Regardless of having required standards methods on the plugin API, the question of having in mind that all methods could be overridable might be considered on a per-estimator basis anyway ?
    - Not having a standard API might close the door to ideas that goes against what a bespoke, fine-grained api enforce (example: loop on n_init on KMeans is currently enforced to be ran sequentially)
    - It's possible to move from bespoke API to standard API without breaking existing plugin. The opposite is not true.

## 2022-11-02
### Who is coming

Indicate if you are coming or not by listing your name/github handle
* Tim / @betatim
* Olivier / @ogrisel
* Julien / @jjerphan
* ...


### Topics to talk about

Add topics to talk about at the next meeting. This gives people a chance to think about them before, which increases the chances of them saying something useful.

General organization of those meetings:

- Mention what we have been doing in the past week
- What we plan to do next

What we have been doing:

- [name=Julien]
    - I mainly have been reviewing PRs from Franck on `sklearn-numba-dpex`
    - I have not followed on latest discussions on [`#24497`](https://github.com/scikit-learn/scikit-learn/pull/24497)
    - Slightly related: I am focusing on general Cython implementations for scikit-learn (i.e. [`#22587`](https://github.com/scikit-learn/scikit-learn/issues/22587)). Those implementations might be extendable and extended via this plugin system, but there is no use-case for now.
    - Next:
        - [Review K-Means++](https://github.com/soda-inria/sklearn-numba-dpex/pull/37)
        - Improve the native Cython implementations of KMeans within scikit-learn.
            - Improvements might involve (somewhat ordered, probably irrelevant):
                - remove interactions with Python / make it even more Cythonic if possible 
                - study running KMeans++ initialisations in parallel
                - turn the LLoyd and Eklan iterations using `PairwiseDistancesReductions`
            - This item is to be turned into a issue after discussions with @jeremiedbb who designed and reimplemented KMeans.
- [name=Franck]
    - KMeans with k-means++
    - https://github.com/soda-inria/sklearn-numba-dpex/pull/37 (ready review)
    - numba_dpex and interoperability with different GPU vendors
    - low level SYCL compiler should be cross-vendor
            - https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
    - however the Python level stack is only compatible with Intel hardware
            - see e.g. https://github.com/IntelPython/dpctl/issues/947
    - objective: realease a dev version on a dedicated conda channel
        - dependencies on `dppy/label/dev` channel
        - or `conda-forge` but they stalled on old versions
    - current benchmark script:
        - https://github.com/soda-inria/sklearn-numba-dpex/blob/main/benchmark/kmeans.py
        - https://github.com/soda-inria/sklearn-numba-dpex#running-the-benchmarks

- [name=Tim]
    - basic plugin works, uses some cheats
    - benchmarking is tricky, really depends on how you setup the problem
        - maybe use 3M samples, 32 dimensions, and 100 clusters
        - exclude the transfer cost?
    - mostly getting things organised internally
    - next: expose more primitives from RAFT to remove cheats
    - RAFT https://github.com/rapidsai/raft/
    - https://github.com/betatim/scikit-learn-gpu


More specific topics:

* [name=Tim] What is the plan to get https://github.com/scikit-learn/scikit-learn/pull/24497 merged?
    * use a different conda channel to pre-release this w/o merging first
    * [name=Olivier]:
        * 2 engines entry points in scikit-learn and 2 plugins contributing each to at least one entry point
        * at least one dev channel with working packages
        * second engine: `KNeighbors{Classifier, Regressor}`
* [name=Tim] An engine to hook `pairwise_distance`?
    * maybe better to stick to one engine per estimator in order to have good error messages?
* [name=Tim] What estimators/engines to target next?
    * [name=Franck] next with Intel collaboration is k-NN
    * [name=Tim] kNN, nearest neighbours, DBSCAN, dim reduction
    * see above about "second engine"
* [name=Franck] Issue related to using entrypoints?
    * entrypoints does not allow writing an engine in a notebook and registering it
    * for testing and development it is useful to be able to register engines at runtime.
    * maybe as part of an estimator specific API that allows enabling an engine for just this estimator we allow passing a class to help with this use case? -> Create a ticket.
* [name=Tim] raising `NotImplemented`, multi plugin, order
    * Three options:
        * Fallback to default (or alternatively other plugins) automatically
        * make calls fail if the requested plugin does not work/raises `NotImplemented`
        * provide both, with a strict mode that raises an error
    * When several plugins are enabled at the same time:
        * try all providers in the order they were provided in the config context
        * keep failing over to the next provider until one works, or default
