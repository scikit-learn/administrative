# Two Year Radar

Below are notes from a discussion on what topics and ideas people think are important to
keep an eye on when considering a two year timescale. 

Topics were collected at the start of the meeting, then similar ones were grouped together,
followed by a round of voting. We then discussed topics in order for roughly 5min per topic.
After covering all topics with double digit votes we continued in a more free-style order
with topics individual people were interested in. We did not cover all topics.

---

> Find the original pad at: https://hackmd.io/@betatim/skl-2yr-2023

---

Goal: Hear about things you think should be "on our radar" (time scale of ~2years).

Points to consider:
* can be something we already do/think about
* can be a negative thing
* can be a positive thing (new promising technology/tool)
* how do we stay relevant in the future?
* try to estimate the probability of it happening (low, medium, high) and the impact (low, medium, high)

___


* [name=Gael] 13 = +3 +3 +3 +2 +2 Putting to production (if we are hard to put to production, people will use other tools) 
    * [name=Vincent] also thinking how to foster good MLOps with the broader ecosystem
    * [name=Vincent] Good MLOps = Easy to run on every (on premise / cloud) server and hardware + Observability with statistics + Good debugability when things go off the rail.
    * [name=Thomas] Serialization format that does not depend on Python (low probability, high impact)
    * [name=Franck] the ecosystem contains tooling to bootstrap a collaborative ML project codebase that aims at production (similar to MVC frameworks for web project)
    * Discussion:
        * Comment from dataiku Security. Pickle is a problem. Dependencies are a problem. Example solution: exporting scikit-learn as predictors that use only numpy (asked by customers)
        * In some companies, security require 9 months of wait / audit on a scikit-learn release before it can be used in production
* [name=Olivier] [name=Tim] 12 = +1 +2 +1 +2 +3 +3 numpy and pandas are no longer the go-to data containers in data-science pipelines (arrow-based DFs with lazy evaluation / pytorch gaining even more popularity)
    * [name=Franck] comprehensive support for the torch array library (replace numpy as a preferred array library ?)
    * Discussion:
        * arrow, [polars](https://www.pola.rs/), ibis and the like for "big data" 
        * pytorch for compute efficiency & compilation
        * 2 levels of representation:
            * pointers / data structure (= memory layout)
                * More and more chunking appearing
            * python level / metadata
        * lazy APIs (as in lazy polars)
            * More important to accept lazyness in early stages of the pipeline (cross-val, hyper-parameter selection)
        * Experiment in skrub but loop back fast to scikit-learn (eg in cross-validation utilities)
* [name=Denis] 11 = +2 +3 +3 +1 +1 +1 understand & evaluate value of sklearn for different industries (including life science / pharma) to unlock support for open-source development but also business opportunities if sklearn will have a company status.
  * partnering to develop long-term strategic ML / stats solutions
  * MLOps
  * facilitate training / onboarding & development of fit-for purpose solutions
  * Interface & common ground between (classical) biostats and machine learning: build trust in Machine Learning.
  * Discussion:
      * Interaction with regulation is difficult and may lead to abandoning scikit-learn
      * machine-learning is frequently still seen as "black magic" that is hard to trust
      * lack of visible way for business to interact with the project
      * how to make it easier for the scikit-learn dev community to interact with people "far beyond github" (e.g. biostats people)
      * how do "we", scikit-learn, also learn from these usecases
      * C-level decision makers see the orchestration (MLflow) but not what goes below
      * With regards to trust: importance of model cards in scikit-learn
* [name=Gael] 11 = +2 +3 +1 +2 +1 +2 Easy to do "standard" data science things (where standard is a moving target): danger = beginners start with other tools that are easier to use
    * [name=Guillaume] cover corner of data science that are currently ignored but should be used for some use cases: e.g. survival analysis, causal inference
        * [name=Denis] longitudinal analyses
        * [name=Vincent] should we mention them in the user-guide?
    * Discussion:
        * scikit-learn has been strong in being the "start here", but will this stay like this?
        * At the level of the ecosystem how do we make sure that there are good (and findable) ways to answer of the needs of good predictive data science (model explaination...)
        * People expect scikit-learn to do good practices
        * interactions with projects like [econML](https://econml.azurewebsites.net/)
* [name=Olivier] 10 = +3 +3 +2 +2 `torch.compile` makes it possible to implement most ML algorithms efficiently with a high-level Python syntax while being hardware agnostic
    * [name=Franck] and `triton` when slightly lower level is needed
    * [name=Thomas] Make PyTorch a depedency (low probabilty, high impact)
    * [name=Denis] [Mojo](https://www.modular.com/mojo) becomes fully open source and could become a replacement for Cython and simply code base / speed up development.
    * Discussion:
        * Should we prioritize compatibility with pytorch rather than the array API?
        * Start with optional dependency, consider making it mandatory later
        * Pytorch has own decision making that is sometimes internal
* [name=Thomas] 1= +1 Migrate away from Cython (medium probability, high impact)
    - Cython is easier to get started, but we are running into its limits.
* [name=Tim] 3 = +2 +1 cool things get built in scikit-learn, but no one knows about them(medium prob, high impact) 
* [name=Thomas] Migrate to MyST markdown for documentation (low probability, high impact)
    * More people know markdown then RST, which makes it easier for contributors
* [name=Thomas] 1 = +1 Take advantage of CPU instructions (SIMD) (medium probability, high impact)
    - Will be easier when NumPy splits out it's SIMD dispatcher so that we can use it.
    - Discussion:
        - We only use SIMD through BLAS (openblas uses SIMD)
        - Numpy has SIMD dispatching (recent addition). They will making it easier to use their infrastructure
        - where is the performance bottleneck: memory, cpu? can we measure it before getting distracted by optimisations?
        - how to get performance improvements for predict time which is run many many many times
            - internal google N=1 data/paper claims compute time is spent roughly 50:50 on fit vs predict
        - making something feel interactive is useful because it encourages exploration
        - 
* [name=Olivier] GPU runtimes with a vendor agnostic API available on laptops for the majority of our users (e.g. via WebGPU in WASM or native platform): being CPU-only becomes a limitation.
    * [name=Thomas] 2 = +2 WASM becomes a majorly supported (high probability, high impact)
* [name=Guillaume] impact of external policy: AI acts & cyber resilience act
* Share
* [name=Franck] 1 = +1 have float32 be the default in the ML ecosystem
* [name=Olivier] 2 = +1 +1 LLMs make traditional tabular predictive models (GBRT and Logistic Regression with feature engineering) completely obsolete (low probability, high impact)
