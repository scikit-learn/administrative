First of, we would like to thank the reviewers for their pertinent
remarks that have increased the quality of the manuscript. Much effort
has been done to improve the reliability of the benchmarks as pointed out
by Reviewer 2, although due to space constraints it is published on a web
page referenced in the manuscript. 

The various points raised by the reviewers have been addressed within the
size limits. We have highlighted changes in yellow in the new version of
the manuscript. Below we address the reviewers' comments.


Reviewer 1
=========

> - The comparison is solid and appropriate for a short description like this. 
>   Maybe there could be an expanded version on the webpage.

We thank the reviewer for this useful suggestion. We have taken this
remark and published the extended benchmark suite in this web page:
http://scikit-learn.sourceforge.net/ml-benchmarks/ . Source code is also
available through its git repository:
https://github.com/scikit-learn/ml-benchmarks

> - The library is limited to a subset of features that machine learning
>   practitioners are using, some of which are featured in competing 
>   libraries: the paper could, space allowing, mention what is 
>   missing, and which of these features are in development.

Because of space constrains we have opted to not give a full account of
the functionality present in this package nor to discuss further
developments beyond the brief mention in the conclusion. This information
outlined in the web page and updated regularly.

> One hiccup I found: some of the examples appear to make full use of
> parallelization, by spawning a multitude of python processes. I am not 
> sure if this should be default behavior (as it chokes the machine).

Thank you for pointing this out. This has been fixed in the development
version and all examples now use the sequential algorithm by default.


Reviewer 2
============

> But even within the Python-world the differences of scope and purpose
> of scikits.learn in comparison with other relevant software is touched
> in a single sentence only. And frankly, I don't think that the major
> difference between scikits.learn and MDP is the presence of compiled
> code, or that PyMVPA heavily relies on R (which it doesn't). I rather
> think that the main difference of these projects is their respective
> target audience, which determines the desired features set and design,
> as well as implementation differences. I'm sure the reader would
> appreciate some guidelines on when to prefer scikits.learn over the
> others, and maybe also when not.

There is some overlap in both philosophy and target audience between
the various python machine learning packages. For instance, the
audience targeted by scikits.learn and MDP or MLPy is not
significantly different, thus these are are only briefly mentioned
in the introduction. We believe however that the greatest value of the
package lies in its performance, extensive documentation and
reliability, all whom are discussed extensively in the article.

Thank your for pointing out the fact that PyMVPA does not depend on R.
That sentence has been modified in the updated version of the article.

> Moreover, I'd be glad to see some information on whether there is any
> consolidation going on in the "machine learning in Python" world. Every
> project has copy of LIBSVM and uses it with mixed success (as the
> authors pointed out), or rather with varying applicability to different
> datasets (dimensionality, #observations).

Indeed, there is some duplication of efforts between machine learning
packages and it would be in the interest of all to find common points,
like maintaining the numerous forks that exist of LIBSVM as the reviewer
pointed out. However, nothing is yet agreed with the developers of PyMVPA
and thus it seems premature to discuss such conjectures. In addition, in
line with the scope of the MLOSS JMLR track, the goal of the article is
not to discuss community dynamics, but to expose a software package.

> I know that there is at least an initial attempt in PyMVPA to make
> scikit.learn functionality available inside this framework. The authors
> point out the advantages of the framework-free implementation of
> scikits.learn that should make it possible to easily use this code in
> other projects too. Are the authors actively working with others on
> such a consolidation?

We discuss on a regular basis with the PyMVPA and MDP developers. We take
feedback from them on our design to make it easier to use from the
frameworks. We have on several occasion sent patches to these packages
when we have found bugs or sub-optimal code.

Benchmarks
==========

> [...]. It looks like the benchmarks were computed using development
> versions, but at least some date and branch information should be
> given.  Moreover, I believe that a benchmark based on a single dataset
> with a particular dimensionality and sample size cannot be used to make
> comprehensive speed assessments. It is perfectly possible that the
> situation changes for smaller or larger datasets.

The benchmarks have been carefully reworked after these remarks. A lot of
effort has been made to produce the most reliable benchmark suite, taking
into account timing, performance and memory efficiency. However, we
acknowledge that these benchmarks are limited. A full assessment of the
performance of the different toolboxes is beyond the scope of this
article.

The web page http://scikit-learn.sourceforge.net/ml-benchmarks/
contains the latest version of the benchmarks. These now feature:

 * List of used software with version number.

 * Two datasets instead of one: one with #features > #samples and the
   other one with #samples > #features, as suggested by the reviewer.

 * Detailed view and description for each method instead of a single table.

 * Image plot of taken time for easier visualization.

 * A table that evaluates each method on a test dataset.  For
   classification algorithms, it is the fraction of correctly classified
   samples, for regression algorithms it is the mean squared error and for
   k-means it is the inertia criterion optimised by the algorithm.

The benchmark table from the original article has been updated
accordingly, although most new features have been omitted due to length
constrains. However, a link is provided to the extended version.

> The two result charts that are linked from the github page show substantial
> variance -- even for algorithms that should not be affected by convergence
> criteria, e.g. kNN. In contrast to the table in the manuscript, in these
> results:
>
> http://packages.python.org/milk/benchmarks.html
>
> scikits.learn is slower than any other tested implementation of kNN. That
> makes me wonder whether the benchmark reliability is negatively impacted by
> random effects and maybe the number of performance estimate samples is too
> low.

The benchmarks on the milk page are performed with an old version of the
scikit learn. Specifically, the reason of the performance difference 
observed on the kNN between old versions of the scikit and new ones is
that, since February 2011, the scikit switches to a brute force search
rather than a ball-tree based search in high dimension:
https://github.com/scikit-learn/scikit-learn/commit/34cbb6556d59bb340afc8d801a91bc7578a05ae0#scikits/learn/neighbors.py

