# Intermediate data structure (aka "Pandas in, Pandas out")

One of the aims is probably that the meta data should be conserved through the pipeline

## Pandas

### Drawbacks
- Memory copies (2 or 3 memory copies, which lead to multiplying memory usage)
    - Does this lead to time cost?
- Not sure how to deal well with sparse
- Pandas semantics getting more and more different from numpy (for instance na)
- Output type dependent on input (which may or may not be a drawback)

### Benefits
- Object well known to data-science users

## XArray

### Drawbacks
- axis have names
    - do we enforce them?
- heterogeneous data support
    - Dataset
- converting back to DataFrame would be another extra line for the user

### Benefits
- Can well attach arbitrary meta data

## Custom data structure
- It was proposed because we didn't want to depend on pandas


## No Data Structure
- Input either DataFrame or feature names with fit params
- Transformers expose an output feature names
- This creates kind of a new especial case for feature names
  and it's hard to expand and extend it later.

## Arrow
- Apache Arrow is another data structure which is somewhat aliegned
  with what we need
- It handles patching the data, chunks, online learning, etc
    - The summaries, stats etc are updated as you patch the data
- It's pretty much a C++ library with its complexities

## Complex Output
- We could have a flag per estimator to set the output type


# Community Feedback
Is there a way to get useful community feedback?
Maybe we could understad their usecases!
We could focus on the space of challenges the user faces rather
than the solution they want.

# Moving Forward
We'd implement a prototype with xarray, to have a better feeling
about whether it's a suitable option for us or not.
