# sample net

tensorflow re-implementation of the architecture described
[here](https://arxiv.org/pdf/1612.07837v1.pdf) for generating
audio at a sample level.

At this stage I'm not sure how many of the experiments I'll replicate,
mostly I'm just curious about how feasible a solution this actually is.

The first challenge is actually figuring out what the architecture is.
The general idea seems to be:
- divide the input up into non-overlapping windows of size _FS_.
- call one of them _f_.
- use _f_ as the input to the first RNN. As Github markdown has no maths
  we will call the output at a particular timestep of this lowest
  frequency RNN "_a_".
- for as many layers as you can fit:
  - divide the current input window (beginning with _f_) into ever
    smaller non-overlapping windows.
  - upsample the RNN outputs (beginning with _a_) using a learned
    upsampling ("deconvolutional" style) to provide a vector for each
    higher frequency timestep.
  - feed a linear combination (in the paper the inputs are projected
    with learned weights and the output of the previous RNN is added
    on in full) into a new RNN, running at this new increased frequency.
- after doing a few of these, move to a sample-level module, as follows:
  - take as input discretised, embedded and flattened windows of size
    _FS_. Make sure now that these windows overlap by all but 1 sample,
    so we have one timestep per sample.
  - feed the whole thing, along with the upsampled output of the final
    RNN into an MLP.
  - the goal of the MLP is to output a distribution over the possible
    next sample.

So eventually, we get to the point where we are outputting a
distribution over a single audio sample. This is done by a softmax
over possible values. While this is nice, it pretty much limits us to
8 bit (as does the embedding lookup) seeing as the next step up in
terms of quality is 16 bit, and personally I'd rather avoid a softmax
over 2^16 values (or storing a 2^16 x anything embedding matrix).

In summary, this model is still very large. What is interesting about
it is the hierarchical RNNs and the autoregressive MLP to smooth
the output. This seems important for audio data where there really is
a huge difference in temporal scale between the lower and higher
levels of abstraction. This provides strong motivation to go classic
Bengio and use a hierarchical RNN over various time scales.

There also seems to be a slight bootstrapping issue when using the
model to generate sound -- at all stages a decent window of samples
is required, but you won't start off with one. You could start it off
with something and let it finish, which might be entertaining or
initialize with a window of *go* symbols and maybe a particular state?
This would require being careful during training to pad the sequences
appropriately (and maybe learn a (convolutional?) MLP to produce
states for one or more of the RNNs representing meta-info like
speakers/accents or style).

## extensions

This model admits a number of obvious extensions:
- the RNNs may not necessarily need to be fed audio
  - eg for text to speech, at least the top levels could receive the
    text/speaker info etc.
  - there are probably alignment issues to resolve, we may not want to
    just feed in a phoneme per frame, although with a big enough frame
    size at the top level and the learned upsampling we might get away
    with it.
- translate between sequences
  - the RNN inputs might not be the inputs to the autoregressive MLP.
  - not going to be able to do things like wave-to-wave language
    translation because you'll run into all sorts of alignment issues.
  - might be able to do things like transfer accents/voices which
    could be a laugh.
