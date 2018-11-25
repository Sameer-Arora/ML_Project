# Numeric Computing (GPU)
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours


def compile(self, arguments, function):
    """Build a Theano function that will run the specified expression on the GPU.
    """
    return theano.function(list(arguments), function, on_unused_input='ignore')


def do_extract_patches(self, layers, size=3, stride=1):
    """This function builds a Theano expression that will get compiled an run on the GPU. It extracts 3x3 patches
    from the intermediate outputs in the model.
    """
    results = []
    for l, f in layers:
        # Use a Theano helper function to extract "neighbors" of specific size, seems a bit slower than doing
        # it manually but much simpler!
        patches = theano.tensor.nnet.neighbours.images2neibs(f, (size, size), (stride, stride), mode='valid')
        # Make sure the patches are in the shape required to insert them into the model as another layer.
        patches = patches.reshape((-1, patches.shape[0] // f.shape[1], size, size)).dimshuffle((1, 0, 2, 3))
        # Calculate the magnitude that we'll use for normalization at runtime, then store...
        results.extend([patches] + self.compute_norms(T, l, patches))
    return results


def do_match_patches(self, layer):
    # Use node in the model to compute the result of the normalized cross-correlation, using results from the
    # nearest-neighbor layers called 'nn3_1' and 'nn4_1'.
    dist = self.matcher_outputs[layer]
    dist = dist.reshape((dist.shape[1], -1))
    # Compute the score of each patch, taking into account statistics from previous iteration. This equalizes
    # the chances of the patches being selected when the user requests more variety.
    offset = self.matcher_history[layer].reshape((-1, 1))
    scores = (dist - offset * args.variety)
    # Pick the best style patches for each patch in the current image, the result is an array of indices.
    # Also return the maximum value along both axis, used to compare slices and add patch variety.
    return [scores.argmax(axis=0), scores.max(axis=0), dist.max(axis=1)]
